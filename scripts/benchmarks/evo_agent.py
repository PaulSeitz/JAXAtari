import os
import time

import jax
import jax.numpy as jnp
import flax.linen as nn
import hydra
import optax
from omegaconf import OmegaConf
import wandb
from rtpt import RTPT

from evosax.algorithms import algorithms
from evosax.algorithms.population_based import population_based_algorithms
from train_utils import save_params, save_params_atomic

import jaxatari
from jaxatari.wrappers import (
    AtariWrapper,
    ObjectCentricWrapper,
    FlattenObservationWrapper,
    NormalizeObservationWrapper,
)

ALG_NAME_ALIASES = {
    "OpenAIES": "Open_ES",
    "OpenES": "Open_ES",
    "SuchGA": "SimpleGA",
    "DeepGA": "SimpleGA",
}


def resolve_alg_name(config):
    return ALG_NAME_ALIASES.get(
        config.get("ALG_NAME", "Open_ES"),
        config.get("ALG_NAME", "Open_ES"),
    )


def is_population_based(alg_name):
    return alg_name in population_based_algorithms


def make_strategy(config, solution):
    alg_name = resolve_alg_name(config)
    if alg_name not in algorithms:
        raise ValueError(
            f"Unknown algorithm '{alg_name}'. Available: {sorted(algorithms.keys())}"
        )

    strategy_class = algorithms[alg_name]
    es_params_config = config.get("ES_PARAMS", {})

    kwargs = {
        "population_size": config["POPSIZE"],
        "solution": solution,
    }

    learning_rate = es_params_config.get("learning_rate")
    if learning_rate is not None and alg_name == "PGPE":
        from evosax.core.optimizer import clipup

        kwargs["optimizer"] = clipup(learning_rate=learning_rate, max_velocity=0.02)
    elif learning_rate is not None and alg_name == "ARS":
        kwargs["optimizer"] = optax.adam(learning_rate=learning_rate)
    elif learning_rate is not None:
        kwargs["optimizer"] = optax.sgd(learning_rate=learning_rate)

    sigma_init = es_params_config.get("sigma_init")
    params_overrides = {
        key: value
        for key, value in es_params_config.items()
        if key not in ("learning_rate", "sigma_init", "crossover_rate")
    }
    if sigma_init is not None:
        if alg_name == "PGPE":
            params_overrides["std_init"] = sigma_init
        else:
            kwargs["std_schedule"] = optax.constant_schedule(sigma_init)

    strategy = strategy_class(**kwargs)

    if alg_name == "SimpleGA":
        num_elite_parents = config.get("NUM_ELITE_PARENTS", 10)
        strategy.elite_ratio = num_elite_parents / config["POPSIZE"]
        if "crossover_rate" in es_params_config:
            params_overrides["crossover_rate"] = es_params_config["crossover_rate"]
        else:
            params_overrides.setdefault("crossover_rate", 0.0)

    es_params = strategy.default_params

    if params_overrides:
        es_params = es_params.replace(**params_overrides)

    return strategy, es_params, alg_name


def apply_elitism(population, best_solution_flat, unravel_solution, generation_counter):
    """Copy the best-ever solution into slot 0 without mutation (Such et al.)."""
    elite = unravel_solution(best_solution_flat)

    def _replace(pop, el):
        return jax.tree.map(lambda leaf, elite_leaf: leaf.at[0].set(elite_leaf), pop, el)

    return jax.lax.cond(generation_counter > 0, lambda: _replace(population, elite), lambda: population)


class PolicyNetwork(nn.Module):
    action_dim: int
    hidden_size: int = 128
    num_layers: int = 2

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        for _ in range(self.num_layers):
            x = nn.Dense(self.hidden_size)(x)
            x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        return x


def build_wrapped_env(config, mods=None):
    env_name = config["ENV_NAME"].lower()
    env = jaxatari.make(env_name, mods=mods)
    env = AtariWrapper(
        env,
        episodic_life=config.get("EPISODIC_LIFE", True),
        first_fire=config.get("FIRST_FIRE", True),
        noop_max=config.get("NOOP_MAX", 30),
        max_frames_per_episode=config.get("MAX_FRAMES_PER_EPISODE", 108_000),
    )
    env = ObjectCentricWrapper(
        env,
        frame_stack_size=config.get("FRAME_STACK_SIZE", 4),
        frame_skip=config.get("FRAME_SKIP", 4),
        clip_reward=config.get("CLIP_REWARD", True),
    )
    env = NormalizeObservationWrapper(env)
    env = FlattenObservationWrapper(env)
    return env


def build_policy(config, env):
    return PolicyNetwork(
        action_dim=env.action_space().n,
        hidden_size=config.get("HIDDEN_SIZE", 128),
        num_layers=config.get("NUM_LAYERS", 2),
    )


def unwrap_state_for_render(env_state):
    state_for_render = env_state
    while hasattr(state_for_render, "atari_state"):
        state_for_render = state_for_render.atari_state
    if hasattr(state_for_render, "env_state"):
        state_for_render = state_for_render.env_state
    return state_for_render


def prepare_run_save_dir(config):
    save_root = config.get("SAVE_PATH")
    if not save_root:
        return None

    env_name = config["ENV_NAME"].lower()
    alg_name = config.get("ALG_NAME", "ES").lower()
    save_dir = os.path.join(
        save_root,
        env_name,
        time.strftime("%Y-%m-%d_%H-%M-%S"),
    )
    os.makedirs(save_dir, exist_ok=True)
    OmegaConf.save(
        config,
        os.path.join(save_dir, f"{alg_name}_{env_name}_config.yaml"),
    )
    print(f"Saving checkpoints to {save_dir}")
    return save_dir


def make_checkpoint_callback(save_dir, unravel_solution, checkpoint_every, keep_checkpoints):
    def checkpoint_callback(generation, best_solution_flat, max_return, mean_return):
        gen = int(generation)
        is_periodic = checkpoint_every > 0 and gen % checkpoint_every == 0
        is_first = gen == 1
        if not (is_periodic or is_first):
            return

        params = unravel_solution(jax.device_get(best_solution_flat))
        save_params_atomic(params, os.path.join(save_dir, "latest.safetensors"))
        if keep_checkpoints:
            save_params_atomic(
                params,
                os.path.join(save_dir, f"checkpoint_gen{gen:05d}.safetensors"),
            )
        print(
            f"Checkpoint generation {gen}: "
            f"return/max={float(max_return):.2f}, return/mean={float(mean_return):.2f}"
        )

    return checkpoint_callback


def build_generation_metrics(
    fitness,
    episode_steps,
    es_state,
    es_alg_metrics,
    config,
    total_env_steps,
    alg_name,
    strategy,
):
    """Scalar metrics for one ES generation. Wandb x-axis = generation (not env steps)."""
    popsize = config["POPSIZE"]
    frame_skip = config.get("FRAME_SKIP", 4)
    env_steps_this_gen = jnp.sum(episode_steps)
    total_env_steps = total_env_steps + env_steps_this_gen

    metrics = {
        # Primary x-axis for wandb is generation (see callback_fn below).
        "generation": es_state.generation_counter,
        # Episode return statistics (higher is better).
        "return/max": jnp.max(fitness),
        "return/mean": jnp.mean(fitness),
        "return/min": jnp.min(fitness),
        "return/std": jnp.std(fitness),
        "return/median": jnp.median(fitness),
        "return/best_ever": -es_state.best_fitness,
        "return/frac_positive": jnp.mean(fitness > 0),
        "return/frac_nonzero": jnp.mean(fitness != 0),
        # Environment interaction budget (compare to PQN's env_step axis).
        "env_steps/episode_mean": jnp.mean(episode_steps),
        "env_steps/episode_max": jnp.max(episode_steps),
        "env_steps/generation": env_steps_this_gen,
        "env_steps/game_frames_generation": env_steps_this_gen * frame_skip,
        "env_steps/total": total_env_steps,
        # ES / evosax internal state (fitness is minimized inside evosax).
        "es/best_fitness": es_state.best_fitness,
        "es/best_solution_norm": es_alg_metrics["best_solution_norm"],
        "es/best_fitness_in_generation": es_alg_metrics["best_fitness_in_generation"],
        "es/mean_norm": es_alg_metrics["mean_norm"] if "mean_norm" in es_alg_metrics else jnp.nan,
        "pop/size": popsize,
    }

    if hasattr(es_state, "std"):
        metrics["es/std_mean"] = jnp.mean(es_state.std)
        metrics["es/std_max"] = jnp.max(es_state.std)

    if alg_name == "SimpleGA":
        metrics["ga/num_elite_parents"] = strategy.num_elites
        metrics["ga/mutation_std"] = es_state.std
        metrics["ga/crossover_rate"] = config.get("ES_PARAMS", {}).get("crossover_rate", 0.0)

    return metrics, total_env_steps


def make_train(config, rtpt_instance, save_dir=None):
    env = build_wrapped_env(config)
    network = build_policy(config, env)

    dummy_obs = jnp.zeros((1, *env.observation_space().shape))
    dummy_key = jax.random.PRNGKey(0)
    policy_params = network.init(dummy_key, dummy_obs)
    solution = policy_params["params"]
    _, unravel_solution = jax.flatten_util.ravel_pytree(solution)
    checkpoint_every = config.get("CHECKPOINT_EVERY", 500)
    keep_checkpoints = config.get("KEEP_CHECKPOINTS", False)
    checkpoint_callback = None
    if save_dir is not None:
        checkpoint_callback = make_checkpoint_callback(
            save_dir,
            unravel_solution,
            checkpoint_every,
            keep_checkpoints,
        )

    def train(rng):
        strategy, es_params, alg_name = make_strategy(config, solution)
        use_elitism = config.get("ELITISM", False) and alg_name == "SimpleGA"
        pop_based = is_population_based(alg_name)

        def rollout_episode(rng_input, network_params):
            def cond_fn(val):
                _, _, done, _, _ = val
                return ~done

            def step_fn(val):
                obs, state, _, cum_reward, n_steps = val

                logits = network.apply({"params": network_params}, obs)
                action = jnp.argmax(logits, axis=-1)

                next_obs, next_state, reward, terminated, truncated, _ = env.step(state, action)
                done = jnp.logical_or(terminated, truncated)

                return next_obs, next_state, done, cum_reward + reward, n_steps + 1

            obs, state = env.reset(rng_input)
            init_val = (obs, state, False, 0.0, 0)

            final_val = jax.lax.while_loop(cond_fn, step_fn, init_val)
            return final_val[3], final_val[4]

        vmap_rollout = jax.vmap(rollout_episode, in_axes=(0, 0))

        population_init = jax.tree.map(
            lambda x: jnp.repeat(x[None, ...], config["POPSIZE"], axis=0),
            solution,
        )

        rng, init_rng, eval_rng = jax.random.split(rng, 3)
        if pop_based:
            rngs_eval = jax.random.split(eval_rng, config["POPSIZE"])
            fitness_init, _ = vmap_rollout(rngs_eval, population_init)
            es_state = strategy.init(init_rng, population_init, -fitness_init, es_params)
        else:
            es_state = strategy.init(init_rng, solution, es_params)

        def generation_step(carry, _):
            es_state, rng, total_env_steps = carry
            rng, ask_rng, eval_rng, tell_rng = jax.random.split(rng, 4)

            population, es_state = strategy.ask(ask_rng, es_state, es_params)
            if use_elitism:
                population = apply_elitism(
                    population,
                    es_state.best_solution,
                    unravel_solution,
                    es_state.generation_counter,
                )

            rngs_eval = jax.random.split(eval_rng, config["POPSIZE"])
            fitness, episode_steps = vmap_rollout(rngs_eval, population)

            es_state, es_alg_metrics = strategy.tell(
                tell_rng, population, -fitness, es_state, es_params
            )

            metrics, total_env_steps = build_generation_metrics(
                fitness,
                episode_steps,
                es_state,
                es_alg_metrics,
                config,
                total_env_steps,
                alg_name,
                strategy,
            )
            generation = metrics["generation"]
            log_every = config.get("WANDB_LOG_EVERY", 1)

            def callback_fn(m, gen, log_interval):
                if config.get("WANDB_MODE") != "disabled" and (
                    int(gen) % int(log_interval) == 0 or int(gen) == 1
                ):
                    # x-axis is ES generation, not env steps (see env_steps/* metrics).
                    host_metrics = {k: float(v) for k, v in m.items()}
                    wandb.log(host_metrics, step=int(gen))
                if rtpt_instance is not None:
                    rtpt_instance.step()

            jax.debug.callback(callback_fn, metrics, generation, log_every)
            if checkpoint_callback is not None:
                jax.debug.callback(
                    checkpoint_callback,
                    es_state.generation_counter,
                    es_state.best_solution,
                    metrics["return/max"],
                    metrics["return/mean"],
                )

            return (es_state, rng, total_env_steps), metrics

        (es_state, rng, _), metrics = jax.lax.scan(
            generation_step,
            (es_state, rng, jnp.array(0, dtype=jnp.int32)),
            None,
            length=config["NUM_GENERATIONS"],
        )

        best_params = unravel_solution(es_state.best_solution)
        return {"es_state": es_state, "metrics": metrics, "best_params": best_params}

    return train


def rtpt_experiment_name(config):
    alg = resolve_alg_name(config).lower()
    env = config["ENV_NAME"].lower()
    return f"evo_{alg}_{env}"


def single_run(config):
    wandb.init(
        entity=config.get("ENTITY"),
        project=config.get("PROJECT", "JAXAtari-Evo"),
        name=f"{config.get('ALG_NAME', 'ES')}_{config['ENV_NAME']}",
        config=config,
        mode=config.get("WANDB_MODE", "online"),
    )
    wandb.define_metric("generation")
    wandb.define_metric("*", step_metric="generation")

    rtpt = RTPT(
        name_initials=config.get("RTPT_INITIALS", "XX"),
        experiment_name=rtpt_experiment_name(config),
        max_iterations=config["NUM_GENERATIONS"],
    )
    rtpt.start()

    save_dir = prepare_run_save_dir(config)
    rng = jax.random.PRNGKey(config.get("SEED", 42))

    print(f"Compiling and running {config.get('ALG_NAME')} on {config['ENV_NAME']}...")

    train_jit = jax.jit(make_train(config, rtpt, save_dir))
    outs = jax.block_until_ready(train_jit(rng))
    print("Training complete.")

    best_params = jax.device_get(outs["best_params"])
    if save_dir is not None:
        env_name = config["ENV_NAME"].lower()
        alg_name = config.get("ALG_NAME", "ES").lower()
        final_path = os.path.join(
            save_dir,
            f'{alg_name}_{env_name}_seed{config["SEED"]}_final.safetensors',
        )
        save_params_atomic(best_params, final_path)
        save_params_atomic(best_params, os.path.join(save_dir, "latest.safetensors"))
        print(f"Final policy saved to {final_path}")

    if config.get("RECORD_FINAL_VIDEO", False):
        from evo_video import generate_evo_video

        generate_evo_video(
            config,
            best_params,
            seed_idx=0,
            video_label="train",
            generation=config["NUM_GENERATIONS"],
        )

    wandb.finish()


@hydra.main(version_base=None, config_path="./config/evo_config", config_name="evo_config")
def main(config):
    config_dict = OmegaConf.to_container(config, resolve=True)
    single_run(config_dict)


if __name__ == "__main__":
    main()
