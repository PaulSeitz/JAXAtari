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
}


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


def make_strategy(config, solution):
    alg_name = ALG_NAME_ALIASES.get(config.get("ALG_NAME", "Open_ES"), config.get("ALG_NAME", "Open_ES"))
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
    if learning_rate is not None:
        if alg_name == "PGPE":
            from evosax.core.optimizer import clipup

            kwargs["optimizer"] = clipup(learning_rate=learning_rate, max_velocity=0.02)
        elif alg_name == "ARS":
            kwargs["optimizer"] = optax.adam(learning_rate=learning_rate)
        else:
            kwargs["optimizer"] = optax.sgd(learning_rate=learning_rate)

    sigma_init = es_params_config.get("sigma_init")
    params_overrides = {
        key: value
        for key, value in es_params_config.items()
        if key not in ("learning_rate", "sigma_init")
    }
    if sigma_init is not None:
        if alg_name == "PGPE":
            params_overrides["std_init"] = sigma_init
        else:
            kwargs["std_schedule"] = optax.constant_schedule(sigma_init)

    strategy = strategy_class(**kwargs)
    es_params = strategy.default_params

    if params_overrides:
        es_params = es_params.replace(**params_overrides)

    return strategy, es_params


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
            f"max_return={float(max_return):.2f}, mean_return={float(mean_return):.2f}"
        )

    return checkpoint_callback


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
        strategy, es_params = make_strategy(config, solution)

        rng, init_rng = jax.random.split(rng)
        es_state = strategy.init(init_rng, solution, es_params)

        def rollout_episode(rng_input, network_params):
            def cond_fn(val):
                _, _, done, _ = val
                return ~done

            def step_fn(val):
                obs, state, _, cum_reward = val

                logits = network.apply({"params": network_params}, obs)
                action = jnp.argmax(logits, axis=-1)

                next_obs, next_state, reward, terminated, truncated, _ = env.step(state, action)
                done = jnp.logical_or(terminated, truncated)

                return next_obs, next_state, done, cum_reward + reward

            obs, state = env.reset(rng_input)
            init_val = (obs, state, False, 0.0)

            final_val = jax.lax.while_loop(cond_fn, step_fn, init_val)
            return final_val[3]

        vmap_rollout = jax.vmap(rollout_episode, in_axes=(0, 0))

        def generation_step(carry, _):
            es_state, rng = carry
            rng, ask_rng, eval_rng, tell_rng = jax.random.split(rng, 4)

            population, es_state = strategy.ask(ask_rng, es_state, es_params)

            rngs_eval = jax.random.split(eval_rng, config["POPSIZE"])
            fitness = vmap_rollout(rngs_eval, population)

            es_state, _ = strategy.tell(tell_rng, population, -fitness, es_state, es_params)

            metrics = {
                "max_return": jnp.max(fitness),
                "mean_return": jnp.mean(fitness),
            }

            def callback_fn(m):
                if config.get("WANDB_MODE") != "disabled":
                    wandb.log(m)
                if rtpt_instance is not None:
                    rtpt_instance.step()

            jax.debug.callback(callback_fn, metrics)
            if checkpoint_callback is not None:
                jax.debug.callback(
                    checkpoint_callback,
                    es_state.generation_counter,
                    es_state.best_solution,
                    metrics["max_return"],
                    metrics["mean_return"],
                )

            return (es_state, rng), metrics

        (es_state, rng), metrics = jax.lax.scan(
            generation_step,
            (es_state, rng),
            None,
            length=config["NUM_GENERATIONS"],
        )

        best_params = unravel_solution(es_state.best_solution)
        return {"es_state": es_state, "metrics": metrics, "best_params": best_params}

    return train


def single_run(config):
    wandb.init(
        entity=config.get("ENTITY"),
        project=config.get("PROJECT", "JAXAtari-Evo"),
        name=f"{config.get('ALG_NAME', 'ES')}_{config['ENV_NAME']}",
        config=config,
        mode=config.get("WANDB_MODE", "online"),
    )

    rtpt = RTPT(
        name_initials=config.get("RTPT_INITIALS", "XX"),
        experiment_name=config["ENV_NAME"],
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
