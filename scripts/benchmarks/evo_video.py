"""Generate and log evaluation videos for trained evo_agent policies."""

import os

import hydra
import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import numpy as np
import wandb
from omegaconf import OmegaConf

from evo_agent import build_policy, build_wrapped_env, unwrap_state_for_render
from train_utils import load_params


def _policy_obs(obs):
    obs = jnp.asarray(obs)
    if obs.ndim == 1:
        return obs[None, ...]
    return obs


def generate_evo_video(
    config,
    params,
    seed_idx=0,
    video_label="eval",
    generation=None,
    log_to_wandb=True,
    save_path=None,
):
    """Roll out a greedy policy and log an mp4 to wandb (PQN-style naming)."""
    env = build_wrapped_env(config)
    renderer = env.renderer
    network = build_policy(config, env)

    rng = jax.random.PRNGKey(config["SEED"] + seed_idx + 1000)
    rng, reset_rng = jax.random.split(rng)
    obs, env_state = env.reset(reset_rng)

    frames = []
    total_reward = 0.0
    max_steps = config.get("VIDEO_MAX_STEPS", 5000)

    for _ in range(max_steps):
        logits = network.apply({"params": params}, _policy_obs(obs))
        action = int(jnp.argmax(logits, axis=-1)[0])

        rng, step_rng = jax.random.split(rng)
        obs, env_state, reward, terminated, truncated, _ = env.step(env_state, action)
        done = bool(terminated or truncated)
        total_reward += float(reward)

        frame = renderer.render(unwrap_state_for_render(env_state))
        frames.append(np.array(frame, dtype=np.uint8))

        if done:
            break

    print(
        f"Final video ({video_label}): {len(frames)} frames, "
        f"total reward: {total_reward:.1f}"
    )

    if len(frames) == 0:
        return total_reward

    frames = np.stack(frames, axis=0)
    frames_for_wandb = np.transpose(frames, (0, 3, 1, 2))
    fps = config.get("VIDEO_FPS", 30)
    video = wandb.Video(frames_for_wandb, fps=fps, format="mp4")

    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        imageio.mimsave(save_path, frames, fps=fps)
        print(f"Video saved to {save_path}")

    if log_to_wandb and wandb.run is not None:
        log_payload = {
            f"final_video_seed{seed_idx}_{video_label}": video,
            f"final_return_seed{seed_idx}_{video_label}": total_reward,
        }
        if generation is not None:
            wandb.log(log_payload, step=int(generation))
        else:
            wandb.log(log_payload)
        print(f"Video '{video_label}' logged to wandb.")

    return total_reward


def single_run(config):
    params_path = config.get("PARAMS_PATH")
    if not params_path:
        raise ValueError("PARAMS_PATH is required for evo_video.py")

    wandb.init(
        entity=config.get("ENTITY"),
        project=config.get("PROJECT", "JAXAtari-Evo"),
        name=f"{config.get('ALG_NAME', 'ES')}_{config['ENV_NAME']}_video",
        config=config,
        mode=config.get("WANDB_MODE", "online"),
    )

    params = load_params(params_path)
    generate_evo_video(
        config,
        params,
        seed_idx=config.get("VIDEO_SEED_IDX", 0),
        video_label=config.get("VIDEO_LABEL", "eval"),
        generation=config.get("VIDEO_STEP"),
        save_path=config.get("VIDEO_SAVE_PATH"),
    )
    wandb.finish()


@hydra.main(version_base=None, config_path="./config/evo_config", config_name="evo_video")
def main(config):
    config_dict = OmegaConf.to_container(config, resolve=True)
    single_run(config_dict)


if __name__ == "__main__":
    main()
