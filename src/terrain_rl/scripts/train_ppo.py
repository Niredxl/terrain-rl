"""Train the baseline PPO policy for 100,000 environment steps."""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from wandb.integration.sb3 import WandbCallback

from terrain_rl.environment import TerrainEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator", choices=["perlin", "cellular", "gan", "diffusion"], default="perlin")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--project", default="terrain-rl")
    args = parser.parse_args()
    import wandb

    run = wandb.init(project=args.project, config={"generator": args.generator, "timesteps": args.timesteps})
    env = make_vec_env(lambda: TerrainEnv(generator=args.generator, physics_backend="none"), n_envs=1)
    Path("artifacts").mkdir(exist_ok=True)
    model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=f"runs/{run.id}")
    model.learn(
        total_timesteps=args.timesteps,
        callback=WandbCallback(gradient_save_freq=0, model_save_path="artifacts", verbose=2),
    )
    model.save("artifacts/ppo-terrain")
    run.finish()


if __name__ == "__main__":
    main()
