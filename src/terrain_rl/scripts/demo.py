"""Plot all four terrain generators and render one PPO episode."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

from terrain_rl.environment import TerrainEnv


def main() -> None:
    names = ["perlin", "cellular", "gan", "diffusion"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    for axis, name in zip(axes.flat, names):
        terrain = TerrainEnv(generator=name, resolution=64, physics_backend="none").generator.generate(seed=7)
        axis.imshow(terrain.elevation, cmap="terrain")
        axis.set_title(name)
        axis.axis("off")
    fig.savefig("terrain-generators.png", dpi=150)

    env = TerrainEnv(generator="perlin", render_mode="human")
    model = PPO.load("artifacts/ppo-terrain", env=env)
    observation, _ = env.reset(seed=7)
    for _ in range(500):
        action, _ = model.predict(observation, deterministic=True)
        observation, _, terminated, truncated, _ = env.step(action)
        env.render()
        if terminated or truncated:
            break
    env.close()


if __name__ == "__main__":
    main()
