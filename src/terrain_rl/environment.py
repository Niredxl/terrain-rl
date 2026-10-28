"""Gymnasium environment for terrain navigation with optional PyBullet collisions."""

from __future__ import annotations

from typing import Literal

import numpy as np
from gymnasium import Env, spaces

from .generators import (
    CellularAutomataTerrainGenerator,
    DiffusionTerrainGenerator,
    GANTerrainGenerator,
    PerlinTerrainGenerator,
)
from .terrain import HeightmapTerrain, TerrainOutput

GeneratorName = Literal["perlin", "cellular", "gan", "diffusion"]


class TerrainEnv(Env[dict[str, np.ndarray], np.ndarray]):
    """A small continuous-control task over a generated heightfield."""

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        generator: GeneratorName | HeightmapTerrain = "perlin",
        resolution: int = 32,
        size: float = 10.0,
        render_mode: str | None = None,
        physics_backend: Literal["auto", "pybullet", "none"] = "auto",
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.resolution, self.size = resolution, size
        self.generator = self._make_generator(generator)
        self.observation_space = spaces.Dict(
            {
                "heightmap": spaces.Box(-1.0, 1.0, (resolution, resolution), dtype=np.float32),
                "state": spaces.Box(np.array([-1.0, -1.0], dtype=np.float32), np.array([1.0, 1.0], dtype=np.float32)),
            }
        )
        self.action_space = spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)
        self.physics_backend = physics_backend
        self._client: int | None = None
        self._terrain_body: int | None = None
        self.terrain: TerrainOutput | None = None
        self.position = np.zeros(2, dtype=np.float32)
        self.target = np.array([0.8, 0.8], dtype=np.float32)

    def _make_generator(self, generator: GeneratorName | HeightmapTerrain) -> HeightmapTerrain:
        if not isinstance(generator, str):
            return generator
        classes = {
            "perlin": PerlinTerrainGenerator,
            "cellular": CellularAutomataTerrainGenerator,
            "gan": GANTerrainGenerator,
            "diffusion": DiffusionTerrainGenerator,
        }
        if generator not in classes:
            raise ValueError(f"unknown terrain generator: {generator}")
        return classes[generator](self.resolution, self.size)

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[dict[str, np.ndarray], dict]:
        super().reset(seed=seed)
        self.terrain = self.generator.generate(seed)
        self.position = np.array([-0.8, -0.8], dtype=np.float32)
        self._create_physics()
        return self._observation(), {"terrain": self.terrain}

    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], float, bool, bool, dict]:
        action = np.asarray(action, dtype=np.float32).clip(-1, 1)
        previous_distance = float(np.linalg.norm(self.position - self.target))
        self.position = np.clip(self.position + action * 0.08, -1, 1)
        distance = float(np.linalg.norm(self.position - self.target))
        reward = previous_distance - distance - 0.01 * float(np.linalg.norm(action))
        terminated = distance < 0.12
        return self._observation(), reward, terminated, False, {"distance": distance}

    def _observation(self) -> dict[str, np.ndarray]:
        if self.terrain is None:
            raise RuntimeError("reset must be called before requesting an observation")
        return {"heightmap": self.terrain.elevation, "state": self.position.copy()}

    def _create_physics(self) -> None:
        if self.physics_backend == "none":
            return
        try:
            import pybullet as p
        except ImportError:
            if self.physics_backend == "pybullet":
                raise ImportError("pybullet is required for physics_backend='pybullet'")
            return
        if self._client is None:
            self._client = p.connect(p.GUI if self.render_mode == "human" else p.DIRECT)
        if self._terrain_body is not None:
            p.removeBody(self._terrain_body, physicsClientId=self._client)
        assert self.terrain is not None
        shape = p.createCollisionShape(
            p.GEOM_HEIGHTFIELD,
            meshScale=[self.size / (self.resolution - 1)] * 2 + [1.0],
            heightfieldData=self.terrain.elevation.flatten().tolist(),
            numHeightfieldRows=self.resolution,
            numHeightfieldColumns=self.resolution,
            physicsClientId=self._client,
        )
        self._terrain_body = p.createMultiBody(0, shape, physicsClientId=self._client)

    def render(self) -> None:
        if self.render_mode == "human" and self._client is not None:
            import pybullet as p

            p.stepSimulation(physicsClientId=self._client)

    def close(self) -> None:
        if self._client is not None:
            import pybullet as p

            p.disconnect(self._client)
            self._client = None
