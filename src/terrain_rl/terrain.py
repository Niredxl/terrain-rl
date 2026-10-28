"""Shared terrain data structures and generator interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TerrainOutput:
    """A terrain heightfield and derived maps, all shaped ``(height, width)``."""

    elevation: np.ndarray
    slope: np.ndarray
    roughness: np.ndarray

    def __post_init__(self) -> None:
        shapes = {self.elevation.shape, self.slope.shape, self.roughness.shape}
        if len(shapes) != 1 or self.elevation.ndim != 2:
            raise ValueError("terrain maps must be equally shaped two-dimensional arrays")


class HeightmapTerrain(ABC):
    """Base interface shared by procedural and learned heightmap generators."""

    def __init__(self, resolution: int = 64, size: float = 10.0, seed: int | None = None) -> None:
        if resolution < 4:
            raise ValueError("resolution must be at least 4")
        self.resolution = resolution
        self.size = float(size)
        self.seed = seed

    @abstractmethod
    def generate(self, seed: int | None = None) -> TerrainOutput:
        """Generate one terrain without changing model parameters."""

    def __call__(self, seed: int | None = None) -> TerrainOutput:
        return self.generate(seed)

    def _rng(self, seed: int | None) -> np.random.Generator:
        return np.random.default_rng(self.seed if seed is None else seed)

    def _derived_maps(self, elevation: np.ndarray) -> TerrainOutput:
        spacing = self.size / max(self.resolution - 1, 1)
        dy, dx = np.gradient(elevation, spacing, spacing)
        slope = np.sqrt(dx * dx + dy * dy).astype(np.float32)
        roughness = np.abs(np.gradient(dx, axis=1)) + np.abs(np.gradient(dy, axis=0))
        return TerrainOutput(
            elevation=np.asarray(elevation, dtype=np.float32),
            slope=slope,
            roughness=np.asarray(roughness, dtype=np.float32),
        )

    @staticmethod
    def as_numpy(value: Any) -> np.ndarray:
        """Convert a tensor-like map to a detached CPU NumPy array."""
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=np.float32)
