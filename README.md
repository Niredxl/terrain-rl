# terrain-rl

Phase 2 provides four heightmap generators (`perlin`, `cellular`, `gan`, and
`diffusion`), a typed Gymnasium environment, and a PPO baseline. The GAN and
diffusion implementations are inference-only architectures; training loops are
intentionally out of scope.

```python
from terrain_rl import TerrainEnv

env = TerrainEnv(generator="diffusion", physics_backend="none")
observation, info = env.reset(seed=0)
```

Install with `pip install -e .`, train with `terrain-ppo`, and create the
2x2 generator plot plus a rendered episode with `terrain-demo`.
