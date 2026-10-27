"""Import `gym` robustly. Old `gym` breaks on NumPy>=2 (np.bool8) and is unmaintained; the environments only
need `Env` and `spaces.Box`, so fall back to gymnasium or a tiny shim instead of failing."""
import numpy as np

try:                                   # preferred, maintained
    import gymnasium as gym
    from gymnasium import spaces
except Exception:
    try:
        import gym
        from gym import spaces
    except Exception:                  # minimal shim (enough for these environments)
        class _Box:
            def __init__(self, low=None, high=None, shape=None, dtype=float):
                self.shape = tuple(shape)
                self.low = np.full(self.shape, low, dtype=float)
                self.high = np.full(self.shape, high, dtype=float)
                self.dtype = dtype

        class spaces:                  # noqa: N801
            Box = _Box

        class gym:                     # noqa: N801
            Env = object
            spaces = spaces
