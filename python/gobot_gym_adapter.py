import math
import random


class GobotBox:
    def __init__(self, low, high, names=None, units=None):
        self.low = [float(value) for value in low]
        self.high = [float(value) for value in high]
        self.shape = (len(self.low),)
        self.names = list(names or [])
        self.units = list(units or [])

    def sample(self):
        values = []
        for lower, upper in zip(self.low, self.high):
            if math.isfinite(lower) and math.isfinite(upper):
                values.append(random.uniform(lower, upper))
            else:
                values.append(0.0)
        return values


def space_from_spec(spec):
    names = list(spec.get("names", []))
    lower_bounds = list(spec.get("lower_bounds", []))
    upper_bounds = list(spec.get("upper_bounds", []))
    units = list(spec.get("units", []))
    try:
        import numpy as np
        from gymnasium import spaces

        return spaces.Box(
            low=np.asarray(lower_bounds, dtype=np.float32),
            high=np.asarray(upper_bounds, dtype=np.float32),
            dtype=np.float32,
        )
    except Exception:
        return GobotBox(lower_bounds, upper_bounds, names=names, units=units)


class GobotGymEnv:
    def __init__(self, scene_path="", robot="robot", backend="null", env=None):
        import gobot

        self.env = env if env is not None else gobot.RLEnvironment(scene_path, robot=robot, backend=backend)
        self._refresh_spaces()

    def _refresh_spaces(self):
        self.observation_spec = self.env.get_observation_spec()
        self.action_spec = self.env.get_action_spec()
        self.observation_space = space_from_spec(self.observation_spec)
        self.action_space = space_from_spec(self.action_spec)

    def reset(self, seed=None, options=None):
        seed_value = 0 if seed is None else int(seed)
        observation, info = self.env.reset(seed=seed_value)
        self._refresh_spaces()
        return observation, info

    def step(self, action):
        return self.env.step(action)

    def close(self):
        pass
