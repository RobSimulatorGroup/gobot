class GobotGymEnv:
    def __init__(self, scene_path="", robot="robot", backend="null", env=None):
        import gobot

        self.env = env if env is not None else gobot.RLEnvironment(scene_path, robot=robot, backend=backend)
        self.observation_space = self.env.get_observation_spec()
        self.action_space = self.env.get_action_spec()

    def reset(self, seed=None, options=None):
        seed_value = 0 if seed is None else int(seed)
        observation, info = self.env.reset(seed=seed_value)
        return observation, info

    def step(self, action):
        return self.env.step(action)

    def close(self):
        pass
