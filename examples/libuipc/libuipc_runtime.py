"""Native libuipc factory and FR3 controller; no scene objects cross the worker boundary."""
from __future__ import annotations


class GraspController:
    def __init__(self, provider):
        from libuipc_demo import FR3_JOINT_NAMES

        self.provider = provider
        self.joints = {}
        for name in FR3_JOINT_NAMES:
            matches = [entry["path"] for entry in provider.joints if entry["path"].endswith("/" + name)]
            if len(matches) != 1:
                raise ValueError(f"FR3 grasp demo has no unique {name} joint")
            self.joints[name] = matches[0]
        self.reset((0,))

    def _set_targets(self):
        from libuipc_demo import _fr3_motion_targets

        for name, target in _fr3_motion_targets(self.time).items():
            self.provider.set_joint_target(self.joints[name], target)

    def step(self, dt):
        self._set_targets()
        self.time += dt

    def reset(self, environments):
        if tuple(environments) != (0,):
            raise ValueError("FR3 demo resets its single environment")
        self.time = 0.
        self._set_targets()

    def apply_commands(self, commands):
        if commands:
            raise ValueError("FR3 demo controller does not accept external commands")


def create(artifact, config, affine_paths):
    from gobot.ipc import LibuipcConfig, LibuipcProvider, LibuipcSceneOutput
    from gobot.sim import RuntimeComponents

    provider = LibuipcProvider(artifact, config=LibuipcConfig(**config))
    try:
        return RuntimeComponents(provider, GraspController(provider),
                LibuipcSceneOutput(provider, affine_paths=affine_paths))
    except Exception:
        provider.close()
        raise
