"""Data-only soft-press control, constructed and executed on the physics worker."""
from __future__ import annotations


class PressController:
    def __init__(self, provider, trajectory):
        import torch

        self.provider = provider
        self.view = provider.create_robot_view(robot_name="press", base_link="press_head",
                joint_names=("press_slide",), link_names=("press_head",))
        control = provider.arrays["ctrl"]
        self.scales = torch.tensor(trajectory["depth_scales"], dtype=control.dtype,
                device=control.device).unsqueeze(1)
        self.command = torch.zeros_like(self.scales)
        self.trajectory = trajectory
        self.tick = 0
        self.view.set_position_targets(self.command)

    def step(self, dt):
        settings = self.trajectory
        tick = self.tick - settings["settle_ticks"]
        if tick < 0:
            progress = 0.
        elif tick < settings["press_ticks"]:
            value = (tick + 1) / settings["press_ticks"]
            progress = value * value * (3 - 2 * value)
        elif tick < settings["press_ticks"] + settings["hold_ticks"]:
            progress = 1.
        else:
            value = min(1., (tick - settings["press_ticks"] - settings["hold_ticks"] + 1) / settings["release_ticks"])
            progress = 1 - value * value * (3 - 2 * value)
        self.command.copy_(self.scales).mul_(-settings["press_depth"] * progress)
        self.view.set_position_targets(self.command)
        self.tick += 1

    def reset(self, environments):
        if set(environments) != set(range(self.provider.num_envs)):
            raise ValueError("shared soft-press cycle resets the full batch")
        self.tick = 0
        self.command.zero_()
        self.view.set_position_targets(self.command)

    def apply_commands(self, commands):
        if commands:
            raise ValueError("the soft-press controller has no external command fields")


def create(artifact, solver_config, num_envs, environments_per_shard, trajectory):
    import torch
    from gobot.ipc import LibuipcBatchConfig, LibuipcConfig
    from gobot.sim import RuntimeComponents, SceneStateOutput
    from gobot.sim.providers import CompiledMuJoCoIpcArtifact, MuJoCoIpcConfig, MuJoCoIpcProvider

    compiled = CompiledMuJoCoIpcArtifact.from_mapping(artifact)
    options = dict(solver_config)
    options["solver"] = LibuipcConfig(**options["solver"])
    provider = MuJoCoIpcProvider(compiled,
            config=MuJoCoIpcConfig(num_envs=num_envs, device="cuda:0",
                    environments_per_shard=environments_per_shard, capture_mujoco_graphs=True),
            libuipc_config=LibuipcBatchConfig(**options),
            mujoco_options={"nconmax": 32, "njmax": 64, "overflow_check_interval": 0})
    try:
        provider.reset(torch.ones(num_envs, dtype=torch.bool, device=provider.arrays["ctrl"].device))
        controller = PressController(provider, trajectory)
        output = SceneStateOutput(provider,
                robots=({"field": "robot.press.link_pose", "robot_name": "press",
                         "base_link": "press_head", "joint_names": ("press_slide",),
                         "link_names": ("press_head",)},),
                deformables=compiled.ipc.deformable_bodies,
                source_bodies=provider.ipc_solver.deformable_bodies,
                positions_field="ipc_positions", refresh_positions=provider.refresh_state)
        return RuntimeComponents(provider, controller, output)
    except Exception:
        provider.close()
        raise
