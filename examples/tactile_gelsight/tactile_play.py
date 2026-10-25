"""GelSight Mini editor Play entry point. Physics lives on the simulation worker."""

from __future__ import annotations
import gobot
from tactile_runtime import FIXED_DT, FIELDS
from press_geometry import PRESS_HEIGHT


class Script(gobot.NodeScript):
    def _ready(self):
        self.runtime = None
        self.session = None
        root = self.get_root()
        self.sensor = next(n for n in root.children if n.type_name == "TactileSensor3D")
        self.press = next(n for n in root.children if n.type_name == "RigidBody3D")
        artifact = self.context.compile_ipc_scene_artifact()
        from gobot.ipc import CompiledIpcSceneArtifact

        compiled = CompiledIpcSceneArtifact.from_mapping(artifact)
        sensor_path = compiled.tactile_sensors[0]["path"]
        affine = [
            link["path"]
            for robot in compiled.robots
            for link in robot["links"]
            if link["collision_shapes"]
        ]
        if len(affine) != 1:
            raise ValueError("the tactile bench must contain one press head")
        self.sync = gobot.sim.SceneSnapshotSync(
            self.context, links={"affine.link_pose": [[self.press]]}
        )
        config = dict(
            fixed_time_step=FIXED_DT,
            gravity=(0.0, 0.0, 0.0),
            friction_coefficient=0.5,
            contact_activation_distance=0.0005,
            contact_resistance=1e10,
            affine_stiffness=1e8,
            kinematic_strength=10000.0,
        )
        spec = gobot.sim.SimulationRuntimeSpec(
            "tactile_runtime:create",
            dict(
                artifact=artifact,
                config=config,
                sensor_path=sensor_path,
                press_path=affine[0],
            ),
        )
        try:
            self.runtime = gobot.sim.AsyncSimulationSession(
                spec,
                fields=("affine.link_pose",) + tuple("tactile." + f for f in FIELDS),
                max_hz=30.0,
            )
            self.session = gobot.sim.ProviderPlaySession(
                self.context,
                self.runtime,
                fixed_dt=FIXED_DT,
                max_sub_steps=1,
                before_step=self._commands,
                sync_scene=self._sync,
            ).start()
            self.context.register_runtime_pose_target(
                self.press, (-0.03, -0.03, PRESS_HEIGHT), (0.03, 0.03, 0.03)
            )
            self.session.set_status("GelSight Mini — press, slide, release")
            print(
                "Select PressHead (the torus) and drag the Move gizmo for manual contact. Reset restores the automatic cycle."
            )
        except Exception:
            self._exit_tree()
            raise

    def _commands(self, dt):
        target = self.context.get_runtime_pose_target(self.press)
        return {} if target is None else {"press_target": target}

    def _sync(self, snapshot):
        import numpy as np

        self.sync(snapshot)
        frame = {
            field: np.asarray(snapshot.buffer("tactile." + field))[0]
            for field in FIELDS
        }
        self.sensor._apply_runtime_frame(frame)
        if int(frame["clock"][0]) == 0:
            self.context.clear_runtime_pose_target(self.press)

    def _exit_tree(self):
        if getattr(self, "session", None) is not None:
            self.session.close()
            self.session = None
        elif getattr(self, "runtime", None) is not None:
            self.runtime.close()
        self.runtime = None
