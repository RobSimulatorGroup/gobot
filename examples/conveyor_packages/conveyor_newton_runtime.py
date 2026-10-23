"""Newton workcell recipe shared by editor Play and ordinary Python runs."""
from __future__ import annotations

from dataclasses import asdict
import json
import numpy as np

from conveyor_config import (HAND_JOINT_NAMES_BY_SIDE, LEAP_ROBOT_NAMES,
                             BELT_ROBOT_NAME, BELT_LINK_NAME, BELT_SURFACE_LENGTH)
import conveyor_profile as original

# Opposed index/thumb pads approach the raised front weld from the side.
# Fit the pad faces, not their closest corners: the lower pad supports the
# seam while the upper pad closes. Palm collision geometry clears the table.
PINCH_STAGE = (
    (-0.01055048769, -0.744002220028, -0.33298020614, 1.355710999671, 0.0, 0.0),
    (-0.08193926788, -0.70187316304, -0.426581114803, 1.348228796269, 0.0, 0.0),
)
PINCH_FINGERS = (
    (2.026777561894, -0.000808592877, -0.279833058501, -0.365955878316) + (0.,) * 8
    + (1.485165189231, 0.018687257831, 0.42869568056, -0.220284584032),
    (2.037156758796, 0.000416534464, -0.30565407384, -0.365778638331) + (0.,) * 8
    + (1.498833574158, -0.016276476395, 0.435012850046, -0.216927887423),
)
OPEN_FINGERS = (
    (1.977401356679, -0.000671767424, -0.505999, -0.365999) + (0.,) * 8
    + (1.485132624691, 0.018408573774, 0.294118249038, -0.296343210223),
    (1.972625782744, 0.000800093619, -0.505999, -0.365999) + (0.,) * 8
    + (1.498946517543, -0.017201115047, 0.301320911733, -0.29469178596),
)
RUNNING_BELT_SPEED = .12
BELT_RAMP_SECONDS = .5


def _smoothstep(fraction):
    fraction = float(np.clip(fraction, 0., 1.))
    return fraction ** 3 * (10. + fraction * (-15. + 6. * fraction))


def _ramped_belt_motion(elapsed):
    """Speed and integrated travel for the same smooth motor startup."""
    fraction = float(np.clip(elapsed / BELT_RAMP_SECONDS, 0., 1.))
    integral = fraction ** 4 * (2.5 - 3. * fraction + fraction ** 2)
    return (RUNNING_BELT_SPEED * _smoothstep(fraction),
            RUNNING_BELT_SPEED * (BELT_RAMP_SECONDS * integral
                                 + max(0., elapsed - BELT_RAMP_SECONDS)))


def _phase_starts(segments):
    starts, elapsed = {}, 0.
    for name, duration, _ in segments:
        starts[name] = elapsed
        elapsed += duration
    return starts


def default_config(snapshot=None):
    from gobot.sim.providers.newton_vbd import NewtonVBDConfig
    from conveyor_sharpa_runtime import is_sharpa
    arms = snapshot is not None and is_sharpa(snapshot)
    return NewtonVBDConfig(substeps=8, contact_damping=1000.,
                           rigid_soft_contact_damping=2., soft_contact_damping=.3,
                           strain_relaxation_time=.02,
                           joint_linear_damping=500. if arms else 50.,
                           joint_angular_damping=2. if arms else .2,
                           soft_contact_capacity=32768,
                           body_particle_contact_capacity=16384)


def prepare_snapshot(snapshot, *, carton_drop_height=.25):
    """Explicit initial conditions for the drop trial, derived from the authored scene."""
    from scipy.spatial.transform import Rotation
    def array_data(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(f"unsupported snapshot value {type(value).__name__}")
    # Worker recipes contain plain owned data; engine transforms include NumPy matrices.
    result = json.loads(json.dumps(snapshot, default=array_data, allow_nan=False))
    from conveyor_sharpa_runtime import is_sharpa
    if is_sharpa(result):
        return result
    for robot in result["robots"]:
        if robot["name"] not in ("carton_wide", "carton_tall"):
            continue
        link = robot["links"][0]
        pose = np.asarray(link["global_transform"]["matrix"], dtype=float)
        delta = np.eye(4)
        if robot["name"] == "carton_tall":
            delta[:3, :3] = Rotation.from_euler("xy", [24, 20], degrees=True).as_matrix()
        delta[:3, 3] = pose[:3, 3] - delta[:3, :3] @ pose[:3, 3] + (0., 0., carton_drop_height)
        for transform in [link["global_transform"], robot["transform"]] + [s["transform"] for s in link["shapes"]]:
            value = delta @ np.asarray(transform["matrix"], dtype=float)
            transform["matrix"] = value.tolist()
            transform["position"] = value[:3, 3].tolist()
            q = Rotation.from_matrix(value[:3, :3]).as_quat()
            transform["quaternion"] = q[[3, 0, 1, 2]].tolist()
    return result


def _targets(stage, fingers=None):
    fingers = np.zeros((2, 16)) if fingers is None else np.asarray(fingers)
    return np.concatenate((np.asarray(stage), fingers), axis=1)


def trajectory():
    """Named keyframes control only hand joints; bags have no motion constraints."""
    home = _targets(original.HAND_STAGE_HOME_TARGETS)
    grip = _targets(PINCH_STAGE, PINCH_FINGERS)
    contact = _targets(PINCH_STAGE, OPEN_FINGERS)
    behind = contact.copy(); behind[:, 1] -= .06
    approach = behind.copy(); approach[:, 2] += .16
    lift = grip.copy(); lift[:, 2] += .46
    # Stop over the already-running belt and open at a fixed wrist pose.
    # The pouch falls freely; neither hand follows the belt or lowers the held
    # weld into contact. Withdraw only after the pads have opened.
    transfer = lift.copy(); transfer[:, 1] += 1.13
    release = transfer.copy(); release[:, 6:] = OPEN_FINGERS
    clear = release.copy(); clear[:, 1] -= .14; clear[:, 2] += .25
    opened_clear = clear.copy(); opened_clear[:, 6:] = 0.
    # Press with the open fingertip rows over the parcel interior, then recover.
    press = _targets(original.HAND_STAGE_BLUE_FLIP_CONTACT_TARGETS)
    press[:, 1] += .052
    press[:, 2] -= .105
    press_above = press.copy(); press_above[:, 2] += .16
    return (
        ("settle_and_carton_drop", .6, home),
        ("press_approach", .6, press_above),
        ("press", .8, press),
        ("press_hold", .4, press),
        ("press_release", .6, press_above),
        ("recover", .8, press_above),
        ("grasp_approach", .8, approach),
        ("grasp_descend", .8, behind),
        ("grasp_insert", .8, contact),
        ("pinch", .8, grip),
        ("pinch_hold", .4, grip),
        ("lift", 2.4, lift),
        ("hold", 2., lift),
        ("transfer", 3., transfer),
        ("drop_hover", .4, transfer),
        ("release", .35, release),
        ("clear", 1., clear),
        ("drop_settle", 1.65, clear),
        ("settle_after_release", .8, opened_clear),
        ("outfeed", 2., home),
    )


class ConveyorNewtonController:
    def __init__(self, provider, *, drop_only=False):
        self.provider = provider
        self.segments = trajectory()
        self.phase_starts = _phase_starts(self.segments)
        self.duration = 1.6 if drop_only else sum(duration for _, duration, _ in self.segments)
        self.drop_only = drop_only
        self.reset((0,))

    def reset(self, environments):
        if tuple(environments) != (0,):
            raise ValueError("conveyor trial resets environment 0")
        self.time = 0.
        self.phase = "settle_and_carton_drop"
        self._last_tick = -1
        self.belt_speed = 0.
        self.provider.set_surface_velocity(BELT_ROBOT_NAME, BELT_LINK_NAME, (0., 0., 0.))

    def step(self, dt):
        tick = self.provider.tick
        if tick != self._last_tick:
            local = 0. if self.drop_only else self.time
            previous = self.segments[0][2]
            command = previous
            for name, duration, target in self.segments:
                self.phase = name
                if local < duration:
                    # Zero velocity and acceleration at both endpoints avoid
                    # exciting the hanging pouch when a hand starts or stops.
                    fraction = _smoothstep(local / duration)
                    command = previous + fraction * (target - previous)
                    break
                local -= duration
                previous = command = target
            for side, names, targets in zip(LEAP_ROBOT_NAMES, HAND_JOINT_NAMES_BY_SIDE, command, strict=True):
                self.provider.set_joint_position_targets(side, names, targets)
            self.belt_speed = (0. if self.drop_only else
                               _ramped_belt_motion(self.time - self.phase_starts["transfer"])[0])
            self.provider.set_surface_velocity(BELT_ROBOT_NAME, BELT_LINK_NAME, (self.belt_speed, 0., 0.))
            self._last_tick = tick
        self.time += dt

    def apply_commands(self, commands):
        if commands:
            raise ValueError("the scripted conveyor trial has no external command fields")


def create(snapshot, config=None, drop_only=False):
    from gobot.sim import RuntimeComponents
    from gobot.sim.providers.newton_vbd import NewtonVBDConfig, NewtonVBDProvider
    options = {**asdict(default_config(snapshot)), **(config or {})}
    provider = NewtonVBDProvider(snapshot, config=NewtonVBDConfig(**options))
    try:
        from conveyor_sharpa_runtime import is_sharpa, SharpaController
        controller = SharpaController if is_sharpa(snapshot) else ConveyorNewtonController
        return RuntimeComponents(provider, controller(provider, drop_only=drop_only))
    except Exception:
        provider.close()
        raise


def scene_sync(context, root, snapshot, *, drop_only=False):
    import gobot
    nodes = {}
    pending = [root]
    while pending:
        node = pending.pop()
        nodes[node.path] = node
        pending.extend(node.children)
    def resolve(path):
        if path in nodes:
            return nodes[path]
        # The compiler's canonical paths are independent of editor parent nodes.
        parts = path.split('/')
        index = parts.index(root.name)
        relative = '/'.join(parts[index + 1:])
        node = root.find(relative) if relative else root
        if node is None:
            raise ValueError(f"scene snapshot has no display node {path}")
        return node
    links = {f"robot.{r['name']}.link_pose": [[resolve(link["path"]) for link in r["links"]]]
             for r in snapshot["robots"]}
    soft = [[resolve(b["path"]) for b in snapshot["deformables"]]]
    sync = gobot.sim.SceneSnapshotSync(context, links=links, deformables=soft,
                vertex_counts=[len(b["vertices"]) for b in snapshot["deformables"]])
    markers = [(node, tuple(node.position)) for node in nodes.values()
               if node.name.startswith("belt_marker_")]
    from conveyor_sharpa_runtime import is_sharpa
    sharpa = is_sharpa(snapshot)
    belt_start = 0. if sharpa else _phase_starts(trajectory())["transfer"]
    belt_length = BELT_SURFACE_LENGTH
    if sharpa:
        from conveyor_sharpa import BELT_LENGTH, BELT_SPEED
        belt_length = BELT_LENGTH

    def apply(state):
        sync(state)
        # Use completed simulation time, never render cadence. Reset and
        # skipped editor frames therefore retain the correct belt phase.
        clocks = getattr(state, "clocks", ())
        if clocks:
            travel = 0. if drop_only else _ramped_belt_motion(clocks[0].time - belt_start)[1]
            if sharpa:
                travel = BELT_SPEED * clocks[0].time
            for marker, origin in markers:
                x = (origin[0] + travel + .5 * belt_length) % belt_length - .5 * belt_length
                marker.position = (x, origin[1], origin[2])
    return apply, tuple(links) + ("deformable.local_vertices",)


class NewtonConveyorPlay:
    def __init__(self, script):
        self.script = script
        self.play = None
        self.runtime = None
        self._reset_pressed = False

    def start(self):
        import gobot
        import os
        context, root = self.script.context, self.script.get_root()
        snapshot = prepare_snapshot(context.compile_physics_scene_snapshot())
        drop_only = os.environ.get("GOBOT_CONVEYOR_DROP_ONLY", "0") == "1"
        sync, fields = scene_sync(context, root, snapshot, drop_only=drop_only)
        recipe = gobot.sim.SimulationRuntimeSpec("conveyor_newton_runtime:create", {
            "snapshot": snapshot, "config": asdict(default_config(snapshot)),
            "drop_only": drop_only})
        self.runtime = gobot.sim.AsyncSimulationSession(recipe,
                fields=fields, environments=(0,))
        self.play = gobot.sim.ProviderPlaySession(context, self.runtime,
                fixed_dt=original.FIXED_DT, max_sub_steps=1, sync_scene=sync).start()
        self.play.set_status("Newton VBD | CUDA | soft parcels and rigid cartons")
        print("Newton VBD conveyor: building GPU model; Stop and Reset remain available.")

    def process(self):
        state = getattr(self.script.context, "input", None)
        pressed = state is not None and state.is_key_pressed("P")
        if pressed and not self._reset_pressed:
            self.play.reset()
        self._reset_pressed = pressed

    def close(self):
        if self.play is not None:
            self.play.close()
        elif self.runtime is not None:
            self.runtime.close()
        self.play = self.runtime = None
