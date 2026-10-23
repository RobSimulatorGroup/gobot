"""Joint-only sorting controller and measurements for the Sharpa workcell."""
import numpy as np

from conveyor_sharpa import (Kinematics, SIDES, ROBOT_NAMES, BELT_SPEED,
                             BELT_CENTER, BELT_LENGTH, BELT_WIDTH,
                             TABLE_TOP, TABLE_LEFT, PARCEL_POSITIONS, PUSH_ROW_Y)
from conveyor_sharpa_motion import motion_table


def is_sharpa(snapshot):
    return all(name in {r["name"] for r in snapshot["robots"]} for name in ROBOT_NAMES)


class SharpaController:
    def __init__(self, provider, *, drop_only=False):
        if drop_only:
            raise ValueError("--drop-only belongs to the LEAP carton-drop scene; Sharpa uses table-height sorting")
        self.provider = provider
        self.segments, self.times, self.targets = motion_table()
        self.kinematics = tuple(Kinematics(side) for side in SIDES)
        self.names = tuple(kin.names for kin in self.kinematics)
        self.duration = float(self.times[-1])
        self.phase_starts = {}
        self.phase_durations = {}
        elapsed = 0.
        for name, duration, _ in self.segments:
            self.phase_starts[name] = elapsed
            self.phase_durations[name] = duration
            elapsed += duration
        self.drop_only = False
        self.reset((0,))

    def reset(self, environments):
        if tuple(environments) != (0,):
            raise ValueError("Sharpa workcell resets environment 0")
        self.time, self._last_tick, self.phase = 0., -1, "settle"
        self.push_lateral_shift = 0.
        self.push_joint_correction = np.zeros(7)
        self._push_correction_goal = np.zeros(7)
        self._push_feedback_time = -float("inf")
        self._command_time = 0.
        self._retreat_lateral_shift = 0.
        self.belt_speed = BELT_SPEED
        self.provider.set_surface_velocity("conveyor", "belt_surface", (BELT_SPEED, 0., 0.))

    def step(self, dt):
        if self.provider.tick != self._last_tick:
            index = min(len(self.times)-2, max(0, int(np.searchsorted(self.times, self.time)-1)))
            alpha = np.clip((self.time-self.times[index])/(self.times[index+1]-self.times[index]), 0., 1.)
            targets = self.targets[index] + alpha*(self.targets[index+1]-self.targets[index])
            previous_phase = self.phase
            self.phase = next((name for name, start in reversed(tuple(self.phase_starts.items())) if self.time >= start), "settle")
            if self.phase == "out_retreat" and previous_phase != self.phase:
                self._retreat_lateral_shift = self.push_lateral_shift
            self._follow_white_parcel(targets)
            for robot, names, target in zip(ROBOT_NAMES, self.names, targets, strict=True):
                self.provider.set_joint_position_targets(robot, names, target)
            self._command_time = self.time
            self._last_tick = self.provider.tick
        self.time += dt

    def _follow_white_parcel(self, targets):
        """Keep the leading pushing finger near the parcel's measured center.

        A fixed lateral line can slide off a parcel after it rotates during the
        handoff. Observe its position at 25 Hz and correct only arm commands;
        the shell, contents and their velocities remain entirely simulated.
        """
        active = self.phase in ("white_push", "out_clear", "out_retreat")
        if active and self.time-self._push_feedback_time >= .04-1e-8:
            if self.phase == "white_push":
                state = self.provider.read_state()
                parcel = state["deformables"].get("soft_mailer_blue")
                if parcel is not None:
                    # Follow with the row center; the wrist yaw changes along the push.
                    desired = float(parcel["positions"][:, 1].mean())-PUSH_ROW_Y
                    desired = np.clip(desired, -.12, .08)
                    self.push_lateral_shift += float(np.clip((desired-self.push_lateral_shift)/3., -.008, .008))
            elif self.phase == "out_retreat":
                u = np.clip((self.time-self.phase_starts[self.phase])/self.phase_durations[self.phase], 0., 1.)
                blend = u**3*(10.+u*(-15.+6.*u))
                self.push_lateral_shift = (1.-blend)*self._retreat_lateral_shift
            arm, self.push_lateral_shift = self._reachable_push_pose(targets[1], self.push_lateral_shift)
            self._push_correction_goal = arm-targets[1, :7]
            self._push_feedback_time = self.time
        elif not active:
            self.push_lateral_shift = 0.
            self._push_correction_goal.fill(0.)
        # Preserve smooth commands between the slower observation/IK updates.
        blend = 1.-np.exp(-max(0., self.time-self._command_time)/.05)
        self.push_joint_correction += blend*(self._push_correction_goal-self.push_joint_correction)
        kin = self.kinematics[1]
        targets[1, :7] = np.clip(targets[1, :7]+self.push_joint_correction,
                                kin.limits[:7, 0], kin.limits[:7, 1])

    def _reachable_push_pose(self, target, shift):
        """Saturate parcel following at the current arm's reachable range.

        A constant Cartesian shift bound alone does not protect the whole
        push: the arm has less lateral room near the end of its travel.
        Back off an infeasible correction while retaining the nominal motion,
        joint-branch bound and worktop clearance. Never move the parcel itself.
        """
        kin = self.kinematics[1]
        # FK already includes gravity feedforward; do not compensate twice.
        palm = kin.forward(target, hand=False)["right_hand_C_MC"]

        def solve(fraction):
            moved = palm.copy()
            moved[1, 3] += fraction*shift
            try:
                arm = kin.arm_pose(moved, target[:7])
            except ValueError:
                return None
            if np.max(np.abs(arm-target[:7])) > .5:
                return None
            clearance, _ = kin.arm_worktop_clearance(dict(zip(kin.names[:7], arm)))
            return arm if clearance >= .005 else None

        arm = solve(1.)
        if arm is not None:
            return arm, shift
        # The authored nominal path is already checked by motion_table.
        low, high, best = 0., 1., target[:7].copy()
        for _ in range(7):
            middle = (low+high)/2.
            arm = solve(middle)
            if arm is None:
                high = middle
            else:
                low, best = middle, arm
        return best, low*shift

    def apply_commands(self, commands):
        if commands:
            raise ValueError("the Sharpa sorting sequence has no external command fields")


class SharpaMetrics:
    def __init__(self, snapshot):
        self.samples = []
        self.peak_penetration = 0.
        self.minimum_volume_ratio = 1.
        self.reference = {}
        for body in snapshot["deformables"]:
            if body["name"] not in PARCEL_POSITIONS:
                continue
            local = np.asarray(body["vertices"])
            pose = np.asarray(body["transform"]["matrix"])
            world = local @ pose[:3, :3].T + pose[:3, 3]
            self.reference[body["name"]] = world-world.mean(0)
        self.carton_reference = {
            f"{robot['name']}/{link['name']}": np.asarray(link["global_transform"]["matrix"])[:3, 3]
            for robot in snapshot["robots"] if robot["name"].startswith("carton_") for link in robot["links"]}
        self.peak_carton_displacement = 0.
        self.kinematics = [Kinematics(side) for side in SIDES]
        self.maximum_wrist_target_error = 0.
        self.maximum_palm_normal_z = -1.

    def update(self, state, phase):
        self.peak_penetration = max(self.peak_penetration, state["peak_soft_penetration"])
        self.minimum_volume_ratio = min(self.minimum_volume_ratio, state["minimum_tet_volume_ratio"])
        parcels = {}
        for name, body in state["deformables"].items():
            q, v = body["positions"], body["velocities"]
            parcels[name] = dict(center=q.mean(0).tolist(), min=q.min(0).tolist(), max=q.max(0).tolist(),
                height=float(np.ptp(q[:, 2])), rms_speed_m_s=float(np.sqrt(np.mean(np.sum(v*v, axis=1)))),
                vertical_rms_speed_m_s=float(np.sqrt(np.mean(v[:, 2]**2))))
            if name in self.reference:
                u, _, vh = np.linalg.svd(self.reference[name].T @ (q-q.mean(0)))
                correction = np.diag((1., 1., np.linalg.det(vh.T@u.T)))
                rotation = vh.T @ correction @ u.T
                parcels[name]["face_normal_z"] = float(rotation[2, 2])
        for name, initial in self.carton_reference.items():
            self.peak_carton_displacement = max(self.peak_carton_displacement,
                float(np.linalg.norm(state["links"][name][:3]-initial)))
        palm_normals = {}
        for i, kin in enumerate(self.kinematics):
            # Compare against the drive targets after configured velocity
            # limiting, including the parcel-following corrections.
            targets = {name: state["joints"][f"{ROBOT_NAMES[i]}/{name}"]["target"] for name in kin.names}
            commanded = kin.forward(targets, hand=False)[f"{kin.side}_hand_C_MC"]
            measured = state["links"][f"{ROBOT_NAMES[i]}/{kin.side}_hand_C_MC"]
            # Newton poses store xyzw quaternions. Sharpa's local +X points
            # out of the palm; measure the simulated orientation, not IK.
            x, y, z, w = measured[3:7]
            palm_normals[kin.side] = float(2.*(x*z-y*w))
            self.maximum_palm_normal_z = max(self.maximum_palm_normal_z, palm_normals[kin.side])
            error = float(np.linalg.norm(commanded[:3, 3]-measured[:3]))
            self.maximum_wrist_target_error = max(self.maximum_wrist_target_error, error)
            if error > .05:
                raise RuntimeError(f"{kin.side} arm lost tracking: {error:.4f} m at {state['time']:.3f}s")
        sample = dict(time=state["time"], phase=phase, parcels=parcels,
                      palm_normal_z=palm_normals,
                      max_soft_penetration_m=state["max_soft_penetration"], soft_contact_count=state["soft_contact_count"])
        self.samples.append(sample)
        return sample

    def report(self, *, drop_only=False):
        final = self.samples[-1]["parcels"] if self.samples else {}
        summary = {}
        for name, initial in PARCEL_POSITIONS.items():
            records = [s["parcels"][name] for s in self.samples if s["time"] > .6]
            summary[name] = dict(final_center=final[name]["center"] if name in final else None,
                final_flip_degrees=float(np.degrees(np.arccos(np.clip(final[name]["face_normal_z"], -1., 1.)))) if name in final else None,
                leftward_travel_m=initial[0]-final[name]["center"][0] if name in final else 0.,
                max_bottom_clearance_m=max((p["min"][2]-TABLE_TOP for p in records), default=0.))
        quiet = [s for s in self.samples if s["phase"] == "outfeed"]
        quiet_speed = max((s["parcels"][n]["vertical_rms_speed_m_s"] for s in quiet
                           for n in s["parcels"]), default=None)
        def on_belt(name):
            if name not in final:
                return False
            parcel = final[name]
            return (parcel["center"][0] < TABLE_LEFT-.14
                    and parcel["min"][0] >= BELT_CENTER[0]-BELT_LENGTH/2-.005
                    and parcel["max"][0] <= TABLE_LEFT+.005
                    and parcel["min"][1] >= BELT_CENTER[1]-BELT_WIDTH/2-.005
                    and parcel["max"][1] <= BELT_CENTER[1]+BELT_WIDTH/2+.005
                    and TABLE_TOP-.005 <= parcel["min"][2] <= TABLE_TOP+.02)
        acceptance = dict(contact_penetration=self.peak_penetration <= .001,
            positive_volume=self.minimum_volume_ratio > 0.,
            both_parcels_on_outfeed=all(on_belt(n) for n in PARCEL_POSITIONS),
            both_parcels_flipped=all(n in final and final[n]["face_normal_z"] < -.7 for n in PARCEL_POSITIONS),
            cartons_remain_on_table=self.peak_carton_displacement < .015,
            arms_track_commands=self.maximum_wrist_target_error < .025,
            palms_face_down=bool(self.samples) and self.maximum_palm_normal_z < -.1,
            no_outfeed_launch=bool(quiet) and all(s["parcels"][n]["min"][2]-TABLE_TOP < .02
                                                 for s in quiet for n in PARCEL_POSITIONS),
            quiet_outfeed=quiet_speed is not None and quiet_speed < .03)
        return dict(acceptance=acceptance, max_soft_penetration_m=self.peak_penetration,
                    minimum_tet_volume_ratio=self.minimum_volume_ratio, parcels=summary,
                    quiet_vertical_rms_speed_m_s=quiet_speed,
                    maximum_carton_displacement_m=self.peak_carton_displacement,
                    maximum_palm_normal_z=self.maximum_palm_normal_z,
                    maximum_wrist_target_error_m=self.maximum_wrist_target_error, samples=self.samples)
