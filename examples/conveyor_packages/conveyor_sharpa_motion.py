"""Alternating, low table-height parcel sorting by the two articulated arms."""
from functools import lru_cache
import numpy as np
from scipy.spatial.transform import Rotation

from conveyor_sharpa import Kinematics, SIDES, WORK_HEIGHT_OFFSET, WORK_ROW_OFFSET, PUSH_ROW_Y
from conveyor_sharpa_grasp import (down_palm, pinch_fingers, pinch_palm,
                                   push_fingers, push_palm, ready_fingers, ready_palm)

# Interpolate palm frames throughout the OpenArm motion, including unloaded
# transitions: endpoint-only joint interpolation can roll a palm upward.


def keyframes():
    kin = [Kinematics(side) for side in SIDES]
    # Contact paths use the original local worktable coordinates; translate
    # the whole task to the authored workcell once, after making all frames.
    push_row = PUSH_ROW_Y-WORK_ROW_OFFSET
    palms = np.array([ready_palm(k, center) for k, center in zip(kin,
        ((.24, -.28, .665), (-.24, -.28, .665)))])
    fingers = np.array([ready_fingers(k) for k in kin])
    home, home_fingers = palms.copy(), fingers.copy()
    frames = []

    def pose(side, center, opening=.08, *, down=False, roll=0., push=False):
        palms[side] = (push_palm(kin[side], center) if push else
                       down_palm(kin[side], center) if down else
                       pinch_palm(kin[side], center, roll))
        fingers[side] = push_fingers(kin[side]) if push else pinch_fingers(kin[side], opening)

    def add(name, duration):
        frames.append((name, duration, palms.copy(), fingers.copy()))

    def move(name, duration, center, opening=.08, *, side=1, down=False, roll=0., push=False):
        pose(side, center, opening, down=down, roll=roll, push=push)
        add(name, duration)

    # Rotate above the front seal before descending. A diagonal descent across
    # the bulge pushes the parcel away and grasps its center instead of its edge.
    add("settle", .4)
    move("yellow_approach", 1., (-.24, -.40, .70), down=True)
    move("yellow_orient", 1.2, (-.24, -.40, .67))
    move("yellow_over_seal", .9, (-.24, -.385, .67))
    move("yellow_descend", .7, (-.24, -.385, .5915))
    move("yellow_insert", .5, (-.24, -.335, .5915))
    move("yellow_hook", .5, (-.24, -.335, .6035))
    move("yellow_pinch", .9, (-.24, -.335, .6035), 0.)
    add("yellow_grip", .3)
    move("yellow_lift", 1.2, (-.24, -.16, .66), 0., roll=-20.)
    move("yellow_turn", .9, (-.24, -.14, .66), 0., roll=-20.)
    move("yellow_release", .2, (-.24, -.14, .66), roll=-20.)
    move("yellow_clear", .8, (-.24, -.22, .69), roll=-20.)

    # The left hand prepares the next parcel while the right hand guides
    # the flipped yellow parcel sideways onto the continuously running belt.
    pose(0, (.24, -.44, .70), down=True)
    move("yellow_reorient", .8, (-.24, -.30, .70), down=True)
    pose(0, (.24, -.43, .67))
    move("yellow_push_above", 1.2, (-.055, -.20, .69), push=True)
    move("yellow_push_contact", .6, (-.055, push_row, .590), push=True)
    pose(0, (.24, -.43, .5915))
    move("yellow_push", 3., (-.48, push_row, .590), push=True)
    pose(0, (.24, -.410, .5915))
    move("yellow_push_clear", .8, (-.44, -.26, .65), push=True)
    move("right_retreat", 1.2, (-.24, -.32, .68), down=True)
    move("white_hook", .5, (.24, -.410, .6035), side=0)
    palms[1], fingers[1] = home[1], home_fingers[1]
    move("white_pinch", .9, (.24, -.410, .6035), 0., side=0)
    add("white_grip", .3)
    move("white_lift", 1.3, (.24, -.18, .66), 0., side=0, roll=-20.)
    move("white_turn", 1., (.24, -.16, .66), 0., side=0, roll=-20.)
    move("white_release", .2, (.24, -.16, .66), side=0, roll=-20.)
    move("white_clear", .8, (.24, -.22, .69), side=0, roll=-20.)

    # Palm-down side pushes keep the wrist above the parcel. The right hand
    # receives the larger parcel near the center of the table.
    move("white_reorient", .8, (.24, -.30, .70), side=0, down=True)
    move("white_push_above", 1.2, (.47, -.20, .69), side=0, push=True)
    move("white_push_contact", .6, (.47, push_row, .590), side=0, push=True)
    # Hand over near the centerline, before the lowered left wrist reaches
    # across the torso and runs into its yaw stop.
    move("white_handoff", 2.8, (.08, push_row, .590), side=0, push=True)
    move("white_handoff_clear", .8, (.24, -.32, .67), side=0, push=True)
    palms[0], fingers[0] = home[0], home_fingers[0]
    move("receive_approach", .9, (.08, -.20, .72), push=True)
    move("receive_contact", .7, (.08, push_row, .590), push=True)
    move("white_push", 3., (-.48, push_row, .590), push=True)
    move("out_clear", .8, (-.44, -.26, .65), push=True)
    move("out_retreat", 1.2, (-.24, -.32, .68), down=True)
    palms[:], fingers[:] = home, home_fingers
    add("return", .9)
    add("outfeed", 1.5)
    for _, _, palms, _ in frames:
        palms[:, 2, 3] += WORK_HEIGHT_OFFSET
        palms[:, 1, 3] += WORK_ROW_OFFSET
    return frames


@lru_cache(maxsize=1)
def home_configuration():
    """Solve the authored waiting pose independently of the sorting program."""
    _, _, palms, fingers = keyframes()[0]
    return np.array([np.r_[kin.compensated_arm_pose(palm, hand, kin.home_seed), hand]
                     for kin, palm, hand in zip((Kinematics(s) for s in SIDES), palms, fingers, strict=True)])


@lru_cache(maxsize=1)
def motion_table():
    frames = keyframes()
    kinematics = [Kinematics(side) for side in SIDES]
    initial = home_configuration().copy()
    previous = [q[:7].copy() for q in initial]
    result, times, samples = [], [0.], [initial]
    _, _, last_palms, last_fingers = frames[0]
    elapsed = 0.
    previous_palms = last_palms.copy()
    for name, duration, palms, fingers in frames:
        count = max(1, round(duration/.04),
                    int(np.ceil(1.875*np.max(np.abs(fingers-last_fingers))/.20)))
        delta_rotations = [Rotation.from_matrix(palms[i,:3,:3] @ last_palms[i,:3,:3].T).as_rotvec() for i in range(2)]
        fractions = list(np.arange(1, count+1)/count)
        previous_u, sample_index = 0., 0
        while sample_index < len(fractions):
            u = fractions[sample_index]
            blend = u**3 * (10. + u*(-15. + 6.*u))
            current = last_palms + blend*(palms-last_palms)
            current_fingers = last_fingers + blend*(fingers-last_fingers)
            targets = []
            failure = None
            for i, kin in enumerate(kinematics):
                current[i,:3,:3] = Rotation.from_rotvec(blend*delta_rotations[i]).as_matrix() @ last_palms[i,:3,:3]
                try:
                    angles = (previous[i] if np.max(abs(current[i]-previous_palms[i])) < 1e-10 else
                              kin.compensated_arm_pose(current[i], current_fingers[i], previous[i]))
                except ValueError as error:
                    failure = f"{name}, {kin.side}, {elapsed+duration*u:.3f}s: {error}"
                    break
                if np.max(np.abs(angles-previous[i])) > .35:
                    failure = f"discontinuous IK in {name}: {kin.side}, at {current[i,:3,3]}, {previous[i]} -> {angles}"
                    break
                wrist = kin.forward(dict(zip(kin.names[:7], angles)), hand=False)[f"{kin.side}_hand_C_MC"]
                if wrist[2, 0] > -.10:
                    raise ValueError(f"{name}, {kin.side}: palm turns upward during the motion")
                clearance, link = kin.arm_worktop_clearance(dict(zip(kin.names[:7], angles)))
                if clearance < .005:
                    raise ValueError(f"{name}, {kin.side}, {elapsed+duration*u:.3f}s: {link} worktop clearance {clearance:.4f} m")
                targets.append(np.r_[angles, current_fingers[i]])
            if failure is not None:
                # Near a wrist singularity a short Cartesian interval may
                # need more than one IK solve to remain on the same branch.
                # Retry from the last accepted pose; never relax the guard or
                # commit one arm while the other arm's sample was rejected.
                if u-previous_u < 1./65536:
                    raise ValueError(failure)
                fractions.insert(sample_index, (previous_u+u)/2)
                continue
            targets = np.asarray(targets)
            # A smooth Cartesian path can still move an individual joint too
            # quickly near a redundant-arm turn. Stretch those intervals before
            # the provider's 1.8 rad/s limit can introduce tracking lag.
            step_duration = max(duration*(u-previous_u),
                float(np.max(np.abs(targets[:, :7]-samples[-1][:, :7])))/1.65,
                float(np.max(np.abs(targets[:, 7:]-samples[-1][:, 7:])))/6.)
            samples.append(targets); times.append(times[-1]+step_duration)
            previous = [q[:7].copy() for q in targets]
            previous_palms = current.copy()
            previous_u = u
            sample_index += 1
        result.append((name, times[-1]-elapsed, samples[-1]))
        elapsed = times[-1]
        last_palms, last_fingers = palms, fingers
    return tuple(result), np.asarray(times), np.asarray(samples)


def trajectory():
    return motion_table()[0]
