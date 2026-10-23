"""Contact geometry and hand postures for the supplied Sharpa meshes."""
import numpy as np

from conveyor_sharpa import PALM_DOWN, transform

# Fits the actual pad surfaces, within the restricted upstream joint limits.
PINCH = np.array((1.83611126, .13089, -.460794934, .34909, .00001,
                  1.31345814, .03490, .125183542, .150324754))
# Abduct the entire thumb outside the front seal; curling only the tip back
# leaves the proximal phalanx inside the filled bulge during insertion.
OPEN_THUMB = np.zeros(5)
PINCH_PRELOAD = .035  # Index MCP overtravel: about 3 mm at the opposed pads.


def pinch_fingers(kin, opening=0., *, preload=True):
    values = np.zeros(22)
    values[:9] = PINCH
    fraction = np.clip(opening/.08, 0., 1.)
    values[:5] += fraction*(OPEN_THUMB-values[:5])
    values[5] -= .25*opening
    # A zero-force geometric pad fit lets a thin seal slip under the parcel's
    # weight. The compliant drive supplies the squeeze; contacts determine the
    # actual finger position. Keep the tool reference at the unloaded fit.
    if preload:
        values[5] += PINCH_PRELOAD*(1.-fraction)
    for i, name in enumerate(kin.finger_names):
        if any(finger in name for finger in ("middle", "ring", "pinky")):
            if "MCP_FE" in name:
                values[i] = 1.35
            elif name.endswith("PIP"):
                values[i] = 1.3
            elif name.endswith("DIP"):
                values[i] = 1.
    return values


def pinch_frame(kin):
    """Palm-local pinch center and a wrist orientation for a horizontal seam."""
    poses = kin.forward(dict(zip(kin.finger_names, pinch_fingers(kin, preload=False))), world=False)
    inverse = np.linalg.inv(poses[f"{kin.side}_hand_C_MC"])
    index = inverse @ poses[f"{kin.side}_index_elastomer"]
    thumb = inverse @ poses[f"{kin.side}_thumb_elastomer"]
    sign = 1 if kin.side == "left" else -1
    a = (index @ (0., -.019*sign, -.00488894737, 1.))[:3]
    b = (thumb @ (0., -.022*sign, -.00391412, 1.))[:3]
    center = (a+b)/2
    up = index[:3, :3] @ (0., -.1046848*sign, -.99450545)
    up /= np.linalg.norm(up)
    forward = -(center-up*np.dot(center, up))
    forward /= np.linalg.norm(forward)
    orientation = np.array((np.cross(forward, up), forward, up))
    return center, orientation


def pinch_palm(kin, center, roll=0.):
    from scipy.spatial.transform import Rotation
    offset, orientation = pinch_frame(kin)
    # The palm tilts down while the index enters the front seal. The static
    # mounting bracket brings OpenArm over the table without changing its axes.
    orientation = (Rotation.from_euler("z", 180, degrees=True).as_matrix()
                   @ Rotation.from_euler("x", 45.+roll, degrees=True).as_matrix() @ orientation)
    return transform(np.asarray(center)-orientation@offset, orientation)


def down_palm(kin, center):
    """Palm facing the worktop, with a slight outward wrist angle.

    Keep the same thumb/index reference point as ``pinch_palm`` so approaching
    and turning the hand do not silently change the authored tool position.
    """
    from scipy.spatial.transform import Rotation
    offset, _ = pinch_frame(kin)
    yaw = 30. if kin.side == "left" else -30.
    orientation = Rotation.from_euler("z", yaw, degrees=True).as_matrix() @ PALM_DOWN
    return transform(np.asarray(center)-orientation@offset, orientation)


def ready_fingers(kin):
    """Relaxed open hand for the low bimanual waiting/return posture."""
    values = np.zeros(22)
    values[:5] = (.25, -.20, .15, 0., .15)
    for i, name in enumerate(kin.finger_names[5:], 5):
        if "MCP_FE" in name:
            values[i] = .25
        elif name.endswith("PIP"):
            values[i] = .30
        elif name.endswith("DIP"):
            values[i] = .16
    return values


def ready_palm(kin, center):
    """Place the relaxed finger row above the parcel, with palms down."""
    from scipy.spatial.transform import Rotation
    poses = kin.forward(dict(zip(kin.finger_names, ready_fingers(kin))), world=False)
    inverse = np.linalg.inv(poses[f"{kin.side}_hand_C_MC"])
    sign = 1 if kin.side == "left" else -1
    points = [(inverse @ poses[f"{kin.side}_{finger}_elastomer"] @
               (0., -.019*sign, -.00488894737, 1.))[:3]
              for finger in ("index", "middle", "ring", "pinky")]
    orientation = Rotation.from_euler("z", 12.*sign, degrees=True).as_matrix() @ PALM_DOWN
    return transform(np.asarray(center)-orientation@np.mean(points, axis=0), orientation)


def push_fingers(kin):
    """Spread the four fingers into a pushing face; keep the thumb clear.

    The opposed thumb/index pinch can hook the film during a side push.
    A separate hand shape distributes that contact across the finger row.
    """
    values = np.zeros(22)
    for i, name in enumerate(kin.finger_names):
        if "thumb" in name:
            continue
        if "MCP_FE" in name:
            values[i] = 1.15
        elif name.endswith(("PIP", "DIP")):
            values[i] = .15
    return values


def push_palm(kin, center):
    """Palm-down pose referenced to the middle of the four contact pads."""
    from scipy.spatial.transform import Rotation
    poses = kin.forward(dict(zip(kin.finger_names, push_fingers(kin))), world=False)
    inverse = np.linalg.inv(poses[f"{kin.side}_hand_C_MC"])
    sign = 1 if kin.side == "left" else -1
    points = [(inverse @ poses[f"{kin.side}_{finger}_elastomer"] @
               (0., -.019*sign, -.00488894737, 1.))[:3]
              for finger in ("index", "middle", "ring", "pinky")]
    offset = np.mean(points, axis=0)
    # Turn the finger row with the side push, leaving the wrist on its own
    # side of the torso instead of exhausting OpenArm's limited wrist yaw.
    yaw = (np.clip((center[0]-.25)*100., -30., 30.) if kin.side == "left"
           else np.clip((center[0]+.12)*100., -40., 20.))
    orientation = Rotation.from_euler("z", yaw, degrees=True).as_matrix() @ PALM_DOWN
    return transform(np.asarray(center)-orientation@offset, orientation)
