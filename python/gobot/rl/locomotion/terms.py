"""Array-independent parameter resolution for velocity task terms."""
import re
import numpy as np

def posture_std(names, mapping, hip_thigh, calf):
    if not mapping:
        return np.asarray([calf if "calf" in name else hip_thigh for name in names], dtype=np.float32)
    result = []
    for name in names:
        values = [value for pattern, value in mapping.items() if re.fullmatch(pattern, name)]
        if len(values) != 1 or values[0] <= 0:
            raise ValueError(f"posture std must match {name!r} exactly once with a positive value")
        result.append(values[0])
    return np.asarray(result, dtype=np.float32)


def locomotion_failure_mask(gravity_z, clearance, illegal_contacts, trunk_contacts, *,
                            fall_angle=None, min_clearance=None, terminate_on_trunk=False):
    """Physical failure predicate shared by NumPy and device tensors.

    Height is measured above the terrain under the pelvis. A low, upright
    seated body is a failure even when the orientation test alone passes.
    """
    import math
    failed = illegal_contacts > 0
    if fall_angle is not None:
        failed = failed | (gravity_z > -math.cos(fall_angle))
    if min_clearance is not None:
        failed = failed | (clearance < min_clearance)
    if terminate_on_trunk:
        failed = failed | (trunk_contacts > 0)
    return failed
