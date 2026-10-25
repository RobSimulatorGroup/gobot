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
