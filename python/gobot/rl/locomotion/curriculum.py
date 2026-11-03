"""Command-progress curriculum shared by NumPy and Torch execution.

Adapted from mjlab 8bb2bc9e (Apache-2.0). Inputs retain their array backend.
"""
def progress_masks(distance, commanded_distance, patch_length, *, promotion_fraction=.8, min_commanded_distance=1.):
    required = (commanded_distance * promotion_fraction).clip(max=patch_length / 2)
    eligible = commanded_distance >= min_commanded_distance
    up = (distance > required) & eligible
    down = (distance < commanded_distance * .5) & ~up & eligible
    return up, down


def directed_increment(displacement, world_command):
    """Signed progress in the commanded direction; turns never cancel requested path length."""
    speed = (world_command.square().sum(-1) if hasattr(world_command, "square") else (world_command ** 2).sum(-1)) ** .5
    return (displacement * world_command).sum(-1) / speed.clip(min=1.e-6)
