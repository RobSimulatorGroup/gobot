"""Command-progress curriculum shared by NumPy and Torch execution.

Adapted from mjlab 8bb2bc9e (Apache-2.0). Inputs retain their array backend.
"""
def progress_masks(distance, commanded_distance, patch_length, *, promotion_fraction=.8, min_commanded_distance=1.):
    required = (commanded_distance * promotion_fraction).clip(max=patch_length / 2)
    up = (distance > required) & (commanded_distance >= min_commanded_distance)
    down = (distance < commanded_distance * .5) & ~up
    return up, down
