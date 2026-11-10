"""Command-displacement curriculum shared by NumPy and Torch execution.

Adapted from mjlab dcdf6ce5 (Apache-2.0). Inputs retain their array backend.
"""


def command_displacement_step(command, quaternion, dt, xp):
    """Integrate the held planar command using the robot's current heading."""
    w, x, y, z = (quaternion[..., i] for i in range(4))
    heading = xp.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    cosine, sine = xp.cos(heading), xp.sin(heading)
    vx, vy = command[..., 0], command[..., 1]
    velocity = xp.stack((cosine * vx - sine * vy, sine * vx + cosine * vy), axis=-1)
    return velocity * dt[..., None]


def displacement_masks(distance, commanded_displacement, command, remaining_time, patch_length):
    """mjlab terrain promotion: traveled distance versus the full episode's command."""
    required = (commanded_displacement ** 2).sum(-1) ** .5
    required += (command[..., :2] ** 2).sum(-1) ** .5 * remaining_time.clip(min=0.)
    up = (distance > (.8 * required).clip(max=patch_length / 2)) & (required >= 1.)
    down = (distance < .5 * required) & ~up
    return up, down
