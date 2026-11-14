"""Contact snapshots and debug arrows for the native mjlab display adapter.

Each row: valid, position[3] (m), normal[3], force[3] (N), receiver body index,
other body index, foot code (1 left / 2 right / 0 other), self-contact flag.
Normals and forces act ON the receiver, in world coordinates. No impulses or
inverse-dynamics force estimates are used.
"""
import numpy as np

CONTACT_COLUMNS = 14


def pack_contacts(*, body_ids, body_names, geom_bodies, geoms, positions,
                  normals, forces, active, capacity):
    by_id = {int(body): index for index, body in enumerate(body_ids)}
    output = np.zeros((capacity, CONTACT_COLUMNS), dtype=np.float32)
    count = 0
    for index in range(len(geoms)):
        g0, g1 = map(int, geoms[index])
        if not active[index] or min(g0, g1) < 0:
            continue
        b0, b1 = int(geom_bodies[g0]), int(geom_bodies[g1])
        r0, r1 = b0 in by_id, b1 in by_id
        if not (r0 or r1):
            continue
        # MuJoCo reports force on geom[1]; invert if the robot is geom[0].
        receiver, other, sign = (b1, b0, 1.0) if r1 else (b0, b1, -1.0)
        force = np.asarray(forces[index], dtype=np.float64) * sign
        normal = np.asarray(normals[index], dtype=np.float64) * sign
        if not np.isfinite(np.concatenate((positions[index], force, normal))).all():
            raise FloatingPointError("non-finite contact snapshot")
        name = body_names[by_id[receiver]]
        foot = 0
        if not (r0 and r1):
            if name == "left_ankle_roll_link":
                foot = 1
            elif name == "right_ankle_roll_link":
                foot = 2
        if count >= capacity:
            raise OverflowError("contact snapshot capacity exceeded")
        output[count] = [1, *positions[index], *normal, *force, by_id[receiver],
                         by_id.get(other, -1), foot, int(r0 and r1)]
        count += 1
    return output


def foot_resultants(contacts):
    valid = contacts[:, 0] > 0
    return np.stack([contacts[valid & (contacts[:, 12] == side), 7:10].sum(axis=0)
                     for side in (1, 2)])


def contact_arrows(contacts, *, offset, settings):
    from gobot.render import DebugArrow

    arrows = []
    draw_points = bool(settings["draw_contacts"])
    draw_forces = bool(settings["draw_contact_forces"])
    scale = max(0.0, float(settings["contact_force_scale"]))
    maximum = max(0.0, float(settings["contact_force_max_length"]))
    offset = np.asarray(offset, dtype=np.float64)
    rows = contacts[contacts[:, 0] > 0]
    for row in rows:
        position, normal, force = row[1:4] + offset, row[4:7], row[7:10]
        if draw_points:
            arrows.append(DebugArrow(position, normal * 0.09, (0.25, 0.95, 1.0, 1.0)))
        magnitude = float(np.linalg.norm(force))
        if draw_forces and magnitude > 1e-5 and scale > 0 and maximum > 0:
            length = min(magnitude * scale, maximum)
            arrows.append(DebugArrow(position, force / magnitude * length,
                                     (1.0, 0.65, 0.1, 1.0)))
    # Resultants are drawn separately from the individual contact forces and
    # labeled explicitly; a multi-point sole should not be mistaken for one contact.
    if draw_forces and scale > 0 and maximum > 0:
        for code, label in ((1, "L foot sum"), (2, "R foot sum")):
            foot = rows[rows[:, 12] == code]
            if not len(foot):
                continue
            force = foot[:, 7:10].sum(axis=0)
            magnitude = float(np.linalg.norm(force))
            if magnitude <= 1e-5:
                continue
            weights = np.linalg.norm(foot[:, 7:10], axis=1)
            position = np.average(foot[:, 1:4], weights=weights, axis=0) + offset
            length = min(magnitude * scale, maximum)
            arrows.append(DebugArrow(position, force / magnitude * length,
                (1.0, 0.25, 0.1, 1.0), label=f"{label}: {magnitude:.1f} N"))
    return arrows
