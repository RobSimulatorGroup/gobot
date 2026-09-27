"""Authored conveyor workcell names, geometry and material configuration."""

import math
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
SCENE_NAME = "conveyor_packages.jscn"
PLAY_SCRIPT_NAME = "conveyor_packages_play.py"
PLAY_SCRIPT_PATH = "res://" + PLAY_SCRIPT_NAME
PLAY_SCRIPT_RESOURCE_ID = "conveyor_packages_play_script"

LEAP_ASSET_ROOT = HERE / "assets" / "leap_hand"
LEAP_RESOURCE_ROOT = "res://assets/leap_hand"
HAND_SIDES = ("left", "right")
# Place the anatomical left/right models on their matching world sides so both
# thumbs point inward toward the workpiece.
LEAP_MJCF_BY_SIDE = ("left_hand.xml", "right_hand.xml")
LEAP_ROBOT_NAMES = tuple(f"leap_{side}" for side in HAND_SIDES)
LEAP_FINGER_JOINT_NAMES = (
    "if_mcp",
    "if_rot",
    "if_pip",
    "if_dip",
    "mf_mcp",
    "mf_rot",
    "mf_pip",
    "mf_dip",
    "rf_mcp",
    "rf_rot",
    "rf_pip",
    "rf_dip",
    "th_cmc",
    "th_axl",
    "th_mcp",
    "th_ipl",
)
LEAP_FINGER_CLOSE_TARGETS = (
    1.08,
    0.0,
    0.82,
    0.78,
    1.08,
    0.0,
    0.82,
    0.78,
    1.08,
    0.0,
    0.82,
    0.78,
    0.82,
    0.50,
    0.82,
    0.78,
)
LEAP_CONTACT_LINK_NAMES = (
    "palm",
    "if_bs",
    "if_px",
    "if_md",
    "if_ds",
    "mf_bs",
    "mf_px",
    "mf_md",
    "mf_ds",
    "rf_bs",
    "rf_px",
    "rf_md",
    "rf_ds",
    "th_mp",
    "th_bs",
    "th_px",
    "th_ds",
)
HAND_STAGE_DOF_NAMES = ("x", "y", "z", "roll", "pitch", "yaw")
HAND_STAGE_JOINT_NAMES_BY_SIDE = tuple(
    tuple(f"leap_{side}_wrist_{name}" for name in HAND_STAGE_DOF_NAMES)
    for side in HAND_SIDES
)
HAND_STAGE_LINK_NAMES_BY_SIDE = tuple(
    (f"leap_{side}_mount",)
    + tuple(
        f"leap_{side}_stage_{name}" for name in HAND_STAGE_DOF_NAMES[:-1]
    )
    for side in HAND_SIDES
)
HAND_BASE_LINK_NAMES = tuple(names[1] for names in HAND_STAGE_LINK_NAMES_BY_SIDE)
HAND_JOINT_NAMES_BY_SIDE = tuple(
    stage_names + LEAP_FINGER_JOINT_NAMES
    for stage_names in HAND_STAGE_JOINT_NAMES_BY_SIDE
)
HAND_LINK_NAMES_BY_SIDE = tuple(
    stage_names[1:] + LEAP_CONTACT_LINK_NAMES
    for stage_names in HAND_STAGE_LINK_NAMES_BY_SIDE
)
MANIPULATION_STATION_X = 0.16
HAND_ROOT_POSITIONS = (
    # The two Menagerie CAD frames are asymmetric. These roots align their
    # fingertips in world Y/Z while leaving room for the inward-facing thumbs.
    (MANIPULATION_STATION_X - 0.18, 0.0433, 1.0930),
    (MANIPULATION_STATION_X + 0.18, -0.0321, 1.1195),
)
# Menagerie presents the hands palm-up. This proper rotation turns the palms
# downward and points the fingers toward the outfeed (+Y).
HAND_PALM_ALIGNMENT_ROTATION = np.asarray(
    ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, -1.0)),
    dtype=np.float64,
)
HAND_STAGE_TRANSLATION_RANGE = 1.24
HAND_STAGE_ROTATION_RANGE = math.pi + 0.12
# Keep enough range for continuous wrist turns during a one-shot manipulation
# cycle. The profile chooses nearby equivalent angles between parcels so high
# transfers never contain a hidden multi-revolution controller jump.
HAND_STAGE_ROLL_RANGE = 6.0 * math.pi + 0.12
HAND_STAGE_LINEAR_STIFFNESS = 9000.0
HAND_STAGE_LINEAR_DAMPING = 240.0
HAND_STAGE_ANGULAR_STIFFNESS = 1400.0
HAND_STAGE_ANGULAR_DAMPING = 85.0
HAND_STAGE_ARMATURE = 0.1
HAND_FINGER_STIFFNESS = 36.0
HAND_FINGER_DAMPING = 1.20
HAND_FRICTION = 3.00
LEAP_TIP_BOUNDS = (
    (-0.0112503, -0.0500004, 0.0023530),
    (0.00958777, -0.0195963, 0.0266643),
)
LEAP_THUMB_TIP_BOUNDS = (
    (-0.0112650, -0.0620958, -0.0266430),
    (0.0095930, -0.0315304, -0.00234675),
)
RIGID_COLLISION_LAYER = 0b0001
HAND_COLLISION_LAYER = 0b0010
DEFORMABLE_COLLISION_LAYER = 0b0100
RIGID_PACKAGE_COLLISION_LAYER = 0b1000
DEFORMABLE_FILL_COLLISION_LAYER = 0b10000
RIGID_COLLISION_MASK = (
    RIGID_COLLISION_LAYER
    | DEFORMABLE_COLLISION_LAYER
    | RIGID_PACKAGE_COLLISION_LAYER
)
HAND_COLLISION_MASK = (
    RIGID_COLLISION_LAYER
    | DEFORMABLE_COLLISION_LAYER
    | RIGID_PACKAGE_COLLISION_LAYER
)
DEFORMABLE_COLLISION_MASK = (
    RIGID_COLLISION_LAYER
    | HAND_COLLISION_LAYER
    | DEFORMABLE_FILL_COLLISION_LAYER
    | RIGID_PACKAGE_COLLISION_LAYER
)
DEFORMABLE_FILL_COLLISION_MASK = DEFORMABLE_COLLISION_LAYER
RIGID_PACKAGE_COLLISION_MASK = (
    RIGID_COLLISION_LAYER
    | HAND_COLLISION_LAYER
    | DEFORMABLE_COLLISION_LAYER
    | RIGID_PACKAGE_COLLISION_LAYER
)

BELT_FRAME_LENGTH = 2.70
BELT_SURFACE_LENGTH = BELT_FRAME_LENGTH
BELT_PROXY_LENGTH = BELT_SURFACE_LENGTH
BELT_WIDTH = 0.72
BELT_THICKNESS = 0.04
BELT_CENTER_Z = 0.54
BELT_TOP_Z = BELT_CENTER_Z + 0.5 * BELT_THICKNESS
BELT_CENTER_X = 0.0
BELT_CENTER_Y = 0.58
WORKTABLE_LENGTH = 2.10
# Preserve the 15 mm handoff gap to the outfeed while extending the sorting
# surface toward the operator. The deeper table supports the full reverse face
# of a pouch after an outward turnover instead of letting a successful flip
# fall past the old front edge.
WORKTABLE_DEPTH = 1.08
WORKTABLE_THICKNESS = 0.08
WORKTABLE_CENTER_X = -0.15
WORKTABLE_CENTER_Y = -0.335
WORKTABLE_TOP_Z = BELT_TOP_Z
# Polished sorting-table laminate against a plastic mailer. Keeping this below
# the hand coefficient lets a down-facing palm sweep the parcel by contact
# friction while the belt remains stationary.
WORKTABLE_SLIDING_FRICTION = 0.30

RIGID_BOX_SPECS = (
    {
        "name": "carton_small",
        "size": (0.25, 0.20, 0.18),
        "mass": 0.62,
        # Incoming rigid parcel waiting on the left side of the static table.
        # Start at the native contact equilibrium instead of dropping a
        # yawed triangle mesh onto one corner before the hands arrive.
        "position": (-1.02, 0.015, WORKTABLE_TOP_Z + 0.091),
        "rotation_degrees": (0.0, 0.0, 0.0),
        "color": (0.70, 0.43, 0.20, 1.0),
    },
    {
        "name": "carton_wide",
        "size": (0.31, 0.24, 0.15),
        "mass": 0.84,
        # Keep the rigid carton ahead of the gripped mailer so the interactive
        # x1 profile demonstrates transport instead of a deliberate rear-end
        # impact during braking.
        "position": (0.60, BELT_CENTER_Y + 0.10, BELT_TOP_Z + 0.077),
        "rotation_degrees": (0.0, 0.0, -7.0),
        "color": (0.82, 0.55, 0.25, 1.0),
    },
    {
        "name": "carton_tall",
        "size": (0.22, 0.18, 0.27),
        "mass": 0.76,
        "position": (0.96, BELT_CENTER_Y - 0.11, BELT_TOP_Z + 0.137),
        "rotation_degrees": (0.0, 0.0, 9.0),
        "color": (0.62, 0.34, 0.16, 1.0),
    },
)

SOFT_PACKAGE_SPECS = (
    {
        "name": "soft_mailer_blue",
        "model": "thin_shell",
        "size": (0.44, 0.32, 0.13),
        "position": (
            MANIPULATION_STATION_X,
            -0.375,
            # The lowest film node starts 1 mm outside its 10 mm point-cloud
            # collider. This avoids a 200 mm impact before the hands arrive.
            WORKTABLE_TOP_Z + 0.0461,
        ),
        "rotation_degrees": (0.0, 0.0, 2.0),
        # Carry most parcel inertia on the closed film so a fingertip grasp
        # moves the package as one object; the light inner core only supports
        # its volume through native deformable contact.
        # The 0.35 kg total models a filled poly mailer rather than the earlier
        # 1.12 kg parcel, which could not be rolled realistically by the two
        # palm contacts while the table supports its weight.
        "density": 675.3040236312168,
        "young_modulus": 1.6e5,
        "poisson_ratio": 0.36,
        "damping": 7.0,
        "thickness": 1.2e-3,
        "bending_stiffness": 1.2e-4,
        "cells": (34, 25),
        "color": (0.10, 0.36, 0.64, 1.0),
        "visible": True,
    },
    {
        "name": "soft_mailer_blue_fill",
        "model": "volumetric",
        "internal_fill": True,
        "size": (0.38, 0.27, 0.070),
        "position": (
            MANIPULATION_STATION_X,
            -0.375,
            # Keep the core entirely inside the asymmetric closed film. It
            # falls about 14 mm onto the lower sheet before the two native
            # deformables make contact and establish the parcel volume.
            WORKTABLE_TOP_Z + 0.0361,
        ),
        "rotation_degrees": (0.0, 0.0, 2.0),
        "density": 8.942111762007034,
        "young_modulus": 7.0e3,
        "poisson_ratio": 0.43,
        "damping": 7.0,
        "cells": (12, 9, 5),
        "color": (0.04, 0.08, 0.12, 0.0),
        "visible": False,
    },
    {
        "name": "soft_pouch_yellow",
        "model": "thin_shell",
        "size": (0.29, 0.205, 0.120),
        # The second mailer starts above the upstream table and settles under
        # its own weight. Its visible film is finer than the blue mailer and
        # encloses a separate soft fill body, avoiding the old solid trapezoid
        # silhouette while retaining real volume during the later turnover.
        "position": (
            -0.42,
            0.055,
            # The generated 8.1059 mm shell radius leaves the same 1 mm
            # authored contact gap as the blue mailer and rigid carton.
            WORKTABLE_TOP_Z + 0.0415059,
        ),
        "rotation_degrees": (0.0, 0.0, 8.0),
        # Scale payload mass with film area. The previous 0.24 kg shell made
        # this compact pouch carry almost twice the areal load of the larger
        # blue mailer, so a real LEAP pinch slid along the seam instead of
        # lifting it. This density gives a 0.14 kg closed film.
        "density": 693.9408647017538,
        "young_modulus": 1.2e5,
        "poisson_ratio": 0.38,
        "damping": 6.0,
        "thickness": 1.1e-3,
        # The fine 36 x 27 film needs enough edge bending resistance to keep
        # the unsupported heat-sealed flange flat instead of rolling outward.
        "bending_stiffness": 1.6e-4,
        "cells": (36, 27),
        "color": (0.88, 0.56, 0.08, 1.0),
        "visible": True,
    },
    {
        "name": "soft_pouch_yellow_fill",
        "model": "volumetric",
        "internal_fill": True,
        "size": (0.25, 0.17, 0.065),
        "position": (
            -0.42,
            0.055,
            # Match the blue mailer's short internal settling distance so the
            # first shell contact stays below the 1 mm penetration budget.
            WORKTABLE_TOP_Z + 0.0365059,
        ),
        "rotation_degrees": (0.0, 0.0, 8.0),
        # The soft contents contribute another 25 g, for a plausible 165 g
        # small parcel while retaining enough volume to form a pillow profile.
        "density": 13.093466369750443,
        "young_modulus": 5.0e3,
        "poisson_ratio": 0.44,
        "damping": 6.0,
        # Extra through-thickness layers resolve a rounded pillow profile:
        # the contents bulge at mid-height and taper toward both film sheets.
        "cells": (14, 11, 8),
        "side_rounding": 0.08,
        "color": (0.08, 0.06, 0.02, 0.0),
        "visible": False,
    },
)

SCENE_ROOT_NAME = "conveyor_packages"
BELT_ROBOT_NAME = "conveyor"
BELT_LINK_NAME = "belt_surface"
BELT_DRIVE_FRICTION = 0.92
RIGID_BOX_NAMES = tuple(spec["name"] for spec in RIGID_BOX_SPECS)
RIGID_BOX_MASSES = tuple(spec["mass"] for spec in RIGID_BOX_SPECS)
SOFT_PACKAGE_NAMES = tuple(spec["name"] for spec in SOFT_PACKAGE_SPECS)
