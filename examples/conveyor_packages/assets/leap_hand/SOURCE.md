# Source

This directory is vendored from Google DeepMind's MuJoCo Menagerie:

- Repository: https://github.com/google-deepmind/mujoco_menagerie
- Path: `leap_hand`
- Commit: `da76818e269b82289eba39808e2fb91d679d6994`
- Model: Carnegie Mellon University LEAP Hand, left and right variants
- License: MIT; see `LICENSE`

The model files and meshes are unchanged. Gobot imports `left_hand.xml` and
`right_hand.xml` and adds its floating wrist drive outside the vendored MJCF.
