# G1 single-motion tracking

This project plays a BeyondMimic policy in real physics alongside the reference
motion. Blue is the controlled robot; green is the reference. The Physics panel
provides native contact points and force visualization.

### Open an installed example

Open **g1_track** in the editor. On first open the project hook requests
`g1-track-deployment.zip` from the `g1-track-dance2-v1` Release of
`RobSimulatorGroup/gobot-example-assets` and checks its SHA-256 before
installing the assets. Subsequent opens use the local copy offline.
Set `GOBOT_G1_TRACK_URL` to use a mirror, and `GOBOT_G1_TRACK_SHA256` when
using a different deployment archive.

For a read-only wheel installation the launcher prepares a writable example
under `~/.gobot/wheel-examples/`. To create a project elsewhere:

```sh
python -m gobot.examples.g1_track.prepare --output ~/gobot-projects/g1_track
gobot_editor --path ~/gobot-projects/g1_track
```

The ZIP for the shipped presentation contains `assets/policy/{policy.pt,config.json}`
and the matching `assets/project/{project.gobot,world.jscn,motion.npz,assets/}`
at the archive root. You can also pass the URL and checksum to the preparation
command with `--download-url` and `--sha256`.

### Prepare your own tracking deployment

Prepare a deployment on the training machine first. Export verifies the original
checkpoint against the training scene, keeps the actor weights unchanged, and
records the scene assets and full physical snapshot for another machine:

```bash
python -m gobot.rl.tracking.export logs/dance2-train/model.pt \
  --output logs/dance2-deployment
```

Copy the deployment directory and its prepared motion project to this computer,
then install them into this example:

```bash
python examples/g1_track/prepare.py logs/dance2-deployment/policy.pt \
  --project-path logs/dance2-assets
uv run gobot_editor --path examples/g1_track
```

Press Play. Inference runs at 50 Hz and physics at 200 Hz. Playback starts at
frame zero and stops at the end of the motion or the first tracking failure.
The blue robot is controlled by the neural policy throughout; no reference
poses are substituted for its physical state. Stop and Play restart the motion.

Weights and motion assets are local files under `assets/` and are not bundled
in Git or wheels. The generated presentation uses `res://` paths, so the whole
prepared example can be moved without embedding a workstation path. Preparation
requires a fresh `assets/` directory; `--output` can prepare another project.

The checked-in presentation layout is for the 29-joint G1. Preparation rebuilds
it from the supplied policy's actual scene. Training commands and input formats
are in the [tracking guide](../../doc/gobot_tracking.md).
