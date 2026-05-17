# Go1 Example

This example keeps the robot project self-contained:

- `go1_scene.xml` is the MJCF source scene.
- `go1.jscn` is the Gobot scene imported from MJCF.
- `go1_env.py` is the rsl_rl-compatible MuJoCo vector environment.
- `train.py` trains a policy into `policies/`.
- `scripts/go1.py` plays a trained policy in `gobot_editor`.

Regenerate the Gobot scene after editing the MJCF:

```python
import gobot

gobot.set_project_path("/home/wqq/gobot/examples/go1")
gobot.import_mjcf_scene(
    "res://go1_scene.xml",
    "res://go1.jscn",
    name="go1",
    script="res://scripts/go1.py",
)
```

Train:

```bash
python train.py --device cpu
```

Run the editor from an editable install and open `go1.jscn`.
