# Go1 Example

This example keeps the robot project self-contained:

- `go1.xml` is the MJCF robot asset.
- `go1_scene.xml` is the MJCF scene that includes `go1.xml`.
- `go1.jscn` is the Gobot robot scene imported from `go1.xml`.
- `go1_scene.jscn` is the Gobot scene that references `go1.jscn`.
- `go1_env.py` is the rsl_rl-compatible MuJoCo vector environment.
- `train.py` trains a policy into `policies/`.
- `scripts/go1.py` is attached to the `go1_scene.jscn` root and plays a trained policy in `gobot_editor`.

Regenerate the Gobot scene after editing the MJCF:

```python
import gobot

gobot.set_project_path("/home/wqq/gobot/examples/go1")
gobot.import_mjcf_scene(
    "res://go1_scene.xml",
    "res://go1_scene.jscn",
    name="go1_scene",
    script="res://scripts/go1.py",
)
```

Train:

```bash
python train.py --device cpu
```

Run the editor from an editable install and open `go1_scene.jscn`.
