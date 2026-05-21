# Go1 Example

This example keeps the robot project self-contained:

- `assets/xml/go1.xml` is the MJCF robot asset used for import.
- `assets/xml/go1_scene.xml` is the MJCF scene that includes `go1.xml`.
- `go1.jscn` is the Gobot robot scene imported from `go1.xml`.
- `go1_scene.jscn` is the Gobot scene that references `go1.jscn`.
- `train/go1_env.py` is the rsl_rl-compatible environment backed by Gobot simulation.
- `train/go1_train.py` trains a policy into `policies/`.
- `scripts/go1.py` is attached to the `go1_scene.jscn` root and plays a trained policy in `gobot_editor`.
- `project.gobot` sets `go1_scene.jscn` as the project main scene.

Regenerate the Gobot scene after editing the MJCF:

```python
import gobot

gobot.set_project_path("/home/wqq/gobot/examples/go1")
gobot.import_mjcf_scene(
    "res://assets/xml/go1_scene.xml",
    "res://go1_scene.jscn",
    name="go1_scene",
    script="res://scripts/go1.py",
)
```

Train:

```bash
cd /home/wqq/gobot/examples/go1/train
python3 go1_train.py --num_envs 256 --iterations 1500 --log_dir logs/go1
```

Run the editor from an editable install and open the `examples/go1` project.
