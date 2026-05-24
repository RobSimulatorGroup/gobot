# Go1 Example

This example keeps the robot project self-contained:

- `assets/xml/go1.xml` is the MJCF robot asset used for import.
- `assets/xml/go1_scene.xml` is the MJCF scene that includes `go1.xml`.
- `go1.jscn` is the Gobot robot scene imported from `go1.xml`.
- `go1_scene.jscn` is the Gobot scene that references `go1.jscn`.
- `train/go1_env.py` is the rsl_rl-compatible environment backed by Gobot simulation.
- `train/go1_train.py` trains a policy into `policies/`.
- `scripts/go1.py` is attached to the `go1_scene.jscn` root and plays a trained policy in `gobot_editor`.
- `policies/go1.onnx` is the default lightweight playback policy.
- `policies/go1.pt` is the training checkpoint fallback for environments with `gobot[train]`.
- `tools/export_policy_onnx.py` converts `policies/go1.pt` to `policies/go1.onnx`.
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

Export a trained checkpoint for lightweight editor playback:

```bash
cd /home/wqq/gobot/examples/go1/tools
python3 export_policy_onnx.py --checkpoint ../policies/go1.pt --output ../policies/go1.onnx
```

The default Gobot install can play ONNX policies. Install `gobot[train]` only
when training or directly loading `.pt` checkpoints.

Run the editor from an editable install and open the `examples/go1` project.
