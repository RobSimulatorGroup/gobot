# Gobot Examples

Gobot ships small, self-contained example projects in the source tree under
`examples/`. Python wheels install the same projects into `gobot/examples/`
inside the Python package.

For a pip install, this means the examples live under the installed package
directory, for example:

```text
<python-env>/lib/pythonX.Y/site-packages/gobot/examples
```

You can query the exact path from Python:

```python
from importlib import resources

print(resources.files("gobot").joinpath("examples"))
```

## Editor Discovery

When the editor opens without a current project, the Resources panel shows an
`Examples` section. The editor discovers example projects by reading
`example_roots` from `~/.gobot/projects.json`:

```json
{
    "example_roots": [
        "/path/to/site-packages/gobot/examples"
    ],
    "projects": []
}
```

The packaged Python launcher updates this list before running `gobot_editor`.
It moves the installed package's examples root to the front of the list.
For a source checkout, the editor falls back to the build-time source
`examples/` directory and writes it into `projects.json`.

Each immediate child directory is treated as an example project when it contains
at least one `.jscn` scene file. Opening an example sets that directory as the
current project, so `res://` paths resolve within the example project.
If multiple roots contain the same example directory name, the editor shows the
first one from `example_roots` and hides the later duplicates.

## Cartpole

`examples/cartpole` is a MuJoCo-backed inverted pendulum project. It contains:

- `cartpole.jscn`: the editable Gobot scene.
- `scripts/cartpole.py`: a Python `NodeScript` controller attached to the root
  robot node.

The script uses a hybrid cart controller: it first straightens the pole when the
initial angle is too large, then switches to the cart-position controller once
the state is near the target.

## Packaging Rules

The Python package install step includes:

- `.jscn` scene files.
- `.py` scripts.

Generated Python cache files and directories are excluded.
