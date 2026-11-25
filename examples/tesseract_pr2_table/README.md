# Tesseract PR2 arm around a table

Run the upstream seven-joint right-arm problem with native Tesseract and replay
the six-waypoint seed and solution in Gobot. Pink translucent links follow the
seed. Solid links follow the solution. The display uses the original URDF
collision geometry, including the table. Play interpolates joint positions and
uses poses exported by Tesseract FK.

From the Gobot repository root, prepare the assets once. The native bindings
run in a separate Python 3.10 environment. The editor uses the Gobot environment.

```sh
git clone https://github.com/tesseract-robotics/trajopt.git /tmp/trajopt-pr2-reference
git -C /tmp/trajopt-pr2-reference checkout --detach 4664f89f3157113998e7c02ac2fefc45469a21c4
uv venv --python 3.10 /tmp/tesseract-pr2-venv
uv pip install --python /tmp/tesseract-pr2-venv/bin/python tesseract-robotics==0.5.1
/tmp/tesseract-pr2-venv/bin/python examples/tesseract_pr2_table/prepare.py /tmp/trajopt-pr2-reference
.venv/bin/gobot_editor --path examples/tesseract_pr2_table
```

Click **Play**. The preparation script loads the original PR2 URDF, SRDF and
six seed configurations. It maps the JSON terms into the released bindings'
native API, including the joint velocity cost, cast collision cost (25 mm,
weight 20) and terminal joint target. OSQP solves the convex subproblems.
The independent Bullet cast check uses zero contact margin, as in the upstream
acceptance test. The start and end must match the requested configurations,
the seed must collide, and all five optimized segments must pass before the
script exports a replay. Running Play reuses that export.

The case data is pinned to the source revision above. The solver comes from
the 0.5.1 Python wheel. The replay shows the results of that combination.
[Upstream test](https://github.com/tesseract-robotics/trajopt/blob/4664f89f3157113998e7c02ac2fefc45469a21c4/trajopt/test/planning_unit.cpp#L65-L150)
and [problem definition](https://github.com/tesseract-robotics/trajopt/blob/4664f89f3157113998e7c02ac2fefc45469a21c4/trajopt_common/data/config/arm_around_table.json).
