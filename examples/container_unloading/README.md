# Container unloading with a FANUC M-10iA

Open the project from the repository root and press **Play**. Each run plans a loaded transfer from the pickup slot to the roller conveyor using the edited scene. **Pause** holds the two robots at the same instant. **Stop** restores the scene.

```sh
.venv/bin/gobot_editor --path examples/container_unloading
```

The 400 mm wide carton starts in a 470 mm opening, with 35 mm on each side. Four-high stacks flank the carton and a long shipping box spans the top of the opening. The workcell uses the approximate inside dimensions of a 20-foot dry container, 5.90 m long, 2.35 m wide and 2.39 m high. The default camera looks through the open double doors. Both folded-steel side walls, the roof, the closed end, the door frame and the open leaves have collision envelopes. Five packed tiers fill the back of the container. A fixed joint attaches the carton to the side-mounted vacuum tool during transport. The carton starts 30 mm above its support.

The translucent pink robot and carton follow the straight joint interpolation. The solid yellow robot and carton follow the optimized motion at the same playback speed. Watch the third carton in the left-hand stack as the pink wrist swings. The solid wrist first draws the carton out of the stack, then swings it toward the conveyor. Both trajectories use the same pickup and drop poses.

| Loaded trajectory | Minimum sampled carton-to-environment signed distance | Closest obstacle |
| --- | ---: | --- |
| Straight joint interpolation | −345.3 mm | Third carton in the port stack |
| Optimized | +14.0 mm | Long carton above the slot |

## Planning and playback

The worker computes both endpoint configurations with IK, then calls Gobot's CPU trust-region SQP solver with 14 joint waypoints spaced 0.23 s apart. Only the endpoints are specified. Coal queries signed distances between the robot hulls, the attached carton and the environment. Scene poses and the joint trajectory are recomputed on each Play.

Before playback the planner checks joint and velocity limits and validates collision geometry at 313 joint configurations, including segment interiors. The acceptance thresholds are 8 mm for the environment and 1 mm for self-collision. The table reports the minimum for the *carton* over those samples. Continuous-time clearance between samples is unverified.

At the end of the planned motion, the carton is 17 mm above the rollers. The robot holds its release pose while the carton settles and moves 2.46 m along the rollers and out through the doorway in a kinematic animation. The animation uses the roller envelope and carton dimensions to stop while the carton is still fully supported. Moving the drop outside the handoff area ends the replay at the loaded pose.

## Requirements and assets

Use a Coal-enabled Gobot build and the `planning-sqp` Python extra for OSQP. The Panda example documents the build command and system dependencies. [Build instructions](../panda_trajopt/README.md#build)

The workcell dimensions and tooling are authored for this example. The FANUC M-10iA represents the six-axis industrial arm in the unloading footage. The seven STL meshes and joint frames come from ROS-Industrial's FANUC support package at revision `d8f42bd73584b255df87098395512538882caea1`. [Model source](https://github.com/ros-industrial/fanuc/tree/d8f42bd73584b255df87098395512538882caea1/fanuc_m10ia_support) · [BSD license](meshes/LICENSE)
