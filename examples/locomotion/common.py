"""Example task selection; robot presets stay outside the engine package."""
def task_cfg(robot, *, play=False):
    if robot == 'go1':
        from examples.go1.train.go1_velocity_cfg import go1_velocity_cfg
        return go1_velocity_cfg(play=play)
    if robot == 'g1':
        from examples.g1.config import g1_velocity_cfg
        return g1_velocity_cfg(play=play)
    raise ValueError(f'unknown locomotion robot {robot!r}')
