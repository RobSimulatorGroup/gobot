"""Snapshot, contact, replay and load-transfer contracts for the VBD provider."""
from copy import deepcopy
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import gobot
from gobot.sim.providers.newton_vbd import NewtonVBDConfig, NewtonVBDProvider
from gobot.sim.providers._newton_vbd_model import interacts


EXAMPLE = Path(__file__).resolve().parents[2] / 'examples/conveyor_packages'


def test_engine_snapshot_owns_geometry_and_preserves_authored_transforms():
    context = gobot.app.create_context()
    try:
        context.set_project_path(str(EXAMPLE))
        root = context.load_scene('res://conveyor_packages.jscn')
        snapshot = context.compile_physics_scene_snapshot()
        assert not context.has_world
        assert snapshot['schema_version'] == 1
        assert len(snapshot['deformables']) == 4
        for robot in snapshot['robots']:
            if not robot['name'].startswith('leap_'):
                continue
            for link in robot['links']:
                for shape in link['shapes']:
                    # Retain the grasp pads while allowing the housing to slide.
                    expected = 3. if shape['name'].endswith('_tip_collision') else .4
                    assert shape['material']['friction'] == pytest.approx(expected)
        table = snapshot['robots'][0]['links'][0]
        assert table['mass'] == 300.
        np.testing.assert_allclose(table['shapes'][1]['transform']['matrix'][:3, 3], [-.15, -.335, .52])
        before = deepcopy(snapshot['deformables'][0]['vertices'])
        snapshot['deformables'][0]['vertices'][0] = [99., 99., 99.]
        fresh = context.compile_physics_scene_snapshot()
        np.testing.assert_equal(fresh['deformables'][0]['vertices'], before)
        root.position = (2., 3., 4.)
        moved = context.compile_physics_scene_snapshot()
        np.testing.assert_allclose(moved['robots'][0]['links'][0]['global_transform']['matrix'][:3, 3], [2., 3., 4.])
        context.clear_scene()
        assert len(fresh['deformables'][0]['vertices']) > 0
    finally:
        context.clear_scene()


def test_config_and_collision_filter_contract():
    for values in ({'substeps': 0}, {'iterations': True}, {'particle_radius': float('nan')},
                   {'fixed_time_step': -.1}, {'soft_friction': -1.},
                   {'soft_contact_stiffness': 0.}, {'soft_contact_damping': -1.},
                   {'rigid_soft_contact_damping': -1.}, {'joint_linear_damping': -1.}):
        with pytest.raises(ValueError):
            NewtonVBDConfig(**values)
    assert interacts(dict(collision_layer=1, collision_mask=2), dict(collision_layer=2, collision_mask=0))
    assert not interacts(dict(collision_layer=1, collision_mask=0), dict(collision_layer=2, collision_mask=0))


def test_authored_mailer_seams_clear_self_contact_margin(monkeypatch):
    """A closed film must not start with its opposite sheet in self-contact."""
    monkeypatch.syspath_prepend(str(EXAMPLE))
    from conveyor_newton_runtime import default_config
    context = gobot.app.create_context()
    try:
        context.set_project_path(str(EXAMPLE))
        context.load_scene('res://conveyor_packages.jscn')
        snapshot = context.compile_physics_scene_snapshot()
    finally:
        context.clear_scene()
    margin = 2 * default_config().particle_radius
    for body in snapshot['deformables']:
        if body['model'] != 1:
            continue
        bottom, top = np.split(np.asarray(body['vertices']), 2)
        np.testing.assert_array_equal(bottom[:, :2], top[:, :2])
        assert (top[:, 2] - bottom[:, 2]).min() > margin, body['name']
        if body['name'] == 'soft_mailer_blue':
            # A fingertip extending 15 mm into the front edge must span at
            # least three rows; the coarse edge could balance on one vertex.
            centerline = bottom[np.abs(bottom[:, 0]) < 1.e-7, 1]
            assert np.count_nonzero(centerline - centerline.min() < .015) >= 3


def _small_snapshot():
    from libuipc_test_scenes import _soft_box, _kinematic_box
    context = gobot.app.create_context()
    root = gobot.create_node('Node3D', 'VBDRegression')
    ground = gobot.create_node('Robot3D', 'ground')
    root.add_child(ground)
    _kinematic_box(ground, 'support', (1., 1., .1), (0., 0., -.05))
    root.add_child(_soft_box('soft', (.08, .08, .06), (0., 0., .07)))
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'world.jscn'
        gobot.save_scene(root, str(path))
        context.set_project_path(directory)
        context.load_scene('res://world.jscn')
        snapshot = context.compile_physics_scene_snapshot()
    context.clear_scene()
    return snapshot


def test_bending_hinge_vertices_do_not_share_a_parallel_solver_color():
    import newton
    import warp as wp
    from gobot.sim.providers._newton_vbd_model import ModelCompiler
    snapshot = _small_snapshot()
    snapshot['robots'] = []
    film = snapshot['deformables'][0]
    film.update(model=1, vertices=[[0., 0., 0.], [.1, 0., 0.],
                                  [0., .1, 0.], [.1, .1, .01]],
                triangles=[0, 1, 2, 2, 1, 3], tetrahedra=[],
                thickness=.00012, bending_stiffness=.001)
    model = ModelCompiler(snapshot, NewtonVBDConfig(device='cpu'), newton, wp).compile()
    colors = model.particle_colors.numpy()
    hinges = model.edge_indices.numpy()
    interior = hinges[np.all(hinges >= 0, axis=1)]
    assert len(interior) == 1
    # The opposite corners share no triangle, but both change the same bend.
    # Three-color triangle coloring must not run those two updates together.
    assert len(np.unique(colors[interior[0]])) == 4


def test_folded_film_internal_contacts_have_equal_and_opposite_reactions():
    """A topological exclusion must not leave only one side of an edge pair."""
    import newton
    import warp as wp
    from gobot.sim.providers._newton_vbd_model import ModelCompiler
    from newton._src.solvers.vbd.particle_vbd_kernels import (
        accumulate_self_contact_force_and_hessian,
    )

    snapshot = _small_snapshot()
    snapshot['robots'] = []
    nx, ny = 7, 5
    vertices = [[.005*x, .005*y, 0.] for y in range(ny) for x in range(nx)]
    triangles = []
    for y in range(ny - 1):
        for x in range(nx - 1):
            a = y*nx + x
            triangles.extend((a, a+1, a+nx, a+1, a+nx+1, a+nx))
    snapshot['deformables'][0].update(
        model=1, vertices=vertices, triangles=triangles, tetrahedra=[],
        thickness=.00012, bending_stiffness=.001, self_collision_enabled=True)
    compiler = ModelCompiler(snapshot, NewtonVBDConfig(device='cpu'), newton, wp)
    model = compiler.compile()
    solver = newton.solvers.SolverVBD(
        model, particle_enable_self_contact=True, particle_self_contact_margin=.002,
        particle_self_contact_gap=.002, rigid_compliant_alm=True)
    compiler.finalize_self_contact_filters(solver.trimesh_collision_detector)
    state = model.state()
    q = state.particle_q.numpy()
    angle = (np.arange(len(q)) % nx) / (nx - 1) * 1.85*np.pi
    q[:, 0] = .002*np.cos(angle)
    q[:, 2] = .02 + .002*np.sin(angle)
    q += np.random.default_rng(5).normal(0., .0003, q.shape).astype(np.float32)
    state.particle_q.assign(q)
    solver.particle_q_prev.assign(q)
    solver._collision_detection_penetration_free(state)

    info = solver.trimesh_collision_detector.collision_info
    data = info.edge_colliding_edges.numpy().reshape(-1, 2)
    pairs = set()
    for offset, count, capacity in zip(info.edge_colliding_edges_offsets.numpy()[:-1],
                                      info.edge_colliding_edges_count.numpy(),
                                      info.edge_colliding_edges_buffer_sizes.numpy(), strict=True):
        assert count <= capacity
        pairs.update(map(tuple, data[offset:offset+count]))
    assert pairs
    assert all((b, a) in pairs for a, b in pairs)

    # Evaluate at one common pose, without gravity, supports or external force.
    # The sum must vanish even though individual contacts are under load. This
    # fixture produced a spurious 3.7 N resultant with the asymmetric filter.
    solver.particle_forces.zero_()
    solver.particle_hessians.zero_()
    for color in range(len(model.particle_color_groups)):
        wp.launch(accumulate_self_contact_force_and_hessian,
                  solver.particle_self_contact_evaluation_kernel_launch_size,
                  [.00025, color, solver.particle_q_prev, state.particle_q,
                   model.particle_colors, model.tri_indices, model.edge_indices,
                   solver.trimesh_collision_info, solver.particle_self_contact_margin,
                   model.soft_contact_ke, model.soft_contact_kd, model.soft_contact_mu,
                   solver.friction_epsilon, solver._self_contact_edge_edge_parallel_epsilon],
                  [solver.particle_forces, solver.particle_hessians], device=model.device)
    forces = solver.particle_forces.numpy()
    assert np.linalg.norm(forces, axis=1).sum() > 1.
    np.testing.assert_allclose(forces.sum(axis=0), 0., atol=1.e-4)


@pytest.mark.skipif(os.environ.get('GOBOT_RUN_NEWTON_GPU_TEST') != '1', reason='explicit CUDA opt-in')
def test_cuda_belt_contact_transports_and_stops_without_lifting_the_soft_body():
    provider = NewtonVBDProvider(_small_snapshot(), config=NewtonVBDConfig(
        substeps=8, rigid_soft_contact_damping=2., strain_relaxation_time=.02,
        rigid_contact_capacity=128, soft_contact_capacity=4096, body_particle_contact_capacity=4096))
    try:
        provider.set_surface_velocity('ground', 'support', (0., 0., 0.))
        provider.step(nsteps=200)
        initial = provider.state.particle_q.numpy().mean(0)
        centers, vertical_speed = [], []
        provider.set_surface_velocity('ground', 'support', (.2, 0., 0.))
        for _ in range(30):
            provider.step(nsteps=10)
            centers.append(provider.state.particle_q.numpy().mean(0))
            vertical_speed.append(np.sqrt(np.mean(provider.state.particle_qd.numpy()[:, 2]**2)))
        centers = np.array(centers)
        assert centers[-1, 0] - initial[0] > .06
        assert np.max(np.abs(centers[:, 2] - initial[2])) < .003
        assert max(vertical_speed) < .03
        # Stopped surfaces retain contact friction, with no old servo force
        # left over from the preceding moving-belt time step.
        provider.set_surface_velocity('ground', 'support', (0., 0., 0.))
        provider.step(nsteps=300)
        velocity = provider.state.particle_qd.numpy()
        assert np.sqrt(np.mean(np.sum(velocity**2, axis=1))) < .01
        assert provider.read_state()['minimum_tet_volume_ratio'] > .5
    finally:
        provider.close()


@pytest.mark.skipif(os.environ.get('GOBOT_RUN_NEWTON_GPU_TEST') != '1', reason='explicit CUDA opt-in')
def test_cuda_filled_mailer_rides_a_flat_belt_without_local_jumps(monkeypatch):
    """Exercise the closed film plus fill, not only a solid soft test block."""
    from dataclasses import replace
    monkeypatch.syspath_prepend(str(EXAMPLE))
    from conveyor_newton_runtime import default_config, prepare_snapshot
    context = gobot.app.create_context()
    try:
        context.set_project_path(str(EXAMPLE))
        context.load_scene('res://conveyor_packages.jscn')
        snapshot = prepare_snapshot(context.compile_physics_scene_snapshot())
    finally:
        context.clear_scene()
    snapshot['robots'] = [r for r in snapshot['robots'] if r['name'] in ('warehouse_frame', 'conveyor')]
    snapshot['deformables'] = [b for b in snapshot['deformables'] if b['name'].startswith('soft_mailer_blue')]
    for body in snapshot['deformables']:
        for axis, delta in enumerate((-.9, .955, 0.)):
            body['transform']['matrix'][axis][3] += delta
            body['transform']['position'][axis] += delta
    provider = NewtonVBDProvider(snapshot, config=replace(default_config(),
        rigid_contact_capacity=2048, soft_contact_capacity=32768, body_particle_contact_capacity=16384))
    try:
        provider.set_surface_velocity('conveyor', 'belt_surface', (0., 0., 0.))
        provider.step(nsteps=500)
        mass = provider.model.particle_mass.numpy()
        start = provider.state.particle_q.numpy()
        start_center = np.average(start, axis=0, weights=mass)
        count = len(snapshot['deformables'][0]['vertices'])
        previous = start[:count].copy()
        for i in range(150):
            provider.set_surface_velocity('conveyor', 'belt_surface', (.35 * min(i * .02 / .5, 1.), 0., 0.))
            provider.step(nsteps=10)
            q = provider.state.particle_q.numpy()
            velocity = provider.state.particle_qd.numpy()
            center = np.average(q, axis=0, weights=mass)
            assert abs(center[2] - start_center[2]) < .002
            assert np.max(q[:count, 2] - previous[:, 2]) < .005
            assert np.sqrt(np.mean(velocity[:count, 2]**2)) < .03
            assert q[:count, 2].min() < .565
            previous = q[:count].copy()
        assert center[0] - start_center[0] > .8
        assert provider.read_state()['minimum_tet_volume_ratio'] > .5
    finally:
        provider.close()


@pytest.mark.skipif(os.environ.get('GOBOT_RUN_NEWTON_GPU_TEST') != '1', reason='explicit CUDA opt-in')
def test_cuda_fixed_support_contact_graph_replay_and_reset():
    snapshot = _small_snapshot()
    results = []
    for graph in (False, True):
        config = NewtonVBDConfig(capture_graph=graph, self_contact=False,
                    rigid_contact_capacity=128, soft_contact_capacity=4096, body_particle_contact_capacity=4096)
        provider = NewtonVBDProvider(snapshot, config=config)
        try:
            initial = provider.snapshot_arrays(provider.scene_fields, (0,))
            for _ in range(150):
                provider.step()
            state = provider.read_state()
            np.testing.assert_allclose(state['links']['ground/support'], [0, 0, -.05, 0, 0, 0, 1], atol=1.e-8)
            q = state['deformables']['soft']['positions']
            assert q[:, 2].min() > -.001
            assert q[:, 2].max() < .065
            assert state['soft_contact_count'] > 0
            assert state['max_soft_penetration'] < .001
            assert state['peak_soft_penetration'] >= state['max_soft_penetration'] - 1.e-6
            assert state['deformables']['soft']['contact_forces'][:, 2].sum() > 0
            provider.reset()
            restored = provider.snapshot_arrays(provider.scene_fields, (0,))
            assert provider.read_state()['peak_soft_penetration'] == 0.
            for field in initial:
                np.testing.assert_array_equal(initial[field], restored[field])
            provider.step(nsteps=25)
            first = provider.read_state()['deformables']['soft']['positions']
            provider.advance_microstep(.0005)
            provider.reset()
            provider.step(nsteps=25)
            np.testing.assert_allclose(provider.read_state()['deformables']['soft']['positions'], first, atol=2.e-6)
            results.append(first)
            with pytest.raises(KeyError):
                provider.snapshot_arrays(('unknown',), (0,))
        finally:
            provider.close()
        with pytest.raises(RuntimeError):
            provider.step()
    np.testing.assert_allclose(*results, atol=2.e-6)


@pytest.mark.skipif(os.environ.get('GOBOT_RUN_NEWTON_GPU_TEST') != '1', reason='explicit CUDA opt-in')
def test_cuda_light_film_and_contents_settle_without_energy_growth():
    """Rigid-body contact impedance previously launched the 30 g film/core pair."""
    context = gobot.app.create_context()
    try:
        context.set_project_path(str(EXAMPLE))
        context.load_scene('res://conveyor_packages.jscn')
        snapshot = context.compile_physics_scene_snapshot()
    finally:
        context.clear_scene()
    snapshot['robots'] = [r for r in snapshot['robots'] if r['name'] in ('warehouse_frame', 'conveyor')]
    snapshot['deformables'] = [b for b in snapshot['deformables'] if b['name'].startswith('soft_mailer_blue')]
    provider = NewtonVBDProvider(snapshot, config=NewtonVBDConfig(
        substeps=8, contact_damping=1000., rigid_soft_contact_damping=2.,
        soft_contact_damping=.3, strain_relaxation_time=.02))
    try:
        mass = provider.model.particle_mass.numpy()
        initial = np.average(provider.state.particle_q.numpy(), axis=0, weights=mass)
        provider.step(nsteps=500)
        state = provider.read_state()
        position = provider.state.particle_q.numpy()
        velocity = provider.state.particle_qd.numpy()
        center = np.average(position, axis=0, weights=mass)
        np.testing.assert_allclose(center[:2], initial[:2], atol=.01)
        assert .56 < center[2] < initial[2]
        assert float(np.sum(mass[:, None] * velocity ** 2) / 2) < 1.e-4
        assert state['minimum_tet_volume_ratio'] > .5
        assert state['max_soft_penetration'] < .001
        count = state['soft_contact_count']
        assert count > 0
        np.testing.assert_array_equal(provider.solver.body_particle_contact_material_kd.numpy()[:count], 2.)
    finally:
        provider.close()


@pytest.mark.skipif(os.environ.get('GOBOT_RUN_NEWTON_GPU_TEST') != '1', reason='explicit CUDA opt-in')
def test_cuda_unattended_yellow_mailer_stays_quiet_after_initial_settling(monkeypatch, record_property):
    """The old 1.95 mm seam developed growing motion after about four seconds."""
    from dataclasses import replace
    monkeypatch.syspath_prepend(str(EXAMPLE))
    from conveyor_newton_runtime import default_config, prepare_snapshot
    from conveyor_newton_batch import TrialMetrics
    context = gobot.app.create_context()
    try:
        context.set_project_path(str(EXAMPLE))
        context.load_scene('res://conveyor_packages.jscn')
        snapshot = prepare_snapshot(context.compile_physics_scene_snapshot())
    finally:
        context.clear_scene()
    snapshot['robots'] = [r for r in snapshot['robots'] if r['name'] in ('warehouse_frame', 'conveyor')]
    snapshot['deformables'] = [b for b in snapshot['deformables'] if b['name'].startswith('soft_pouch_yellow')]
    provider = NewtonVBDProvider(snapshot, config=replace(default_config(),
        rigid_contact_capacity=2048, soft_contact_capacity=32768, body_particle_contact_capacity=16384))
    metrics = TrialMetrics(snapshot)
    try:
        mass = provider.model.particle_mass.numpy().astype(np.float64)
        kinetic_samples = []
        # Sample throughout eight seconds, including the delayed instability
        # window. A quiet final frame alone can hide a large earlier excursion.
        for _ in range(200):
            provider.step(nsteps=20)
            state = provider.read_state()
            metrics.update(state, 'settle_after_release')
            velocity = provider.state.particle_qd.numpy().astype(np.float64)
            kinetic_samples.append((state['time'], float(np.sum(mass[:, None] * velocity ** 2) / 2)))
        # This is motion energy, not a total-energy/conservation calculation:
        # gravity, elastic strain and contact potential are not included.
        record_property('kinetic_energy_samples_s_J', kinetic_samples)
        assert max(energy for t, energy in kinetic_samples if t >= 1.) < 1.e-6
        report = metrics.report()
        assert report['acceptance']['unattended_parcel_stability'], report['unattended_parcel']
        assert report['minimum_tet_volume_ratio'] > .5
        assert report['maximum_soft_penetration_m'] < .001
    finally:
        provider.close()


@pytest.mark.skipif(os.environ.get('GOBOT_RUN_NEWTON_GPU_TEST') != '1', reason='explicit CUDA opt-in')
def test_cuda_carton_drop_dissipates_impact_energy(monkeypatch):
    from dataclasses import replace
    from scipy.spatial.transform import Rotation
    monkeypatch.syspath_prepend(str(EXAMPLE))
    from conveyor_newton_runtime import default_config, prepare_snapshot
    context = gobot.app.create_context()
    try:
        context.set_project_path(str(EXAMPLE))
        context.load_scene('res://conveyor_packages.jscn')
        snapshot = prepare_snapshot(context.compile_physics_scene_snapshot())
    finally:
        context.clear_scene()
    snapshot['robots'] = [r for r in snapshot['robots'] if not r['name'].startswith('leap_')]
    snapshot['deformables'] = []
    provider = NewtonVBDProvider(snapshot, config=replace(default_config(), rigid_contact_capacity=2048,
                                              soft_contact_capacity=64, body_particle_contact_capacity=64))
    cartons = [r for r in snapshot['robots'] if r['name'] in ('carton_wide', 'carton_tall')]
    impacted = set()
    rebound = {r['name']: 0. for r in cartons}
    try:
        provider.set_surface_velocity('conveyor', 'belt_surface', (0., 0., 0.))
        for _ in range(1000):
            provider.step()
            poses = provider.state.body_q.numpy()
            for robot in cartons:
                link = robot['links'][0]
                body = provider.compiler.links[(robot['name'], link['name'])]
                rotation = Rotation.from_quat(poses[body, 3:]).as_matrix()
                clearance = poses[body, 2] - np.abs(rotation[2]) @ np.asarray(link['shapes'][0]['size']) / 2 - .56
                assert clearance > -.002
                if clearance < .002:
                    impacted.add(robot['name'])
                if robot['name'] in impacted:
                    rebound[robot['name']] = max(rebound[robot['name']], clearance)
        assert len(impacted) == 2
        assert rebound['carton_wide'] < .01
        assert rebound['carton_tall'] < .025
        velocity = provider.state.body_qd.numpy()
        for robot in cartons:
            body = provider.compiler.links[(robot['name'], robot['links'][0]['name'])]
            assert np.linalg.norm(velocity[body, :3]) < .02
    finally:
        provider.close()


@pytest.mark.skipif(os.environ.get('GOBOT_RUN_NEWTON_GPU_TEST') != '1', reason='explicit CUDA opt-in')
def test_cuda_conveyor_articulation_and_async_scene_lifecycle(monkeypatch):
    import time
    monkeypatch.syspath_prepend(str(EXAMPLE))
    monkeypatch.setenv('GOBOT_CONVEYOR_BACKEND', 'newton')
    from conveyor_packages_play import Script
    context = gobot.app.create_context()
    context.set_project_path(str(EXAMPLE))
    root = context.load_scene('res://conveyor_packages.jscn')
    script = Script()
    script._attach(root, root, context)
    script._ready()
    play = script._newton_play.play
    runtime = script._newton_play.runtime
    def complete():
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            result = play._poll()
            if result is not None:
                assert not result.error, result.error
                assert not result.presentation_error, result.presentation_error
                play.sync_scene()
                return result
            time.sleep(.005)
        raise AssertionError('Newton runtime did not finish within 90 seconds')
    try:
        ready = complete()
        assert runtime.info.provider_name == 'NewtonVBD'
        fields = ready.snapshot.fields
        initial = {name: np.asarray(ready.snapshot.buffer(name)).copy() for name in fields}
        runtime.subscribe(fields, environments=(0,), max_hz=1.e9)
        assert complete().operation == 'subscribe'
        for _ in range(40):
            assert play._step(.002)
            stepped = complete()
        script._process(.016)
        palm_field = 'robot.leap_left.link_pose'
        # A stationary hand must not spin as quaternion signs cross zero.
        q = np.asarray(stepped.snapshot.buffer(palm_field))[0, :, 3:]
        rest = initial[palm_field][0, :, 3:]
        assert np.min(np.abs(np.sum(q * rest, axis=1))) > .995
        play.reset()
        reset = complete()
        assert reset.clocks[0].tick == 0
        for name in fields:
            np.testing.assert_allclose(reset.snapshot.buffer(name), initial[name], atol=1.e-6)
        assert root.find('soft_mailer_blue').surface_mesh.vertex_count == 1820
    finally:
        script._exit_tree()
        context.clear_scene()
