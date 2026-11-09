"""EPA overflow stays visible, and enlarged scratch space survives graph capture."""
import os

import pytest
import torch

from gobot.sim.providers.base import CompiledSceneArtifact, SimulationCapacityError, _artifact_digest
from gobot.sim.providers.mujoco_warp import MuJoCoWarpProvider


def test_epa_overflow_is_not_a_solver_iteration_limit():
    provider = MuJoCoWarpProvider.__new__(MuJoCoWarpProvider)
    provider._closed = False
    provider._torch = torch
    provider._arrays = {'overflow': torch.tensor([(1 << 8) | (1 << 9)], dtype=torch.int32)}
    with pytest.raises(SimulationCapacityError, match='EPA needs a larger horizon buffer') as caught:
        provider.assert_no_overflow()
    assert 'nconmax/njmax do not control it' in str(caught.value)
    assert 'env 0: EPA horizon buffer' in str(caught.value)


@pytest.mark.skipif(os.environ.get('GOBOT_RUN_WARP_EPA_GPU_TEST') != '1', reason='requires CUDA')
def test_epa_capacity_restores_contact_with_cuda_graphs(monkeypatch):
    import mujoco
    import mujoco_warp as mjw
    import warp as wp
    from mujoco_warp._src import collision_convex

    xml = '''<mujoco>
      <option gravity="0 0 0" ccd_iterations="500" ccd_tolerance="0.000001"/>
      <worldbody>
        <geom type="cylinder" size="1 .5"/>
        <body pos="-.4143508375 -.2631894946 .3012744784"
              quat=".8294911981 -.1058898270 -.2205973268 -.5020642281">
          <freejoint/><geom type="cylinder" size="1 .5"/>
        </body>
      </worldbody>
    </mujoco>'''
    cpu = mujoco.MjModel.from_xml_string(xml)
    cpu_data = mujoco.MjData(cpu)
    mujoco.mj_forward(cpu, cpu_data)
    assert cpu_data.ncon > 0
    wp.init()
    with wp.ScopedDevice('cuda:0'):
        model = mjw.put_model(cpu)
        data = mjw.put_data(cpu, mujoco.MjData(cpu), nworld=1, nconmax=8, njmax=40)
        # Force a reproducible capacity failure, independent of a robot's fall pose.
        monkeypatch.setattr(collision_convex, 'MJ_MAX_EPAHORIZON', 5)
        mjw.forward(model, data)
        wp.synchronize()
        assert data.overflow.numpy()[0] & int(mjw.OverflowType.EPA_HORIZON)
        assert data.nacon.numpy()[0] == 0

    monkeypatch.setattr(collision_convex, 'MJ_MAX_EPAHORIZON', 24)
    artifact = CompiledSceneArtifact(
        schema_version=3, producer='mujoco', format='mjcf', content=xml,
        content_digest=_artifact_digest(xml), producer_version=mujoco.mj_versionString(),
        dimensions={'nq': cpu.nq, 'nv': cpu.nv, 'nu': cpu.nu}, robots=(), controls=(), terrain_geom_groups=())
    provider = MuJoCoWarpProvider(artifact, num_envs=1, nconmax=8, njmax=40, capture_graphs=True)
    try:
        assert provider.capacities['epa_horizon'] == 64
        assert provider._model.opt.warn_overflow & int(mjw.OverflowType.EPA_HORIZON)
        mask = torch.ones(1, device='cuda:0', dtype=torch.bool)
        qpos = torch.as_tensor(cpu.qpos0, device='cuda:0', dtype=torch.float32).reshape(1, -1)
        provider.reset(mask, qpos=qpos, qvel=torch.zeros_like(provider.arrays['qvel']))
        provider.forward()
        provider.synchronize()
        provider.assert_no_overflow()
        provider.assert_finite()
        assert provider._data.nacon.numpy()[0] > 0
        assert provider._data.contact.dist.numpy()[0] == pytest.approx(cpu_data.contact[0].dist, abs=2.e-4)
        for _ in range(3):
            provider.step()
        provider.synchronize()
        provider.assert_no_overflow()
        provider.assert_finite()
    finally:
        provider.close()
