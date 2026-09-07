from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def probe():
    pytest.importorskip("torch")
    pytest.importorskip("mujoco")
    spec = importlib.util.spec_from_file_location("ipc_articulation_ab_test", ROOT / "benchmark/ipc_articulation_ab.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_probe_scene_compiles_from_gobot_and_has_coupled_inertia(probe, tmp_path):
    import gobot
    import mujoco
    context = gobot.app.create_context()
    try:
        path = tmp_path / "probe.jscn"
        gobot.save_scene(probe.create_scene(), str(path))
        context.load_scene(str(path))
        context.fixed_time_step = probe.DT
        artifact = probe.CompiledMuJoCoIpcArtifact.from_context(context)
        job = probe.probe_job(artifact, tmp_path)
        assert len(job["joints"]) == 2
        assert len(job["soft"]["vertices"]) == 64
        assert len([link for link in job["links"] if not link["fixed"]]) == 2
        assert not artifact.ipc.deformable_attachments
        model = mujoco.MjModel.from_xml_string(artifact.mujoco.content)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        mass = np.zeros((model.nv, model.nv))
        mujoco.mj_fullM(model, data, mass)
        np.testing.assert_allclose(mass, [[3., 2.], [2., 2.]], atol=1.e-6)
        assert np.linalg.eigvalsh(mass).min() > 0.
        q_ids, dof_ids = probe.joint_layout(model, job)
        assert set(q_ids) == set(dof_ids) == {0, 1}
        job["joints"].reverse()
        reverse_q, reverse_dof = probe.joint_layout(model, job)
        np.testing.assert_array_equal(reverse_q, q_ids[::-1])
        np.testing.assert_array_equal(reverse_dof, dof_ids[::-1])
    finally:
        context.clear_scene()


@pytest.mark.skipif(os.environ.get("GOBOT_RUN_IPC_ARTICULATION_GPU_TEST") != "1",
                    reason="requires the explicit native articulation probe target and CUDA")
def test_joint_mass_prediction_without_hand_contact_and_invalid_mass(probe, tmp_path):
    import gobot
    context = gobot.app.create_context()
    native = None
    try:
        path = tmp_path / "probe.jscn"
        gobot.save_scene(probe.create_scene(contact=False), str(path))
        context.load_scene(str(path))
        context.fixed_time_step = probe.DT
        artifact = probe.CompiledMuJoCoIpcArtifact.from_context(context)
        job = probe.probe_job(artifact, tmp_path)
        module = ROOT / "build/cp313-cp313-linux_x86_64/benchmark/libgobot_ipc_articulation_probe.so"
        with pytest.raises(RuntimeError, match="exactly two"):
            probe.ArticulationProbe(module, {**job, "joints": job["joints"][:1]})
        native = probe.ArticulationProbe(module, job)
        for invalid in (np.array([[1., 2.], [2., 1.]]), np.array([[1., 2.], [0., 1.]]),
                        np.full((2, 2), np.nan)):
            with pytest.raises(RuntimeError, match="positive definite"):
                native.step(invalid, np.zeros(2))
        mass = np.array([[3., 2.], [2., 2.]])
        for prediction in (np.zeros(2), np.array([-.0001, .00003]), np.array([.00002, -.00004])):
            result = native.step(mass, prediction)
            np.testing.assert_allclose(result["delta"], prediction, atol=1.e-5, rtol=0.)
            assert np.isfinite(result["positions"]).all()
        with pytest.raises(ValueError):
            native.step(np.eye(3), np.zeros(2))
    finally:
        if native is not None:
            native.close()
        context.clear_scene()
