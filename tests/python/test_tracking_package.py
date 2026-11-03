from examples.locomotion.package import source_files


def test_source_overlay_includes_native_and_provider_changes_but_excludes_logs(tmp_path):
    wanted=('src/gobot/physics/backends/new.cpp','include/gobot/scene/new.hpp',
            'python/gobot/sim/providers/new.py','examples/g1/g1_29dof.meshes/mesh.ply',
            'examples/locomotion/protocols/tracking.json','examples/locomotion/cluster/train.sbatch',
            'examples/locomotion/cluster/bootstrap.sh','tests/python/test_tracking.py')
    ignored=('examples/g1/logs/private.json','examples/g1/assets/motions/tracking/large.csv',
             'python/gobot/__pycache__/cache.py','examples/locomotion/logs/run/source.cpp')
    for name in (*wanted,*ignored):
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('fixture')
    selected={str(path.relative_to(tmp_path)) for path in source_files(tmp_path)}
    assert set(wanted)<=selected
    assert not set(ignored)&selected
