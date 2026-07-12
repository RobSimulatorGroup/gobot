from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
INCLUDE_RE = re.compile(r'^\s*#\s*include\s*[<"]([^>"]+)[>"]')


def source_files(root: Path, suffixes: set[str]) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.suffix in suffixes)


def main() -> int:
    violations: list[str] = []
    physics_headers = source_files(ROOT / "include/gobot/physics", {".hpp", ".h"})
    physics_sources = source_files(ROOT / "src/gobot/physics", {".cpp", ".cc"})

    for path in [*physics_headers, *physics_sources]:
        if path.name == "physics_scene_compiler.cpp":
            continue
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            match = INCLUDE_RE.match(line)
            if match and match.group(1).startswith("gobot/scene/"):
                violations.append(
                    f"{path.relative_to(ROOT)}:{line_number}: physics runtime depends on Scene header "
                    f"{match.group(1)!r}"
                )

    backend_root = ROOT / "src/gobot/physics/backends"
    for path in source_files(backend_root, {".cpp", ".cc", ".hpp", ".h"}):
        text = path.read_text(encoding="utf-8")
        if "source_path" in text:
            violations.append(
                f"{path.relative_to(ROOT)}: physics backend references authoring import provenance 'source_path'"
            )

    for root in (ROOT / "include/gobot/physics", ROOT / "src/gobot/physics"):
        for path in source_files(root, {".cpp", ".cc", ".hpp", ".h"}):
            if "BuildFromScene" in path.read_text(encoding="utf-8"):
                violations.append(
                    f"{path.relative_to(ROOT)}: physics API reintroduces direct Scene traversal through BuildFromScene"
                )

    python_binding_root = ROOT / "src/gobot/python"
    forbidden_binding_tokens = {
        "mjModel": "MuJoCo model pointer",
        "mjData": "MuJoCo data pointer",
        "mj_loadXML": "direct MuJoCo XML runtime path",
        "ModelForEnvironment": "backend runtime pointer accessor",
        "joint_bindings_": "MuJoCo joint binding storage",
        "link_bindings_": "MuJoCo link binding storage",
        "sensor_bindings_": "MuJoCo sensor binding storage",
        "command_rng_": "locomotion command RNG ownership",
        "ResampleCommand(": "locomotion command sampling implementation",
        "LocomotionCommandRuntime command_runtime_": "standalone locomotion command storage",
        "foot_contact_time_": "locomotion contact-history storage",
        "foot_air_time_": "locomotion contact-history storage",
        "foot_peak_height_": "locomotion contact-history storage",
        "void CopyPhysicsState(": "locomotion robot-state extraction",
        "std::vector<float> base_position_;": "locomotion base-state storage",
        "std::vector<float> base_quaternion_;": "locomotion base-state storage",
        "std::vector<float> base_linear_velocity_body_;": "locomotion base-state storage",
        "std::vector<float> joint_position_;": "locomotion joint-state storage",
        "std::vector<float> link_position_;": "locomotion link-state storage",
        "std::vector<float> foot_position_;": "locomotion foot-state storage",
        "std::vector<float> height_scan_;": "locomotion sensor-state storage",
        "reward_terms_": "task reward scratch storage",
        "actor_obs_": "task observation scratch storage",
        "critic_obs_": "task observation scratch storage",
    }
    for path in source_files(python_binding_root, {".cpp", ".cc", ".hpp", ".h"}):
        text = path.read_text(encoding="utf-8")
        for line_number, line in enumerate(text.splitlines(), 1):
            match = INCLUDE_RE.match(line)
            if match and (
                match.group(1).startswith("mujoco/")
                or match.group(1).startswith("gobot/physics/backends/")
            ):
                violations.append(
                    f"{path.relative_to(ROOT)}:{line_number}: Python binding includes backend header "
                    f"{match.group(1)!r}"
                )
        for token, description in forbidden_binding_tokens.items():
            if token in text:
                violations.append(
                    f"{path.relative_to(ROOT)}: Python binding references {description} ({token})"
                )

    locomotion_runtime_files = (
        ROOT / "include/gobot/simulation/locomotion_command_runtime.hpp",
        ROOT / "src/gobot/simulation/locomotion_command_runtime.cpp",
        ROOT / "include/gobot/simulation/locomotion_batch_runtime.hpp",
        ROOT / "src/gobot/simulation/locomotion_batch_runtime.cpp",
    )
    for path in locomotion_runtime_files:
        text = path.read_text(encoding="utf-8")
        for token in ("pybind11", "py::", "numpy"):
            if token in text:
                violations.append(
                    f"{path.relative_to(ROOT)}: simulation locomotion runtime depends on Python token {token!r}"
                )

    physics_server_source = ROOT / "src/gobot/physics/physics_server.cpp"
    physics_server_text = physics_server_source.read_text(encoding="utf-8")
    if "gobot/physics/backends/" in physics_server_text:
        violations.append(
            "src/gobot/physics/physics_server.cpp: physics service directly includes a backend implementation"
        )

    app_context_binding = ROOT / "src/gobot/python/manual_bindings/manual_binding_app_context.cpp"
    if "GOBOT_HAS_MUJOCO" in app_context_binding.read_text(encoding="utf-8"):
        violations.append(
            "src/gobot/python/manual_bindings/manual_binding_app_context.cpp: "
            "batch binding branches on a backend compile definition"
        )

    cmake_text = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    for target_name in (
        "gobot_physics_core",
        "gobot_simulation_core",
        "gobot_physics_mujoco_cpu",
        "gobot_resource_mjcf",
    ):
        if f"add_library({target_name} OBJECT" not in cmake_text:
            violations.append(
                f"CMakeLists.txt: missing isolated engine object target {target_name!r}"
            )

    example_specific_python_tokens = ("GO1_", "go1_rough_velocity")
    for path in source_files(ROOT / "python/gobot", {".py", ".pyi"}):
        text = path.read_text(encoding="utf-8")
        for token in example_specific_python_tokens:
            if token in text:
                violations.append(
                    f"{path.relative_to(ROOT)}: core Python package contains example-specific token {token!r}"
                )

    if violations:
        print("Architecture dependency violations:")
        for violation in violations:
            print(f"- {violation}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
