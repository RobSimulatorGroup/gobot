# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build

```bash
# From repo root, first time setup
git submodule update --init --recursive
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

Build outputs land in `build/bin/` and `build/lib/`.

### CMake options

| Flag | Default | Purpose |
|------|---------|---------|
| `GOB_BUILD_TESTS` | OFF | Build unit tests |
| `GOB_BUILD_EXPERIMENT` | OFF | Build experimental imgui/glfw sandbox |

### System dependencies required

```bash
# Eigen3, nlohmann_json, libpng, SDL2 must be installed
sudo apt install libeigen3-dev nlohmann-json3-dev libpng-dev libsdl2-dev
```

CPM-managed deps (fmt, spdlog, magic_enum, cxxopts) are fetched automatically.

## Tests

```bash
cmake .. -DGOB_BUILD_TESTS=On
make -j$(nproc)
ctest                        # run all tests
ctest -R test_node           # run a single test by name pattern
```

Tests use Google Test and live in `tests/core/` and `tests/scene/`.

## Architecture

The project builds two artifacts:
- `libgobot.so` — the engine shared library (everything below)
- `gobot_editor` — thin executable that links `libgobot.so` via `src/gobot/platform/linux/`

### Module layout inside `libgobot.so`

| Directory | Role |
|-----------|------|
| `src/gobot/core/` | Reflection (RTTR), resource system, RID, project settings, string utils |
| `src/gobot/scene/` | Node/Node3D scene graph, meshes, materials, cameras |
| `src/gobot/drivers/` | OpenGL (via glad) and SDL2 window/input backends |
| `src/gobot/rendering/` | Render pipeline and storage |
| `src/gobot/editor/` | ImGui-based editor panels (property inspector, scene view, console) |
| `src/gobot/main/` | Engine bootstrap / main loop |

Public headers mirror this layout under `include/gobot/`.

### Key design patterns

- **Reflection via RTTR** — types are registered with RTTR for the property inspector and serialization. Look for `RTTR_REGISTRATION` blocks.
- **Scene graph** — `Node` is the base; `Node3D` adds transform. The tree is owned/managed through the scene system.
- **Resource system** — assets (meshes, textures, shaders, materials) are reference-counted resources identified by RIDs.
- **Shaders** — GLSL sources live in `shader/glsl/`; a CMake target (`GobotShaderCompile`) generates headers into `shader/generated/` at build time, which `libgobot.so` depends on.

### Third-party (git submodules in `3rdparty/`)

imgui, glad, rttr, googletest, meshoptimizer, gli, stb, basis_universal — all built as part of the CMake tree via `3rdparty/CMakeLists.txt`.
