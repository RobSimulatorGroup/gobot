#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>

#include "gobot/core/types.hpp"
#include "gobot/physics/physics_types.hpp"

namespace gobot::python {

namespace py = pybind11;

enum class NativeVectorActionMode {
    NormalizedPosition,
    Position,
    Velocity,
    Effort
};

struct NativeVectorEnvConfig {
    std::string scene;
    std::string robot;
    PhysicsBackendType backend{PhysicsBackendType::MuJoCoCpu};
    int num_envs{1};
    int batch_size{0};
    int num_workers{0};
    RealType physics_dt{1.0 / 240.0};
    int decimation{1};
    int max_episode_steps{500};
    bool auto_reset{true};
    std::vector<std::string> controlled_joints;
    std::uint64_t seed{0};
    std::string task_json;
};

struct NativeVectorActionConfig {
    std::string name;
    std::string joint;
    NativeVectorActionMode mode{NativeVectorActionMode::NormalizedPosition};
    RealType scale{1.0};
    RealType offset{0.0};
    RealType lower{-1.0};
    RealType upper{1.0};
    std::string unit{"normalized"};
    std::vector<std::string> passive_joints;
};

void RegisterNativeVectorEnv(py::module_& module);

} // namespace gobot::python
