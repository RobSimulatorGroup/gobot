/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "gobot/physics/physics_types.hpp"

#ifdef GOBOT_HAS_MUJOCO
#include <mujoco/mujoco.h>
#include <memory>
#include <algorithm>
#include <cmath>
#include <cctype>

namespace gobot::mujoco_detail {

inline constexpr RealType kMuJoCoActuatorEpsilon = 1.0e-9;
inline constexpr int kGobotTerrainGeomGroup = 5;
inline constexpr std::size_t kMuJoCoErrorBufferSize = 1024;

struct ModelDeleter {
    void operator()(mjModel* model) const { mj_deleteModel(model); }
};

struct CompiledModel {
    std::unique_ptr<mjModel, ModelDeleter> model;
    PhysicsSceneArtifact artifact;
    std::vector<std::string> robot_prefixes;
};

bool CompileModel(const PhysicsSceneSnapshot& snapshot,
                  const PhysicsWorldSettings& settings,
                  CompiledModel* result,
                  std::string* error);

inline bool EndsWith(const std::string& value, const std::string& suffix) {
    return value.size() >= suffix.size() &&
           value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

inline std::string SanitizeMuJoCoName(std::string name) {
    for (char& c : name) {
        const bool valid = std::isalnum(static_cast<unsigned char>(c)) || c == '_';
        if (!valid) {
            c = '_';
        }
    }

    if (name.empty()) {
        return "robot";
    }

    return name;
}

inline void SetMuJoCoVector3(double* target, const Vector3& value) {
    target[0] = value.x();
    target[1] = value.y();
    target[2] = value.z();
}

inline std::string SensorComponentName(const std::string& prefix,
                                const PhysicsSensorSnapshot& sensor,
                                std::string_view component) {
    return prefix + sensor.link_name + "_" + sensor.name + "_" + std::string(component);
}

inline void ApplyMuJoCoOptions(mjOption* option, const PhysicsWorldSettings& settings) {
    static_assert(static_cast<int>(PhysicsSolverType::ProjectedGaussSeidel) == mjSOL_PGS);
    static_assert(static_cast<int>(PhysicsSolverType::ConjugateGradient) == mjSOL_CG);
    static_assert(static_cast<int>(PhysicsSolverType::Newton) == mjSOL_NEWTON);
    static_assert(static_cast<int>(PhysicsIntegratorType::Euler) == mjINT_EULER);
    static_assert(static_cast<int>(PhysicsIntegratorType::RungeKutta4) == mjINT_RK4);
    static_assert(static_cast<int>(PhysicsIntegratorType::Implicit) == mjINT_IMPLICIT);
    static_assert(static_cast<int>(PhysicsIntegratorType::ImplicitFast) == mjINT_IMPLICITFAST);
    static_assert(static_cast<int>(PhysicsFrictionConeType::Pyramidal) == mjCONE_PYRAMIDAL);
    static_assert(static_cast<int>(PhysicsFrictionConeType::Elliptic) == mjCONE_ELLIPTIC);
    static_assert(static_cast<int>(PhysicsJacobianType::Dense) == mjJAC_DENSE);
    static_assert(static_cast<int>(PhysicsJacobianType::Sparse) == mjJAC_SPARSE);
    static_assert(static_cast<int>(PhysicsJacobianType::Auto) == mjJAC_AUTO);

    if (!option) {
        return;
    }

    option->timestep = settings.fixed_time_step;
    SetMuJoCoVector3(option->gravity, settings.gravity);
    option->solver = static_cast<int>(settings.mujoco_solver.solver);
    option->integrator = static_cast<int>(settings.mujoco_solver.integrator);
    option->cone = static_cast<int>(settings.mujoco_solver.cone);
    option->jacobian = static_cast<int>(settings.mujoco_solver.jacobian);
    option->iterations = settings.mujoco_solver.iterations;
    option->ls_iterations = settings.mujoco_solver.line_search_iterations;
    option->noslip_iterations = settings.mujoco_solver.no_slip_iterations;
    option->ccd_iterations = settings.mujoco_solver.convex_collision_iterations;
    option->tolerance = settings.mujoco_solver.tolerance;
    option->ls_tolerance = settings.mujoco_solver.line_search_tolerance;
    option->noslip_tolerance = settings.mujoco_solver.no_slip_tolerance;
    option->ccd_tolerance = settings.mujoco_solver.convex_collision_tolerance;
    option->impratio = settings.mujoco_solver.impedance_ratio;
}

} // namespace gobot::mujoco_detail
#endif
