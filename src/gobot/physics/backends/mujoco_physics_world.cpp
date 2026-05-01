/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
*/

#include "gobot/physics/backends/mujoco_physics_world.hpp"

#include "gobot/core/registration.hpp"
#include "gobot/log.hpp"

#ifdef GOBOT_HAS_MUJOCO
#include <mujoco/mujoco.h>
#endif

namespace gobot {

MuJoCoPhysicsWorld::MuJoCoPhysicsWorld()
    : available_(IsBackendAvailable()) {
    if (!available_) {
        SetLastError(GetUnavailableReason());
    }
}

MuJoCoPhysicsWorld::~MuJoCoPhysicsWorld() = default;

bool MuJoCoPhysicsWorld::IsBackendAvailable() {
#ifdef GOBOT_HAS_MUJOCO
    return true;
#else
    return false;
#endif
}

std::string MuJoCoPhysicsWorld::GetUnavailableReason() {
#ifdef GOBOT_HAS_MUJOCO
    return {};
#else
    return "MuJoCo support is not enabled. Configure with -DGOB_BUILD_MUJOCO=ON and provide the MuJoCo SDK/package.";
#endif
}

PhysicsBackendType MuJoCoPhysicsWorld::GetBackendType() const {
    return PhysicsBackendType::MuJoCoCpu;
}

bool MuJoCoPhysicsWorld::IsAvailable() const {
    return available_;
}

const std::string& MuJoCoPhysicsWorld::GetLastError() const {
    return last_error_;
}

bool MuJoCoPhysicsWorld::BuildFromScene(const Node* scene_root) {
    if (!CaptureSceneSnapshot(scene_root)) {
        return false;
    }

    if (!available_) {
        SetLastError(GetUnavailableReason());
        return false;
    }

#ifdef GOBOT_HAS_MUJOCO
    // Next phase: translate PhysicsSceneSnapshot into mjSpec/mjModel.
    LOG_INFO("MuJoCo physics scene captured: robots={}, links={}, joints={}, collision_shapes={}",
             scene_snapshot_.robots.size(),
             scene_snapshot_.total_link_count,
             scene_snapshot_.total_joint_count,
             scene_snapshot_.total_collision_shape_count);
    last_error_.clear();
    return true;
#else
    return false;
#endif
}

void MuJoCoPhysicsWorld::Reset() {
    if (!available_) {
        return;
    }
}

void MuJoCoPhysicsWorld::Step(RealType delta_time) {
    GOB_UNUSED(delta_time);
    if (!available_) {
        return;
    }
}

} // namespace gobot

GOBOT_REGISTRATION {

    Class_<MuJoCoPhysicsWorld>("MuJoCoPhysicsWorld")
            .constructor()(CtorAsRawPtr)
            .method("is_backend_available", &MuJoCoPhysicsWorld::IsBackendAvailable);

    gobot::Type::register_wrapper_converter_for_base_classes<Ref<MuJoCoPhysicsWorld>, Ref<PhysicsWorld>>();

};
