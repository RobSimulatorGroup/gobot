#pragma once

#include <variant>
#include "gobot/physics/physics_types.hpp"

namespace gobot {

class PhysicsWorld;

struct PhysicsJointCommand {
    std::size_t robot = 0;
    std::size_t joint = 0;
    PhysicsJointControlMode mode = PhysicsJointControlMode::Passive;
    RealType target = 0;
};

struct PhysicsJointResetCommand {
    std::size_t robot = 0;
    std::size_t joint = 0;
    RealType position = 0;
    RealType velocity = 0;
};

struct PhysicsLinkResetCommand {
    std::size_t robot = 0;
    std::size_t link = 0;
    Vector3 position = Vector3::Zero();
    Quaternion orientation = Quaternion::Identity();
    Vector3 linear_velocity = Vector3::Zero();
    Vector3 angular_velocity = Vector3::Zero();
};

struct PhysicsLinkForceCommand {
    std::size_t robot = 0;
    std::size_t link = 0;
    Vector3 point = Vector3::Zero();
    Vector3 force = Vector3::Zero();
    bool spring = false;
    Vector3 target = Vector3::Zero();
};

struct PhysicsDeformableForceCommand {
    std::uint64_t stable_id = 0;
    std::vector<Vector3> forces;
};

struct PhysicsClearForcesCommand {
    bool deformables_only = false;
};

using PhysicsCommand = std::variant<PhysicsJointCommand, PhysicsJointResetCommand,
        PhysicsLinkResetCommand, PhysicsLinkForceCommand, PhysicsDeformableForceCommand,
        PhysicsClearForcesCommand>;
using PhysicsCommands = std::vector<PhysicsCommand>;

// Indices address the immutable compiled topology, never SceneTree objects.
GOBOT_EXPORT bool ValidatePhysicsLinkForceTarget(const PhysicsSceneSnapshot& snapshot,
        const std::string& robot, const std::string& link, std::string* error);
GOBOT_EXPORT bool ValidatePhysicsCommands(const PhysicsSceneSnapshot& snapshot,
        const PhysicsCommands& commands, std::string* error);
GOBOT_EXPORT bool ApplyPhysicsCommands(PhysicsWorld& world, const PhysicsCommands& commands,
        std::string* error);
GOBOT_EXPORT std::size_t PhysicsCommandsBytes(const PhysicsCommands& commands);

} // namespace gobot
