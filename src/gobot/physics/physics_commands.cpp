#include "gobot/physics/physics_commands.hpp"

#include <algorithm>
#include <cmath>
#include <type_traits>
#include "gobot/physics/physics_world.hpp"

namespace gobot {
namespace {

bool LinkHasMotionDofs(const PhysicsRobotSnapshot& robot, const PhysicsLinkSnapshot& link) {
    if (robot.standalone_rigid_body) return std::isfinite(link.mass) && link.mass > 0;
    const auto* current = &link;
    // A fixed child can still move with an articulated ancestor. Root mass alone
    // does not determine mobility, and malformed cycles must not hang validation.
    for (std::size_t depth = 0; depth < robot.links.size(); ++depth) {
        const auto joint = std::find_if(robot.joints.begin(), robot.joints.end(),
                [&](const auto& candidate) { return candidate.child_link == current->name; });
        if (joint == robot.joints.end()) return false;
        switch (static_cast<JointType>(joint->joint_type)) {
            case JointType::Revolute:
            case JointType::Continuous:
            case JointType::Prismatic:
            case JointType::Floating:
            case JointType::Planar:
                return true;
            case JointType::Fixed:
                break;
            default:
                return false;
        }
        const auto parent = std::find_if(robot.links.begin(), robot.links.end(),
                [&](const auto& candidate) { return candidate.name == joint->parent_link; });
        if (parent == robot.links.end()) return false;
        current = &*parent;
    }
    return false;
}

} // namespace

bool ValidatePhysicsLinkForceTarget(const PhysicsSceneSnapshot& snapshot,
        const std::string& robot_name, const std::string& link_name, std::string* error) {
    const auto robot = std::find_if(snapshot.robots.begin(), snapshot.robots.end(),
            [&](const auto& candidate) { return candidate.name == robot_name; });
    if (robot != snapshot.robots.end()) {
        const auto link = std::find_if(robot->links.begin(), robot->links.end(),
                [&](const auto& candidate) { return candidate.name == link_name; });
        if (link != robot->links.end()) {
            if (LinkHasMotionDofs(*robot, *link)) {
                if (error) error->clear();
                return true;
            }
            if (error) *error = "Cannot apply external force to static link '" + robot_name + "::" + link_name + "'.";
            return false;
        }
    }
    if (error) *error = "Cannot apply external force to missing link '" + robot_name + "::" + link_name + "'.";
    return false;
}

bool ValidatePhysicsCommands(const PhysicsSceneSnapshot& snapshot,
        const PhysicsCommands& commands, std::string* error) {
    if (error) error->clear();
    for (const auto& command : commands) {
        const bool valid = std::visit([&](const auto& value) {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, PhysicsClearForcesCommand>) {
                return true;
            } else if constexpr (std::is_same_v<T, PhysicsDeformableForceCommand>) {
                const auto body = std::find_if(snapshot.deformables.begin(), snapshot.deformables.end(),
                        [&](const auto& candidate) { return candidate.stable_id == value.stable_id; });
                return body != snapshot.deformables.end() && body->vertices.size() == value.forces.size() &&
                        std::all_of(value.forces.begin(), value.forces.end(),
                                [](const Vector3& force) { return force.allFinite(); });
            } else {
                if (value.robot >= snapshot.robots.size()) return false;
                const auto& robot = snapshot.robots[value.robot];
                if constexpr (std::is_same_v<T, PhysicsJointCommand>) {
                    return value.joint < robot.joints.size() && std::isfinite(value.target) &&
                            (value.mode == PhysicsJointControlMode::Passive ||
                             value.mode == PhysicsJointControlMode::Position ||
                             value.mode == PhysicsJointControlMode::Velocity ||
                             value.mode == PhysicsJointControlMode::Effort);
                } else if constexpr (std::is_same_v<T, PhysicsJointResetCommand>) {
                    return value.joint < robot.joints.size() && std::isfinite(value.position) &&
                            std::isfinite(value.velocity);
                } else if constexpr (std::is_same_v<T, PhysicsLinkResetCommand>) {
                    return value.link < robot.links.size() && value.position.allFinite() &&
                            value.orientation.coeffs().allFinite() && value.orientation.norm() > CMP_EPSILON &&
                            value.linear_velocity.allFinite() && value.angular_velocity.allFinite();
                } else {
                    return value.link < robot.links.size() && value.point.allFinite() &&
                            value.force.allFinite() && value.target.allFinite() &&
                            ValidatePhysicsLinkForceTarget(snapshot, robot.name, robot.links[value.link].name, error);
                }
            }
        }, command);
        if (!valid) {
            if (error && error->empty()) {
                *error = "Physics command has an invalid topology index, shape, mode or non-finite value.";
            }
            return false;
        }
    }
    if (error) error->clear();
    return true;
}

bool ApplyPhysicsCommands(PhysicsWorld& world, const PhysicsCommands& commands, std::string* error) {
    if (!ValidatePhysicsCommands(world.GetSceneSnapshot(), commands, error)) return false;
    for (const auto& command : commands) {
        const bool applied = std::visit([&](const auto& value) {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, PhysicsClearForcesCommand>) {
                if (value.deformables_only) world.ClearDeformableExternalForces();
                else world.ClearExternalForces();
                return true;
            } else if constexpr (std::is_same_v<T, PhysicsDeformableForceCommand>) {
                return world.SetDeformableExternalForces(value.stable_id, value.forces);
            } else {
                const auto& robot = world.GetSceneSnapshot().robots[value.robot];
                if constexpr (std::is_same_v<T, PhysicsJointCommand>) {
                    return world.SetJointControl(robot.name, robot.joints[value.joint].name, value.mode, value.target);
                } else if constexpr (std::is_same_v<T, PhysicsJointResetCommand>) {
                    return world.ResetJointState(robot.name, robot.joints[value.joint].name, value.position, value.velocity);
                } else if constexpr (std::is_same_v<T, PhysicsLinkResetCommand>) {
                    return world.ResetLinkState(robot.name, robot.links[value.link].name, value.position,
                            value.orientation, value.linear_velocity, value.angular_velocity);
                } else {
                    return value.spring
                            ? world.SetLinkSpringForce(robot.name, robot.links[value.link].name,
                                    value.point, value.target, value.force)
                            : world.SetLinkExternalForce(robot.name, robot.links[value.link].name,
                                    value.point, value.force);
                }
            }
        }, command);
        if (!applied) {
            if (error) *error = world.GetLastError().empty() ? "Physics backend rejected a command." : world.GetLastError();
            return false;
        }
    }
    if (error) error->clear();
    return true;
}

std::size_t PhysicsCommandsBytes(const PhysicsCommands& commands) {
    std::size_t bytes = commands.size() * sizeof(PhysicsCommand);
    for (const auto& command : commands) {
        if (const auto* forces = std::get_if<PhysicsDeformableForceCommand>(&command)) {
            bytes += forces->forces.size() * sizeof(Vector3);
        }
    }
    return bytes;
}

} // namespace gobot
