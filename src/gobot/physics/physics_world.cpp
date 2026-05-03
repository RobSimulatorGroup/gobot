/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
*/

#include "gobot/physics/physics_world.hpp"

#include <utility>

#include "gobot/core/registration.hpp"
#include "gobot/scene/collision_shape_3d.hpp"
#include "gobot/scene/joint_3d.hpp"
#include "gobot/scene/link_3d.hpp"
#include "gobot/scene/mesh_instance_3d.hpp"
#include "gobot/scene/node.hpp"
#include "gobot/scene/resources/box_shape_3d.hpp"
#include "gobot/scene/resources/cylinder_shape_3d.hpp"
#include "gobot/scene/resources/sphere_shape_3d.hpp"
#include "gobot/scene/robot_3d.hpp"

namespace gobot {
namespace {

Affine3 ResolveNodeGlobalTransform(const Node3D* node, const Affine3& parent_global_transform) {
    if (node == nullptr) {
        return parent_global_transform;
    }

    return node->IsInsideTree() ? node->GetGlobalTransform() : parent_global_transform * node->GetTransform();
}

bool HasVisualMeshDescendant(const Node* node, const Node* root_node) {
    if (!node) {
        return false;
    }

    if (node != root_node && Object::PointerCastTo<Link3D>(node)) {
        return false;
    }

    if (auto mesh_instance = Object::PointerCastTo<MeshInstance3D>(node)) {
        return mesh_instance->GetMesh().IsValid();
    }

    for (std::size_t i = 0; i < node->GetChildCount(); ++i) {
        if (HasVisualMeshDescendant(node->GetChild(static_cast<int>(i)), root_node)) {
            return true;
        }
    }

    return false;
}

bool HasCollisionShapeDescendant(const Node* node, const Node* root_node) {
    if (!node) {
        return false;
    }

    if (node != root_node && Object::PointerCastTo<Link3D>(node)) {
        return false;
    }

    if (Object::PointerCastTo<CollisionShape3D>(node)) {
        return true;
    }

    for (std::size_t i = 0; i < node->GetChildCount(); ++i) {
        if (HasCollisionShapeDescendant(node->GetChild(static_cast<int>(i)), root_node)) {
            return true;
        }
    }

    return false;
}

bool IsImplicitVirtualRootLink(const Link3D* link, const PhysicsRobotSnapshot& robot_snapshot) {
    if (link == nullptr || link->GetRole() == LinkRole::VirtualRoot) {
        return link != nullptr;
    }

    const bool has_inertial = link->HasInertial() ||
                              link->GetMass() > 0.0 ||
                              !link->GetInertiaDiagonal().isZero(CMP_EPSILON) ||
                              !link->GetInertiaOffDiagonal().isZero(CMP_EPSILON);
    if (has_inertial) {
        return false;
    }

    for (const PhysicsJointSnapshot& joint : robot_snapshot.joints) {
        if (joint.child_link == link->GetName()) {
            return false;
        }
    }

    if (HasCollisionShapeDescendant(link, link)) {
        return false;
    }

    return !HasVisualMeshDescendant(link, link);
}

PhysicsShapeSnapshot CaptureShapeSnapshot(const CollisionShape3D* collision_shape,
                                          const Affine3& global_transform) {
    PhysicsShapeSnapshot snapshot;
    snapshot.node = collision_shape;
    snapshot.global_transform = global_transform;
    snapshot.disabled = collision_shape->IsDisabled();

    const Ref<Shape3D>& shape = collision_shape->GetShape();
    if (!shape.IsValid()) {
        return snapshot;
    }

    if (auto box = dynamic_pointer_cast<BoxShape3D>(shape)) {
        snapshot.type = PhysicsShapeType::Box;
        snapshot.box_size = box->GetSize();
    } else if (auto sphere = dynamic_pointer_cast<SphereShape3D>(shape)) {
        snapshot.type = PhysicsShapeType::Sphere;
        snapshot.radius = sphere->GetRadius();
    } else if (auto cylinder = dynamic_pointer_cast<CylinderShape3D>(shape)) {
        snapshot.type = PhysicsShapeType::Cylinder;
        snapshot.radius = cylinder->GetRadius();
        snapshot.height = cylinder->GetHeight();
    }

    return snapshot;
}

void CollectRobotNodes(const Node* node,
                       PhysicsRobotSnapshot* robot_snapshot,
                       std::vector<PhysicsShapeSnapshot>* loose_collision_shapes,
                       const Affine3& parent_global_transform) {
    if (!node) {
        return;
    }

    const Node3D* node_3d = Object::PointerCastTo<Node3D>(node);
    const Affine3 node_global_transform = ResolveNodeGlobalTransform(node_3d, parent_global_transform);

    if (auto link = Object::PointerCastTo<Link3D>(node)) {
        PhysicsLinkSnapshot link_snapshot;
        link_snapshot.node = link;
        link_snapshot.name = link->GetName();
        link_snapshot.role = link->GetRole() == LinkRole::VirtualRoot
                                     ? PhysicsLinkRole::VirtualRoot
                                     : PhysicsLinkRole::Physical;
        link_snapshot.global_transform = node_global_transform;
        link_snapshot.mass = link->GetMass();
        link_snapshot.center_of_mass = link->GetCenterOfMass();
        link_snapshot.inertia_diagonal = link->GetInertiaDiagonal();
        link_snapshot.inertia_off_diagonal = link->GetInertiaOffDiagonal();

        for (std::size_t i = 0; i < node->GetChildCount(); ++i) {
            if (auto collision_shape = Object::PointerCastTo<CollisionShape3D>(node->GetChild(static_cast<int>(i)))) {
                link_snapshot.collision_shapes.emplace_back(CaptureShapeSnapshot(
                        collision_shape,
                        ResolveNodeGlobalTransform(collision_shape, node_global_transform)));
            }
        }

        robot_snapshot->links.emplace_back(std::move(link_snapshot));
    } else if (auto joint = Object::PointerCastTo<Joint3D>(node)) {
        PhysicsJointSnapshot joint_snapshot;
        joint_snapshot.node = joint;
        joint_snapshot.name = joint->GetName();
        joint_snapshot.parent_link = joint->GetParentLink();
        joint_snapshot.child_link = joint->GetChildLink();
        joint_snapshot.global_transform = node_global_transform;
        joint_snapshot.axis = joint->GetAxis();
        joint_snapshot.lower_limit = joint->GetLowerLimit();
        joint_snapshot.upper_limit = joint->GetUpperLimit();
        joint_snapshot.effort_limit = joint->GetEffortLimit();
        joint_snapshot.velocity_limit = joint->GetVelocityLimit();
        joint_snapshot.joint_position = joint->GetJointPosition();
        joint_snapshot.joint_type = static_cast<int>(joint->GetJointType());
        robot_snapshot->joints.emplace_back(std::move(joint_snapshot));
    } else if (auto collision_shape = Object::PointerCastTo<CollisionShape3D>(node)) {
        if (!Object::PointerCastTo<Link3D>(node->GetParent())) {
            loose_collision_shapes->emplace_back(CaptureShapeSnapshot(collision_shape, node_global_transform));
        }
    }

    for (std::size_t i = 0; i < node->GetChildCount(); ++i) {
        CollectRobotNodes(node->GetChild(static_cast<int>(i)),
                          robot_snapshot,
                          loose_collision_shapes,
                          node_global_transform);
    }
}

void CollectSceneNodes(const Node* node,
                       PhysicsSceneSnapshot* snapshot,
                       const Affine3& parent_global_transform) {
    if (!node) {
        return;
    }

    const Node3D* node_3d = Object::PointerCastTo<Node3D>(node);
    const Affine3 node_global_transform = ResolveNodeGlobalTransform(node_3d, parent_global_transform);

    if (auto robot = Object::PointerCastTo<Robot3D>(node)) {
        PhysicsRobotSnapshot robot_snapshot;
        robot_snapshot.node = robot;
        robot_snapshot.name = robot->GetName();
        robot_snapshot.source_path = robot->GetSourcePath();
        CollectRobotNodes(node, &robot_snapshot, &snapshot->loose_collision_shapes, parent_global_transform);
        for (PhysicsLinkSnapshot& link : robot_snapshot.links) {
            if (IsImplicitVirtualRootLink(link.node, robot_snapshot)) {
                link.role = PhysicsLinkRole::VirtualRoot;
            }
        }
        snapshot->total_link_count += robot_snapshot.links.size();
        snapshot->total_joint_count += robot_snapshot.joints.size();
        for (const PhysicsLinkSnapshot& link : robot_snapshot.links) {
            snapshot->total_collision_shape_count += link.collision_shapes.size();
        }
        snapshot->robots.emplace_back(std::move(robot_snapshot));
        return;
    }

    if (auto collision_shape = Object::PointerCastTo<CollisionShape3D>(node)) {
        snapshot->loose_collision_shapes.emplace_back(CaptureShapeSnapshot(collision_shape, node_global_transform));
        ++snapshot->total_collision_shape_count;
    }

    for (std::size_t i = 0; i < node->GetChildCount(); ++i) {
        CollectSceneNodes(node->GetChild(static_cast<int>(i)), snapshot, node_global_transform);
    }
}

const PhysicsRobotState* FindRobotState(const PhysicsSceneState& state, const std::string& robot_name) {
    for (const PhysicsRobotState& robot_state : state.robots) {
        if (robot_state.name == robot_name) {
            return &robot_state;
        }
    }

    return nullptr;
}

const PhysicsLinkState* FindLinkState(const PhysicsRobotState& robot_state, const std::string& link_name) {
    for (const PhysicsLinkState& link_state : robot_state.links) {
        if (link_state.link_name == link_name) {
            return &link_state;
        }
    }

    return nullptr;
}

const PhysicsJointState* FindPreviousJointState(const PhysicsRobotState& robot_state, const std::string& joint_name) {
    for (const PhysicsJointState& joint_state : robot_state.joints) {
        if (joint_state.joint_name == joint_name) {
            return &joint_state;
        }
    }

    return nullptr;
}

} // namespace

const PhysicsWorldSettings& PhysicsWorld::GetSettings() const {
    return settings_;
}

void PhysicsWorld::SetSettings(const PhysicsWorldSettings& settings) {
    settings_ = settings;
}

bool PhysicsWorld::BuildFromScene(const Node* scene_root) {
    if (!CaptureSceneSnapshot(scene_root)) {
        return false;
    }

    ResetSceneStateFromSnapshot();
    return true;
}

bool PhysicsWorld::RestoreCompatibleState(const PhysicsSceneState& previous_state) {
    for (PhysicsRobotState& robot_state : scene_state_.robots) {
        const PhysicsRobotState* previous_robot_state = FindRobotState(previous_state, robot_state.name);
        if (previous_robot_state == nullptr) {
            continue;
        }

        for (PhysicsLinkState& link_state : robot_state.links) {
            const PhysicsLinkState* previous_link_state = FindLinkState(*previous_robot_state, link_state.link_name);
            if (previous_link_state == nullptr || previous_link_state->role != link_state.role) {
                continue;
            }

            link_state.global_transform = previous_link_state->global_transform;
            link_state.linear_velocity = previous_link_state->linear_velocity;
            link_state.angular_velocity = previous_link_state->angular_velocity;
        }

        for (PhysicsJointState& joint_state : robot_state.joints) {
            const PhysicsJointState* previous_joint_state =
                    FindPreviousJointState(*previous_robot_state, joint_state.joint_name);
            if (previous_joint_state == nullptr || previous_joint_state->joint_type != joint_state.joint_type) {
                continue;
            }

            joint_state.position = previous_joint_state->position;
            joint_state.velocity = previous_joint_state->velocity;
            joint_state.effort = previous_joint_state->effort;
            joint_state.control_mode = previous_joint_state->control_mode;
            joint_state.target_position = previous_joint_state->target_position;
            joint_state.target_velocity = previous_joint_state->target_velocity;
            joint_state.target_effort = previous_joint_state->target_effort;
        }
    }

    last_error_.clear();
    return true;
}

void PhysicsWorld::Reset() {
    ResetSceneStateFromSnapshot();
}

void PhysicsWorld::Step(RealType delta_time) {
    GOB_UNUSED(delta_time);
}

bool PhysicsWorld::SetJointControl(const std::string& robot_name,
                                   const std::string& joint_name,
                                   PhysicsJointControlMode control_mode,
                                   RealType target) {
    PhysicsJointState* joint_state = FindJointState(robot_name, joint_name);
    if (!joint_state) {
        SetLastError(fmt::format("Cannot set control for missing joint '{}::{}'.", robot_name, joint_name));
        return false;
    }

    joint_state->control_mode = control_mode;
    switch (control_mode) {
        case PhysicsJointControlMode::Passive:
            joint_state->target_position = joint_state->position;
            joint_state->target_velocity = 0.0;
            joint_state->target_effort = 0.0;
            break;
        case PhysicsJointControlMode::Position:
            joint_state->target_position = target;
            break;
        case PhysicsJointControlMode::Velocity:
            joint_state->target_velocity = target;
            break;
        case PhysicsJointControlMode::Effort:
            joint_state->target_effort = target;
            break;
    }

    last_error_.clear();
    return true;
}

const PhysicsSceneSnapshot& PhysicsWorld::GetSceneSnapshot() const {
    return scene_snapshot_;
}

const PhysicsSceneState& PhysicsWorld::GetSceneState() const {
    return scene_state_;
}

bool PhysicsWorld::CaptureSceneSnapshot(const Node* scene_root) {
    scene_snapshot_ = {};
    scene_state_ = {};
    if (!scene_root) {
        SetLastError("Cannot build a physics world from a null scene root.");
        return false;
    }

    CollectSceneNodes(scene_root, &scene_snapshot_, Affine3::Identity());
    last_error_.clear();
    return true;
}

PhysicsJointState* PhysicsWorld::FindJointState(const std::string& robot_name,
                                                const std::string& joint_name) {
    for (PhysicsRobotState& robot_state : scene_state_.robots) {
        if (robot_state.name != robot_name) {
            continue;
        }

        for (PhysicsJointState& joint_state : robot_state.joints) {
            if (joint_state.joint_name == joint_name) {
                return &joint_state;
            }
        }
    }

    return nullptr;
}

void PhysicsWorld::ResetSceneStateFromSnapshot() {
    scene_state_ = {};
    scene_state_.robots.reserve(scene_snapshot_.robots.size());

    for (const PhysicsRobotSnapshot& robot_snapshot : scene_snapshot_.robots) {
        PhysicsRobotState robot_state;
        robot_state.node = robot_snapshot.node;
        robot_state.name = robot_snapshot.name;
        robot_state.links.reserve(robot_snapshot.links.size());
        robot_state.joints.reserve(robot_snapshot.joints.size());

        for (const PhysicsLinkSnapshot& link_snapshot : robot_snapshot.links) {
            PhysicsLinkState link_state;
            link_state.node = link_snapshot.node;
            link_state.robot_name = robot_snapshot.name;
            link_state.link_name = link_snapshot.name;
            link_state.role = link_snapshot.role;
            link_state.global_transform = link_snapshot.global_transform;
            robot_state.links.emplace_back(std::move(link_state));
            ++scene_state_.total_link_count;
        }

        for (const PhysicsJointSnapshot& joint_snapshot : robot_snapshot.joints) {
            PhysicsJointState joint_state;
            joint_state.node = joint_snapshot.node;
            joint_state.robot_name = robot_snapshot.name;
            joint_state.joint_name = joint_snapshot.name;
            joint_state.joint_type = joint_snapshot.joint_type;
            joint_state.position = joint_snapshot.joint_position;
            joint_state.target_position = joint_snapshot.joint_position;
            robot_state.joints.emplace_back(std::move(joint_state));
            ++scene_state_.total_joint_count;
        }

        scene_state_.robots.emplace_back(std::move(robot_state));
    }
}

void PhysicsWorld::SetLastError(std::string error) {
    last_error_ = std::move(error);
}

} // namespace gobot

GOBOT_REGISTRATION {

    QuickEnumeration_<PhysicsBackendType>("PhysicsBackendType");
    QuickEnumeration_<PhysicsShapeType>("PhysicsShapeType");
    QuickEnumeration_<PhysicsJointControlMode>("PhysicsJointControlMode");

    Class_<PhysicsWorld>("PhysicsWorld")
            .method("is_available", &PhysicsWorld::IsAvailable)
            .method("get_last_error", &PhysicsWorld::GetLastError)
            .method("reset", &PhysicsWorld::Reset)
            .method("step", &PhysicsWorld::Step)
            .method("set_joint_control", &PhysicsWorld::SetJointControl);

    gobot::Type::register_wrapper_converter_for_base_classes<Ref<PhysicsWorld>, Ref<RefCounted>>();

};
