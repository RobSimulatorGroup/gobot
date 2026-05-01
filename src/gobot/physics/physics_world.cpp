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
#include "gobot/scene/node.hpp"
#include "gobot/scene/resources/box_shape_3d.hpp"
#include "gobot/scene/resources/cylinder_shape_3d.hpp"
#include "gobot/scene/resources/sphere_shape_3d.hpp"
#include "gobot/scene/robot_3d.hpp"

namespace gobot {
namespace {

PhysicsShapeSnapshot CaptureShapeSnapshot(const CollisionShape3D* collision_shape) {
    PhysicsShapeSnapshot snapshot;
    snapshot.node = collision_shape;
    snapshot.global_transform = collision_shape->IsInsideTree()
            ? collision_shape->GetGlobalTransform()
            : collision_shape->GetTransform();
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
                       std::vector<PhysicsShapeSnapshot>* loose_collision_shapes) {
    if (!node) {
        return;
    }

    if (auto link = Object::PointerCastTo<Link3D>(node)) {
        PhysicsLinkSnapshot link_snapshot;
        link_snapshot.node = link;
        link_snapshot.name = link->GetName();
        link_snapshot.global_transform = link->IsInsideTree() ? link->GetGlobalTransform() : link->GetTransform();
        link_snapshot.mass = link->GetMass();
        link_snapshot.center_of_mass = link->GetCenterOfMass();
        link_snapshot.inertia_diagonal = link->GetInertiaDiagonal();
        link_snapshot.inertia_off_diagonal = link->GetInertiaOffDiagonal();

        for (std::size_t i = 0; i < node->GetChildCount(); ++i) {
            if (auto collision_shape = Object::PointerCastTo<CollisionShape3D>(node->GetChild(static_cast<int>(i)))) {
                link_snapshot.collision_shapes.emplace_back(CaptureShapeSnapshot(collision_shape));
            }
        }

        robot_snapshot->links.emplace_back(std::move(link_snapshot));
    } else if (auto joint = Object::PointerCastTo<Joint3D>(node)) {
        PhysicsJointSnapshot joint_snapshot;
        joint_snapshot.node = joint;
        joint_snapshot.name = joint->GetName();
        joint_snapshot.parent_link = joint->GetParentLink();
        joint_snapshot.child_link = joint->GetChildLink();
        joint_snapshot.global_transform = joint->IsInsideTree() ? joint->GetGlobalTransform() : joint->GetTransform();
        joint_snapshot.axis = joint->GetAxis();
        joint_snapshot.lower_limit = joint->GetLowerLimit();
        joint_snapshot.upper_limit = joint->GetUpperLimit();
        joint_snapshot.effort_limit = joint->GetEffortLimit();
        joint_snapshot.velocity_limit = joint->GetVelocityLimit();
        joint_snapshot.joint_position = joint->GetJointPosition();
        joint_snapshot.joint_type = static_cast<int>(joint->GetJointType());
        robot_snapshot->joints.emplace_back(std::move(joint_snapshot));
    }

    for (std::size_t i = 0; i < node->GetChildCount(); ++i) {
        CollectRobotNodes(node->GetChild(static_cast<int>(i)), robot_snapshot, loose_collision_shapes);
    }
}

void CollectSceneNodes(const Node* node, PhysicsSceneSnapshot* snapshot) {
    if (!node) {
        return;
    }

    if (auto robot = Object::PointerCastTo<Robot3D>(node)) {
        PhysicsRobotSnapshot robot_snapshot;
        robot_snapshot.node = robot;
        robot_snapshot.name = robot->GetName();
        robot_snapshot.source_path = robot->GetSourcePath();
        CollectRobotNodes(node, &robot_snapshot, &snapshot->loose_collision_shapes);
        snapshot->total_link_count += robot_snapshot.links.size();
        snapshot->total_joint_count += robot_snapshot.joints.size();
        for (const PhysicsLinkSnapshot& link : robot_snapshot.links) {
            snapshot->total_collision_shape_count += link.collision_shapes.size();
        }
        snapshot->robots.emplace_back(std::move(robot_snapshot));
        return;
    }

    if (auto collision_shape = Object::PointerCastTo<CollisionShape3D>(node)) {
        snapshot->loose_collision_shapes.emplace_back(CaptureShapeSnapshot(collision_shape));
        ++snapshot->total_collision_shape_count;
    }

    for (std::size_t i = 0; i < node->GetChildCount(); ++i) {
        CollectSceneNodes(node->GetChild(static_cast<int>(i)), snapshot);
    }
}

} // namespace

const PhysicsWorldSettings& PhysicsWorld::GetSettings() const {
    return settings_;
}

void PhysicsWorld::SetSettings(const PhysicsWorldSettings& settings) {
    settings_ = settings;
}

bool PhysicsWorld::BuildFromScene(const Node* scene_root) {
    return CaptureSceneSnapshot(scene_root);
}

void PhysicsWorld::Reset() {
}

void PhysicsWorld::Step(RealType delta_time) {
    GOB_UNUSED(delta_time);
}

const PhysicsSceneSnapshot& PhysicsWorld::GetSceneSnapshot() const {
    return scene_snapshot_;
}

bool PhysicsWorld::CaptureSceneSnapshot(const Node* scene_root) {
    scene_snapshot_ = {};
    if (!scene_root) {
        SetLastError("Cannot build a physics world from a null scene root.");
        return false;
    }

    CollectSceneNodes(scene_root, &scene_snapshot_);
    last_error_.clear();
    return true;
}

void PhysicsWorld::SetLastError(std::string error) {
    last_error_ = std::move(error);
}

} // namespace gobot

GOBOT_REGISTRATION {

    QuickEnumeration_<PhysicsBackendType>("PhysicsBackendType");
    QuickEnumeration_<PhysicsShapeType>("PhysicsShapeType");

    Class_<PhysicsWorld>("PhysicsWorld")
            .method("is_available", &PhysicsWorld::IsAvailable)
            .method("get_last_error", &PhysicsWorld::GetLastError)
            .method("reset", &PhysicsWorld::Reset)
            .method("step", &PhysicsWorld::Step);

    gobot::Type::register_wrapper_converter_for_base_classes<Ref<PhysicsWorld>, Ref<RefCounted>>();

};
