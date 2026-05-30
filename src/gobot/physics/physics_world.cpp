/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
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
#include "gobot/scene/resources/capsule_shape_3d.hpp"
#include "gobot/scene/resources/cylinder_shape_3d.hpp"
#include "gobot/scene/resources/sphere_shape_3d.hpp"
#include "gobot/scene/robot_3d.hpp"
#include "gobot/scene/sensor_3d.hpp"
#include "gobot/scene/terrain_3d.hpp"

namespace gobot {
namespace {

PhysicsJointControlMode JointDriveModeToControlMode(int drive_mode) {
    switch (static_cast<JointDriveMode>(drive_mode)) {
        case JointDriveMode::Position:
            return PhysicsJointControlMode::Position;
        case JointDriveMode::Velocity:
            return PhysicsJointControlMode::Velocity;
        case JointDriveMode::Motor:
            return PhysicsJointControlMode::Effort;
        case JointDriveMode::Passive:
            return PhysicsJointControlMode::Passive;
    }

    return PhysicsJointControlMode::Passive;
}

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
    snapshot.friction = collision_shape->GetFriction();
    snapshot.contype = collision_shape->GetContactType();
    snapshot.conaffinity = collision_shape->GetContactAffinity();
    snapshot.condim = collision_shape->GetContactDimension();
    snapshot.solref = collision_shape->GetSolref();
    snapshot.solimp = collision_shape->GetSolimp();
    snapshot.margin = collision_shape->GetMargin();
    snapshot.gap = collision_shape->GetGap();

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
    } else if (auto capsule = dynamic_pointer_cast<CapsuleShape3D>(shape)) {
        snapshot.type = PhysicsShapeType::Capsule;
        snapshot.radius = capsule->GetRadius();
        snapshot.height = capsule->GetHeight();
    } else if (auto cylinder = dynamic_pointer_cast<CylinderShape3D>(shape)) {
        snapshot.type = PhysicsShapeType::Cylinder;
        snapshot.radius = cylinder->GetRadius();
        snapshot.height = cylinder->GetHeight();
    }

    return snapshot;
}

std::vector<std::string> ChannelNamesForSensorType(PhysicsSensorType type) {
    switch (type) {
        case PhysicsSensorType::IMU:
            return {
                    "orientation_w",
                    "orientation_x",
                    "orientation_y",
                    "orientation_z",
                    "angular_velocity_x",
                    "angular_velocity_y",
                    "angular_velocity_z",
                    "linear_velocity_x",
                    "linear_velocity_y",
                    "linear_velocity_z",
                    "linear_acceleration_x",
                    "linear_acceleration_y",
                    "linear_acceleration_z"};
        case PhysicsSensorType::AngularMomentum:
            return {
                    "angular_momentum_x",
                    "angular_momentum_y",
                    "angular_momentum_z"};
        case PhysicsSensorType::Contact:
            return {"contact_strength"};
        case PhysicsSensorType::Unknown:
            break;
    }

    return {};
}

PhysicsSensorSnapshot CaptureSensorSnapshot(const Sensor3D* sensor,
                                            const std::string& link_name,
                                            const Affine3& global_transform) {
    PhysicsSensorSnapshot snapshot;
    snapshot.node = sensor;
    snapshot.name = sensor->GetName();
    snapshot.link_name = link_name;
    snapshot.global_transform = global_transform;
    snapshot.enabled = sensor->IsEnabled();
    snapshot.sensor_period = sensor->GetSensorPeriod();
    snapshot.noise_stddev = sensor->GetNoiseStddev();
    snapshot.visualize_debug = sensor->ShouldVisualizeDebug();

    if (Object::PointerCastTo<IMUSensor3D>(sensor) != nullptr) {
        snapshot.type = PhysicsSensorType::IMU;
    } else if (Object::PointerCastTo<AngularMomentumSensor3D>(sensor) != nullptr) {
        snapshot.type = PhysicsSensorType::AngularMomentum;
    } else if (auto* contact_sensor = Object::PointerCastTo<ContactSensor3D>(sensor)) {
        snapshot.type = PhysicsSensorType::Contact;
        snapshot.radius = contact_sensor->GetRadius();
        snapshot.min_threshold = contact_sensor->GetMinThreshold();
        snapshot.max_threshold = contact_sensor->GetMaxThreshold();
    }

    snapshot.channel_names = ChannelNamesForSensorType(snapshot.type);
    return snapshot;
}

PhysicsTerrainSnapshot CaptureTerrainSnapshot(const Terrain3D* terrain,
                                              const Affine3& global_transform) {
    PhysicsTerrainSnapshot snapshot;
    snapshot.node = terrain;
    snapshot.name = terrain->GetName();
    snapshot.surface_color = terrain->GetSurfaceColor();
    snapshot.friction = terrain->GetFriction();
    snapshot.contype = terrain->GetContactType();
    snapshot.conaffinity = terrain->GetContactAffinity();
    snapshot.condim = terrain->GetContactDimension();
    snapshot.solref = terrain->GetSolref();
    snapshot.solimp = terrain->GetSolimp();
    snapshot.margin = terrain->GetMargin();
    snapshot.gap = terrain->GetGap();
    snapshot.spawn_origins = terrain->GetSpawnOrigins();

    for (const TerrainBox& box : terrain->GetBoxes()) {
        Affine3 local = Affine3::Identity();
        local.translation() = box.center;
        local.SetEulerAngle({
                DEG_TO_RAD(box.rotation_degrees.x()),
                DEG_TO_RAD(box.rotation_degrees.y()),
                DEG_TO_RAD(box.rotation_degrees.z())
        }, EulerOrder::SXYZ);

        PhysicsTerrainBoxSnapshot box_snapshot;
        box_snapshot.global_transform = global_transform * local;
        box_snapshot.size = box.size;
        snapshot.boxes.push_back(std::move(box_snapshot));
    }

    for (const TerrainHeightField& heightfield : terrain->GetHeightFields()) {
        Affine3 local = Affine3::Identity();
        local.translation() = heightfield.center;

        PhysicsTerrainHeightFieldSnapshot heightfield_snapshot;
        heightfield_snapshot.global_transform = global_transform * local;
        heightfield_snapshot.size = heightfield.size;
        heightfield_snapshot.rows = heightfield.rows;
        heightfield_snapshot.cols = heightfield.cols;
        heightfield_snapshot.heights = heightfield.heights;
        heightfield_snapshot.normalized_elevation = heightfield.normalized_elevation;
        heightfield_snapshot.base_thickness = heightfield.base_thickness;
        heightfield_snapshot.z_offset = heightfield.z_offset;
        snapshot.heightfields.push_back(std::move(heightfield_snapshot));
    }

    for (const TerrainMeshPatch& mesh_patch : terrain->GetMeshPatches()) {
        Affine3 local = Affine3::Identity();
        local.translation() = mesh_patch.center;
        local.SetEulerAngle({
                DEG_TO_RAD(mesh_patch.rotation_degrees.x()),
                DEG_TO_RAD(mesh_patch.rotation_degrees.y()),
                DEG_TO_RAD(mesh_patch.rotation_degrees.z())
        }, EulerOrder::SXYZ);

        PhysicsTerrainMeshPatchSnapshot mesh_patch_snapshot;
        mesh_patch_snapshot.global_transform = global_transform * local;
        mesh_patch_snapshot.vertices = mesh_patch.vertices;
        mesh_patch_snapshot.indices = mesh_patch.indices;
        mesh_patch_snapshot.color = mesh_patch.color;
        snapshot.mesh_patches.push_back(std::move(mesh_patch_snapshot));
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
            } else if (auto sensor = Object::PointerCastTo<Sensor3D>(node->GetChild(static_cast<int>(i)))) {
                robot_snapshot->sensors.emplace_back(CaptureSensorSnapshot(
                        sensor,
                        link_snapshot.name,
                        ResolveNodeGlobalTransform(sensor, node_global_transform)));
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
        joint_snapshot.damping = joint->GetDamping();
        joint_snapshot.joint_position = joint->GetJointPosition();
        joint_snapshot.initial_position = joint->GetInitialPosition();
        joint_snapshot.drive_mode = static_cast<int>(joint->GetDriveMode());
        joint_snapshot.drive_stiffness = joint->GetDriveStiffness();
        joint_snapshot.drive_damping = joint->GetDriveDamping();
        joint_snapshot.control_lower_limit = joint->GetControlLowerLimit();
        joint_snapshot.control_upper_limit = joint->GetControlUpperLimit();
        joint_snapshot.force_lower_limit = joint->GetForceLowerLimit();
        joint_snapshot.force_upper_limit = joint->GetForceUpperLimit();
        joint_snapshot.gear = joint->GetGear();
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
        snapshot->total_sensor_count += robot_snapshot.sensors.size();
        for (const PhysicsLinkSnapshot& link : robot_snapshot.links) {
            snapshot->total_collision_shape_count += link.collision_shapes.size();
        }
        snapshot->robots.emplace_back(std::move(robot_snapshot));
        return;
    }

    if (auto collision_shape = Object::PointerCastTo<CollisionShape3D>(node)) {
        snapshot->loose_collision_shapes.emplace_back(CaptureShapeSnapshot(collision_shape, node_global_transform));
        ++snapshot->total_collision_shape_count;
    } else if (auto terrain = Object::PointerCastTo<Terrain3D>(node)) {
        PhysicsTerrainSnapshot terrain_snapshot = CaptureTerrainSnapshot(terrain, node_global_transform);
        snapshot->total_terrain_count += 1;
        snapshot->total_collision_shape_count += terrain_snapshot.boxes.size() +
                                                 terrain_snapshot.heightfields.size() +
                                                 terrain_snapshot.mesh_patches.size();
        snapshot->terrains.emplace_back(std::move(terrain_snapshot));
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

const PhysicsSensorState* FindPreviousSensorState(const PhysicsRobotState& robot_state,
                                                  const std::string& sensor_name,
                                                  const std::string& link_name) {
    for (const PhysicsSensorState& sensor_state : robot_state.sensors) {
        if (sensor_state.sensor_name == sensor_name && sensor_state.link_name == link_name) {
            return &sensor_state;
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

        for (PhysicsSensorState& sensor_state : robot_state.sensors) {
            const PhysicsSensorState* previous_sensor_state =
                    FindPreviousSensorState(*previous_robot_state,
                                            sensor_state.sensor_name,
                                            sensor_state.link_name);
            if (previous_sensor_state == nullptr || previous_sensor_state->type != sensor_state.type) {
                continue;
            }

            sensor_state.values = previous_sensor_state->values;
            sensor_state.timestamp = previous_sensor_state->timestamp;
        }
    }

    last_error_.clear();
    return true;
}

void PhysicsWorld::Reset() {
    ResetSceneStateFromSnapshot();
    ClearExternalForces();
}

void PhysicsWorld::Step(RealType delta_time) {
    GOB_UNUSED(delta_time);
}

bool PhysicsWorld::ConfigureEnvironmentBatch(std::size_t environment_count) {
    if (environment_count == 0) {
        SetLastError("Physics environment batch size must be greater than zero.");
        return false;
    }

    if (environment_count != 1) {
        SetLastError("This physics backend does not support batched environments.");
        return false;
    }

    last_error_.clear();
    return true;
}

std::size_t PhysicsWorld::GetEnvironmentCount() const {
    return 1;
}

const PhysicsSceneState* PhysicsWorld::GetEnvironmentState(std::size_t environment_index) const {
    if (environment_index != 0) {
        return nullptr;
    }

    return &scene_state_;
}

bool PhysicsWorld::ResetEnvironment(std::size_t environment_index) {
    if (environment_index != 0) {
        SetLastError(fmt::format("Environment index {} is out of range.", environment_index));
        return false;
    }

    Reset();
    last_error_.clear();
    return true;
}

bool PhysicsWorld::StepEnvironment(std::size_t environment_index, RealType delta_time) {
    if (environment_index != 0) {
        SetLastError(fmt::format("Environment index {} is out of range.", environment_index));
        return false;
    }

    Step(delta_time);
    last_error_.clear();
    return true;
}

bool PhysicsWorld::ResetJointState(const std::string& robot_name,
                                   const std::string& joint_name,
                                   RealType position,
                                   RealType velocity) {
    return ResetJointStateIn(scene_state_, robot_name, joint_name, position, velocity);
}

bool PhysicsWorld::ResetEnvironmentJointState(std::size_t environment_index,
                                              const std::string& robot_name,
                                              const std::string& joint_name,
                                              RealType position,
                                              RealType velocity) {
    if (environment_index != 0) {
        SetLastError(fmt::format("Environment index {} is out of range.", environment_index));
        return false;
    }

    return ResetJointState(robot_name, joint_name, position, velocity);
}

bool PhysicsWorld::ResetLinkState(const std::string& robot_name,
                                  const std::string& link_name,
                                  const Vector3& position,
                                  const Quaternion& orientation,
                                  const Vector3& linear_velocity,
                                  const Vector3& angular_velocity) {
    return ResetLinkStateIn(scene_state_,
                            robot_name,
                            link_name,
                            position,
                            orientation,
                            linear_velocity,
                            angular_velocity);
}

bool PhysicsWorld::ResetEnvironmentLinkState(std::size_t environment_index,
                                             const std::string& robot_name,
                                             const std::string& link_name,
                                             const Vector3& position,
                                             const Quaternion& orientation,
                                             const Vector3& linear_velocity,
                                             const Vector3& angular_velocity) {
    if (environment_index != 0) {
        SetLastError(fmt::format("Environment index {} is out of range.", environment_index));
        return false;
    }

    return ResetLinkState(robot_name, link_name, position, orientation, linear_velocity, angular_velocity);
}

bool PhysicsWorld::SetJointControl(const std::string& robot_name,
                                   const std::string& joint_name,
                                   PhysicsJointControlMode control_mode,
                                   RealType target) {
    return SetJointControlIn(scene_state_, robot_name, joint_name, control_mode, target);
}

bool PhysicsWorld::SetEnvironmentJointControl(std::size_t environment_index,
                                              const std::string& robot_name,
                                              const std::string& joint_name,
                                              PhysicsJointControlMode control_mode,
                                              RealType target) {
    if (environment_index != 0) {
        SetLastError(fmt::format("Environment index {} is out of range.", environment_index));
        return false;
    }

    return SetJointControl(robot_name, joint_name, control_mode, target);
}

bool PhysicsWorld::SetLinkExternalForce(const std::string& robot_name,
                                        const std::string& link_name,
                                        const Vector3& point,
                                        const Vector3& force) {
    if (FindMutableLinkState(robot_name, link_name) == nullptr) {
        SetLastError(fmt::format("Cannot apply external force to missing link '{}::{}'.",
                                 robot_name,
                                 link_name));
        return false;
    }

    for (PhysicsExternalForce& external_force : external_forces_) {
        if (external_force.robot_name == robot_name && external_force.link_name == link_name) {
            external_force.point = point;
            external_force.local_point = Vector3::Zero();
            external_force.target_point = point;
            external_force.force = force;
            external_force.use_spring = false;
            last_error_.clear();
            return true;
        }
    }

    PhysicsExternalForce external_force;
    external_force.robot_name = robot_name;
    external_force.link_name = link_name;
    external_force.point = point;
    external_force.local_point = Vector3::Zero();
    external_force.target_point = point;
    external_force.force = force;
    external_force.use_spring = false;
    external_forces_.push_back(external_force);
    last_error_.clear();
    return true;
}

bool PhysicsWorld::SetLinkSpringForce(const std::string& robot_name,
                                      const std::string& link_name,
                                      const Vector3& local_point,
                                      const Vector3& target_point,
                                      const Vector3& force_hint) {
    PhysicsLinkState* link_state = FindMutableLinkState(robot_name, link_name);
    if (link_state == nullptr) {
        SetLastError(fmt::format("Cannot apply external force to missing link '{}::{}'.",
                                 robot_name,
                                 link_name));
        return false;
    }

    const Vector3 point = link_state->global_transform * local_point;
    for (PhysicsExternalForce& external_force : external_forces_) {
        if (external_force.robot_name == robot_name && external_force.link_name == link_name) {
            external_force.point = point;
            external_force.local_point = local_point;
            external_force.target_point = target_point;
            external_force.force = force_hint;
            external_force.use_spring = true;
            last_error_.clear();
            return true;
        }
    }

    PhysicsExternalForce external_force;
    external_force.robot_name = robot_name;
    external_force.link_name = link_name;
    external_force.point = point;
    external_force.local_point = local_point;
    external_force.target_point = target_point;
    external_force.force = force_hint;
    external_force.use_spring = true;
    external_forces_.push_back(external_force);
    last_error_.clear();
    return true;
}

void PhysicsWorld::ClearExternalForces() {
    external_forces_.clear();
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
    return FindJointStateIn(scene_state_, robot_name, joint_name);
}

PhysicsJointState* PhysicsWorld::FindJointStateIn(PhysicsSceneState& scene_state,
                                                  const std::string& robot_name,
                                                  const std::string& joint_name) {
    for (PhysicsRobotState& robot_state : scene_state.robots) {
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

PhysicsLinkState* PhysicsWorld::FindMutableLinkState(const std::string& robot_name,
                                                     const std::string& link_name) {
    return FindMutableLinkStateIn(scene_state_, robot_name, link_name);
}

PhysicsLinkState* PhysicsWorld::FindMutableLinkStateIn(PhysicsSceneState& scene_state,
                                                       const std::string& robot_name,
                                                       const std::string& link_name) {
    for (PhysicsRobotState& robot_state : scene_state.robots) {
        if (robot_state.name != robot_name) {
            continue;
        }

        for (PhysicsLinkState& link_state : robot_state.links) {
            if (link_state.link_name == link_name) {
                return &link_state;
            }
        }
    }

    return nullptr;
}

bool PhysicsWorld::ResetJointStateIn(PhysicsSceneState& scene_state,
                                     const std::string& robot_name,
                                     const std::string& joint_name,
                                     RealType position,
                                     RealType velocity) {
    PhysicsJointState* joint_state = FindJointStateIn(scene_state, robot_name, joint_name);
    if (!joint_state) {
        SetLastError(fmt::format("Cannot reset state for missing joint '{}::{}'.", robot_name, joint_name));
        return false;
    }

    joint_state->position = position;
    joint_state->velocity = velocity;
    joint_state->effort = 0.0;
    joint_state->control_mode = PhysicsJointControlMode::Passive;
    joint_state->target_position = position;
    joint_state->target_velocity = 0.0;
    joint_state->target_effort = 0.0;
    last_error_.clear();
    return true;
}

bool PhysicsWorld::ResetLinkStateIn(PhysicsSceneState& scene_state,
                                    const std::string& robot_name,
                                    const std::string& link_name,
                                    const Vector3& position,
                                    const Quaternion& orientation,
                                    const Vector3& linear_velocity,
                                    const Vector3& angular_velocity) {
    PhysicsLinkState* link_state = FindMutableLinkStateIn(scene_state, robot_name, link_name);
    if (!link_state) {
        SetLastError(fmt::format("Cannot reset state for missing link '{}::{}'.", robot_name, link_name));
        return false;
    }

    Quaternion normalized_orientation = orientation;
    if (normalized_orientation.norm() <= CMP_EPSILON) {
        normalized_orientation = Quaternion::Identity();
    } else {
        normalized_orientation.normalize();
    }

    link_state->global_transform.translation() = position;
    link_state->global_transform.linear() = normalized_orientation.toRotationMatrix();
    link_state->linear_velocity = linear_velocity;
    link_state->angular_velocity = angular_velocity;
    last_error_.clear();
    return true;
}

bool PhysicsWorld::SetJointControlIn(PhysicsSceneState& scene_state,
                                     const std::string& robot_name,
                                     const std::string& joint_name,
                                     PhysicsJointControlMode control_mode,
                                     RealType target) {
    PhysicsJointState* joint_state = FindJointStateIn(scene_state, robot_name, joint_name);
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

PhysicsSceneState PhysicsWorld::MakeSceneStateFromSnapshot() const {
    PhysicsSceneState scene_state;
    scene_state.robots.reserve(scene_snapshot_.robots.size());

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
            ++scene_state.total_link_count;
        }

        for (const PhysicsJointSnapshot& joint_snapshot : robot_snapshot.joints) {
            PhysicsJointState joint_state;
            joint_state.node = joint_snapshot.node;
            joint_state.robot_name = robot_snapshot.name;
            joint_state.joint_name = joint_snapshot.name;
            joint_state.joint_type = joint_snapshot.joint_type;
            joint_state.position = joint_snapshot.joint_position;
            joint_state.target_position = joint_snapshot.joint_position;
            joint_state.control_mode = JointDriveModeToControlMode(joint_snapshot.drive_mode);
            robot_state.joints.emplace_back(std::move(joint_state));
            ++scene_state.total_joint_count;
        }

        for (const PhysicsSensorSnapshot& sensor_snapshot : robot_snapshot.sensors) {
            PhysicsSensorState sensor_state;
            sensor_state.node = sensor_snapshot.node;
            sensor_state.robot_name = robot_snapshot.name;
            sensor_state.link_name = sensor_snapshot.link_name;
            sensor_state.sensor_name = sensor_snapshot.name;
            sensor_state.type = sensor_snapshot.type;
            sensor_state.enabled = sensor_snapshot.enabled;
            sensor_state.channel_names = sensor_snapshot.channel_names;
            sensor_state.values.assign(sensor_state.channel_names.size(), 0.0);
            if (sensor_snapshot.type == PhysicsSensorType::IMU && sensor_state.values.size() >= 4) {
                const Quaternion orientation(sensor_snapshot.global_transform.linear());
                sensor_state.values[0] = orientation.w();
                sensor_state.values[1] = orientation.x();
                sensor_state.values[2] = orientation.y();
                sensor_state.values[3] = orientation.z();
            }
            robot_state.sensors.emplace_back(std::move(sensor_state));
            ++scene_state.total_sensor_count;
        }

        scene_state.robots.emplace_back(std::move(robot_state));
    }

    return scene_state;
}

void PhysicsWorld::ResetSceneStateFromSnapshot() {
    scene_state_ = MakeSceneStateFromSnapshot();
}

void PhysicsWorld::SetLastError(std::string error) {
    last_error_ = std::move(error);
}

} // namespace gobot

GOBOT_REGISTRATION {

    QuickEnumeration_<PhysicsBackendType>("PhysicsBackendType");
    QuickEnumeration_<PhysicsShapeType>("PhysicsShapeType");
    QuickEnumeration_<PhysicsSensorType>("PhysicsSensorType");
    QuickEnumeration_<PhysicsJointControlMode>("PhysicsJointControlMode");

    Class_<PhysicsBackendInfo>("PhysicsBackendInfo")
            .constructor()
            .property("type", &PhysicsBackendInfo::type)
            .property("name", &PhysicsBackendInfo::name)
            .property("available", &PhysicsBackendInfo::available)
            .property("cpu", &PhysicsBackendInfo::cpu)
            .property("gpu", &PhysicsBackendInfo::gpu)
            .property("robotics_focused", &PhysicsBackendInfo::robotics_focused)
            .property("status", &PhysicsBackendInfo::status);

    Class_<MuJoCoSolverSettings>("MuJoCoSolverSettings")
            .constructor()
            .property("solver", &MuJoCoSolverSettings::solver)
            .property("integrator", &MuJoCoSolverSettings::integrator)
            .property("cone", &MuJoCoSolverSettings::cone)
            .property("jacobian", &MuJoCoSolverSettings::jacobian)
            .property("iterations", &MuJoCoSolverSettings::iterations)
            .property("line_search_iterations", &MuJoCoSolverSettings::line_search_iterations)
            .property("no_slip_iterations", &MuJoCoSolverSettings::no_slip_iterations)
            .property("convex_collision_iterations", &MuJoCoSolverSettings::convex_collision_iterations)
            .property("tolerance", &MuJoCoSolverSettings::tolerance)
            .property("line_search_tolerance", &MuJoCoSolverSettings::line_search_tolerance)
            .property("no_slip_tolerance", &MuJoCoSolverSettings::no_slip_tolerance)
            .property("convex_collision_tolerance", &MuJoCoSolverSettings::convex_collision_tolerance)
            .property("impedance_ratio", &MuJoCoSolverSettings::impedance_ratio);

    Class_<PhysicsWorldSettings>("PhysicsWorldSettings")
            .constructor()
            .property("gravity", &PhysicsWorldSettings::gravity)
            .property("fixed_time_step", &PhysicsWorldSettings::fixed_time_step)
            .property("default_joint_gains", &PhysicsWorldSettings::default_joint_gains)
            .property("mujoco_solver", &PhysicsWorldSettings::mujoco_solver);

    Class_<PhysicsWorld>("PhysicsWorld")
            .method("is_available", &PhysicsWorld::IsAvailable)
            .method("get_last_error", &PhysicsWorld::GetLastError)
            .method("reset", &PhysicsWorld::Reset)
            .method("step", &PhysicsWorld::Step)
            .method("reset_link_state", &PhysicsWorld::ResetLinkState)
            .method("set_joint_control", &PhysicsWorld::SetJointControl);

    gobot::Type::register_wrapper_converter_for_base_classes<Ref<PhysicsWorld>, Ref<RefCounted>>();

};
