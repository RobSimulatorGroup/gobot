/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/physics/physics_scene_compiler.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <optional>
#include <unordered_set>
#include <utility>

#include <fmt/format.h>

#include "gobot/core/registration.hpp"
#include "gobot/scene/collision_shape_3d.hpp"
#include "gobot/scene/deformable_body_3d.hpp"
#include "gobot/scene/joint_3d.hpp"
#include "gobot/scene/link_3d.hpp"
#include "gobot/scene/mesh_instance_3d.hpp"
#include "gobot/scene/node.hpp"
#include "gobot/scene/physics_coupling.hpp"
#include "gobot/scene/resources/box_shape_3d.hpp"
#include "gobot/scene/resources/capsule_shape_3d.hpp"
#include "gobot/scene/resources/convex_mesh_shape_3d.hpp"
#include "gobot/scene/resources/cylinder_shape_3d.hpp"
#include "gobot/scene/resources/sphere_shape_3d.hpp"
#include "gobot/scene/robot_3d.hpp"
#include "gobot/scene/sensor_3d.hpp"
#include "gobot/scene/terrain_3d.hpp"

namespace gobot {
namespace {

std::string CanonicalScenePath(const Node* node) {
    std::vector<std::string> names;
    for (const Node* current = node; current != nullptr; current = current->GetParent()) {
        names.push_back(current->GetName());
    }
    std::string path;
    for (auto iterator = names.rbegin(); iterator != names.rend(); ++iterator) {
        path.push_back('/');
        path += *iterator;
    }
    return path.empty() ? std::string{"/"} : path;
}

const Node* ResolveNodePath(const Node& source, const NodePath& path) {
    if (path.IsEmpty() || path.GetSubNameCount() != 0) {
        return nullptr;
    }
    const Node* current = &source;
    std::size_t name_index = 0;
    const std::vector<std::string> names = path.GetNames();
    if (path.IsAbsolute()) {
        while (current->GetParent() != nullptr) {
            current = current->GetParent();
        }
        if (!names.empty() && names.front() == current->GetName()) {
            name_index = 1;
        }
    }
    for (; name_index < names.size(); ++name_index) {
        const std::string& name = names[name_index];
        if (name == ".") {
            continue;
        }
        if (name == "..") {
            current = current->GetParent();
            if (current == nullptr) {
                return nullptr;
            }
            continue;
        }
        const Node* child = nullptr;
        for (std::size_t index = 0; index < current->GetChildCount(); ++index) {
            const Node* candidate = current->GetChild(static_cast<int>(index));
            if (candidate->GetName() == name) {
                child = candidate;
                break;
            }
        }
        if (child == nullptr) {
            return nullptr;
        }
        current = child;
    }
    return current;
}

PhysicsMaterialSnapshot CapturePhysicsMaterial(const Ref<PhysicsMaterial3D>& material) {
    PhysicsMaterialSnapshot snapshot;
    if (!material.IsValid()) {
        return snapshot;
    }
    snapshot.sliding_friction = material->GetSlidingFriction();
    snapshot.torsional_friction = material->GetTorsionalFriction();
    snapshot.rolling_friction = material->GetRollingFriction();
    snapshot.restitution = material->GetRestitution();
    snapshot.contact_compliance = material->GetContactCompliance();
    snapshot.contact_damping = material->GetContactDamping();
    return snapshot;
}

Affine3 ResolveNodeGlobalTransform(const Node3D* node, const Affine3& parent_global_transform) {
    if (node == nullptr) {
        return parent_global_transform;
    }
    return node->IsInsideTree() ? node->GetGlobalTransform() : parent_global_transform * node->GetTransform();
}

bool HasVisualMeshDescendant(const Node* node, const Node* root_node) {
    if (node == nullptr) {
        return false;
    }
    if (node != root_node && Object::PointerCastTo<Link3D>(node) != nullptr) {
        return false;
    }
    if (const auto* mesh_instance = Object::PointerCastTo<MeshInstance3D>(node)) {
        return mesh_instance->GetMesh().IsValid();
    }
    for (std::size_t index = 0; index < node->GetChildCount(); ++index) {
        if (HasVisualMeshDescendant(node->GetChild(static_cast<int>(index)), root_node)) {
            return true;
        }
    }
    return false;
}

bool HasCollisionShapeDescendant(const Node* node, const Node* root_node) {
    if (node == nullptr) {
        return false;
    }
    if (node != root_node && Object::PointerCastTo<Link3D>(node) != nullptr) {
        return false;
    }
    if (Object::PointerCastTo<CollisionShape3D>(node) != nullptr) {
        return true;
    }
    for (std::size_t index = 0; index < node->GetChildCount(); ++index) {
        if (HasCollisionShapeDescendant(node->GetChild(static_cast<int>(index)), root_node)) {
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

void ExpandXYBounds(const Vector3& point, Vector2* xy_min, Vector2* xy_max, bool* initialized) {
    const Vector2 xy{point.x(), point.y()};
    if (!*initialized) {
        *xy_min = xy;
        *xy_max = xy;
        *initialized = true;
        return;
    }
    xy_min->x() = std::min(xy_min->x(), xy.x());
    xy_min->y() = std::min(xy_min->y(), xy.y());
    xy_max->x() = std::max(xy_max->x(), xy.x());
    xy_max->y() = std::max(xy_max->y(), xy.y());
}

void SetBoxXYBounds(PhysicsTerrainBoxSnapshot* box) {
    const Vector3 half = box->size.cwiseMax(Vector3::Zero()) * 0.5;
    bool initialized = false;
    for (int x_sign : {-1, 1}) {
        for (int y_sign : {-1, 1}) {
            for (int z_sign : {-1, 1}) {
                ExpandXYBounds(box->global_transform *
                                       Vector3(static_cast<RealType>(x_sign) * half.x(),
                                               static_cast<RealType>(y_sign) * half.y(),
                                               static_cast<RealType>(z_sign) * half.z()),
                               &box->xy_min,
                               &box->xy_max,
                               &initialized);
            }
        }
    }
    box->has_xy_bounds = initialized;
}

void SetHeightFieldXYBounds(PhysicsTerrainHeightFieldSnapshot* heightfield) {
    const RealType half_x = heightfield->size.x() * 0.5;
    const RealType half_y = heightfield->size.y() * 0.5;
    bool initialized = false;
    for (int x_sign : {-1, 1}) {
        for (int y_sign : {-1, 1}) {
            ExpandXYBounds(heightfield->global_transform *
                                   Vector3(static_cast<RealType>(x_sign) * half_x,
                                           static_cast<RealType>(y_sign) * half_y,
                                           0.0),
                           &heightfield->xy_min,
                           &heightfield->xy_max,
                           &initialized);
        }
    }
    heightfield->has_xy_bounds = initialized;
}

void SetMeshPatchXYBounds(PhysicsTerrainMeshPatchSnapshot* mesh_patch) {
    bool initialized = false;
    for (const Vector3& vertex : mesh_patch->vertices) {
        ExpandXYBounds(mesh_patch->global_transform * vertex,
                       &mesh_patch->xy_min,
                       &mesh_patch->xy_max,
                       &initialized);
    }
    mesh_patch->has_xy_bounds = initialized;
}

PhysicsShapeSnapshot CaptureShapeSnapshot(const CollisionShape3D* collision_shape,
                                          const Affine3& global_transform) {
    PhysicsShapeSnapshot snapshot;
    snapshot.name = collision_shape->GetName();
    snapshot.scene_path = CanonicalScenePath(collision_shape);
    snapshot.global_transform = global_transform;
    snapshot.disabled = collision_shape->IsDisabled();
    snapshot.material = CapturePhysicsMaterial(collision_shape->GetPhysicsMaterial());
    snapshot.collision_layer = collision_shape->GetCollisionLayer();
    snapshot.collision_mask = collision_shape->GetCollisionMask();
    snapshot.contact_offset = collision_shape->GetContactOffset();
    snapshot.rest_offset = collision_shape->GetRestOffset();

    const Ref<Shape3D>& shape = collision_shape->GetShape();
    if (!shape.IsValid()) {
        return snapshot;
    }
    if (const auto box = dynamic_pointer_cast<BoxShape3D>(shape)) {
        snapshot.type = PhysicsShapeType::Box;
        snapshot.box_size = box->GetSize();
    } else if (const auto sphere = dynamic_pointer_cast<SphereShape3D>(shape)) {
        snapshot.type = PhysicsShapeType::Sphere;
        snapshot.radius = sphere->GetRadius();
    } else if (const auto capsule = dynamic_pointer_cast<CapsuleShape3D>(shape)) {
        snapshot.type = PhysicsShapeType::Capsule;
        snapshot.radius = capsule->GetRadius();
        snapshot.height = capsule->GetHeight();
    } else if (const auto cylinder = dynamic_pointer_cast<CylinderShape3D>(shape)) {
        snapshot.type = PhysicsShapeType::Cylinder;
        snapshot.radius = cylinder->GetRadius();
        snapshot.height = cylinder->GetHeight();
    } else if (const auto convex_mesh = dynamic_pointer_cast<ConvexMeshShape3D>(shape)) {
        const Ref<Mesh>& mesh = convex_mesh->GetMesh();
        const std::shared_ptr<const MeshSurfaceList> surfaces =
                mesh.IsValid() ? mesh->GetSurfaceData() : nullptr;
        if (!surfaces) {
            return snapshot;
        }

        snapshot.type = PhysicsShapeType::Mesh;
        for (const MeshSurfaceData& surface : *surfaces) {
            const std::uint32_t vertex_offset =
                    static_cast<std::uint32_t>(snapshot.vertices.size());
            snapshot.vertices.insert(snapshot.vertices.end(),
                                     surface.vertices.begin(),
                                     surface.vertices.end());

            if (surface.indices.empty()) {
                const std::size_t triangle_vertex_count =
                        surface.vertices.size() - surface.vertices.size() % 3;
                for (std::size_t index = 0; index < triangle_vertex_count; ++index) {
                    snapshot.indices.push_back(vertex_offset + static_cast<std::uint32_t>(index));
                }
                continue;
            }

            for (std::size_t index = 0; index + 2 < surface.indices.size(); index += 3) {
                const std::uint32_t a = surface.indices[index];
                const std::uint32_t b = surface.indices[index + 1];
                const std::uint32_t c = surface.indices[index + 2];
                if (a >= surface.vertices.size() ||
                    b >= surface.vertices.size() ||
                    c >= surface.vertices.size()) {
                    continue;
                }
                snapshot.indices.insert(snapshot.indices.end(),
                                        {vertex_offset + a, vertex_offset + b, vertex_offset + c});
            }
        }
    }
    return snapshot;
}

std::vector<std::string> ChannelNamesForSensorType(PhysicsSensorType type) {
    switch (type) {
        case PhysicsSensorType::IMU:
            return {
                    "orientation_w", "orientation_x", "orientation_y", "orientation_z",
                    "angular_velocity_x", "angular_velocity_y", "angular_velocity_z",
                    "linear_velocity_x", "linear_velocity_y", "linear_velocity_z",
                    "linear_acceleration_x", "linear_acceleration_y", "linear_acceleration_z"};
        case PhysicsSensorType::AngularMomentum:
            return {"angular_momentum_x", "angular_momentum_y", "angular_momentum_z"};
        case PhysicsSensorType::Contact:
            return {"contact_strength"};
        case PhysicsSensorType::RayCast:
        case PhysicsSensorType::TerrainHeight:
        case PhysicsSensorType::HeightScanner:
        case PhysicsSensorType::Unknown:
            return {};
    }
    return {};
}

PhysicsSensorSnapshot CaptureSensorSnapshot(const Sensor3D* sensor,
                                            const std::string& link_name,
                                            const Affine3& global_transform) {
    PhysicsSensorSnapshot snapshot;
    snapshot.name = sensor->GetName();
    snapshot.scene_path = CanonicalScenePath(sensor);
    snapshot.link_name = link_name;
    snapshot.global_transform = global_transform;
    snapshot.local_transform = sensor->GetTransform();
    snapshot.enabled = sensor->IsEnabled();
    snapshot.sensor_period = sensor->GetSensorPeriod();
    snapshot.noise_stddev = sensor->GetNoiseStddev();
    if (const Ref<SensorNoiseModel>& noise_model = sensor->GetNoiseModel();
        noise_model.IsValid()) {
        snapshot.has_noise_model = true;
        snapshot.noise_stddev = noise_model->GetWhiteNoiseStddev();
        snapshot.noise_bias_mean = noise_model->GetBiasMean();
        snapshot.noise_bias_stddev = noise_model->GetBiasStddev();
        snapshot.noise_random_walk_stddev = noise_model->GetRandomWalkStddev();
        snapshot.noise_quantization_step = noise_model->GetQuantizationStep();
        snapshot.noise_clip_min = noise_model->GetClipMin();
        snapshot.noise_clip_max = noise_model->GetClipMax();
        snapshot.noise_seed_offset = noise_model->GetSeedOffset();
    }
    snapshot.visualize_debug = sensor->ShouldVisualizeDebug();
    snapshot.visible = sensor->IsInsideTree() ? sensor->IsVisibleInTree() : sensor->IsVisible();
    snapshot.debug_marker_radius = sensor->GetDebugMarkerRadius();

    if (Object::PointerCastTo<IMUSensor3D>(sensor) != nullptr) {
        snapshot.type = PhysicsSensorType::IMU;
    } else if (Object::PointerCastTo<AngularMomentumSensor3D>(sensor) != nullptr) {
        snapshot.type = PhysicsSensorType::AngularMomentum;
    } else if (const auto* contact_sensor = Object::PointerCastTo<ContactSensor3D>(sensor)) {
        snapshot.type = PhysicsSensorType::Contact;
        snapshot.radius = contact_sensor->GetRadius();
        snapshot.min_threshold = contact_sensor->GetMinThreshold();
        snapshot.max_threshold = contact_sensor->GetMaxThreshold();
    } else if (const auto* height_scanner = Object::PointerCastTo<HeightScanner3D>(sensor)) {
        snapshot.type = PhysicsSensorType::HeightScanner;
        snapshot.sample_offsets = height_scanner->GetResolvedSampleOffsets();
        snapshot.ray_direction = height_scanner->GetRayDirection();
        snapshot.ray_direction_world_space = height_scanner->IsRayDirectionWorldSpace();
        snapshot.max_distance = height_scanner->GetMaxDistance();
        snapshot.reduction_mode = height_scanner->GetReductionMode();
        snapshot.pattern_mode = height_scanner->GetPatternMode();
        snapshot.grid_size = height_scanner->GetGridSize();
        snapshot.grid_resolution = height_scanner->GetGridResolution();
        snapshot.ray_alignment = height_scanner->GetRayAlignment();
    } else if (const auto* terrain_height = Object::PointerCastTo<TerrainHeightSensor3D>(sensor)) {
        snapshot.type = PhysicsSensorType::TerrainHeight;
        snapshot.sample_offsets = terrain_height->GetResolvedSampleOffsets();
        snapshot.ray_direction = terrain_height->GetRayDirection();
        snapshot.ray_direction_world_space = terrain_height->IsRayDirectionWorldSpace();
        snapshot.max_distance = terrain_height->GetMaxDistance();
        snapshot.reduction_mode = terrain_height->GetReductionMode();
        snapshot.pattern_mode = terrain_height->GetPatternMode();
        snapshot.grid_size = terrain_height->GetGridSize();
        snapshot.grid_resolution = terrain_height->GetGridResolution();
        snapshot.ray_alignment = terrain_height->GetRayAlignment();
    } else if (const auto* raycast = Object::PointerCastTo<RayCastSensor3D>(sensor)) {
        snapshot.type = PhysicsSensorType::RayCast;
        snapshot.sample_offsets = raycast->GetResolvedSampleOffsets();
        snapshot.ray_direction = raycast->GetRayDirection();
        snapshot.ray_direction_world_space = raycast->IsRayDirectionWorldSpace();
        snapshot.max_distance = raycast->GetMaxDistance();
        snapshot.pattern_mode = raycast->GetPatternMode();
        snapshot.grid_size = raycast->GetGridSize();
        snapshot.grid_resolution = raycast->GetGridResolution();
        snapshot.ray_alignment = raycast->GetRayAlignment();
    }

    snapshot.channel_names = ChannelNamesForSensorType(snapshot.type);
    if (snapshot.type == PhysicsSensorType::RayCast || snapshot.type == PhysicsSensorType::HeightScanner) {
        for (std::size_t index = 0; index < snapshot.sample_offsets.size(); ++index) {
            snapshot.channel_names.push_back(fmt::format("distance_{}", index));
        }
    } else if (snapshot.type == PhysicsSensorType::TerrainHeight) {
        if (snapshot.reduction_mode == RayReductionMode::None) {
            for (std::size_t index = 0; index < snapshot.sample_offsets.size(); ++index) {
                snapshot.channel_names.push_back(fmt::format("height_{}", index));
            }
        } else {
            snapshot.channel_names.push_back("height");
        }
    }
    return snapshot;
}

PhysicsTerrainSnapshot CaptureTerrainSnapshot(const Terrain3D* terrain,
                                              const Affine3& global_transform) {
    PhysicsTerrainSnapshot snapshot;
    snapshot.name = terrain->GetName();
    snapshot.scene_path = CanonicalScenePath(terrain);
    snapshot.surface_color = terrain->GetSurfaceColor();
    snapshot.material = CapturePhysicsMaterial(terrain->GetPhysicsMaterial());
    snapshot.collision_layer = terrain->GetCollisionLayer();
    snapshot.collision_mask = terrain->GetCollisionMask();
    snapshot.contact_offset = terrain->GetContactOffset();
    snapshot.rest_offset = terrain->GetRestOffset();
    snapshot.spawn_origins = terrain->GetSpawnOrigins();

    for (const TerrainBox& box : terrain->GetBoxes()) {
        Affine3 local = Affine3::Identity();
        local.translation() = box.center;
        local.SetEulerAngle({DEG_TO_RAD(box.rotation_degrees.x()),
                             DEG_TO_RAD(box.rotation_degrees.y()),
                             DEG_TO_RAD(box.rotation_degrees.z())},
                            EulerOrder::SXYZ);
        PhysicsTerrainBoxSnapshot box_snapshot;
        box_snapshot.global_transform = global_transform * local;
        box_snapshot.size = box.size;
        SetBoxXYBounds(&box_snapshot);
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
        SetHeightFieldXYBounds(&heightfield_snapshot);
        snapshot.heightfields.push_back(std::move(heightfield_snapshot));
    }
    for (const TerrainMeshPatch& mesh_patch : terrain->GetMeshPatches()) {
        Affine3 local = Affine3::Identity();
        local.translation() = mesh_patch.center;
        local.SetEulerAngle({DEG_TO_RAD(mesh_patch.rotation_degrees.x()),
                             DEG_TO_RAD(mesh_patch.rotation_degrees.y()),
                             DEG_TO_RAD(mesh_patch.rotation_degrees.z())},
                            EulerOrder::SXYZ);
        PhysicsTerrainMeshPatchSnapshot mesh_patch_snapshot;
        mesh_patch_snapshot.global_transform = global_transform * local;
        mesh_patch_snapshot.vertices = mesh_patch.vertices;
        mesh_patch_snapshot.indices = mesh_patch.indices;
        mesh_patch_snapshot.color = mesh_patch.color;
        SetMeshPatchXYBounds(&mesh_patch_snapshot);
        snapshot.mesh_patches.push_back(std::move(mesh_patch_snapshot));
    }
    return snapshot;
}

void CollectRobotNodes(const Node* node,
                       PhysicsRobotSnapshot* robot_snapshot,
                       PhysicsRobotSceneBinding* scene_binding,
                       std::vector<PhysicsShapeSnapshot>* loose_collision_shapes,
                       const Affine3& parent_global_transform,
                       std::optional<std::size_t> parent_link_index = std::nullopt) {
    const Node3D* node_3d = Object::PointerCastTo<Node3D>(node);
    const Affine3 global_transform = ResolveNodeGlobalTransform(node_3d, parent_global_transform);
    std::optional<std::size_t> link_index = parent_link_index;

    if (const auto* link = Object::PointerCastTo<Link3D>(node)) {
        PhysicsLinkSnapshot snapshot;
        snapshot.name = link->GetName();
        snapshot.scene_path = CanonicalScenePath(link);
        snapshot.role = link->GetRole() == LinkRole::VirtualRoot
                                ? PhysicsLinkRole::VirtualRoot
                                : PhysicsLinkRole::Physical;
        snapshot.global_transform = global_transform;
        snapshot.mass = link->GetMass();
        snapshot.center_of_mass = link->GetCenterOfMass();
        snapshot.inertia_orientation = link->GetInertiaOrientation();
        snapshot.inertia_diagonal = link->GetInertiaDiagonal();
        snapshot.inertia_off_diagonal = link->GetInertiaOffDiagonal();
        link_index = robot_snapshot->links.size();
        robot_snapshot->links.push_back(std::move(snapshot));
        scene_binding->link_ids.push_back(link->GetInstanceId());
    } else if (const auto* joint = Object::PointerCastTo<Joint3D>(node)) {
        PhysicsJointSnapshot snapshot;
        snapshot.name = joint->GetName();
        snapshot.scene_path = CanonicalScenePath(joint);
        snapshot.parent_link = joint->GetParentLink();
        snapshot.child_link = joint->GetChildLink();
        snapshot.global_transform = global_transform;
        snapshot.axis = joint->GetAxis();
        snapshot.lower_limit = joint->GetLowerLimit();
        snapshot.upper_limit = joint->GetUpperLimit();
        snapshot.effort_limit = joint->GetEffortLimit();
        snapshot.velocity_limit = joint->GetVelocityLimit();
        snapshot.damping = joint->GetDamping();
        snapshot.armature = joint->GetArmature();
        snapshot.friction_loss = joint->GetFrictionLoss();
        if (const Ref<JointActuatorConfig>& config = joint->GetActuatorConfig();
            config.IsValid()) {
            snapshot.actuator_model.command_delay_steps = config->GetCommandDelaySteps();
            snapshot.actuator_model.command_deadband = config->GetCommandDeadband();
            snapshot.actuator_model.command_slew_rate = config->GetCommandSlewRate();
            snapshot.actuator_model.strength_scale = config->GetStrengthScale();
            snapshot.actuator_model.motor_velocity_limit = config->GetMotorVelocityLimit();
            snapshot.actuator_model.motor_stall_effort = config->GetMotorStallEffort();
        }
        snapshot.joint_position = joint->GetJointPosition();
        snapshot.initial_position = joint->GetInitialPosition();
        snapshot.drive_mode = static_cast<int>(joint->GetDriveMode());
        snapshot.drive_stiffness = joint->GetDriveStiffness();
        snapshot.drive_damping = joint->GetDriveDamping();
        snapshot.control_lower_limit = joint->GetControlLowerLimit();
        snapshot.control_upper_limit = joint->GetControlUpperLimit();
        snapshot.force_lower_limit = joint->GetForceLowerLimit();
        snapshot.force_upper_limit = joint->GetForceUpperLimit();
        snapshot.gear = joint->GetGear();
        snapshot.affine_actuator_enabled = joint->IsAffineActuatorEnabled();
        snapshot.affine_actuator_control_gain = joint->GetAffineActuatorControlGain();
        snapshot.affine_actuator_force_offset = joint->GetAffineActuatorForceOffset();
        snapshot.affine_actuator_position_gain = joint->GetAffineActuatorPositionGain();
        snapshot.affine_actuator_velocity_gain = joint->GetAffineActuatorVelocityGain();
        snapshot.affine_actuator_inherit_range = joint->GetAffineActuatorInheritRange();
        snapshot.joint_type = static_cast<int>(joint->GetJointType());
        robot_snapshot->joints.push_back(std::move(snapshot));
        scene_binding->joint_ids.push_back(joint->GetInstanceId());
    } else if (const auto* collision_shape = Object::PointerCastTo<CollisionShape3D>(node)) {
        PhysicsShapeSnapshot snapshot = CaptureShapeSnapshot(collision_shape, global_transform);
        if (link_index.has_value()) {
            robot_snapshot->links[*link_index].collision_shapes.push_back(std::move(snapshot));
        } else {
            loose_collision_shapes->push_back(std::move(snapshot));
        }
    } else if (const auto* sensor = Object::PointerCastTo<Sensor3D>(node)) {
        const std::string link_name = link_index.has_value()
                                              ? robot_snapshot->links[*link_index].name
                                              : std::string{};
        robot_snapshot->sensors.push_back(CaptureSensorSnapshot(sensor, link_name, global_transform));
    }

    for (std::size_t index = 0; index < node->GetChildCount(); ++index) {
        CollectRobotNodes(node->GetChild(static_cast<int>(index)),
                          robot_snapshot,
                          scene_binding,
                          loose_collision_shapes,
                          global_transform,
                          link_index);
    }
}

void CollectSceneNodes(const Node* node,
                       PhysicsSceneSnapshot* snapshot,
                       PhysicsSceneBindings* bindings,
                       const Affine3& parent_global_transform) {
    const Node3D* node_3d = Object::PointerCastTo<Node3D>(node);
    const Affine3 global_transform = ResolveNodeGlobalTransform(node_3d, parent_global_transform);

    if (const auto* robot = Object::PointerCastTo<Robot3D>(node)) {
        PhysicsRobotSnapshot robot_snapshot;
        robot_snapshot.name = robot->GetName();
        robot_snapshot.scene_path = CanonicalScenePath(robot);
        PhysicsRobotSceneBinding scene_binding;
        scene_binding.robot_id = robot->GetInstanceId();
        CollectRobotNodes(node,
                          &robot_snapshot,
                          &scene_binding,
                          &snapshot->loose_collision_shapes,
                          parent_global_transform);
        for (std::size_t index = 0; index < robot_snapshot.links.size(); ++index) {
            const auto* link = Object::PointerCastTo<Link3D>(
                    ObjectDB::GetInstance(scene_binding.link_ids[index]));
            if (IsImplicitVirtualRootLink(link, robot_snapshot)) {
                robot_snapshot.links[index].role = PhysicsLinkRole::VirtualRoot;
            }
        }
        snapshot->total_link_count += robot_snapshot.links.size();
        snapshot->total_joint_count += robot_snapshot.joints.size();
        snapshot->total_sensor_count += robot_snapshot.sensors.size();
        for (const PhysicsLinkSnapshot& link : robot_snapshot.links) {
            snapshot->total_collision_shape_count += link.collision_shapes.size();
        }
        snapshot->robots.push_back(std::move(robot_snapshot));
        bindings->robots.push_back(std::move(scene_binding));
        return;
    }

    if (const auto* collision_shape = Object::PointerCastTo<CollisionShape3D>(node)) {
        snapshot->loose_collision_shapes.push_back(CaptureShapeSnapshot(collision_shape, global_transform));
        ++snapshot->total_collision_shape_count;
    } else if (const auto* sensor = Object::PointerCastTo<Sensor3D>(node)) {
        snapshot->loose_sensors.push_back(CaptureSensorSnapshot(sensor, {}, global_transform));
        ++snapshot->total_sensor_count;
    } else if (const auto* terrain = Object::PointerCastTo<Terrain3D>(node)) {
        PhysicsTerrainSnapshot terrain_snapshot = CaptureTerrainSnapshot(terrain, global_transform);
        ++snapshot->total_terrain_count;
        snapshot->total_collision_shape_count += terrain_snapshot.boxes.size() +
                                                 terrain_snapshot.heightfields.size() +
                                                 terrain_snapshot.mesh_patches.size();
        snapshot->terrains.push_back(std::move(terrain_snapshot));
    } else if (const auto* deformable = Object::PointerCastTo<DeformableBody3D>(node)) {
        PhysicsDeformableSnapshot deformable_snapshot;
        deformable_snapshot.name = deformable->GetName();
        deformable_snapshot.scene_path = CanonicalScenePath(deformable);
        deformable_snapshot.global_transform = global_transform;
        if (const Ref<TetrahedralMesh>& mesh = deformable->GetMesh(); mesh.IsValid()) {
            deformable_snapshot.vertices = mesh->GetVertices();
            deformable_snapshot.tetrahedra = mesh->GetTetrahedra();
            deformable_snapshot.surface_triangles = mesh->GetResolvedSurfaceTriangles();
        }
        deformable_snapshot.density = deformable->GetDensity();
        deformable_snapshot.young_modulus = deformable->GetYoungModulus();
        deformable_snapshot.poisson_ratio = deformable->GetPoissonRatio();
        deformable_snapshot.damping = deformable->GetDamping();
        deformable_snapshot.kinematic = deformable->IsKinematic();
        deformable_snapshot.collision_layer = deformable->GetCollisionLayer();
        deformable_snapshot.collision_mask = deformable->GetCollisionMask();
        deformable_snapshot.self_collision_enabled = deformable->IsSelfCollisionEnabled();
        snapshot->deformables.push_back(std::move(deformable_snapshot));
        ++snapshot->total_deformable_count;
    } else if (const auto* coupling = Object::PointerCastTo<PhysicsCoupling>(node)) {
        PhysicsCouplingSnapshot coupling_snapshot;
        coupling_snapshot.name = coupling->GetName();
        coupling_snapshot.scene_path = CanonicalScenePath(coupling);
        coupling_snapshot.enabled = coupling->IsEnabled();
        const Node* target = ResolveNodePath(*coupling, coupling->GetRigidLinkPath());
        coupling_snapshot.rigid_link_path = Object::PointerCastTo<Link3D>(target) != nullptr
                ? CanonicalScenePath(target)
                : std::string{};
        coupling_snapshot.mode = static_cast<int>(coupling->GetMode());
        coupling_snapshot.force_scale = coupling->GetForceScale();
        coupling_snapshot.torque_scale = coupling->GetTorqueScale();
        snapshot->couplings.push_back(std::move(coupling_snapshot));
        ++snapshot->total_coupling_count;
    }

    for (std::size_t index = 0; index < node->GetChildCount(); ++index) {
        CollectSceneNodes(node->GetChild(static_cast<int>(index)), snapshot, bindings, global_transform);
    }
}

void AssignStableIds(PhysicsSceneSnapshot* snapshot) {
    struct Entry {
        std::string_view path;
        std::uint64_t* id;
    };
    std::vector<Entry> entries;
    for (PhysicsRobotSnapshot& robot : snapshot->robots) {
        entries.push_back({robot.scene_path, &robot.stable_id});
        for (PhysicsLinkSnapshot& link : robot.links) {
            entries.push_back({link.scene_path, &link.stable_id});
            for (PhysicsShapeSnapshot& shape : link.collision_shapes) {
                entries.push_back({shape.scene_path, &shape.stable_id});
            }
        }
        for (PhysicsJointSnapshot& joint : robot.joints) {
            entries.push_back({joint.scene_path, &joint.stable_id});
        }
        for (PhysicsSensorSnapshot& sensor : robot.sensors) {
            entries.push_back({sensor.scene_path, &sensor.stable_id});
        }
    }
    for (PhysicsShapeSnapshot& shape : snapshot->loose_collision_shapes) {
        entries.push_back({shape.scene_path, &shape.stable_id});
    }
    for (PhysicsSensorSnapshot& sensor : snapshot->loose_sensors) {
        entries.push_back({sensor.scene_path, &sensor.stable_id});
    }
    for (PhysicsTerrainSnapshot& terrain : snapshot->terrains) {
        entries.push_back({terrain.scene_path, &terrain.stable_id});
    }
    for (PhysicsDeformableSnapshot& deformable : snapshot->deformables) {
        entries.push_back({deformable.scene_path, &deformable.stable_id});
    }
    for (PhysicsCouplingSnapshot& coupling : snapshot->couplings) {
        entries.push_back({coupling.scene_path, &coupling.stable_id});
    }
    for (const Entry& entry : entries) {
        std::uint64_t stable_id = UINT64_C(14695981039346656037);
        for (const unsigned char byte : entry.path) {
            stable_id ^= byte;
            stable_id *= UINT64_C(1099511628211);
        }
        *entry.id = stable_id;
    }
}

void AddDiagnostic(CompiledPhysicsScene* compiled_scene,
                   PhysicsSceneCompileSeverity severity,
                   std::string path,
                   std::string message) {
    compiled_scene->diagnostics.push_back({severity, std::move(path), std::move(message)});
}

bool ValidateContactParameters(const PhysicsMaterialSnapshot& material,
                               RealType contact_offset,
                               RealType rest_offset,
                               std::string_view path,
                               CompiledPhysicsScene* compiled_scene) {
    const std::array<RealType, 6> values{
            material.sliding_friction,
            material.torsional_friction,
            material.rolling_friction,
            material.restitution,
            material.contact_compliance,
            material.contact_damping};
    const bool valid_material = std::ranges::all_of(values, [](RealType value) {
        return std::isfinite(value) && value >= 0.0;
    });
    if (!valid_material || material.restitution > 1.0 ||
        !std::isfinite(contact_offset) || contact_offset < 0.0 ||
        !std::isfinite(rest_offset) || rest_offset > contact_offset) {
        AddDiagnostic(compiled_scene,
                      PhysicsSceneCompileSeverity::Error,
                      std::string(path),
                      "Physics material values must be finite and non-negative, restitution "
                      "must be in [0, 1], and rest_offset must not exceed contact_offset.");
        return false;
    }
    return true;
}

bool InsertUnique(const std::string& name,
                  const std::string& kind,
                  const std::string& path,
                  std::unordered_set<std::string>* names,
                  CompiledPhysicsScene* compiled_scene) {
    if (names->insert(name).second) {
        return true;
    }
    AddDiagnostic(compiled_scene,
                  PhysicsSceneCompileSeverity::Error,
                  path,
                  fmt::format("Duplicate {} name '{}'. Runtime names must be unique within their owner.", kind, name));
    return false;
}

bool ValidateSensorNoise(const PhysicsSensorSnapshot& sensor,
                         CompiledPhysicsScene* compiled_scene) {
    bool valid = std::isfinite(sensor.noise_stddev) && sensor.noise_stddev >= 0.0;
    if (sensor.has_noise_model) {
        const std::array<RealType, 3> non_negative_values{
                sensor.noise_bias_stddev,
                sensor.noise_random_walk_stddev,
                sensor.noise_quantization_step};
        valid = valid && std::ranges::all_of(non_negative_values, [](RealType value) {
            return std::isfinite(value) && value >= 0.0;
        }) && std::isfinite(sensor.noise_bias_mean) &&
                !std::isnan(sensor.noise_clip_min) &&
                !std::isnan(sensor.noise_clip_max) &&
                sensor.noise_clip_min <= sensor.noise_clip_max;
    }
    if (!valid) {
        AddDiagnostic(compiled_scene,
                      PhysicsSceneCompileSeverity::Error,
                      sensor.scene_path,
                      "SensorNoiseModel standard deviations and quantization must be "
                      "finite and non-negative, with an ordered clip range.");
    }
    return valid;
}

bool ValidateCompiledScene(CompiledPhysicsScene* compiled_scene) {
    bool valid = true;
    std::unordered_set<std::string> stable_paths;
    std::unordered_set<std::uint64_t> stable_ids;
    const auto validate_identity = [&](const std::string& path,
                                       std::uint64_t stable_id,
                                       std::string_view kind) {
        if (path.empty() || stable_id == 0) {
            AddDiagnostic(compiled_scene,
                          PhysicsSceneCompileSeverity::Error,
                          path,
                          fmt::format("{} requires a canonical scene path and non-zero stable id.",
                                      kind));
            valid = false;
            return;
        }
        if (!stable_paths.insert(path).second) {
            AddDiagnostic(compiled_scene,
                          PhysicsSceneCompileSeverity::Error,
                          path,
                          "Duplicate canonical scene path; physics node names must be unique "
                          "within their parent.");
            valid = false;
        }
        if (!stable_ids.insert(stable_id).second) {
            AddDiagnostic(compiled_scene,
                          PhysicsSceneCompileSeverity::Error,
                          path,
                          "Duplicate physics stable id; canonical scene paths must be unique.");
            valid = false;
        }
    };

    for (const PhysicsRobotSnapshot& robot : compiled_scene->snapshot.robots) {
        validate_identity(robot.scene_path, robot.stable_id, "Robot3D");
        for (const PhysicsLinkSnapshot& link : robot.links) {
            validate_identity(link.scene_path, link.stable_id, "Link3D");
            for (const PhysicsShapeSnapshot& shape : link.collision_shapes) {
                validate_identity(shape.scene_path, shape.stable_id, "CollisionShape3D");
            }
        }
        for (const PhysicsJointSnapshot& joint : robot.joints) {
            validate_identity(joint.scene_path, joint.stable_id, "Joint3D");
        }
        for (const PhysicsSensorSnapshot& sensor : robot.sensors) {
            validate_identity(sensor.scene_path, sensor.stable_id, "Sensor3D");
        }
    }
    for (const PhysicsShapeSnapshot& shape : compiled_scene->snapshot.loose_collision_shapes) {
        validate_identity(shape.scene_path, shape.stable_id, "CollisionShape3D");
    }
    for (const PhysicsSensorSnapshot& sensor : compiled_scene->snapshot.loose_sensors) {
        validate_identity(sensor.scene_path, sensor.stable_id, "Sensor3D");
    }
    for (const PhysicsTerrainSnapshot& terrain : compiled_scene->snapshot.terrains) {
        validate_identity(terrain.scene_path, terrain.stable_id, "Terrain3D");
    }
    for (const PhysicsDeformableSnapshot& deformable : compiled_scene->snapshot.deformables) {
        validate_identity(deformable.scene_path, deformable.stable_id, "DeformableBody3D");
    }
    for (const PhysicsCouplingSnapshot& coupling : compiled_scene->snapshot.couplings) {
        validate_identity(coupling.scene_path, coupling.stable_id, "PhysicsCoupling");
    }

    std::unordered_set<std::string> robot_names;
    for (const PhysicsRobotSnapshot& robot : compiled_scene->snapshot.robots) {
        const std::string robot_path = fmt::format("/robots/{}", robot.name);
        valid = InsertUnique(robot.name, "robot", robot_path, &robot_names, compiled_scene) && valid;

        std::unordered_set<std::string> link_names;
        std::unordered_set<std::string> shape_names;
        for (const PhysicsLinkSnapshot& link : robot.links) {
            valid = InsertUnique(link.name,
                                 "link",
                                 fmt::format("{}/links/{}", robot_path, link.name),
                                 &link_names,
                                 compiled_scene) && valid;
            for (const PhysicsShapeSnapshot& shape : link.collision_shapes) {
                valid = ValidateContactParameters(shape.material,
                                                  shape.contact_offset,
                                                  shape.rest_offset,
                                                  shape.scene_path,
                                                  compiled_scene) && valid;
                if (shape.name.empty()) {
                    continue;
                }
                valid = InsertUnique(shape.name,
                                     "collision shape",
                                     fmt::format("{}/links/{}/collision_shapes/{}",
                                                 robot_path,
                                                 link.name,
                                                 shape.name),
                                     &shape_names,
                                     compiled_scene) && valid;
            }
        }

        std::unordered_set<std::string> joint_names;
        for (const PhysicsJointSnapshot& joint : robot.joints) {
            const std::string joint_path = fmt::format("{}/joints/{}", robot_path, joint.name);
            valid = InsertUnique(joint.name, "joint", joint_path, &joint_names, compiled_scene) && valid;
            const std::array<RealType, 5> actuator_values{
                    joint.actuator_model.command_deadband,
                    joint.actuator_model.command_slew_rate,
                    joint.actuator_model.strength_scale,
                    joint.actuator_model.motor_velocity_limit,
                    joint.actuator_model.motor_stall_effort};
            if (!std::ranges::all_of(actuator_values, [](RealType value) {
                    return std::isfinite(value) && value >= 0.0;
                })) {
                AddDiagnostic(compiled_scene,
                              PhysicsSceneCompileSeverity::Error,
                              joint.scene_path,
                              "JointActuatorConfig values must be finite and non-negative.");
                valid = false;
            }
            if (!joint.parent_link.empty() && !link_names.contains(joint.parent_link)) {
                AddDiagnostic(compiled_scene,
                              PhysicsSceneCompileSeverity::Warning,
                              joint_path,
                              fmt::format("Parent link '{}' is not authored in this robot.", joint.parent_link));
            }
            if (!joint.child_link.empty() && !link_names.contains(joint.child_link)) {
                AddDiagnostic(compiled_scene,
                              PhysicsSceneCompileSeverity::Warning,
                              joint_path,
                              fmt::format("Child link '{}' is not authored in this robot.", joint.child_link));
            }
        }

        std::unordered_set<std::string> sensor_names;
        for (const PhysicsSensorSnapshot& sensor : robot.sensors) {
            const std::string sensor_path = fmt::format("{}/sensors/{}", robot_path, sensor.name);
            valid = InsertUnique(sensor.name, "sensor", sensor_path, &sensor_names, compiled_scene) && valid;
            valid = ValidateSensorNoise(sensor, compiled_scene) && valid;
            if (!sensor.link_name.empty() && !link_names.contains(sensor.link_name)) {
                AddDiagnostic(compiled_scene,
                              PhysicsSceneCompileSeverity::Error,
                              sensor_path,
                              fmt::format("Sensor link '{}' is not authored in this robot.", sensor.link_name));
                valid = false;
            }
        }
    }

    for (const PhysicsShapeSnapshot& shape : compiled_scene->snapshot.loose_collision_shapes) {
        valid = ValidateContactParameters(shape.material,
                                          shape.contact_offset,
                                          shape.rest_offset,
                                          shape.scene_path,
                                          compiled_scene) && valid;
    }
    for (const PhysicsSensorSnapshot& sensor : compiled_scene->snapshot.loose_sensors) {
        valid = ValidateSensorNoise(sensor, compiled_scene) && valid;
    }
    for (const PhysicsTerrainSnapshot& terrain : compiled_scene->snapshot.terrains) {
        valid = ValidateContactParameters(terrain.material,
                                          terrain.contact_offset,
                                          terrain.rest_offset,
                                          terrain.scene_path,
                                          compiled_scene) && valid;
    }
    std::unordered_set<std::string> coupled_links;
    for (const PhysicsCouplingSnapshot& coupling : compiled_scene->snapshot.couplings) {
        if (!coupling.enabled) {
            continue;
        }
        if (coupling.rigid_link_path.empty() || !std::isfinite(coupling.force_scale) ||
            coupling.force_scale < 0.0 || !std::isfinite(coupling.torque_scale) ||
            coupling.torque_scale < 0.0) {
            AddDiagnostic(compiled_scene,
                          PhysicsSceneCompileSeverity::Error,
                          coupling.scene_path,
                          "Enabled PhysicsCoupling requires a valid Link3D path and finite, "
                          "non-negative force scales.");
            valid = false;
        } else if (!coupled_links.insert(coupling.rigid_link_path).second) {
            AddDiagnostic(compiled_scene,
                          PhysicsSceneCompileSeverity::Error,
                          coupling.scene_path,
                          "Multiple enabled PhysicsCoupling nodes target the same Link3D.");
            valid = false;
        }
    }

    if (compiled_scene->snapshot.robots.size() != compiled_scene->bindings.robots.size()) {
        AddDiagnostic(compiled_scene,
                      PhysicsSceneCompileSeverity::Error,
                      "/bindings",
                      "Physics snapshot and scene binding robot counts differ.");
        valid = false;
    }
    return valid;
}

} // namespace

bool PhysicsSceneCompiler::Compile(const Node* scene_root,
                                   CompiledPhysicsScene* compiled_scene,
                                   std::string* error) {
    if (compiled_scene == nullptr) {
        if (error != nullptr) {
            *error = "Cannot compile a physics scene into a null output.";
        }
        return false;
    }

    *compiled_scene = {};
    if (scene_root == nullptr) {
        if (error != nullptr) {
            *error = "Cannot compile a physics scene from a null scene root.";
        }
        return false;
    }

    compiled_scene->bindings.scene_root_id = scene_root->GetInstanceId();
    CollectSceneNodes(scene_root,
                      &compiled_scene->snapshot,
                      &compiled_scene->bindings,
                      Affine3::Identity());
    AssignStableIds(&compiled_scene->snapshot);
    if (!ValidateCompiledScene(compiled_scene)) {
        if (error != nullptr) {
            const auto diagnostic = std::find_if(
                    compiled_scene->diagnostics.begin(),
                    compiled_scene->diagnostics.end(),
                    [](const PhysicsSceneCompileDiagnostic& candidate) {
                        return candidate.severity == PhysicsSceneCompileSeverity::Error;
                    });
            *error = diagnostic != compiled_scene->diagnostics.end()
                             ? fmt::format("{}: {}", diagnostic->path, diagnostic->message)
                             : "Physics scene compilation failed validation.";
        }
        return false;
    }

    if (error != nullptr) {
        error->clear();
    }
    return true;
}

} // namespace gobot
