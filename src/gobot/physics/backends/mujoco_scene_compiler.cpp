/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/physics/backends/mujoco_scene_compiler.hpp"
#include "mujoco_compiled_model.hpp"
#include "gobot/log.hpp"

#include <array>
#include <set>
#include <unordered_map>

namespace gobot {
#ifdef GOBOT_HAS_MUJOCO
namespace mujoco_detail {
namespace {
std::string ArtifactDigest(std::string_view content) {
    std::uint64_t digest = 14695981039346656037ULL;
    for (const unsigned char byte : content) {
        digest ^= byte;
        digest *= 1099511628211ULL;
    }
    return fmt::format("fnv1a64:{:016x}", digest);
}

bool SerializeMuJoCoSpec(const mjSpec* spec, std::string* xml, std::string* error_message) {
    std::array<char, kMuJoCoErrorBufferSize> error{};
    std::array<char, 1> probe{};
    const int required_size =
            mj_saveXMLString(spec, probe.data(), static_cast<int>(probe.size()), error.data(), error.size());
    if (required_size <= 0) {
        if (error_message != nullptr) {
            *error_message = error.data()[0] != '\0'
                                     ? error.data()
                                     : "MuJoCo failed to determine serialized MJCF size.";
        }
        return false;
    }

    std::vector<char> buffer(static_cast<std::size_t>(required_size) + 1, '\0');
    error.fill('\0');
    if (mj_saveXMLString(spec,
                        buffer.data(),
                        static_cast<int>(buffer.size()),
                        error.data(),
                        error.size()) != 0) {
        if (error_message != nullptr) {
            *error_message = error.data()[0] != '\0'
                                     ? error.data()
                                     : "MuJoCo failed to serialize the compiled scene.";
        }
        return false;
    }
    *xml = buffer.data();
    return true;
}

const PhysicsJointSnapshot* FindParentJointForLink(const PhysicsRobotSnapshot& robot, const std::string& link_name) {
    for (const PhysicsJointSnapshot& joint : robot.joints) {
        if (joint.child_link == link_name) {
            return &joint;
        }
    }
    return nullptr;
}

void SetMuJoCoArray(double* target, const std::vector<RealType>& value, int max_count) {
    if (!target) {
        return;
    }
    const int count = std::min<int>(static_cast<int>(value.size()), max_count);
    for (int index = 0; index < count; ++index) {
        target[index] = value[index];
    }
}

void SetMuJoCoQuaternion(double* target, const Matrix3& rotation) {
    const Quaternion quaternion(rotation);
    target[0] = quaternion.w();
    target[1] = quaternion.x();
    target[2] = quaternion.y();
    target[3] = quaternion.z();
}

void SetMuJoCoQuaternion(double* target, const Quaternion& quaternion) {
    Quaternion normalized = quaternion;
    if (normalized.norm() <= CMP_EPSILON) {
        normalized = Quaternion::Identity();
    } else {
        normalized.normalize();
    }
    target[0] = normalized.w();
    target[1] = normalized.x();
    target[2] = normalized.y();
    target[3] = normalized.z();
}

void SetMuJoCoPose(mjsBody* body, const Affine3& local_transform) {
    if (!body) {
        return;
    }
    SetMuJoCoVector3(body->pos, local_transform.translation());
    SetMuJoCoQuaternion(body->quat, local_transform.linear());
}

void SetMuJoCoGeomPose(mjsGeom* geom, const Affine3& local_transform) {
    if (!geom) {
        return;
    }
    SetMuJoCoVector3(geom->pos, local_transform.translation());
    SetMuJoCoQuaternion(geom->quat, local_transform.linear());
}

Affine3 RelativeTransform(const Affine3& parent, const Affine3& child) {
    return parent.inverse() * child;
}

std::string SensorSiteName(const std::string& prefix, const PhysicsSensorSnapshot& sensor) {
    return prefix + sensor.link_name + "_" + sensor.name + "_site";
}

mjsSite* AddSensorSiteToBody(mjsBody* body,
                             const PhysicsSensorSnapshot& sensor,
                             const PhysicsLinkSnapshot& link,
                             const std::string& site_name) {
    if (!body || !sensor.enabled) {
        return nullptr;
    }

    mjsSite* site = mjs_addSite(body, nullptr);
    if (!site) {
        return nullptr;
    }

    mjs_setName(site->element, site_name.c_str());
    SetMuJoCoVector3(site->pos, RelativeTransform(link.global_transform, sensor.global_transform).translation());
    SetMuJoCoQuaternion(site->quat, RelativeTransform(link.global_transform, sensor.global_transform).linear());
    site->type = mjGEOM_SPHERE;
    const double radius = sensor.radius > 0.0 ? static_cast<double>(sensor.radius) : 0.01;
    site->size[0] = radius;
    site->size[1] = radius;
    site->size[2] = radius;
    site->rgba[0] = 0.1f;
    site->rgba[1] = 0.55f;
    site->rgba[2] = 0.95f;
    site->rgba[3] = sensor.visualize_debug ? 1.0f : 0.0f;
    return site;
}

mjsSensor* AddSiteSensor(mjSpec* spec,
                         const std::string& sensor_name,
                         mjtSensor type,
                         const std::string& site_name,
                         const PhysicsSensorSnapshot& sensor) {
    if (!spec) {
        return nullptr;
    }

    mjsSensor* mujoco_sensor = mjs_addSensor(spec);
    if (!mujoco_sensor) {
        return nullptr;
    }

    mjs_setName(mujoco_sensor->element, sensor_name.c_str());
    mujoco_sensor->type = type;
    mujoco_sensor->objtype = mjOBJ_SITE;
    mjs_setString(mujoco_sensor->objname, site_name.c_str());
    mujoco_sensor->noise = 0.0;
    if (sensor.sensor_period > 0.0) {
        mujoco_sensor->interval[0] = sensor.sensor_period;
    }
    return mujoco_sensor;
}

mjsSensor* AddBodySensor(mjSpec* spec,
                         const std::string& sensor_name,
                         mjtSensor type,
                         const std::string& body_name,
                         const PhysicsSensorSnapshot& sensor) {
    if (!spec) {
        return nullptr;
    }

    mjsSensor* mujoco_sensor = mjs_addSensor(spec);
    if (!mujoco_sensor) {
        return nullptr;
    }

    mjs_setName(mujoco_sensor->element, sensor_name.c_str());
    mujoco_sensor->type = type;
    mujoco_sensor->objtype = mjOBJ_BODY;
    mjs_setString(mujoco_sensor->objname, body_name.c_str());
    mujoco_sensor->noise = 0.0;
    if (sensor.sensor_period > 0.0) {
        mujoco_sensor->interval[0] = sensor.sensor_period;
    }
    return mujoco_sensor;
}

void AddSensorToSpec(mjSpec* spec,
                     mjsBody* body,
                     const PhysicsSensorSnapshot& sensor,
                     const PhysicsLinkSnapshot& link,
                     const std::string& prefix) {
    if (!spec || !body || !sensor.enabled || sensor.type == PhysicsSensorType::Unknown) {
        return;
    }

    const std::string site_name = SensorSiteName(prefix, sensor);
    if (sensor.type != PhysicsSensorType::AngularMomentum &&
        sensor.type != PhysicsSensorType::RayCast &&
        sensor.type != PhysicsSensorType::TerrainHeight &&
        sensor.type != PhysicsSensorType::HeightScanner) {
        if (AddSensorSiteToBody(body, sensor, link, site_name) == nullptr) {
            return;
        }
    }

    switch (sensor.type) {
        case PhysicsSensorType::IMU:
            AddSiteSensor(spec, SensorComponentName(prefix, sensor, "orientation"), mjSENS_FRAMEQUAT, site_name, sensor);
            AddSiteSensor(spec, SensorComponentName(prefix, sensor, "angular_velocity"), mjSENS_GYRO, site_name, sensor);
            AddSiteSensor(spec, SensorComponentName(prefix, sensor, "linear_velocity"), mjSENS_VELOCIMETER, site_name, sensor);
            AddSiteSensor(spec, SensorComponentName(prefix, sensor, "linear_acceleration"), mjSENS_ACCELEROMETER,
                          site_name, sensor);
            break;
        case PhysicsSensorType::AngularMomentum:
            AddBodySensor(spec,
                          SensorComponentName(prefix, sensor, "angular_momentum"),
                          mjSENS_SUBTREEANGMOM,
                          prefix + link.name,
                          sensor);
            break;
        case PhysicsSensorType::Contact:
            if (mjsSensor* touch_sensor =
                        AddSiteSensor(spec, SensorComponentName(prefix, sensor, "contact"), mjSENS_TOUCH, site_name, sensor)) {
                if (sensor.max_threshold > 0.0) {
                    touch_sensor->cutoff = static_cast<double>(sensor.max_threshold);
                }
            }
            break;
        case PhysicsSensorType::RayCast:
            break;
        case PhysicsSensorType::TerrainHeight:
            break;
        case PhysicsSensorType::HeightScanner:
            break;
        case PhysicsSensorType::Unknown:
            break;
    }
}

bool IsControllableMuJoCoJoint(const PhysicsJointSnapshot& joint) {
    const auto type = static_cast<JointType>(joint.joint_type);
    return type == JointType::Revolute ||
           type == JointType::Continuous ||
           type == JointType::Prismatic;
}

bool IsFixedMuJoCoJoint(const PhysicsJointSnapshot& joint) {
    return static_cast<JointType>(joint.joint_type) == JointType::Fixed;
}

double PositiveOrDefault(RealType value, double fallback) {
    return value > 0.0 ? static_cast<double>(value) : fallback;
}

bool HasControlRange(const PhysicsJointSnapshot& joint) {
    return joint.control_upper_limit > joint.control_lower_limit;
}

bool HasForceRange(const PhysicsJointSnapshot& joint) {
    return joint.force_upper_limit > joint.force_lower_limit;
}

bool HasUsableAuthoredModel(const PhysicsRobotSnapshot& robot) {
    return !robot.links.empty() || !robot.joints.empty();
}

int ContactDimension(const PhysicsMaterialSnapshot& material) {
    if (material.rolling_friction > 0.0) {
        return 6;
    }
    if (material.torsional_friction > 0.0) {
        return 4;
    }
    return material.sliding_friction > 0.0 ? 3 : 1;
}

void ConfigureMaterialContact(mjsGeom* geom,
                              const PhysicsMaterialSnapshot& material,
                              std::uint32_t collision_layer,
                              std::uint32_t collision_mask,
                              RealType contact_offset,
                              RealType rest_offset) {
    if (!geom) {
        return;
    }

    geom->contype = static_cast<int>(collision_layer);
    geom->conaffinity = static_cast<int>(collision_mask);
    geom->condim = ContactDimension(material);
    geom->friction[0] = material.sliding_friction;
    geom->friction[1] = material.torsional_friction;
    geom->friction[2] = material.rolling_friction;

    const RealType compliance_time = material.contact_compliance > 0.0
            ? std::max<RealType>(0.001, std::sqrt(material.contact_compliance))
            : 0.02;
    const RealType restitution_damping = std::max<RealType>(
            0.001, material.contact_damping * (1.0 - material.restitution));
    geom->solref[0] = compliance_time;
    geom->solref[1] = restitution_damping;
    geom->margin = contact_offset;
    geom->gap = std::max<RealType>(0.0, contact_offset - rest_offset);
    geom->priority = 0;
}

void ConfigureGeomContact(mjsGeom* geom, const PhysicsShapeSnapshot& shape) {
    ConfigureMaterialContact(geom,
                             shape.material,
                             shape.collision_layer,
                             shape.collision_mask,
                             shape.contact_offset,
                             shape.rest_offset);
}

void ConfigureGeomContact(mjsGeom* geom, const PhysicsTerrainSnapshot& terrain) {
    if (!geom) {
        return;
    }

    ConfigureMaterialContact(geom,
                             terrain.material,
                             terrain.collision_layer,
                             terrain.collision_mask,
                             terrain.contact_offset,
                             terrain.rest_offset);
    geom->group = kGobotTerrainGeomGroup;
}

void SetMuJoCoGeomColor(mjsGeom* geom, const Color& color) {
    if (!geom) {
        return;
    }
    geom->rgba[0] = color.red();
    geom->rgba[1] = color.green();
    geom->rgba[2] = color.blue();
    geom->rgba[3] = color.alpha();
}

bool IsValidTriangleMesh(const std::vector<Vector3>& vertices,
                         const std::vector<std::uint32_t>& indices) {
    if (vertices.empty() || indices.size() < 3 || indices.size() % 3 != 0) {
        return false;
    }

    return std::all_of(indices.begin(), indices.end(),
                       [&vertices](std::uint32_t index) { return index < vertices.size(); });
}

bool SetMuJoCoMeshData(mjsMesh* mesh,
                       const std::vector<Vector3>& mesh_vertices,
                       const std::vector<std::uint32_t>& mesh_indices) {
    if (!mesh || !IsValidTriangleMesh(mesh_vertices, mesh_indices)) {
        return false;
    }

    std::vector<float> vertices;
    vertices.reserve(mesh_vertices.size() * 3);
    for (const Vector3& vertex : mesh_vertices) {
        vertices.push_back(static_cast<float>(vertex.x()));
        vertices.push_back(static_cast<float>(vertex.y()));
        vertices.push_back(static_cast<float>(vertex.z()));
    }

    std::vector<int> indices;
    indices.reserve(mesh_indices.size());
    for (std::uint32_t index : mesh_indices) {
        indices.push_back(static_cast<int>(index));
    }

    const int vertex_count = static_cast<int>(mesh_vertices.size());
    const int face_count = static_cast<int>(indices.size() / 3);
    if (vertex_count <= 0 || face_count <= 0) {
        return false;
    }

    mjs_setFloat(mesh->uservert, vertices.data(), static_cast<int>(vertices.size()));
    mjs_setInt(mesh->userface, indices.data(), static_cast<int>(indices.size()));
    return true;
}

bool SetMuJoCoMeshData(mjsMesh* mesh,
                       const PhysicsTerrainMeshPatchSnapshot& mesh_patch) {
    return SetMuJoCoMeshData(mesh, mesh_patch.vertices, mesh_patch.indices);
}

bool IsSupportedShapeType(PhysicsShapeType type) {
    switch (type) {
        case PhysicsShapeType::Box:
        case PhysicsShapeType::Sphere:
        case PhysicsShapeType::Cylinder:
        case PhysicsShapeType::Capsule:
        case PhysicsShapeType::Mesh:
            return true;
        case PhysicsShapeType::Unknown:
            return false;
    }
    return false;
}

bool AddShapeMeshAsset(mjSpec* spec,
                       const PhysicsShapeSnapshot& shape,
                       const std::string& geom_name,
                       std::string* mesh_name) {
    mesh_name->clear();
    if (shape.type != PhysicsShapeType::Mesh) {
        return true;
    }
    if (!spec || !IsValidTriangleMesh(shape.vertices, shape.indices)) {
        return false;
    }

    mjsMesh* mesh = mjs_addMesh(spec, nullptr);
    if (!mesh) {
        return false;
    }
    *mesh_name = geom_name + "_mesh";
    mjs_setName(mesh->element, mesh_name->c_str());
    return SetMuJoCoMeshData(mesh, shape.vertices, shape.indices);
}

bool ConfigureShapeGeometry(mjsGeom* geom,
                            const PhysicsShapeSnapshot& shape,
                            const std::string& mesh_name) {
    if (!geom) {
        return false;
    }
    switch (shape.type) {
        case PhysicsShapeType::Box:
            geom->type = mjGEOM_BOX;
            geom->size[0] = shape.box_size.x() * 0.5;
            geom->size[1] = shape.box_size.y() * 0.5;
            geom->size[2] = shape.box_size.z() * 0.5;
            return true;
        case PhysicsShapeType::Sphere:
            geom->type = mjGEOM_SPHERE;
            geom->size[0] = shape.radius;
            return true;
        case PhysicsShapeType::Cylinder:
            geom->type = mjGEOM_CYLINDER;
            geom->size[0] = shape.radius;
            geom->size[1] = shape.height * 0.5;
            return true;
        case PhysicsShapeType::Capsule:
            geom->type = mjGEOM_CAPSULE;
            geom->size[0] = shape.radius;
            geom->size[1] = shape.height * 0.5;
            return true;
        case PhysicsShapeType::Mesh:
            if (mesh_name.empty()) {
                return false;
            }
            geom->type = mjGEOM_MESH;
            mjs_setString(geom->meshname, mesh_name.c_str());
            return true;
        case PhysicsShapeType::Unknown:
            return false;
    }
    return false;
}

std::vector<float> NormalizeHeightFieldData(const PhysicsTerrainHeightFieldSnapshot& heightfield,
                                            RealType* min_height,
                                            RealType* height_range) {
    std::vector<float> data;
    const std::size_t expected_count = static_cast<std::size_t>(heightfield.rows) *
                                       static_cast<std::size_t>(heightfield.cols);
    data.resize(expected_count, 0.0f);

    if (expected_count == 0) {
        if (min_height) {
            *min_height = 0.0;
        }
        if (height_range) {
            *height_range = CMP_EPSILON;
        }
        return data;
    }

    RealType min_value = 0.0;
    RealType max_value = 0.0;
    if (!heightfield.heights.empty()) {
        const auto minmax = std::minmax_element(heightfield.heights.begin(), heightfield.heights.end());
        min_value = *minmax.first;
        max_value = *minmax.second;
    }
    const RealType range = std::max<RealType>(max_value - min_value, CMP_EPSILON);
    const bool has_normalized_elevation =
            heightfield.normalized_elevation.size() == expected_count;

    for (int row = 0; row < heightfield.rows; ++row) {
        const int source_row = heightfield.rows - 1 - row;
        for (int col = 0; col < heightfield.cols; ++col) {
            const std::size_t destination_index =
                    static_cast<std::size_t>(row * heightfield.cols + col);
            const std::size_t source_index =
                    static_cast<std::size_t>(source_row * heightfield.cols + col);
            if (has_normalized_elevation) {
                data[destination_index] = static_cast<float>(std::clamp<RealType>(
                        heightfield.normalized_elevation[source_index], 0.0, 1.0));
            } else {
                const RealType value = source_index < heightfield.heights.size()
                                               ? heightfield.heights[source_index]
                                               : min_value;
                data[destination_index] = static_cast<float>((value - min_value) / range);
            }
        }
    }

    if (min_height) {
        *min_height = min_value + heightfield.z_offset;
    }
    if (height_range) {
        *height_range = range;
    }
    return data;
}

void AddShapeGeomToBody(mjSpec* spec,
                        mjsBody* body,
                        const PhysicsShapeSnapshot& shape,
                        const PhysicsLinkSnapshot& link,
                        const std::string& name) {
    if (!spec || !body || shape.disabled || !IsSupportedShapeType(shape.type)) {
        return;
    }

    std::string mesh_name;
    if (!AddShapeMeshAsset(spec, shape, name, &mesh_name)) {
        return;
    }

    mjsGeom* geom = mjs_addGeom(body, nullptr);
    if (!geom) {
        return;
    }

    mjs_setName(geom->element, name.c_str());
    if (!ConfigureShapeGeometry(geom, shape, mesh_name)) {
        return;
    }
    SetMuJoCoGeomPose(geom, RelativeTransform(link.global_transform, shape.global_transform));
    ConfigureGeomContact(geom, shape);
    geom->rgba[0] = 0.72f;
    geom->rgba[1] = 0.78f;
    geom->rgba[2] = 0.84f;
    geom->rgba[3] = 1.0f;
}

void ConfigureBodyInertial(mjsBody* body, const PhysicsLinkSnapshot& link) {
    if (!body || link.role == PhysicsLinkRole::VirtualRoot) {
        return;
    }

    const double mass = PositiveOrDefault(link.mass, 1.0);
    body->mass = mass;
    SetMuJoCoVector3(body->ipos, link.center_of_mass);
    SetMuJoCoQuaternion(body->iquat, link.inertia_orientation);
    body->inertia[0] = PositiveOrDefault(link.inertia_diagonal.x(), mass * 0.01);
    body->inertia[1] = PositiveOrDefault(link.inertia_diagonal.y(), mass * 0.01);
    body->inertia[2] = PositiveOrDefault(link.inertia_diagonal.z(), mass * 0.01);
    body->explicitinertial = true;
}

void AddJointToBody(mjsBody* body,
                    const PhysicsJointSnapshot& joint,
                    const Affine3& child_link_global_transform,
                    const std::string& prefixed_name) {
    if (!body || IsFixedMuJoCoJoint(joint)) {
        return;
    }

    if (static_cast<JointType>(joint.joint_type) == JointType::Floating) {
        mjsJoint* free_joint = mjs_addFreeJoint(body);
        if (free_joint) {
            mjs_setName(free_joint->element, prefixed_name.c_str());
        }
        return;
    }

    mjsJoint* mujoco_joint = mjs_addJoint(body, nullptr);
    if (!mujoco_joint) {
        return;
    }

    mjs_setName(mujoco_joint->element, prefixed_name.c_str());
    const auto type = static_cast<JointType>(joint.joint_type);
    mujoco_joint->type = type == JointType::Prismatic ? mjJNT_SLIDE : mjJNT_HINGE;
    SetMuJoCoVector3(mujoco_joint->pos,
                     RelativeTransform(child_link_global_transform, joint.global_transform).translation());
    const Vector3 world_axis = joint.global_transform.linear() * joint.axis;
    SetMuJoCoVector3(mujoco_joint->axis, child_link_global_transform.linear().transpose() * world_axis);
    mujoco_joint->ref = 0.0;
    if (type == JointType::Revolute || type == JointType::Prismatic) {
        mujoco_joint->limited = joint.upper_limit > joint.lower_limit ? mjLIMITED_TRUE : mjLIMITED_FALSE;
        mujoco_joint->range[0] = joint.lower_limit;
        mujoco_joint->range[1] = joint.upper_limit;
    }
    if (HasForceRange(joint)) {
        mujoco_joint->actfrclimited = mjLIMITED_TRUE;
        mujoco_joint->actfrcrange[0] = joint.force_lower_limit;
        mujoco_joint->actfrcrange[1] = joint.force_upper_limit;
    } else if (joint.effort_limit > 0.0) {
        mujoco_joint->actfrclimited = mjLIMITED_TRUE;
        mujoco_joint->actfrcrange[0] = -joint.effort_limit;
        mujoco_joint->actfrcrange[1] = joint.effort_limit;
    }
    mujoco_joint->damping[0] = static_cast<double>(joint.damping);
    mujoco_joint->armature = static_cast<double>(joint.armature);
    mujoco_joint->frictionloss = static_cast<double>(joint.friction_loss);
}

void ConfigureActuatorLimits(mjsActuator* actuator, const PhysicsJointSnapshot& joint, bool control_is_position) {
    if (!actuator) {
        return;
    }
    if (control_is_position) {
        actuator->inheritrange = 0.0;
        actuator->ctrllimited = mjLIMITED_FALSE;
    }
    if (HasControlRange(joint)) {
        actuator->ctrllimited = mjLIMITED_TRUE;
        actuator->ctrlrange[0] = joint.control_lower_limit;
        actuator->ctrlrange[1] = joint.control_upper_limit;
    }
    if (!control_is_position && !HasControlRange(joint) && joint.effort_limit > 0.0) {
        actuator->ctrllimited = mjLIMITED_TRUE;
        actuator->ctrlrange[0] = -joint.effort_limit;
        actuator->ctrlrange[1] = joint.effort_limit;
    }
    if (HasForceRange(joint)) {
        actuator->forcelimited = mjLIMITED_TRUE;
        actuator->forcerange[0] = joint.force_lower_limit;
        actuator->forcerange[1] = joint.force_upper_limit;
    } else if (joint.effort_limit > 0.0) {
        actuator->forcelimited = mjLIMITED_TRUE;
        actuator->forcerange[0] = -joint.effort_limit;
        actuator->forcerange[1] = joint.effort_limit;
    }
}

mjsActuator* AddJointActuator(mjSpec* spec,
                              const PhysicsJointSnapshot& joint,
                              const std::string& prefixed_name,
                              const std::string& suffix,
                              bool control_is_position) {
    if (!spec || !IsControllableMuJoCoJoint(joint)) {
        return nullptr;
    }

    mjsActuator* actuator = mjs_addActuator(spec, nullptr);
    if (!actuator) {
        return nullptr;
    }

    const std::string actuator_name = prefixed_name + suffix;
    mjs_setName(actuator->element, actuator_name.c_str());
    actuator->trntype = mjTRN_JOINT;
    mjs_setString(actuator->target, prefixed_name.c_str());
    SetMuJoCoArray(actuator->gear, joint.gear, 6);
    ConfigureActuatorLimits(actuator, joint, control_is_position);
    return actuator;
}

void AddJointActuators(mjSpec* spec, const PhysicsJointSnapshot& joint, const std::string& prefixed_name) {
    const auto drive_mode = static_cast<JointDriveMode>(joint.drive_mode);
    if (drive_mode == JointDriveMode::Motor) {
        mjsActuator* motor = AddJointActuator(spec, joint, prefixed_name, "_motor", false);
        if (motor) {
            mjs_setToMotor(motor);
        }
    } else if (drive_mode == JointDriveMode::Position) {
        mjsActuator* position = AddJointActuator(spec, joint, prefixed_name, "_position", true);
        if (position) {
            double kv = static_cast<double>(joint.drive_damping);
            const char* error = mjs_setToPosition(position,
                                                  joint.drive_stiffness,
                                                  kv > 0.0 ? &kv : nullptr,
                                                  nullptr,
                                                  nullptr,
                                                  0.0);
            if (error != nullptr && error[0] != '\0') {
                LOG_WARN("Failed to configure MuJoCo position actuator '{}': {}", prefixed_name, error);
            }
            ConfigureActuatorLimits(position, joint, true);
        }
    } else if (drive_mode == JointDriveMode::Velocity) {
        mjsActuator* velocity = AddJointActuator(spec, joint, prefixed_name, "_velocity", false);
        if (velocity) {
            mjs_setToVelocity(velocity, joint.drive_damping);
            ConfigureActuatorLimits(velocity, joint, false);
        }
    }

    if (!joint.affine_actuator_enabled) {
        return;
    }

    mjsActuator* affine = AddJointActuator(spec, joint, prefixed_name, "_affine", false);
    if (!affine) {
        return;
    }
    affine->dyntype = mjDYN_NONE;
    affine->gaintype = mjGAIN_FIXED;
    affine->biastype = mjBIAS_AFFINE;
    affine->gainprm[0] = static_cast<double>(joint.affine_actuator_control_gain);
    affine->biasprm[0] = static_cast<double>(joint.affine_actuator_force_offset);
    affine->biasprm[1] = static_cast<double>(joint.affine_actuator_position_gain);
    affine->biasprm[2] = static_cast<double>(joint.affine_actuator_velocity_gain);
    affine->inheritrange = static_cast<double>(joint.affine_actuator_inherit_range);

    // The joint-level actuator force range clamps the combined primary and
    // affine drives. Keep this auxiliary actuator's force range unbounded so
    // importing it does not apply the same limit twice.
    affine->forcelimited = mjLIMITED_FALSE;
    if (!HasControlRange(joint)) {
        affine->ctrllimited = joint.affine_actuator_inherit_range > 0.0
                                      ? mjLIMITED_AUTO
                                      : mjLIMITED_FALSE;
    }
}

class ModelBuilder {
public:
    ModelBuilder(const PhysicsSceneSnapshot& snapshot, const PhysicsWorldSettings& settings,
                 CompiledModel& result, std::string* error)
        : scene_snapshot_(snapshot), settings_(settings), result_(result), error_(error) {}
    bool Compile();
private:
    bool AddAuthoredRobotToSpec(void* spec, const PhysicsRobotSnapshot& robot,
                               std::size_t robot_index, const std::string& prefix);
    void AddLooseSceneGeomsToSpec(void* spec);
    void AddTerrainGeomsToSpec(void* spec);
    void SetLastError(std::string error) { if (error_) *error_ = std::move(error); }
    struct MuJoCoRobotBinding { std::size_t robot_index; std::string prefix; };
    const PhysicsSceneSnapshot& scene_snapshot_;
    const PhysicsWorldSettings& settings_;
    CompiledModel& result_;
    std::string* error_;
    PhysicsSceneArtifact scene_artifact_;
    std::vector<MuJoCoRobotBinding> robot_bindings_;
};

bool ModelBuilder::Compile() {
    scene_artifact_ = {};

    std::vector<std::size_t> robot_indices;
    for (std::size_t robot_index = 0; robot_index < scene_snapshot_.robots.size(); ++robot_index) {
        robot_indices.push_back(robot_index);
    }

    if (robot_indices.empty()) {
        SetLastError("MuJoCo backend requires at least one Robot3D or RigidBody3D in the scene.");
        return false;
    }

    mjSpec* parent_spec = mj_makeSpec();
    if (!parent_spec) {
        SetLastError("MuJoCo failed to allocate a parent spec.");
        return false;
    }
    std::unique_ptr<mjSpec, decltype(&mj_deleteSpec)> parent_spec_guard(parent_spec, mj_deleteSpec);
    parent_spec->compiler.degree = 0;
    ApplyMuJoCoOptions(&parent_spec->option, settings_);

    AddLooseSceneGeomsToSpec(parent_spec);
    AddTerrainGeomsToSpec(parent_spec);

    std::set<std::string> used_prefixes;
    robot_bindings_.clear();
    for (const std::size_t robot_index : robot_indices) {
        const PhysicsRobotSnapshot& robot = scene_snapshot_.robots[robot_index];
        std::string base_prefix = SanitizeMuJoCoName(robot.name);
        std::string prefix = base_prefix + "_";
        int duplicate_index = 2;
        while (used_prefixes.contains(prefix)) {
            prefix = fmt::format("{}{}_", base_prefix, duplicate_index++);
        }
        used_prefixes.insert(prefix);

        if (!HasUsableAuthoredModel(robot)) {
            SetLastError(fmt::format(
                    "MuJoCo robot '{}' has no authored links or joints. Import the robot into the Gobot scene before building the runtime model.",
                    robot.name));
            return false;
        }
        if (!AddAuthoredRobotToSpec(parent_spec, robot, robot_index, prefix)) {
            return false;
        }

        robot_bindings_.push_back({robot_index, prefix});
    }

    std::unique_ptr<mjModel, decltype(&mj_deleteModel)> compiled_model(
            mj_compile(parent_spec, nullptr),
            mj_deleteModel);
    if (!compiled_model) {
        const std::string compile_error =
                mjs_getError(parent_spec) ? mjs_getError(parent_spec) : "unknown error";
        SetLastError(fmt::format("MuJoCo failed to compile authored scene model: {}", compile_error));
        return false;
    }

    std::string serialized_mjcf;
    std::string serialization_error;
    if (!SerializeMuJoCoSpec(parent_spec, &serialized_mjcf, &serialization_error)) {
        SetLastError("MuJoCo failed to serialize the authored scene: " + serialization_error);
        return false;
    }

    auto* model = compiled_model.get();
    scene_artifact_.schema_version = MuJoCoSceneCompiler::kArtifactSchemaVersion;
    scene_artifact_.backend = PhysicsBackendType::MuJoCoCpu;
    scene_artifact_.producer = "mujoco";
    scene_artifact_.format = "mjcf";
    scene_artifact_.content = std::move(serialized_mjcf);
    scene_artifact_.content_digest = ArtifactDigest(scene_artifact_.content);
    scene_artifact_.producer_version = mj_versionString();
    scene_artifact_.backend_version = scene_artifact_.producer_version;
    scene_artifact_.nq = static_cast<std::size_t>(model->nq);
    scene_artifact_.nv = static_cast<std::size_t>(model->nv);
    scene_artifact_.nu = static_cast<std::size_t>(model->nu);
    scene_artifact_.nbody = static_cast<std::size_t>(model->nbody);
    scene_artifact_.njoint = static_cast<std::size_t>(model->njnt);
    scene_artifact_.ngeom = static_cast<std::size_t>(model->ngeom);
    scene_artifact_.nsensor = static_cast<std::size_t>(model->nsensor);
    scene_artifact_.nhfield = static_cast<std::size_t>(model->nhfield);
    if (!scene_snapshot_.terrains.empty()) {
        scene_artifact_.terrain_geom_groups = {kGobotTerrainGeomGroup};
    }
    for (const MuJoCoRobotBinding& binding : robot_bindings_) {
        if (binding.robot_index < scene_snapshot_.robots.size()) {
            const std::string& robot_name = scene_snapshot_.robots[binding.robot_index].name;
            scene_artifact_.robot_names.push_back(robot_name);
            scene_artifact_.robot_prefixes.push_back(binding.prefix);
            scene_artifact_.robots.push_back({robot_name, binding.prefix, {}, {}, {}});
        }
    }

    auto find_robot_topology = [&](std::string_view runtime_name)
            -> PhysicsArtifactRobotTopology* {
        PhysicsArtifactRobotTopology* result = nullptr;
        std::size_t matched_length = 0;
        for (PhysicsArtifactRobotTopology& robot : scene_artifact_.robots) {
            if (!robot.runtime_prefix.empty() &&
                runtime_name.starts_with(robot.runtime_prefix) &&
                robot.runtime_prefix.size() > matched_length) {
                result = &robot;
                matched_length = robot.runtime_prefix.size();
            }
        }
        return result;
    };

    for (int body_id = 0; body_id < model->nbody; ++body_id) {
        const char* name = mj_id2name(model, mjOBJ_BODY, body_id);
        if (name != nullptr) {
            if (auto* robot = find_robot_topology(name)) {
                robot->body_names.emplace_back(name);
            }
        }
    }
    for (int joint_id = 0; joint_id < model->njnt; ++joint_id) {
        const char* name = mj_id2name(model, mjOBJ_JOINT, joint_id);
        if (name != nullptr) {
            if (auto* robot = find_robot_topology(name)) {
                robot->joint_names.emplace_back(name);
            }
        }
    }
    for (int actuator_id = 0; actuator_id < model->nu; ++actuator_id) {
        const char* actuator_name_value = mj_id2name(model, mjOBJ_ACTUATOR, actuator_id);
        const std::string actuator_name = actuator_name_value != nullptr
                                                  ? actuator_name_value
                                                  : fmt::format("actuator_{}", actuator_id);
        std::string joint_name;
        if (model->actuator_trntype[actuator_id] == mjTRN_JOINT ||
            model->actuator_trntype[actuator_id] == mjTRN_JOINTINPARENT) {
            const int joint_id = model->actuator_trnid[2 * actuator_id];
            const char* joint_name_value = joint_id >= 0
                                                   ? mj_id2name(model, mjOBJ_JOINT, joint_id)
                                                   : nullptr;
            if (joint_name_value != nullptr) {
                joint_name = joint_name_value;
            }
        }
        PhysicsArtifactRobotTopology* robot = find_robot_topology(
                joint_name.empty() ? std::string_view(actuator_name) : std::string_view(joint_name));
        const std::string mode = EndsWith(actuator_name, "_position")
                                         ? "position"
                                 : EndsWith(actuator_name, "_velocity")
                                         ? "velocity"
                                         : "direct";
        scene_artifact_.controls.push_back({
                actuator_id,
                actuator_name,
                joint_name,
                mode,
                robot != nullptr ? robot->name : std::string{}});
        if (robot != nullptr) {
            robot->control_indices.push_back(actuator_id);
        }
    }

    const std::size_t standalone_body_count = static_cast<std::size_t>(
            std::count_if(scene_snapshot_.robots.begin(),
                          scene_snapshot_.robots.end(),
                          [](const PhysicsRobotSnapshot& rigid_system) {
                              return rigid_system.standalone_rigid_body;
                          }));
    LOG_INFO("MuJoCo {} compiled: Robot3D={}, RigidBody3D={}, nq={}, nv={}, joints={}",
             "model",
             scene_snapshot_.robots.size() - standalone_body_count,
             standalone_body_count,
             model->nq,
             model->nv,
             model->njnt);
    result_.model.reset(compiled_model.release());
    result_.artifact = std::move(scene_artifact_);
    for (const auto& binding : robot_bindings_) result_.robot_prefixes.push_back(binding.prefix);
    return true;
}

bool ModelBuilder::AddAuthoredRobotToSpec(void* parent_spec_ptr,
                                                const PhysicsRobotSnapshot& robot,
                                                std::size_t robot_index,
                                                const std::string& prefix) {
    auto* parent_spec = static_cast<mjSpec*>(parent_spec_ptr);
    if (!parent_spec) {
        SetLastError("Cannot add authored robot to a null MuJoCo parent spec.");
        return false;
    }

    mjsBody* world = mjs_findBody(parent_spec, "world");
    if (!world) {
        SetLastError("MuJoCo parent spec has no world body.");
        return false;
    }

    std::unordered_map<std::string, mjsBody*> bodies;
    std::unordered_map<std::string, Affine3> body_global_transforms;
    for (const PhysicsLinkSnapshot& link : robot.links) {
        if (link.name.empty() || link.role == PhysicsLinkRole::VirtualRoot) {
            continue;
        }

        const PhysicsJointSnapshot* parent_joint = FindParentJointForLink(robot, link.name);
        mjsBody* parent_body = world;
        Affine3 parent_global_transform = Affine3::Identity();
        if (parent_joint != nullptr && !parent_joint->parent_link.empty()) {
            const auto parent_iter = bodies.find(parent_joint->parent_link);
            if (parent_iter != bodies.end()) {
                parent_body = parent_iter->second;
                parent_global_transform = body_global_transforms[parent_joint->parent_link];
            }
        }

        mjsBody* body = mjs_addBody(parent_body, nullptr);
        if (!body) {
            SetLastError(fmt::format("MuJoCo failed to create body for Gobot link '{}::{}'.",
                                     robot.name,
                                     link.name));
            return false;
        }

        const std::string body_name = prefix + link.name;
        mjs_setName(body->element, body_name.c_str());
        SetMuJoCoPose(body, RelativeTransform(parent_global_transform, link.global_transform));
        ConfigureBodyInertial(body, link);

        if (robot.standalone_rigid_body) {
            mjsJoint* free_joint = mjs_addFreeJoint(body);
            if (!free_joint) {
                SetLastError(fmt::format(
                        "MuJoCo failed to create a free joint for RigidBody3D '{}'.",
                        robot.name));
                return false;
            }
            mjs_setName(free_joint->element, (prefix + "free_joint").c_str());
        } else if (parent_joint != nullptr) {
            AddJointToBody(body, *parent_joint, link.global_transform, prefix + parent_joint->name);
        }

        for (std::size_t shape_index = 0; shape_index < link.collision_shapes.size(); ++shape_index) {
            const PhysicsShapeSnapshot& shape = link.collision_shapes[shape_index];
            const std::string shape_name = shape.name.empty()
                                                   ? fmt::format("{}{}_geom_{}", prefix, link.name, shape_index)
                                                   : prefix + SanitizeMuJoCoName(shape.name);
            AddShapeGeomToBody(parent_spec,
                               body,
                               shape,
                               link,
                               shape_name);
        }

        for (const PhysicsSensorSnapshot& sensor : robot.sensors) {
            if (sensor.link_name == link.name) {
                AddSensorToSpec(parent_spec, body, sensor, link, prefix);
            }
        }

        bodies[link.name] = body;
        body_global_transforms[link.name] = link.global_transform;
    }

    for (const PhysicsJointSnapshot& joint : robot.joints) {
        AddJointActuators(parent_spec, joint, prefix + joint.name);
    }

    LOG_INFO("Added authored Gobot {} '{}' to MuJoCo spec with prefix '{}'.",
             robot.standalone_rigid_body ? "RigidBody3D" : "Robot3D",
             robot.name,
             prefix);
    GOB_UNUSED(robot_index);
    return true;
}

void ModelBuilder::AddLooseSceneGeomsToSpec(void* spec_ptr) {
    auto* spec = static_cast<mjSpec*>(spec_ptr);
    if (!spec || scene_snapshot_.loose_collision_shapes.empty()) {
        return;
    }

    mjsBody* world = mjs_findBody(spec, "world");
    if (!world) {
        LOG_WARN("MuJoCo spec has no world body; Gobot loose scene geoms were not added.");
        return;
    }

    int added_count = 0;
    for (std::size_t shape_index = 0; shape_index < scene_snapshot_.loose_collision_shapes.size(); ++shape_index) {
        const PhysicsShapeSnapshot& shape = scene_snapshot_.loose_collision_shapes[shape_index];
        if (shape.disabled) {
            continue;
        }

        if (!IsSupportedShapeType(shape.type)) {
            continue;
        }

        const std::string name = fmt::format("gobot_loose_box_{}", shape_index);
        std::string mesh_name;
        if (!AddShapeMeshAsset(spec, shape, name, &mesh_name)) {
            continue;
        }

        mjsGeom* geom = mjs_addGeom(world, nullptr);
        if (!geom) {
            continue;
        }

        mjs_setName(geom->element, name.c_str());
        if (!ConfigureShapeGeometry(geom, shape, mesh_name)) {
            continue;
        }
        geom->pos[0] = shape.global_transform.translation().x();
        geom->pos[1] = shape.global_transform.translation().y();
        geom->pos[2] = shape.global_transform.translation().z();
        const Quaternion rotation(shape.global_transform.linear());
        geom->quat[0] = rotation.w();
        geom->quat[1] = rotation.x();
        geom->quat[2] = rotation.y();
        geom->quat[3] = rotation.z();
        ConfigureGeomContact(geom, shape);
        geom->rgba[0] = 0.28f;
        geom->rgba[1] = 0.30f;
        geom->rgba[2] = 0.32f;
        geom->rgba[3] = 1.0f;
        ++added_count;
    }

    if (added_count > 0) {
        LOG_INFO("Added {} loose Gobot box collision geoms to the MuJoCo world.", added_count);
    }
}

void ModelBuilder::AddTerrainGeomsToSpec(void* spec_ptr) {
    auto* spec = static_cast<mjSpec*>(spec_ptr);
    if (!spec || scene_snapshot_.terrains.empty()) {
        return;
    }

    mjsBody* world = mjs_findBody(spec, "world");
    if (!world) {
        LOG_WARN("MuJoCo spec has no world body; Gobot Terrain3D geoms were not added.");
        return;
    }

    int added_count = 0;
    int hfield_count = 0;
    int mesh_count = 0;
    for (std::size_t terrain_index = 0; terrain_index < scene_snapshot_.terrains.size(); ++terrain_index) {
        const PhysicsTerrainSnapshot& terrain = scene_snapshot_.terrains[terrain_index];
        const std::string terrain_name = SanitizeMuJoCoName(
                terrain.name.empty() ? fmt::format("terrain_{}", terrain_index) : terrain.name);

        for (std::size_t box_index = 0; box_index < terrain.boxes.size(); ++box_index) {
            const PhysicsTerrainBoxSnapshot& box = terrain.boxes[box_index];
            if (box.size.x() <= 0.0 || box.size.y() <= 0.0 || box.size.z() <= 0.0) {
                continue;
            }

            mjsGeom* geom = mjs_addGeom(world, nullptr);
            if (!geom) {
                continue;
            }

            const std::string name = fmt::format("gobot_{}_box_{}", terrain_name, box_index);
            mjs_setName(geom->element, name.c_str());
            geom->type = mjGEOM_BOX;
            geom->size[0] = box.size.x() * 0.5;
            geom->size[1] = box.size.y() * 0.5;
            geom->size[2] = box.size.z() * 0.5;
            SetMuJoCoGeomPose(geom, box.global_transform);
            ConfigureGeomContact(geom, terrain);
            SetMuJoCoGeomColor(geom, terrain.surface_color);
            ++added_count;
        }

        for (std::size_t hfield_index = 0; hfield_index < terrain.heightfields.size(); ++hfield_index) {
            const PhysicsTerrainHeightFieldSnapshot& heightfield = terrain.heightfields[hfield_index];
            const std::size_t expected_count = static_cast<std::size_t>(heightfield.rows) *
                                               static_cast<std::size_t>(heightfield.cols);
            if (heightfield.rows < 2 || heightfield.cols < 2 || expected_count == 0) {
                continue;
            }

            RealType min_height = 0.0;
            RealType height_range = CMP_EPSILON;
            std::vector<float> normalized_heights =
                    NormalizeHeightFieldData(heightfield, &min_height, &height_range);

            mjsHField* mujoco_hfield = mjs_addHField(spec);
            if (!mujoco_hfield) {
                continue;
            }

            const std::string hfield_name = fmt::format("gobot_{}_hfield_{}", terrain_name, hfield_index);
            mjs_setName(mujoco_hfield->element, hfield_name.c_str());
            mujoco_hfield->nrow = heightfield.rows;
            mujoco_hfield->ncol = heightfield.cols;
            mujoco_hfield->size[0] = heightfield.size.x() * 0.5;
            mujoco_hfield->size[1] = heightfield.size.y() * 0.5;
            mujoco_hfield->size[2] = height_range;
            mujoco_hfield->size[3] = std::max<RealType>(heightfield.base_thickness, 0.0);
            mjs_setFloat(mujoco_hfield->userdata,
                         normalized_heights.data(),
                         static_cast<int>(normalized_heights.size()));

            mjsGeom* geom = mjs_addGeom(world, nullptr);
            if (!geom) {
                continue;
            }

            const std::string geom_name = fmt::format("{}_geom", hfield_name);
            mjs_setName(geom->element, geom_name.c_str());
            geom->type = mjGEOM_HFIELD;
            mjs_setString(geom->hfieldname, hfield_name.c_str());
            SetMuJoCoGeomPose(geom, heightfield.global_transform);
            geom->pos[2] += min_height;
            ConfigureGeomContact(geom, terrain);
            SetMuJoCoGeomColor(geom, terrain.surface_color);
            ++added_count;
            ++hfield_count;
        }

        for (std::size_t mesh_index = 0; mesh_index < terrain.mesh_patches.size(); ++mesh_index) {
            const PhysicsTerrainMeshPatchSnapshot& mesh_patch = terrain.mesh_patches[mesh_index];
            if (mesh_patch.vertices.empty() || mesh_patch.indices.size() < 3) {
                continue;
            }

            mjsMesh* mujoco_mesh = mjs_addMesh(spec, nullptr);
            if (!mujoco_mesh) {
                continue;
            }

            const std::string mesh_name = fmt::format("gobot_{}_mesh_{}", terrain_name, mesh_index);
            mjs_setName(mujoco_mesh->element, mesh_name.c_str());
            if (!SetMuJoCoMeshData(mujoco_mesh, mesh_patch)) {
                continue;
            }

            mjsGeom* geom = mjs_addGeom(world, nullptr);
            if (!geom) {
                continue;
            }

            const std::string geom_name = fmt::format("{}_geom", mesh_name);
            mjs_setName(geom->element, geom_name.c_str());
            geom->type = mjGEOM_MESH;
            mjs_setString(geom->meshname, mesh_name.c_str());
            SetMuJoCoGeomPose(geom, mesh_patch.global_transform);
            ConfigureGeomContact(geom, terrain);
            SetMuJoCoGeomColor(geom, mesh_patch.color);
            ++added_count;
            ++mesh_count;
        }
    }

    if (added_count > 0) {
        LOG_INFO("Added {} Gobot Terrain3D geom(s) to the MuJoCo world ({} hfield asset(s), {} mesh asset(s)).",
                 added_count,
                 hfield_count,
                 mesh_count);
    }
}

} // namespace

bool CompileModel(const PhysicsSceneSnapshot& snapshot, const PhysicsWorldSettings& settings,
                  CompiledModel* result, std::string* error) {
    if (!result) {
        if (error) *error = "MuJoCo compiler requires an output model.";
        return false;
    }
    CompiledModel compiled;
    ModelBuilder builder(snapshot, settings, compiled, error);
    if (!builder.Compile()) return false;
    *result = std::move(compiled);
    if (error) error->clear();
    return true;
}
} // namespace mujoco_detail
#endif

bool MuJoCoSceneCompiler::Compile(PhysicsSceneSnapshot scene_snapshot,
                                  const PhysicsWorldSettings& settings,
                                  PhysicsSceneArtifact* artifact,
                                  std::string* error) {
    if (!artifact) {
        if (error) *error = "MuJoCo scene compiler requires an output artifact.";
        return false;
    }
#ifdef GOBOT_HAS_MUJOCO
    mujoco_detail::CompiledModel compiled;
    if (!mujoco_detail::CompileModel(scene_snapshot, settings, &compiled, error)) return false;
    *artifact = std::move(compiled.artifact);
    return true;
#else
    GOB_UNUSED(scene_snapshot);
    GOB_UNUSED(settings);
    if (error) *error = "MuJoCo CPU support is not compiled into this build.";
    return false;
#endif
}
} // namespace gobot
