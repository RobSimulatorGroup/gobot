/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/physics/backends/mujoco_physics_world.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <memory>
#include <set>
#include <string_view>
#include <unordered_map>

#include "gobot/core/config/project_setting.hpp"
#include "gobot/core/registration.hpp"
#include "gobot/log.hpp"
#include "gobot/physics/joint_controller.hpp"
#include "gobot/scene/joint_3d.hpp"

#ifdef GOBOT_HAS_MUJOCO
#include <mujoco/mujoco.h>
#endif

namespace gobot {
namespace {

#ifdef GOBOT_HAS_MUJOCO
constexpr int kMuJoCoErrorBufferSize = 1024;
constexpr RealType kMuJoCoActuatorEpsilon = 1.0e-9;

struct MuJoCoJointActuatorIds {
    int motor{-1};
    int position{-1};
    int velocity{-1};
};

bool IsActuatorForJoint(const mjModel* model, int actuator_id, int joint_id) {
    const int transmission_type = model->actuator_trntype[actuator_id];
    const int transmission_id = model->actuator_trnid[2 * actuator_id];
    return (transmission_type == mjTRN_JOINT || transmission_type == mjTRN_JOINTINPARENT) &&
           transmission_id == joint_id;
}

std::string GetMuJoCoName(const mjModel* model, int object_type, int object_id) {
    const char* name = mj_id2name(model, object_type, object_id);
    return name != nullptr ? name : "";
}

bool EndsWith(const std::string& value, const std::string& suffix) {
    return value.size() >= suffix.size() &&
           value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

bool LooksLikeMotorActuator(const mjModel* model, int actuator_id) {
    return model->actuator_gaintype[actuator_id] == mjGAIN_FIXED &&
           model->actuator_biastype[actuator_id] == mjBIAS_NONE;
}

bool LooksLikePositionActuator(const mjModel* model, int actuator_id) {
    const mjtNum* gain = model->actuator_gainprm + mjNGAIN * actuator_id;
    const mjtNum* bias = model->actuator_biasprm + mjNBIAS * actuator_id;
    return model->actuator_gaintype[actuator_id] == mjGAIN_FIXED &&
           model->actuator_biastype[actuator_id] == mjBIAS_AFFINE &&
           std::abs(bias[1]) > kMuJoCoActuatorEpsilon &&
           std::abs(gain[0] + bias[1]) <= std::max<mjtNum>(1.0, std::abs(gain[0])) * 1.0e-6;
}

bool LooksLikeVelocityActuator(const mjModel* model, int actuator_id) {
    const mjtNum* gain = model->actuator_gainprm + mjNGAIN * actuator_id;
    const mjtNum* bias = model->actuator_biasprm + mjNBIAS * actuator_id;
    return model->actuator_gaintype[actuator_id] == mjGAIN_FIXED &&
           model->actuator_biastype[actuator_id] == mjBIAS_AFFINE &&
           std::abs(bias[1]) <= kMuJoCoActuatorEpsilon &&
           std::abs(bias[2]) > kMuJoCoActuatorEpsilon &&
           std::abs(gain[0] + bias[2]) <= std::max<mjtNum>(1.0, std::abs(gain[0])) * 1.0e-6;
}

MuJoCoJointActuatorIds FindActuatorsForJoint(const mjModel* model, int joint_id, const std::string& joint_name) {
    MuJoCoJointActuatorIds ids;
    if (!model || joint_id < 0) {
        return ids;
    }

    int fallback_actuator_id = -1;
    for (int actuator_id = 0; actuator_id < model->nu; ++actuator_id) {
        if (!IsActuatorForJoint(model, actuator_id, joint_id)) {
            continue;
        }

        if (fallback_actuator_id < 0) {
            fallback_actuator_id = actuator_id;
        }

        const std::string actuator_name = GetMuJoCoName(model, mjOBJ_ACTUATOR, actuator_id);
        if (ids.motor < 0 && (EndsWith(actuator_name, "_motor") ||
                              LooksLikeMotorActuator(model, actuator_id))) {
            ids.motor = actuator_id;
            continue;
        }
        if (ids.position < 0 && (EndsWith(actuator_name, "_position") ||
                                 EndsWith(actuator_name, "_pos") ||
                                 LooksLikePositionActuator(model, actuator_id))) {
            ids.position = actuator_id;
            continue;
        }
        if (ids.velocity < 0 && (EndsWith(actuator_name, "_velocity") ||
                                 EndsWith(actuator_name, "_vel") ||
                                 LooksLikeVelocityActuator(model, actuator_id))) {
            ids.velocity = actuator_id;
        }
    }

    const int named_motor_id = mj_name2id(model, mjOBJ_ACTUATOR, (joint_name + "_motor").c_str());
    if (named_motor_id >= 0) {
        ids.motor = named_motor_id;
    }
    const int named_position_id = mj_name2id(model, mjOBJ_ACTUATOR, (joint_name + "_position").c_str());
    if (named_position_id >= 0) {
        ids.position = named_position_id;
    }
    const int named_velocity_id = mj_name2id(model, mjOBJ_ACTUATOR, (joint_name + "_velocity").c_str());
    if (named_velocity_id >= 0) {
        ids.velocity = named_velocity_id;
    }

    const int named_joint_id = mj_name2id(model, mjOBJ_ACTUATOR, joint_name.c_str());
    if (ids.motor < 0 && named_joint_id >= 0) {
        ids.motor = named_joint_id;
    }
    if (ids.motor < 0 && ids.position < 0 && ids.velocity < 0) {
        ids.motor = fallback_actuator_id;
    }
    return ids;
}

void ZeroActuatorTerms(mjModel* model, int actuator_id) {
    if (!model || actuator_id < 0 || actuator_id >= model->nu) {
        return;
    }
    for (int index = 0; index < mjNGAIN; ++index) {
        model->actuator_gainprm[mjNGAIN * actuator_id + index] = 0.0;
    }
    for (int index = 0; index < mjNBIAS; ++index) {
        model->actuator_biasprm[mjNBIAS * actuator_id + index] = 0.0;
    }
}

void DisableActuator(mjModel* model, mjData* data, int actuator_id) {
    if (!model || !data || actuator_id < 0 || actuator_id >= model->nu) {
        return;
    }
    data->ctrl[actuator_id] = 0.0;
    ZeroActuatorTerms(model, actuator_id);
}

void SetMotorActuator(mjModel* model, mjData* data, int actuator_id, RealType effort) {
    if (!model || !data || actuator_id < 0 || actuator_id >= model->nu) {
        return;
    }
    ZeroActuatorTerms(model, actuator_id);
    model->actuator_gainprm[mjNGAIN * actuator_id + 0] = 1.0;
    data->ctrl[actuator_id] = effort;
}

void SetPositionActuator(mjModel* model, mjData* data, int actuator_id, RealType target, RealType stiffness) {
    if (!model || !data || actuator_id < 0 || actuator_id >= model->nu) {
        return;
    }
    ZeroActuatorTerms(model, actuator_id);
    model->actuator_gainprm[mjNGAIN * actuator_id + 0] = stiffness;
    model->actuator_biasprm[mjNBIAS * actuator_id + 1] = -stiffness;
    data->ctrl[actuator_id] = target;
}

void SetVelocityActuator(mjModel* model, mjData* data, int actuator_id, RealType target, RealType damping) {
    if (!model || !data || actuator_id < 0 || actuator_id >= model->nu) {
        return;
    }
    ZeroActuatorTerms(model, actuator_id);
    model->actuator_gainprm[mjNGAIN * actuator_id + 0] = damping;
    model->actuator_biasprm[mjNBIAS * actuator_id + 2] = -damping;
    data->ctrl[actuator_id] = target;
}

std::string ResolvePhysicsSourcePath(const std::string& source_path) {
    if (source_path.empty()) {
        return {};
    }

    if (ProjectSettings::s_singleton) {
        return ProjectSettings::GetInstance()->GlobalizePath(source_path);
    }

    if (source_path.starts_with("res://")) {
        return source_path.substr(std::string_view("res://").size());
    }

    return source_path;
}

const PhysicsLinkSnapshot* FindLinkSnapshot(const PhysicsRobotSnapshot& robot, const std::string& link_name) {
    for (const PhysicsLinkSnapshot& link : robot.links) {
        if (link.name == link_name) {
            return &link;
        }
    }

    return nullptr;
}

const PhysicsJointSnapshot* FindParentJointForLink(const PhysicsRobotSnapshot& robot, const std::string& link_name) {
    for (const PhysicsJointSnapshot& joint : robot.joints) {
        if (joint.child_link == link_name) {
            return &joint;
        }
    }
    return nullptr;
}

std::string SanitizeMuJoCoName(std::string name) {
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

void SetMuJoCoVector3(double* target, const Vector3& value) {
    target[0] = value.x();
    target[1] = value.y();
    target[2] = value.z();
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

std::string SensorComponentName(const std::string& prefix,
                                const PhysicsSensorSnapshot& sensor,
                                std::string_view component) {
    return prefix + sensor.link_name + "_" + sensor.name + "_" + std::string(component);
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
    mujoco_sensor->noise = sensor.noise_stddev;
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
    mujoco_sensor->noise = sensor.noise_stddev;
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
    if (sensor.type != PhysicsSensorType::AngularMomentum) {
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

bool HasBodyJoint(const PhysicsLinkSnapshot& link, const PhysicsRobotSnapshot& robot) {
    const PhysicsJointSnapshot* parent_joint = FindParentJointForLink(robot, link.name);
    return parent_joint != nullptr && !IsFixedMuJoCoJoint(*parent_joint);
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

void ApplyMuJoCoOptions(mjOption* option, const PhysicsWorldSettings& settings) {
    if (!option) {
        return;
    }

    option->timestep = settings.fixed_time_step;
    SetMuJoCoVector3(option->gravity, settings.gravity);
    option->solver = settings.mujoco_solver.solver;
    option->integrator = settings.mujoco_solver.integrator;
    option->cone = settings.mujoco_solver.cone;
    option->jacobian = settings.mujoco_solver.jacobian;
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

void ConfigureGeomContact(mjsGeom* geom, const PhysicsShapeSnapshot& shape) {
    if (!geom) {
        return;
    }

    geom->contype = shape.contype;
    geom->conaffinity = shape.conaffinity;
    geom->condim = shape.condim;
    geom->friction[0] = shape.friction.x();
    geom->friction[1] = shape.friction.y();
    geom->friction[2] = shape.friction.z();
    geom->solref[0] = shape.solref.x();
    geom->solref[1] = shape.solref.y();
    SetMuJoCoArray(geom->solimp, shape.solimp, mjNIMP);
    geom->margin = shape.margin;
    geom->gap = shape.gap;
}

void ConfigureGeomContact(mjsGeom* geom, const PhysicsTerrainSnapshot& terrain) {
    if (!geom) {
        return;
    }

    geom->contype = terrain.contype;
    geom->conaffinity = terrain.conaffinity;
    geom->condim = terrain.condim;
    geom->friction[0] = terrain.friction.x();
    geom->friction[1] = terrain.friction.y();
    geom->friction[2] = terrain.friction.z();
    geom->solref[0] = terrain.solref.x();
    geom->solref[1] = terrain.solref.y();
    SetMuJoCoArray(geom->solimp, terrain.solimp, mjNIMP);
    geom->margin = terrain.margin;
    geom->gap = terrain.gap;
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

bool SetMuJoCoMeshData(mjsMesh* mesh,
                       const PhysicsTerrainMeshPatchSnapshot& mesh_patch) {
    if (!mesh || mesh_patch.vertices.empty() || mesh_patch.indices.empty()) {
        return false;
    }

    std::vector<float> vertices;
    vertices.reserve(mesh_patch.vertices.size() * 3);
    for (const Vector3& vertex : mesh_patch.vertices) {
        vertices.push_back(static_cast<float>(vertex.x()));
        vertices.push_back(static_cast<float>(vertex.y()));
        vertices.push_back(static_cast<float>(vertex.z()));
    }

    std::vector<int> indices;
    indices.reserve(mesh_patch.indices.size());
    for (std::uint32_t index : mesh_patch.indices) {
        if (index >= mesh_patch.vertices.size()) {
            return false;
        }
        indices.push_back(static_cast<int>(index));
    }

    const int vertex_count = static_cast<int>(mesh_patch.vertices.size());
    const int face_count = static_cast<int>(indices.size() / 3);
    if (vertex_count <= 0 || face_count <= 0) {
        return false;
    }

    mjs_setFloat(mesh->uservert, vertices.data(), static_cast<int>(vertices.size()));
    mjs_setInt(mesh->userface, indices.data(), static_cast<int>(indices.size()));
    return true;
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
    if (!heightfield.normalized_elevation.empty()) {
        for (std::size_t index = 0; index < expected_count; ++index) {
            const RealType value = index < heightfield.normalized_elevation.size()
                    ? heightfield.normalized_elevation[index]
                    : 0.0;
            data[index] = static_cast<float>(std::clamp(value,
                                                        static_cast<RealType>(0.0),
                                                        static_cast<RealType>(1.0)));
        }
        if (min_height) {
            *min_height = heightfield.z_offset;
        }
        if (height_range) {
            *height_range = 1.0;
        }
        return data;
    }

    if (!heightfield.heights.empty()) {
        const auto minmax = std::minmax_element(heightfield.heights.begin(), heightfield.heights.end());
        min_value = *minmax.first;
        max_value = *minmax.second;
    }
    const RealType range = std::max<RealType>(max_value - min_value, CMP_EPSILON);

    for (std::size_t index = 0; index < expected_count; ++index) {
        const RealType value = index < heightfield.heights.size() ? heightfield.heights[index] : min_value;
        data[index] = static_cast<float>((value - min_value) / range);
    }

    if (min_height) {
        *min_height = min_value + heightfield.z_offset;
    }
    if (height_range) {
        *height_range = range;
    }
    return data;
}

void AddShapeGeomToBody(mjsBody* body,
                        const PhysicsShapeSnapshot& shape,
                        const PhysicsLinkSnapshot& link,
                        const std::string& name) {
    if (!body || shape.disabled) {
        return;
    }

    mjsGeom* geom = mjs_addGeom(body, nullptr);
    if (!geom) {
        return;
    }

    mjs_setName(geom->element, name.c_str());
    switch (shape.type) {
        case PhysicsShapeType::Box:
            geom->type = mjGEOM_BOX;
            geom->size[0] = shape.box_size.x() * 0.5;
            geom->size[1] = shape.box_size.y() * 0.5;
            geom->size[2] = shape.box_size.z() * 0.5;
            break;
        case PhysicsShapeType::Sphere:
            geom->type = mjGEOM_SPHERE;
            geom->size[0] = shape.radius;
            break;
        case PhysicsShapeType::Cylinder:
            geom->type = mjGEOM_CYLINDER;
            geom->size[0] = shape.radius;
            geom->size[1] = shape.height * 0.5;
            break;
        case PhysicsShapeType::Capsule:
            geom->type = mjGEOM_CAPSULE;
            geom->size[0] = shape.radius;
            geom->size[1] = shape.height * 0.5;
            break;
        default:
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
    body->iquat[0] = 1.0;
    body->iquat[1] = 0.0;
    body->iquat[2] = 0.0;
    body->iquat[3] = 0.0;
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
}

void ConfigureActuatorLimits(mjsActuator* actuator, const PhysicsJointSnapshot& joint, bool control_is_position) {
    if (!actuator) {
        return;
    }
    if (HasControlRange(joint)) {
        actuator->ctrllimited = mjLIMITED_TRUE;
        actuator->ctrlrange[0] = joint.control_lower_limit;
        actuator->ctrlrange[1] = joint.control_upper_limit;
    } else if (control_is_position && joint.upper_limit > joint.lower_limit) {
        actuator->ctrllimited = mjLIMITED_TRUE;
        actuator->ctrlrange[0] = joint.lower_limit;
        actuator->ctrlrange[1] = joint.upper_limit;
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
    if (drive_mode == JointDriveMode::Passive) {
        return;
    }

    if (drive_mode == JointDriveMode::Motor) {
        mjsActuator* motor = AddJointActuator(spec, joint, prefixed_name, "_motor", false);
        if (motor) {
            mjs_setToMotor(motor);
        }
        return;
    }

    if (drive_mode == JointDriveMode::Position) {
        mjsActuator* position = AddJointActuator(spec, joint, prefixed_name, "_position", true);
        if (position) {
            mjs_setToPosition(position, joint.drive_stiffness, nullptr, nullptr, nullptr, 0.0);
            ConfigureActuatorLimits(position, joint, true);
        }
        return;
    }

    if (drive_mode == JointDriveMode::Velocity) {
        mjsActuator* velocity = AddJointActuator(spec, joint, prefixed_name, "_velocity", false);
        if (velocity) {
            mjs_setToVelocity(velocity, joint.drive_damping);
            ConfigureActuatorLimits(velocity, joint, false);
        }
    }
}

std::string GetMuJoCoString(const mjString* value) {
    return value ? mjs_getString(value) : "";
}

std::string ResolveMuJoCoAssetPath(const std::filesystem::path& model_dir,
                                   const std::string& asset_dir,
                                   const std::string& file_path) {
    if (file_path.empty()) {
        return {};
    }

    const std::filesystem::path file(file_path);
    if (file.is_absolute()) {
        return file.lexically_normal().string();
    }

    const std::filesystem::path direct_path = (model_dir / file).lexically_normal();
    if (std::filesystem::exists(direct_path)) {
        return direct_path.string();
    }

    if (model_dir.has_parent_path()) {
        const std::filesystem::path parent_direct_path =
                (model_dir.parent_path() / file).lexically_normal();
        if (std::filesystem::exists(parent_direct_path)) {
            return parent_direct_path.string();
        }
    }

    if (!asset_dir.empty()) {
        const std::filesystem::path asset_path = (model_dir / asset_dir / file).lexically_normal();
        if (std::filesystem::exists(asset_path)) {
            return asset_path.string();
        }

        if (model_dir.has_parent_path()) {
            const std::filesystem::path parent_asset_path =
                    (model_dir.parent_path() / asset_dir / file).lexically_normal();
            if (std::filesystem::exists(parent_asset_path)) {
                return parent_asset_path.string();
            }
        }

        if (model_dir.has_parent_path() && model_dir.parent_path().has_parent_path()) {
            const std::filesystem::path grandparent_asset_path =
                    (model_dir.parent_path().parent_path() / asset_dir / file).lexically_normal();
            if (std::filesystem::exists(grandparent_asset_path)) {
                return grandparent_asset_path.string();
            }
        }

        return asset_path.string();
    }

    return direct_path.string();
}

void NormalizeMuJoCoFileString(mjString* file,
                               const std::filesystem::path& model_dir,
                               const std::string& asset_dir) {
    const std::string existing_path = GetMuJoCoString(file);
    if (existing_path.empty()) {
        return;
    }

    const std::string resolved_path = ResolveMuJoCoAssetPath(model_dir, asset_dir, existing_path);
    if (!resolved_path.empty()) {
        mjs_setString(file, resolved_path.c_str());
    }
}

void NormalizeMuJoCoAssetPaths(mjSpec* spec, const std::filesystem::path& model_path) {
    if (!spec) {
        return;
    }

    const std::filesystem::path model_dir = model_path.parent_path();
    const std::string mesh_dir = GetMuJoCoString(spec->compiler.meshdir);
    const std::string texture_dir = GetMuJoCoString(spec->compiler.texturedir);

    for (mjsElement* element = mjs_firstElement(spec, mjOBJ_MESH);
         element;
         element = mjs_nextElement(spec, element)) {
        mjsMesh* mesh = mjs_asMesh(element);
        if (mesh) {
            NormalizeMuJoCoFileString(mesh->file, model_dir, mesh_dir);
        }
    }

    for (mjsElement* element = mjs_firstElement(spec, mjOBJ_HFIELD);
         element;
         element = mjs_nextElement(spec, element)) {
        mjsHField* hfield = mjs_asHField(element);
        if (hfield) {
            NormalizeMuJoCoFileString(hfield->file, model_dir, mesh_dir);
        }
    }

    for (mjsElement* element = mjs_firstElement(spec, mjOBJ_TEXTURE);
         element;
         element = mjs_nextElement(spec, element)) {
        mjsTexture* texture = mjs_asTexture(element);
        if (texture) {
            NormalizeMuJoCoFileString(texture->file, model_dir, texture_dir);
            if (texture->cubefiles) {
                for (std::string& cube_file : *texture->cubefiles) {
                    if (!cube_file.empty()) {
                        cube_file = ResolveMuJoCoAssetPath(model_dir, texture_dir, cube_file);
                    }
                }
            }
        }
    }

    for (mjsElement* element = mjs_firstElement(spec, mjOBJ_SKIN);
         element;
         element = mjs_nextElement(spec, element)) {
        mjsSkin* skin = mjs_asSkin(element);
        if (skin) {
            NormalizeMuJoCoFileString(skin->file, model_dir, mesh_dir);
        }
    }

    mjs_setString(spec->compiler.meshdir, "");
    mjs_setString(spec->compiler.texturedir, "");
    mjs_setString(spec->modelfiledir, "");
}

#endif

} // namespace

MuJoCoPhysicsWorld::MuJoCoPhysicsWorld()
    : available_(IsBackendAvailable()) {
    if (!available_) {
        SetLastError(GetUnavailableReason());
    }
}

MuJoCoPhysicsWorld::~MuJoCoPhysicsWorld() {
#ifdef GOBOT_HAS_MUJOCO
    FreeModel();
#endif
}

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
    ResetSceneStateFromSnapshot();

    if (!available_) {
        SetLastError(GetUnavailableReason());
        return false;
    }

#ifdef GOBOT_HAS_MUJOCO
    if (!LoadModelFromRobotSources()) {
        return false;
    }

    if (!ConfigureEnvironmentBatch(1)) {
        return false;
    }
    SyncStateToMuJoCo(0);
    SyncStateFromMuJoCo(0);
    last_error_.clear();
    return true;
#else
    return false;
#endif
}

void MuJoCoPhysicsWorld::Reset() {
    PhysicsWorld::Reset();
    if (!available_) {
        return;
    }

#ifdef GOBOT_HAS_MUJOCO
    if (model_ && data_) {
        if (!environment_states_.empty()) {
            environment_states_[0] = scene_state_;
        }
        mj_resetData(static_cast<mjModel*>(model_), static_cast<mjData*>(data_));
        SyncStateToMuJoCo(0);
        SyncStateFromMuJoCo(0);
    }
#endif
}

bool MuJoCoPhysicsWorld::SetLinkExternalForce(const std::string& robot_name,
                                              const std::string& link_name,
                                              const Vector3& point,
                                              const Vector3& force) {
    if (!PhysicsWorld::SetLinkExternalForce(robot_name, link_name, point, force)) {
        return false;
    }
    return true;
}

bool MuJoCoPhysicsWorld::SetLinkSpringForce(const std::string& robot_name,
                                            const std::string& link_name,
                                            const Vector3& local_point,
                                            const Vector3& target_point,
                                            const Vector3& force_hint) {
    if (!PhysicsWorld::SetLinkSpringForce(robot_name, link_name, local_point, target_point, force_hint)) {
        return false;
    }
    return true;
}

void MuJoCoPhysicsWorld::ClearExternalForces() {
    PhysicsWorld::ClearExternalForces();
#ifdef GOBOT_HAS_MUJOCO
    auto* model = static_cast<mjModel*>(model_);
    if (model != nullptr && model->nbody > 0) {
        for (void* environment_data : environment_data_) {
            auto* data = static_cast<mjData*>(environment_data);
            if (data != nullptr) {
                mju_zero(data->xfrc_applied, 6 * model->nbody);
            }
        }
    }
#endif
}

MuJoCoPhysicsWorld::Diagnostics MuJoCoPhysicsWorld::GetDiagnostics() const {
    Diagnostics diagnostics;
#ifdef GOBOT_HAS_MUJOCO
    const auto* model = static_cast<const mjModel*>(model_);
    if (model == nullptr) {
        return diagnostics;
    }

    diagnostics.timestep = static_cast<RealType>(model->opt.timestep);
    diagnostics.solver = model->opt.solver;
    diagnostics.integrator = model->opt.integrator;
    diagnostics.cone = model->opt.cone;
    diagnostics.jacobian = model->opt.jacobian;
    diagnostics.iterations = model->opt.iterations;
    diagnostics.line_search_iterations = model->opt.ls_iterations;
    diagnostics.no_slip_iterations = model->opt.noslip_iterations;
    diagnostics.convex_collision_iterations = model->opt.ccd_iterations;
    diagnostics.tolerance = static_cast<RealType>(model->opt.tolerance);
    diagnostics.line_search_tolerance = static_cast<RealType>(model->opt.ls_tolerance);
    diagnostics.no_slip_tolerance = static_cast<RealType>(model->opt.noslip_tolerance);
    diagnostics.convex_collision_tolerance = static_cast<RealType>(model->opt.ccd_tolerance);
    diagnostics.impedance_ratio = static_cast<RealType>(model->opt.impratio);
    diagnostics.actuator_count = model->nu;

    for (const MuJoCoJointBinding& binding : joint_bindings_) {
        if (diagnostics.first_position_actuator_stiffness <= 0.0 &&
            binding.position_actuator_id >= 0 &&
            binding.position_actuator_id < model->nu) {
            diagnostics.first_position_actuator_stiffness =
                    static_cast<RealType>(std::abs(model->actuator_gainprm[
                            mjNGAIN * binding.position_actuator_id + 0]));
        }
        if (diagnostics.first_controllable_joint_damping <= 0.0 &&
            binding.dof_address >= 0 &&
            binding.dof_address < model->nv) {
            diagnostics.first_controllable_joint_damping =
                    static_cast<RealType>(model->dof_damping[binding.dof_address]);
        }
    }

    if (model->ngeom > 0) {
        diagnostics.first_collision_friction = Vector3{
                static_cast<RealType>(model->geom_friction[0]),
                static_cast<RealType>(model->geom_friction[1]),
                static_cast<RealType>(model->geom_friction[2])};
        diagnostics.first_collision_contact_dimension = model->geom_condim[0];
        diagnostics.first_collision_solref = Vector2{
                static_cast<RealType>(model->geom_solref[0]),
                static_cast<RealType>(model->geom_solref[1])};
        diagnostics.first_collision_solimp.reserve(mjNIMP);
        for (int index = 0; index < mjNIMP; ++index) {
            diagnostics.first_collision_solimp.push_back(
                    static_cast<RealType>(model->geom_solimp[index]));
        }
    }
#endif
    return diagnostics;
}

bool MuJoCoPhysicsWorld::RestoreCompatibleState(const PhysicsSceneState& previous_state) {
    if (!PhysicsWorld::RestoreCompatibleState(previous_state)) {
        return false;
    }

#ifdef GOBOT_HAS_MUJOCO
    if (model_ && data_) {
        if (!environment_states_.empty()) {
            environment_states_[0] = scene_state_;
        }
        SyncStateToMuJoCo(0);
        SyncStateFromMuJoCo(0);
    }
#endif

    return true;
}

void MuJoCoPhysicsWorld::Step(RealType delta_time) {
    if (!available_) {
        return;
    }

#ifdef GOBOT_HAS_MUJOCO
    auto* model = static_cast<mjModel*>(model_);
    auto* data = static_cast<mjData*>(data_);
    if (!model || !data) {
        return;
    }

    ApplyMuJoCoOptions(&model->opt, settings_);
    model->opt.timestep = delta_time > 0.0 ? delta_time : settings_.fixed_time_step;
    ApplyControlsToMuJoCo(0);
    ApplyExternalForcesToMuJoCo(0);
    mj_step(model, data);
    SyncStateFromMuJoCo(0);
#else
    GOB_UNUSED(delta_time);
#endif
}

bool MuJoCoPhysicsWorld::ConfigureEnvironmentBatch(std::size_t environment_count) {
    if (environment_count == 0) {
        SetLastError("MuJoCo environment batch size must be greater than zero.");
        return false;
    }

    if (!available_) {
        SetLastError(GetUnavailableReason());
        return false;
    }

#ifdef GOBOT_HAS_MUJOCO
    auto* model = static_cast<mjModel*>(model_);
    if (!model) {
        SetLastError("MuJoCo model has not been built.");
        return false;
    }

    for (std::size_t i = 1; i < environment_data_.size(); ++i) {
        if (environment_data_[i] != nullptr) {
            mj_deleteData(static_cast<mjData*>(environment_data_[i]));
        }
    }
    environment_data_.clear();

    if (data_ == nullptr) {
        data_ = mj_makeData(model);
        if (data_ == nullptr) {
            SetLastError("MuJoCo failed to allocate runtime data for environment 0.");
            return false;
        }
    }
    environment_data_.push_back(data_);

    for (std::size_t i = 1; i < environment_count; ++i) {
        mjData* data = mj_makeData(model);
        if (data == nullptr) {
            SetLastError(fmt::format("MuJoCo failed to allocate runtime data for environment {}.", i));
            return false;
        }
        environment_data_.push_back(data);
    }

    environment_states_.assign(environment_count, scene_state_);
    for (std::size_t i = 0; i < environment_count; ++i) {
        mj_resetData(model, static_cast<mjData*>(environment_data_[i]));
        SyncStateToMuJoCo(i);
        SyncStateFromMuJoCo(i);
    }

    last_error_.clear();
    return true;
#else
    GOB_UNUSED(environment_count);
    return false;
#endif
}

std::size_t MuJoCoPhysicsWorld::GetEnvironmentCount() const {
#ifdef GOBOT_HAS_MUJOCO
    return environment_data_.empty() ? 0 : environment_data_.size();
#else
    return 0;
#endif
}

const PhysicsSceneState* MuJoCoPhysicsWorld::GetEnvironmentState(std::size_t environment_index) const {
#ifdef GOBOT_HAS_MUJOCO
    if (!IsEnvironmentIndexValid(environment_index)) {
        return nullptr;
    }
    return &EnvironmentState(environment_index);
#else
    GOB_UNUSED(environment_index);
    return nullptr;
#endif
}

bool MuJoCoPhysicsWorld::ResetEnvironment(std::size_t environment_index) {
#ifdef GOBOT_HAS_MUJOCO
    if (!IsEnvironmentIndexValid(environment_index)) {
        SetLastError(fmt::format("Environment index {} is out of range.", environment_index));
        return false;
    }

    auto* model = static_cast<mjModel*>(model_);
    auto* data = static_cast<mjData*>(environment_data_[environment_index]);
    if (!model || !data) {
        SetLastError("MuJoCo environment data is unavailable.");
        return false;
    }

    EnvironmentState(environment_index) = MakeSceneStateFromSnapshot();
    mj_resetData(model, data);
    SyncStateToMuJoCo(environment_index);
    SyncStateFromMuJoCo(environment_index);
    last_error_.clear();
    return true;
#else
    GOB_UNUSED(environment_index);
    return false;
#endif
}

bool MuJoCoPhysicsWorld::StepEnvironment(std::size_t environment_index, RealType delta_time) {
#ifdef GOBOT_HAS_MUJOCO
    if (!IsEnvironmentIndexValid(environment_index)) {
        SetLastError(fmt::format("Environment index {} is out of range.", environment_index));
        return false;
    }

    auto* model = static_cast<mjModel*>(model_);
    auto* data = static_cast<mjData*>(environment_data_[environment_index]);
    if (!model || !data) {
        SetLastError("MuJoCo environment data is unavailable.");
        return false;
    }

    ApplyMuJoCoOptions(&model->opt, settings_);
    model->opt.timestep = delta_time > 0.0 ? delta_time : settings_.fixed_time_step;
    ApplyControlsToMuJoCo(environment_index);
    ApplyExternalForcesToMuJoCo(environment_index);
    mj_step(model, data);
    SyncStateFromMuJoCo(environment_index);
    last_error_.clear();
    return true;
#else
    GOB_UNUSED(environment_index);
    GOB_UNUSED(delta_time);
    return false;
#endif
}

bool MuJoCoPhysicsWorld::ResetJointState(const std::string& robot_name,
                                         const std::string& joint_name,
                                         RealType position,
                                         RealType velocity) {
    if (!PhysicsWorld::ResetJointState(robot_name, joint_name, position, velocity)) {
        return false;
    }

#ifdef GOBOT_HAS_MUJOCO
    if (model_ && data_) {
        if (!environment_states_.empty()) {
            environment_states_[0] = scene_state_;
        }
        SyncStateToMuJoCo(0);
        SyncStateFromMuJoCo(0);
    }
#endif

    return true;
}

bool MuJoCoPhysicsWorld::ResetEnvironmentJointState(std::size_t environment_index,
                                                    const std::string& robot_name,
                                                    const std::string& joint_name,
                                                    RealType position,
                                                    RealType velocity) {
#ifdef GOBOT_HAS_MUJOCO
    if (!IsEnvironmentIndexValid(environment_index)) {
        SetLastError(fmt::format("Environment index {} is out of range.", environment_index));
        return false;
    }

    PhysicsSceneState& state = EnvironmentState(environment_index);
    if (!ResetJointStateIn(state, robot_name, joint_name, position, velocity)) {
        return false;
    }

#ifdef GOBOT_HAS_MUJOCO
    SyncStateToMuJoCo(environment_index);
    SyncStateFromMuJoCo(environment_index);
#endif
    last_error_.clear();
    return true;
#else
    GOB_UNUSED(environment_index);
    GOB_UNUSED(robot_name);
    GOB_UNUSED(joint_name);
    GOB_UNUSED(position);
    GOB_UNUSED(velocity);
    return false;
#endif
}

bool MuJoCoPhysicsWorld::ResetLinkState(const std::string& robot_name,
                                        const std::string& link_name,
                                        const Vector3& position,
                                        const Quaternion& orientation,
                                        const Vector3& linear_velocity,
                                        const Vector3& angular_velocity) {
    if (!PhysicsWorld::ResetLinkState(robot_name,
                                      link_name,
                                      position,
                                      orientation,
                                      linear_velocity,
                                      angular_velocity)) {
        return false;
    }

#ifdef GOBOT_HAS_MUJOCO
    if (model_ && data_) {
        if (!environment_states_.empty()) {
            environment_states_[0] = scene_state_;
        }
        SyncStateToMuJoCo(0);
        SyncStateFromMuJoCo(0);
    }
#endif

    return true;
}

bool MuJoCoPhysicsWorld::ResetEnvironmentLinkState(std::size_t environment_index,
                                                   const std::string& robot_name,
                                                   const std::string& link_name,
                                                   const Vector3& position,
                                                   const Quaternion& orientation,
                                                   const Vector3& linear_velocity,
                                                   const Vector3& angular_velocity) {
#ifdef GOBOT_HAS_MUJOCO
    if (!IsEnvironmentIndexValid(environment_index)) {
        SetLastError(fmt::format("Environment index {} is out of range.", environment_index));
        return false;
    }

    PhysicsSceneState& state = EnvironmentState(environment_index);
    if (!ResetLinkStateIn(state,
                          robot_name,
                          link_name,
                          position,
                          orientation,
                          linear_velocity,
                          angular_velocity)) {
        return false;
    }

#ifdef GOBOT_HAS_MUJOCO
    SyncStateToMuJoCo(environment_index);
    SyncStateFromMuJoCo(environment_index);
#endif
    last_error_.clear();
    return true;
#else
    GOB_UNUSED(environment_index);
    GOB_UNUSED(robot_name);
    GOB_UNUSED(link_name);
    GOB_UNUSED(position);
    GOB_UNUSED(orientation);
    GOB_UNUSED(linear_velocity);
    GOB_UNUSED(angular_velocity);
    return false;
#endif
}

bool MuJoCoPhysicsWorld::SetEnvironmentJointControl(std::size_t environment_index,
                                                    const std::string& robot_name,
                                                    const std::string& joint_name,
                                                    PhysicsJointControlMode control_mode,
                                                    RealType target) {
#ifdef GOBOT_HAS_MUJOCO
    if (!IsEnvironmentIndexValid(environment_index)) {
        SetLastError(fmt::format("Environment index {} is out of range.", environment_index));
        return false;
    }

    PhysicsSceneState& state = EnvironmentState(environment_index);
    if (!SetJointControlIn(state, robot_name, joint_name, control_mode, target)) {
        return false;
    }

    last_error_.clear();
    return true;
#else
    GOB_UNUSED(environment_index);
    GOB_UNUSED(robot_name);
    GOB_UNUSED(joint_name);
    GOB_UNUSED(control_mode);
    GOB_UNUSED(target);
    return false;
#endif
}

#ifdef GOBOT_HAS_MUJOCO
bool MuJoCoPhysicsWorld::LoadModelFromRobotSources() {
    FreeModel();

    std::vector<std::size_t> robot_indices;
    for (std::size_t robot_index = 0; robot_index < scene_snapshot_.robots.size(); ++robot_index) {
        robot_indices.push_back(robot_index);
    }

    if (robot_indices.empty()) {
        SetLastError("MuJoCo backend requires at least one Robot3D in the scene.");
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

        bool attached = false;
        if (HasUsableAuthoredModel(robot)) {
            attached = AddAuthoredRobotToSpec(parent_spec, robot, robot_index, prefix);
        } else if (!robot.source_path.empty()) {
            const std::string model_path = ResolvePhysicsSourcePath(robot.source_path);
            if (!model_path.empty() && std::filesystem::exists(model_path)) {
                attached = AttachRobotModelToSpec(parent_spec, robot, robot_index, prefix);
            } else {
                LOG_WARN("MuJoCo robot '{}' source '{}' is unavailable; using authored Gobot scene data.",
                         robot.name,
                         robot.source_path);
                attached = AddAuthoredRobotToSpec(parent_spec, robot, robot_index, prefix);
            }
        } else {
            attached = AddAuthoredRobotToSpec(parent_spec, robot, robot_index, prefix);
        }
        if (!attached) {
            return false;
        }

        robot_bindings_.push_back({robot_index, prefix});
    }

    mjModel* model = mj_compile(parent_spec, nullptr);
    if (!model) {
        const std::string compile_error = mjs_getError(parent_spec) ? mjs_getError(parent_spec) : "unknown error";
        SetLastError(fmt::format("MuJoCo failed to compile merged scene model: {}", compile_error));
        return false;
    }

    auto* data = mj_makeData(model);
    if (!data) {
        mj_deleteModel(model);
        SetLastError("MuJoCo failed to allocate runtime data for merged scene model.");
        return false;
    }

    model_ = model;
    data_ = data;
    BuildLinkBindings();
    BuildJointBindings();
    BuildSensorBindings();

    LOG_INFO("MuJoCo merged physics model loaded: robots={}, nq={}, nv={}, joints={}",
             robot_bindings_.size(),
             model->nq,
             model->nv,
             model->njnt);
    return true;
}

bool MuJoCoPhysicsWorld::AttachRobotModelToSpec(void* parent_spec_ptr,
                                                const PhysicsRobotSnapshot& robot,
                                                std::size_t robot_index,
                                                const std::string& prefix) {
    auto* parent_spec = static_cast<mjSpec*>(parent_spec_ptr);
    if (!parent_spec) {
        SetLastError("Cannot attach robot to a null MuJoCo parent spec.");
        return false;
    }

    const std::string model_path = ResolvePhysicsSourcePath(robot.source_path);
    if (model_path.empty()) {
        SetLastError(fmt::format("MuJoCo robot '{}' source path is empty after path resolution.", robot.name));
        return false;
    }

    if (!std::filesystem::exists(model_path)) {
        SetLastError(fmt::format("MuJoCo robot '{}' source file does not exist: {}", robot.name, model_path));
        return false;
    }

    char error[kMuJoCoErrorBufferSize] = {};
    mjSpec* child_spec = mj_parseXML(model_path.c_str(), nullptr, error, sizeof(error));
    if (!child_spec) {
        SetLastError(fmt::format("MuJoCo failed to parse robot '{}' source '{}': {}",
                                 robot.name,
                                 model_path,
                                 error));
        return false;
    }
    std::unique_ptr<mjSpec, decltype(&mj_deleteSpec)> child_spec_guard(child_spec, mj_deleteSpec);

    NormalizeMuJoCoAssetPaths(child_spec, model_path);
    AddFloatingBaseJointsToSpec(child_spec, robot);

    mjsBody* world = mjs_findBody(parent_spec, "world");
    if (!world) {
        SetLastError("MuJoCo parent spec has no world body.");
        return false;
    }

    mjsElement* attached = mjs_attach(world->element, child_spec->element, prefix.c_str(), "");
    if (!attached) {
        const std::string attach_error = mjs_getError(parent_spec) ? mjs_getError(parent_spec) : "unknown error";
        SetLastError(fmt::format("MuJoCo failed to attach robot '{}' from '{}': {}",
                                 robot.name,
                                 model_path,
                                 attach_error));
        return false;
    }

    LOG_INFO("Attached MuJoCo robot '{}' from '{}' with prefix '{}'.",
             robot.name,
             model_path,
             prefix);
    GOB_UNUSED(robot_index);
    return true;
}

bool MuJoCoPhysicsWorld::AddAuthoredRobotToSpec(void* parent_spec_ptr,
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

        if (parent_joint != nullptr) {
            AddJointToBody(body, *parent_joint, link.global_transform, prefix + parent_joint->name);
        }

        for (std::size_t shape_index = 0; shape_index < link.collision_shapes.size(); ++shape_index) {
            AddShapeGeomToBody(body,
                               link.collision_shapes[shape_index],
                               link,
                               fmt::format("{}{}_geom_{}", prefix, link.name, shape_index));
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

    LOG_INFO("Added authored Gobot robot '{}' to MuJoCo spec with prefix '{}'.", robot.name, prefix);
    GOB_UNUSED(robot_index);
    return true;
}

void MuJoCoPhysicsWorld::AddLooseSceneGeomsToSpec(void* spec_ptr) {
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

        mjsGeom* geom = mjs_addGeom(world, nullptr);
        if (!geom) {
            continue;
        }

        const std::string name = fmt::format("gobot_loose_box_{}", shape_index);
        mjs_setName(geom->element, name.c_str());
        switch (shape.type) {
            case PhysicsShapeType::Box:
                geom->type = mjGEOM_BOX;
                geom->size[0] = shape.box_size.x() * 0.5;
                geom->size[1] = shape.box_size.y() * 0.5;
                geom->size[2] = shape.box_size.z() * 0.5;
                break;
            case PhysicsShapeType::Sphere:
                geom->type = mjGEOM_SPHERE;
                geom->size[0] = shape.radius;
                break;
            case PhysicsShapeType::Cylinder:
                geom->type = mjGEOM_CYLINDER;
                geom->size[0] = shape.radius;
                geom->size[1] = shape.height * 0.5;
                break;
            case PhysicsShapeType::Capsule:
                geom->type = mjGEOM_CAPSULE;
                geom->size[0] = shape.radius;
                geom->size[1] = shape.height * 0.5;
                break;
            default:
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

void MuJoCoPhysicsWorld::AddTerrainGeomsToSpec(void* spec_ptr) {
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

void MuJoCoPhysicsWorld::AddFloatingBaseJointsToSpec(void* spec_ptr, const PhysicsRobotSnapshot& robot) {
    auto* spec = static_cast<mjSpec*>(spec_ptr);
    if (!spec) {
        return;
    }

    int added_count = 0;
    for (const PhysicsJointSnapshot& joint : robot.joints) {
        if (static_cast<JointType>(joint.joint_type) != JointType::Floating || joint.child_link.empty()) {
            continue;
        }

        mjsBody* child_body = mjs_findBody(spec, joint.child_link.c_str());
        if (!child_body) {
            LOG_WARN("MuJoCo spec has no body '{}' for Gobot floating joint '{}'.",
                     joint.child_link,
                     joint.name);
            continue;
        }

        const std::string free_joint_name = joint.name.empty()
                                                    ? fmt::format("{}_freejoint", joint.child_link)
                                                    : joint.name;
        if (mjs_findElement(spec, mjOBJ_JOINT, free_joint_name.c_str()) != nullptr) {
            continue;
        }

        mjsJoint* free_joint = mjs_addFreeJoint(child_body);
        if (!free_joint) {
            LOG_WARN("Failed to add MuJoCo free joint for Gobot floating joint '{}'.", joint.name);
            continue;
        }

        mjs_setName(free_joint->element, free_joint_name.c_str());
        ++added_count;
    }

    if (added_count > 0) {
        LOG_INFO("Added {} Gobot floating joint(s) to MuJoCo robot '{}'.", added_count, robot.name);
    }
}

void MuJoCoPhysicsWorld::BuildLinkBindings() {
    link_bindings_.clear();

    auto* model = static_cast<mjModel*>(model_);
    if (!model) {
        return;
    }

    for (std::size_t robot_index = 0; robot_index < scene_state_.robots.size(); ++robot_index) {
        PhysicsRobotState& robot_state = scene_state_.robots[robot_index];
        const std::string prefix = GetRobotPrefix(robot_index);
        if (prefix.empty()) {
            LOG_WARN("MuJoCo has no loaded model prefix for Gobot robot '{}'. Its links will not be synchronized.",
                     robot_state.name);
            continue;
        }

        for (std::size_t link_index = 0; link_index < robot_state.links.size(); ++link_index) {
            PhysicsLinkState& link_state = robot_state.links[link_index];
            const PhysicsLinkSnapshot* link_snapshot = robot_index < scene_snapshot_.robots.size()
                                                               ? FindLinkSnapshot(scene_snapshot_.robots[robot_index],
                                                                                  link_state.link_name)
                                                               : nullptr;
            if (link_snapshot != nullptr && link_snapshot->role == PhysicsLinkRole::VirtualRoot) {
                continue;
            }

            const std::string body_name = prefix + link_state.link_name;
            const int body_id = mj_name2id(model, mjOBJ_BODY, body_name.c_str());
            if (body_id < 0) {
                LOG_WARN("MuJoCo model does not contain body '{}' for Gobot link '{}::{}'. It will not be synchronized.",
                         body_name,
                         robot_state.name,
                         link_state.link_name);
                continue;
            }

            MuJoCoLinkBinding binding;
            binding.robot_index = robot_index;
            binding.link_index = link_index;
            binding.body_id = body_id;
            link_bindings_.emplace_back(binding);
        }
    }

    LOG_INFO("MuJoCo link bindings built: {} of {} Gobot links.",
             link_bindings_.size(),
             scene_state_.total_link_count);
}

void MuJoCoPhysicsWorld::BuildJointBindings() {
    joint_bindings_.clear();

    auto* model = static_cast<mjModel*>(model_);
    if (!model) {
        return;
    }

    for (std::size_t robot_index = 0; robot_index < scene_state_.robots.size(); ++robot_index) {
        PhysicsRobotState& robot_state = scene_state_.robots[robot_index];
        const std::string prefix = GetRobotPrefix(robot_index);
        if (prefix.empty()) {
            LOG_WARN("MuJoCo has no loaded model prefix for Gobot robot '{}'. Its joints will not be synchronized.",
                     robot_state.name);
            continue;
        }

        for (std::size_t joint_index = 0; joint_index < robot_state.joints.size(); ++joint_index) {
            PhysicsJointState& joint_state = robot_state.joints[joint_index];
            const std::string joint_name = prefix + joint_state.joint_name;
            const int joint_id = mj_name2id(model, mjOBJ_JOINT, joint_name.c_str());
            if (joint_id < 0) {
                LOG_WARN("MuJoCo model does not contain joint '{}' for Gobot joint '{}::{}'. It will not be synchronized.",
                         joint_name,
                         robot_state.name,
                         joint_state.joint_name);
                continue;
            }

            MuJoCoJointBinding binding;
            binding.robot_index = robot_index;
            binding.joint_index = joint_index;
            binding.mujoco_joint_id = joint_id;
            const MuJoCoJointActuatorIds actuator_ids = FindActuatorsForJoint(model, joint_id, joint_name);
            binding.motor_actuator_id = actuator_ids.motor;
            binding.position_actuator_id = actuator_ids.position;
            binding.velocity_actuator_id = actuator_ids.velocity;
            if (binding.position_actuator_id >= 0) {
                binding.position_stiffness =
                        std::abs(model->actuator_gainprm[mjNGAIN * binding.position_actuator_id + 0]);
            }
            if (binding.velocity_actuator_id >= 0) {
                binding.velocity_damping =
                        std::abs(model->actuator_gainprm[mjNGAIN * binding.velocity_actuator_id + 0]);
            }
            binding.qpos_address = model->jnt_qposadr[joint_id];
            binding.dof_address = model->jnt_dofadr[joint_id];
            binding.joint_type = model->jnt_type[joint_id];
            if (binding.dof_address >= 0 && binding.dof_address < model->nv) {
                binding.passive_damping = static_cast<RealType>(model->dof_damping[binding.dof_address]);
            }
            binding.controller.SetGains(settings_.default_joint_gains);
            joint_bindings_.emplace_back(binding);
        }
    }

    LOG_INFO("MuJoCo joint bindings built: {} of {} Gobot joints.",
             joint_bindings_.size(),
             scene_state_.total_joint_count);
}

void MuJoCoPhysicsWorld::BuildSensorBindings() {
    sensor_bindings_.clear();

    auto* model = static_cast<mjModel*>(model_);
    if (!model) {
        return;
    }

    for (std::size_t robot_index = 0; robot_index < scene_state_.robots.size(); ++robot_index) {
        PhysicsRobotState& robot_state = scene_state_.robots[robot_index];
        const std::string prefix = GetRobotPrefix(robot_index);
        if (prefix.empty()) {
            LOG_WARN("MuJoCo has no loaded model prefix for Gobot robot '{}'. Its sensors will not be synchronized.",
                     robot_state.name);
            continue;
        }

        if (robot_index >= scene_snapshot_.robots.size()) {
            continue;
        }

        const PhysicsRobotSnapshot& robot_snapshot = scene_snapshot_.robots[robot_index];
        for (std::size_t sensor_index = 0; sensor_index < robot_state.sensors.size(); ++sensor_index) {
            if (sensor_index >= robot_snapshot.sensors.size()) {
                continue;
            }

            const PhysicsSensorSnapshot& sensor_snapshot = robot_snapshot.sensors[sensor_index];
            if (!sensor_snapshot.enabled) {
                continue;
            }

            MuJoCoSensorBinding binding;
            binding.robot_index = robot_index;
            binding.sensor_index = sensor_index;

            auto add_component = [&](std::string_view component, std::size_t value_offset) {
                const std::string component_name = SensorComponentName(prefix, sensor_snapshot, component);
                const int sensor_id = mj_name2id(model, mjOBJ_SENSOR, component_name.c_str());
                if (sensor_id < 0) {
                    LOG_WARN("MuJoCo model does not contain sensor '{}' for Gobot sensor '{}::{}::{}'.",
                             component_name,
                             robot_state.name,
                             sensor_snapshot.link_name,
                             sensor_snapshot.name);
                    return;
                }

                binding.components.push_back({sensor_id, value_offset});
            };

            switch (sensor_snapshot.type) {
                case PhysicsSensorType::IMU:
                    add_component("orientation", 0);
                    add_component("angular_velocity", 4);
                    add_component("linear_velocity", 7);
                    add_component("linear_acceleration", 10);
                    break;
                case PhysicsSensorType::AngularMomentum:
                    add_component("angular_momentum", 0);
                    break;
                case PhysicsSensorType::Contact:
                    add_component("contact", 0);
                    break;
                case PhysicsSensorType::Unknown:
                    break;
            }

            if (!binding.components.empty()) {
                sensor_bindings_.emplace_back(std::move(binding));
            }
        }
    }

    LOG_INFO("MuJoCo sensor bindings built: {} of {} Gobot sensors.",
             sensor_bindings_.size(),
             scene_state_.total_sensor_count);
}

std::string MuJoCoPhysicsWorld::GetRobotPrefix(std::size_t robot_index) const {
    for (const MuJoCoRobotBinding& binding : robot_bindings_) {
        if (binding.robot_index == robot_index) {
            return binding.prefix;
        }
    }

    return {};
}

bool MuJoCoPhysicsWorld::IsEnvironmentIndexValid(std::size_t environment_index) const {
    return environment_index < environment_data_.size() &&
           environment_index < environment_states_.size() &&
           environment_data_[environment_index] != nullptr;
}

PhysicsSceneState& MuJoCoPhysicsWorld::EnvironmentState(std::size_t environment_index) {
    return environment_index == 0 ? scene_state_ : environment_states_[environment_index];
}

const PhysicsSceneState& MuJoCoPhysicsWorld::EnvironmentState(std::size_t environment_index) const {
    return environment_index == 0 ? scene_state_ : environment_states_[environment_index];
}

void MuJoCoPhysicsWorld::ApplyControlsToMuJoCo(std::size_t environment_index) {
    auto* model = static_cast<mjModel*>(model_);
    auto* data = IsEnvironmentIndexValid(environment_index)
                         ? static_cast<mjData*>(environment_data_[environment_index])
                         : nullptr;
    if (!model || !data) {
        return;
    }

    PhysicsSceneState& state = EnvironmentState(environment_index);
    for (int velocity_index = 0; velocity_index < model->nv; ++velocity_index) {
        data->qfrc_applied[velocity_index] = 0.0;
    }

    for (MuJoCoJointBinding& binding : joint_bindings_) {
        if (binding.robot_index >= state.robots.size()) {
            continue;
        }

        const PhysicsRobotState& robot_state = state.robots[binding.robot_index];
        if (binding.joint_index >= robot_state.joints.size()) {
            continue;
        }

        const PhysicsJointState& joint_state = robot_state.joints[binding.joint_index];
        if (binding.joint_type == mjJNT_FREE || binding.joint_type == mjJNT_BALL) {
            continue;
        }

        if (binding.dof_address < 0 || binding.dof_address >= model->nv) {
            continue;
        }

        binding.controller.SetGains(settings_.default_joint_gains);
        DisableActuator(model, data, binding.motor_actuator_id);
        DisableActuator(model, data, binding.position_actuator_id);
        DisableActuator(model, data, binding.velocity_actuator_id);

        JointControllerLimits limits;
        if (binding.robot_index < scene_snapshot_.robots.size() &&
            binding.joint_index < scene_snapshot_.robots[binding.robot_index].joints.size()) {
            limits = MakeJointControllerLimits(scene_snapshot_.robots[binding.robot_index].joints[binding.joint_index]);
        }

        const bool has_native_actuator =
                binding.motor_actuator_id >= 0 ||
                binding.position_actuator_id >= 0 ||
                binding.velocity_actuator_id >= 0;

        if (has_native_actuator) {
            switch (joint_state.control_mode) {
                case PhysicsJointControlMode::Passive:
                    break;
                case PhysicsJointControlMode::Effort:
                    if (binding.motor_actuator_id >= 0) {
                        SetMotorActuator(model, data, binding.motor_actuator_id, joint_state.target_effort);
                    } else {
                        data->qfrc_applied[binding.dof_address] = JointController::ClampEffort(
                                joint_state.target_effort,
                                limits.effort_limit);
                    }
                    break;
                case PhysicsJointControlMode::Position:
                    if (binding.position_actuator_id >= 0) {
                        const RealType stiffness = binding.position_stiffness > 0.0
                                                           ? binding.position_stiffness
                                                           : settings_.default_joint_gains.position_stiffness;
                        SetPositionActuator(model,
                                            data,
                                            binding.position_actuator_id,
                                            JointController::ClampTargetPosition(joint_state.target_position, limits),
                                            stiffness);
                        if (binding.velocity_actuator_id >= 0 && settings_.default_joint_gains.velocity_damping > 0.0) {
                            const RealType damping = binding.velocity_damping > 0.0
                                                             ? binding.velocity_damping
                                                             : settings_.default_joint_gains.velocity_damping;
                            SetVelocityActuator(model,
                                                data,
                                                binding.velocity_actuator_id,
                                                joint_state.target_velocity,
                                                damping);
                        } else if (binding.passive_damping <= 0.0 &&
                                   settings_.default_joint_gains.velocity_damping > 0.0) {
                            data->qfrc_applied[binding.dof_address] -=
                                    settings_.default_joint_gains.velocity_damping * joint_state.velocity;
                        }
                    } else {
                        const RealType effort = binding.controller.ComputeEffort(MakeJointControllerState(joint_state),
                                                                                 MakeJointControllerCommand(joint_state),
                                                                                 limits,
                                                                                 settings_.fixed_time_step);
                        if (binding.motor_actuator_id >= 0) {
                            SetMotorActuator(model, data, binding.motor_actuator_id, effort);
                        } else {
                            data->qfrc_applied[binding.dof_address] = effort;
                        }
                    }
                    break;
                case PhysicsJointControlMode::Velocity:
                    if (binding.velocity_actuator_id >= 0) {
                        const RealType damping = binding.velocity_damping > 0.0
                                                         ? binding.velocity_damping
                                                         : settings_.default_joint_gains.velocity_damping;
                        SetVelocityActuator(model,
                                            data,
                                            binding.velocity_actuator_id,
                                            joint_state.target_velocity,
                                            damping);
                    } else {
                        const RealType effort = binding.controller.ComputeEffort(MakeJointControllerState(joint_state),
                                                                                 MakeJointControllerCommand(joint_state),
                                                                                 limits,
                                                                                 settings_.fixed_time_step);
                        if (binding.motor_actuator_id >= 0) {
                            SetMotorActuator(model, data, binding.motor_actuator_id, effort);
                        } else {
                            data->qfrc_applied[binding.dof_address] = effort;
                        }
                    }
                    break;
            }
            continue;
        }

        const RealType effort = binding.controller.ComputeEffort(MakeJointControllerState(joint_state),
                                                                 MakeJointControllerCommand(joint_state),
                                                                 limits,
                                                                 settings_.fixed_time_step);
        data->qfrc_applied[binding.dof_address] = effort;
    }
}

void MuJoCoPhysicsWorld::ApplyExternalForcesToMuJoCo(std::size_t environment_index) {
    auto* model = static_cast<mjModel*>(model_);
    auto* data = IsEnvironmentIndexValid(environment_index)
                         ? static_cast<mjData*>(environment_data_[environment_index])
                         : nullptr;
    if (model == nullptr || data == nullptr || model->nbody <= 0) {
        return;
    }

    const PhysicsSceneState& state = EnvironmentState(environment_index);
    mju_zero(data->xfrc_applied, 6 * model->nbody);
    if (external_forces_.empty()) {
        return;
    }

    for (const PhysicsExternalForce& external_force : external_forces_) {
        auto binding_iter = std::find_if(link_bindings_.begin(),
                                         link_bindings_.end(),
                                         [&](const MuJoCoLinkBinding& binding) {
                                             if (binding.robot_index >= state.robots.size()) {
                                                 return false;
                                             }
                                             const PhysicsRobotState& robot = state.robots[binding.robot_index];
                                             if (binding.link_index >= robot.links.size()) {
                                                 return false;
                                             }
                                             return robot.name == external_force.robot_name &&
                                                    robot.links[binding.link_index].link_name == external_force.link_name;
                                         });
        if (binding_iter == link_bindings_.end() ||
            binding_iter->body_id <= 0 ||
            binding_iter->body_id >= model->nbody) {
            continue;
        }

        const int body_id = binding_iter->body_id;
        const Vector3 body_position{static_cast<RealType>(data->xpos[3 * body_id + 0]),
                                    static_cast<RealType>(data->xpos[3 * body_id + 1]),
                                    static_cast<RealType>(data->xpos[3 * body_id + 2])};
        Matrix3 body_rotation;
        body_rotation << data->xmat[9 * body_id + 0],
                         data->xmat[9 * body_id + 1],
                         data->xmat[9 * body_id + 2],
                         data->xmat[9 * body_id + 3],
                         data->xmat[9 * body_id + 4],
                         data->xmat[9 * body_id + 5],
                         data->xmat[9 * body_id + 6],
                         data->xmat[9 * body_id + 7],
                         data->xmat[9 * body_id + 8];
        const Vector3 body_com{static_cast<RealType>(data->xipos[3 * body_id + 0]),
                               static_cast<RealType>(data->xipos[3 * body_id + 1]),
                               static_cast<RealType>(data->xipos[3 * body_id + 2])};
        const Vector3 world_point = external_force.use_spring
                ? body_position + body_rotation * external_force.local_point
                : external_force.point;

        Vector3 force = external_force.force;
        if (external_force.use_spring) {
            const Vector3 moment_arm = world_point - body_com;
            const Vector3 link_velocity{static_cast<RealType>(data->cvel[6 * body_id + 3]),
                                        static_cast<RealType>(data->cvel[6 * body_id + 4]),
                                        static_cast<RealType>(data->cvel[6 * body_id + 5])};
            const Vector3 link_angular_velocity{static_cast<RealType>(data->cvel[6 * body_id + 0]),
                                                static_cast<RealType>(data->cvel[6 * body_id + 1]),
                                                static_cast<RealType>(data->cvel[6 * body_id + 2])};
            const Vector3 point_velocity = link_velocity + link_angular_velocity.cross(moment_arm);
            const RealType mass = model->body_mass[body_id] > mjMINVAL
                    ? static_cast<RealType>(model->body_mass[body_id])
                    : 1.0;
            constexpr RealType stiffness = 100.0;
            constexpr RealType damping = 10.0;
            force = (external_force.target_point - world_point) * (stiffness * mass) -
                    point_velocity * (damping * mass);
            if (external_force.force.squaredNorm() > CMP_EPSILON2) {
                const Vector3 axis = external_force.force.normalized();
                force = axis * force.dot(axis);
            }
        }

        const Vector3 torque = (world_point - body_com).cross(force);
        data->xfrc_applied[6 * body_id + 0] += force.x();
        data->xfrc_applied[6 * body_id + 1] += force.y();
        data->xfrc_applied[6 * body_id + 2] += force.z();
        data->xfrc_applied[6 * body_id + 3] += torque.x();
        data->xfrc_applied[6 * body_id + 4] += torque.y();
        data->xfrc_applied[6 * body_id + 5] += torque.z();
    }
}

void MuJoCoPhysicsWorld::FreeModel() {
    sensor_bindings_.clear();
    link_bindings_.clear();
    joint_bindings_.clear();

    for (void* environment_data : environment_data_) {
        if (environment_data != nullptr) {
            mj_deleteData(static_cast<mjData*>(environment_data));
        }
    }
    environment_data_.clear();
    environment_states_.clear();
    data_ = nullptr;

    if (model_) {
        mj_deleteModel(static_cast<mjModel*>(model_));
        model_ = nullptr;
    }
}

void MuJoCoPhysicsWorld::SyncStateFromMuJoCo(std::size_t environment_index) {
    auto* model = static_cast<mjModel*>(model_);
    auto* data = IsEnvironmentIndexValid(environment_index)
                         ? static_cast<mjData*>(environment_data_[environment_index])
                         : nullptr;
    if (!model || !data) {
        return;
    }

    PhysicsSceneState& state = EnvironmentState(environment_index);
    for (const MuJoCoLinkBinding& binding : link_bindings_) {
        if (binding.robot_index >= state.robots.size()) {
            continue;
        }

        PhysicsRobotState& robot_state = state.robots[binding.robot_index];
        if (binding.link_index >= robot_state.links.size()) {
            continue;
        }

        if (binding.body_id < 0 || binding.body_id >= model->nbody) {
            continue;
        }

        PhysicsLinkState& link_state = robot_state.links[binding.link_index];
        Affine3 transform = Affine3::Identity();
        transform.translation() = Vector3(data->xpos[3 * binding.body_id + 0],
                                          data->xpos[3 * binding.body_id + 1],
                                          data->xpos[3 * binding.body_id + 2]);
        Matrix3 rotation;
        rotation << data->xmat[9 * binding.body_id + 0],
                    data->xmat[9 * binding.body_id + 1],
                    data->xmat[9 * binding.body_id + 2],
                    data->xmat[9 * binding.body_id + 3],
                    data->xmat[9 * binding.body_id + 4],
                    data->xmat[9 * binding.body_id + 5],
                    data->xmat[9 * binding.body_id + 6],
                    data->xmat[9 * binding.body_id + 7],
                    data->xmat[9 * binding.body_id + 8];
        transform.linear() = rotation;
        link_state.global_transform = transform;
        link_state.angular_velocity = Vector3(data->cvel[6 * binding.body_id + 0],
                                              data->cvel[6 * binding.body_id + 1],
                                              data->cvel[6 * binding.body_id + 2]);
        link_state.linear_velocity = Vector3(data->cvel[6 * binding.body_id + 3],
                                             data->cvel[6 * binding.body_id + 4],
                                             data->cvel[6 * binding.body_id + 5]);
    }

    for (const MuJoCoJointBinding& binding : joint_bindings_) {
        if (binding.robot_index >= state.robots.size()) {
            continue;
        }

        PhysicsRobotState& robot_state = state.robots[binding.robot_index];
        if (binding.joint_index >= robot_state.joints.size()) {
            continue;
        }

        PhysicsJointState& joint_state = robot_state.joints[binding.joint_index];
        if (binding.joint_type == mjJNT_FREE) {
            if (binding.qpos_address >= 0 && binding.qpos_address + 6 < model->nq) {
                joint_state.position = static_cast<RealType>(data->qpos[binding.qpos_address + 2]);
            }
            if (binding.dof_address >= 0 && binding.dof_address + 5 < model->nv) {
                joint_state.velocity = Vector3(data->qvel[binding.dof_address + 0],
                                               data->qvel[binding.dof_address + 1],
                                               data->qvel[binding.dof_address + 2]).norm();
            }
            if (binding.robot_index < scene_snapshot_.robots.size() &&
                binding.joint_index < scene_snapshot_.robots[binding.robot_index].joints.size()) {
                const PhysicsJointSnapshot& joint_snapshot =
                        scene_snapshot_.robots[binding.robot_index].joints[binding.joint_index];
                for (PhysicsLinkState& link_state : robot_state.links) {
                    if (link_state.link_name != joint_snapshot.child_link) {
                        continue;
                    }
                    if (binding.dof_address >= 0 && binding.dof_address + 5 < model->nv) {
                        link_state.linear_velocity = Vector3(data->qvel[binding.dof_address + 0],
                                                             data->qvel[binding.dof_address + 1],
                                                             data->qvel[binding.dof_address + 2]);
                        link_state.angular_velocity = Vector3(data->qvel[binding.dof_address + 3],
                                                              data->qvel[binding.dof_address + 4],
                                                              data->qvel[binding.dof_address + 5]);
                    }
                    break;
                }
            }
            continue;
        }
        if (binding.joint_type == mjJNT_BALL) {
            continue;
        }

        if (binding.qpos_address >= 0 && binding.qpos_address < model->nq) {
            joint_state.position = static_cast<RealType>(data->qpos[binding.qpos_address]);
        }
        if (binding.dof_address >= 0 && binding.dof_address < model->nv) {
            joint_state.velocity = static_cast<RealType>(data->qvel[binding.dof_address]);
        }
    }

    SyncContactsFromMuJoCo(environment_index);
    SyncSensorsFromMuJoCo(environment_index);
    if (environment_index == 0 && !environment_states_.empty()) {
        environment_states_[0] = scene_state_;
    }
}

void MuJoCoPhysicsWorld::SyncStateToMuJoCo(std::size_t environment_index) {
    auto* model = static_cast<mjModel*>(model_);
    auto* data = IsEnvironmentIndexValid(environment_index)
                         ? static_cast<mjData*>(environment_data_[environment_index])
                         : nullptr;
    if (!model || !data) {
        return;
    }

    const PhysicsSceneState& state = EnvironmentState(environment_index);
    for (const MuJoCoJointBinding& binding : joint_bindings_) {
        if (binding.robot_index >= state.robots.size()) {
            continue;
        }

        const PhysicsRobotState& robot_state = state.robots[binding.robot_index];
        if (binding.joint_index >= robot_state.joints.size()) {
            continue;
        }

        const PhysicsJointState& joint_state = robot_state.joints[binding.joint_index];
        if (binding.joint_type == mjJNT_FREE) {
            const PhysicsJointSnapshot* joint_snapshot = nullptr;
            const PhysicsLinkSnapshot* child_link_snapshot = nullptr;
            const PhysicsLinkState* child_link_state = nullptr;
            if (binding.robot_index < scene_snapshot_.robots.size() &&
                binding.joint_index < scene_snapshot_.robots[binding.robot_index].joints.size()) {
                joint_snapshot = &scene_snapshot_.robots[binding.robot_index].joints[binding.joint_index];
                child_link_snapshot = FindLinkSnapshot(scene_snapshot_.robots[binding.robot_index],
                                                       joint_snapshot->child_link);
                for (const PhysicsLinkState& link_state : robot_state.links) {
                    if (link_state.link_name == joint_snapshot->child_link) {
                        child_link_state = &link_state;
                        break;
                    }
                }
            }

            if (joint_snapshot != nullptr && binding.qpos_address >= 0 && binding.qpos_address + 6 < model->nq) {
                const Affine3 initial_transform = child_link_state != nullptr
                                                          ? child_link_state->global_transform
                                                          : (child_link_snapshot != nullptr
                                                                     ? child_link_snapshot->global_transform
                                                                     : joint_snapshot->global_transform);
                const Vector3 position = initial_transform.translation();
                const Quaternion orientation(initial_transform.linear());
                data->qpos[binding.qpos_address + 0] = position.x();
                data->qpos[binding.qpos_address + 1] = position.y();
                data->qpos[binding.qpos_address + 2] = position.z();
                data->qpos[binding.qpos_address + 3] = orientation.w();
                data->qpos[binding.qpos_address + 4] = orientation.x();
                data->qpos[binding.qpos_address + 5] = orientation.y();
                data->qpos[binding.qpos_address + 6] = orientation.z();
            }
            if (child_link_state != nullptr && binding.dof_address >= 0 && binding.dof_address + 5 < model->nv) {
                data->qvel[binding.dof_address + 0] = child_link_state->linear_velocity.x();
                data->qvel[binding.dof_address + 1] = child_link_state->linear_velocity.y();
                data->qvel[binding.dof_address + 2] = child_link_state->linear_velocity.z();
                data->qvel[binding.dof_address + 3] = child_link_state->angular_velocity.x();
                data->qvel[binding.dof_address + 4] = child_link_state->angular_velocity.y();
                data->qvel[binding.dof_address + 5] = child_link_state->angular_velocity.z();
            }
            continue;
        }
        if (binding.joint_type == mjJNT_BALL) {
            continue;
        }

        if (binding.qpos_address >= 0 && binding.qpos_address < model->nq) {
            data->qpos[binding.qpos_address] = joint_state.position;
        }
        if (binding.dof_address >= 0 && binding.dof_address < model->nv) {
            data->qvel[binding.dof_address] = joint_state.velocity;
        }
    }

    mj_forward(model, data);
}

void MuJoCoPhysicsWorld::SyncContactsFromMuJoCo(std::size_t environment_index) {
    auto* model = static_cast<mjModel*>(model_);
    auto* data = IsEnvironmentIndexValid(environment_index)
                         ? static_cast<mjData*>(environment_data_[environment_index])
                         : nullptr;
    PhysicsSceneState& state = EnvironmentState(environment_index);
    state.contacts.clear();
    if (!model || !data) {
        return;
    }

    auto find_link_binding_for_body = [this](int body_id) -> const MuJoCoLinkBinding* {
        for (const MuJoCoLinkBinding& binding : link_bindings_) {
            if (binding.body_id == body_id) {
                return &binding;
            }
        }

        return nullptr;
    };

    for (int contact_index = 0; contact_index < data->ncon; ++contact_index) {
        const mjContact& contact = data->contact[contact_index];
        const int geom_id_a = contact.geom[0];
        const int geom_id_b = contact.geom[1];
        if (geom_id_a < 0 || geom_id_a >= model->ngeom || geom_id_b < 0 || geom_id_b >= model->ngeom) {
            continue;
        }

        const int body_id_a = model->geom_bodyid[geom_id_a];
        const int body_id_b = model->geom_bodyid[geom_id_b];
        const MuJoCoLinkBinding* binding_a = find_link_binding_for_body(body_id_a);
        const MuJoCoLinkBinding* binding_b = find_link_binding_for_body(body_id_b);
        if (binding_a == nullptr && binding_b == nullptr) {
            continue;
        }

        auto add_contact = [&](const MuJoCoLinkBinding& link_binding,
                               const MuJoCoLinkBinding* other_binding,
                               RealType normal_sign) {
            if (link_binding.robot_index >= state.robots.size()) {
                return;
            }

            const PhysicsRobotState& robot_state = state.robots[link_binding.robot_index];
            if (link_binding.link_index >= robot_state.links.size()) {
                return;
            }

            PhysicsContactState contact_state;
            contact_state.robot_name = robot_state.name;
            contact_state.link_name = robot_state.links[link_binding.link_index].link_name;
            if (other_binding != nullptr &&
                other_binding->robot_index < state.robots.size() &&
                other_binding->link_index < state.robots[other_binding->robot_index].links.size()) {
                const PhysicsRobotState& other_robot_state = state.robots[other_binding->robot_index];
                contact_state.other_robot_name = other_robot_state.name;
                contact_state.other_link_name = other_robot_state.links[other_binding->link_index].link_name;
            }
            contact_state.position = Vector3(contact.pos[0], contact.pos[1], contact.pos[2]);
            contact_state.normal = normal_sign * Vector3(contact.frame[0], contact.frame[1], contact.frame[2]);
            contact_state.distance = contact.dist;
            state.contacts.emplace_back(std::move(contact_state));
        };

        if (binding_a != nullptr) {
            add_contact(*binding_a, binding_b, 1.0);
        }
        if (binding_b != nullptr) {
            add_contact(*binding_b, binding_a, -1.0);
        }
    }
}

void MuJoCoPhysicsWorld::SyncSensorsFromMuJoCo(std::size_t environment_index) {
    auto* model = static_cast<mjModel*>(model_);
    auto* data = IsEnvironmentIndexValid(environment_index)
                         ? static_cast<mjData*>(environment_data_[environment_index])
                         : nullptr;
    PhysicsSceneState& state = EnvironmentState(environment_index);
    if (!model || !data) {
        return;
    }

    const RealType timestamp = static_cast<RealType>(data->time);
    for (const MuJoCoSensorBinding& binding : sensor_bindings_) {
        if (binding.robot_index >= state.robots.size()) {
            continue;
        }

        PhysicsRobotState& robot_state = state.robots[binding.robot_index];
        if (binding.sensor_index >= robot_state.sensors.size()) {
            continue;
        }

        PhysicsSensorState& sensor_state = robot_state.sensors[binding.sensor_index];
        if (!sensor_state.enabled) {
            continue;
        }

        for (const MuJoCoSensorComponentBinding& component : binding.components) {
            const int sensor_id = component.sensor_id;
            if (sensor_id < 0 || sensor_id >= model->nsensor) {
                continue;
            }

            const int address = model->sensor_adr[sensor_id];
            const int dimension = model->sensor_dim[sensor_id];
            if (address < 0 || dimension < 0 || address + dimension > model->nsensordata) {
                continue;
            }

            const std::size_t value_offset = component.value_offset;
            const std::size_t writable_count = sensor_state.values.size() > value_offset
                                                       ? sensor_state.values.size() - value_offset
                                                       : 0;
            const std::size_t count = std::min<std::size_t>(static_cast<std::size_t>(dimension), writable_count);
            for (std::size_t value_index = 0; value_index < count; ++value_index) {
                sensor_state.values[value_offset + value_index] =
                        static_cast<RealType>(data->sensordata[address + static_cast<int>(value_index)]);
            }
        }

        sensor_state.timestamp = timestamp;
    }
}
#endif

} // namespace gobot

GOBOT_REGISTRATION {

    Class_<MuJoCoPhysicsWorld>("MuJoCoPhysicsWorld")
            .constructor()(CtorAsRawPtr)
            .method("is_backend_available", &MuJoCoPhysicsWorld::IsBackendAvailable);

    gobot::Type::register_wrapper_converter_for_base_classes<Ref<MuJoCoPhysicsWorld>, Ref<PhysicsWorld>>();

};
