/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/core/io/resource_format_mjcf.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
#include <regex>
#include <sstream>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "gobot/core/config/project_setting.hpp"
#include "gobot/core/io/resource_loader.hpp"
#include "gobot/core/registration.hpp"
#include "gobot/core/string_utils.hpp"
#include "gobot/log.hpp"
#include "gobot/rendering/render_server.hpp"
#include "gobot/scene/collision_shape_3d.hpp"
#include "gobot/scene/joint_3d.hpp"
#include "gobot/scene/link_3d.hpp"
#include "gobot/scene/mesh_instance_3d.hpp"
#include "gobot/scene/resources/array_mesh.hpp"
#include "gobot/scene/resources/box_shape_3d.hpp"
#include "gobot/scene/resources/capsule_shape_3d.hpp"
#include "gobot/scene/resources/cylinder_shape_3d.hpp"
#include "gobot/scene/resources/material.hpp"
#include "gobot/scene/resources/mesh.hpp"
#include "gobot/scene/resources/packed_scene.hpp"
#include "gobot/scene/resources/sphere_shape_3d.hpp"
#include "gobot/scene/robot_3d.hpp"
#include "gobot/scene/sensor_3d.hpp"

#ifdef GOBOT_HAS_MUJOCO
#include <mujoco/mujoco.h>
#endif

namespace gobot {
namespace {

constexpr int kMuJoCoErrorBufferSize = 1024;

std::string ResolveInputPath(const std::string& path) {
    if (path.starts_with("res://")) {
        ProjectSettings* settings = ProjectSettings::s_singleton;
        return settings != nullptr ? settings->GlobalizePath(path) : path.substr(6);
    }
    return path;
}

std::string ReadTextFile(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        LOG_ERROR("Cannot open MJCF file: {}.", path);
        return {};
    }

    std::ostringstream contents;
    contents << file.rdbuf();
    return contents.str();
}

std::string GetAttribute(const std::string& attributes, const std::string& name) {
    const std::regex attr_regex(name + R"(\s*=\s*["']([^"']*)["'])", std::regex::icase);
    std::smatch match;
    if (!std::regex_search(attributes, match, attr_regex)) {
        return {};
    }
    return match[1].str();
}

bool LooksLikeMJCF(const std::string& xml) {
    return std::regex_search(xml, std::regex(R"(<\s*mujoco\b)", std::regex::icase));
}

std::string ReadMJCFModelName(const std::string& xml) {
    std::smatch match;
    if (!std::regex_search(xml, match, std::regex(R"(<\s*mujoco\b([^>]*)>)", std::regex::icase))) {
        return {};
    }
    return GetAttribute(match[1].str(), "model");
}

template <typename T>
void AddProperty(SceneState::NodeData& node_data, std::string name, T value) {
    node_data.properties.push_back({std::move(name), Variant(std::move(value))});
}

template <typename T>
void SetProperty(SceneState::NodeData& node_data, const std::string& name, T value) {
    for (SceneState::PropertyData& property : node_data.properties) {
        if (property.name == name) {
            property.value = Variant(std::move(value));
            return;
        }
    }
    AddProperty(node_data, name, std::move(value));
}

void AddTransformProperties(SceneState::NodeData& node_data, const Vector3& position, const Matrix3& rotation) {
    Affine3 transform = Affine3::Identity();
    transform.translation() = position;
    transform.linear() = rotation;
    AddProperty(node_data, "position", position);
    const Vector3 euler = transform.GetEulerAngle(EulerOrder::RXYZ);
    AddProperty(node_data, "rotation_degrees", Vector3{
            RAD_TO_DEG(euler.x()),
            RAD_TO_DEG(euler.y()),
            RAD_TO_DEG(euler.z())});
}

#ifdef GOBOT_HAS_MUJOCO
Affine3 MakeTransform(const Vector3& position, const Matrix3& rotation) {
    Affine3 transform = Affine3::Identity();
    transform.translation() = position;
    transform.linear() = rotation;
    return transform;
}

std::string GetMuJoCoName(const mjModel* model, int object_type, int object_id, const std::string& fallback) {
    const char* name = mj_id2name(model, object_type, object_id);
    if (name != nullptr && name[0] != '\0') {
        return name;
    }
    return fallback;
}

Vector3 ToVector3(const mjtNum* values) {
    return {
            static_cast<RealType>(values[0]),
            static_cast<RealType>(values[1]),
            static_cast<RealType>(values[2])};
}

Matrix3 ToMatrix3FromMuJoCoQuat(const mjtNum* quat) {
    mjtNum matrix[9] = {};
    mju_quat2Mat(matrix, quat);

    Matrix3 rotation;
    rotation << matrix[0], matrix[1], matrix[2],
                matrix[3], matrix[4], matrix[5],
                matrix[6], matrix[7], matrix[8];
    return rotation;
}

Affine3 GetBodyWorldTransform(const mjData* data, int body_id) {
    return MakeTransform(ToVector3(data->xpos + 3 * body_id),
                         ToMatrix3FromMuJoCoQuat(data->xquat + 4 * body_id));
}

Affine3 GetJointWorldTransform(const mjModel* model, const mjData* data, int joint_id, int child_body_id) {
    if (model->jnt_type[joint_id] == mjJNT_FREE) {
        const int qpos_address = model->jnt_qposadr[joint_id];
        if (qpos_address >= 0 && qpos_address + 6 < model->nq) {
            return MakeTransform(ToVector3(data->qpos + qpos_address),
                                 ToMatrix3FromMuJoCoQuat(data->qpos + qpos_address + 3));
        }
    }

    const Affine3 body_world_transform = GetBodyWorldTransform(data, child_body_id);
    Affine3 joint_local_transform = Affine3::Identity();
    joint_local_transform.translation() = ToVector3(model->jnt_pos + 3 * joint_id);
    return body_world_transform * joint_local_transform;
}

void InvertFrameAccumulation(mjtNum pos[3], mjtNum quat[4], const mjtNum child_pos[3], const mjtNum child_quat[4]) {
    mjtNum child_quat_inv[4] = {};
    mjtNum corrected_quat[4] = {};
    mju_negQuat(child_quat_inv, child_quat);
    mju_mulQuat(corrected_quat, quat, child_quat_inv);
    mju_copy4(quat, corrected_quat);

    mjtNum rotation[9] = {};
    mjtNum rotated_child_pos[3] = {};
    mju_quat2Mat(rotation, quat);
    mju_mulMatVec3(rotated_child_pos, rotation, child_pos);
    mju_subFrom3(pos, rotated_child_pos);
}

Color ToColor(const float* rgba) {
    return {
            static_cast<float>(rgba[0]),
            static_cast<float>(rgba[1]),
            static_cast<float>(rgba[2]),
            static_cast<float>(rgba[3])};
}

Color GetGeomColor(const mjModel* model, int geom_id) {
    const int material_id = model->geom_matid[geom_id];
    if (material_id >= 0 && material_id < model->nmat) {
        return ToColor(model->mat_rgba + 4 * material_id);
    }
    return ToColor(model->geom_rgba + 4 * geom_id);
}

bool IsActuatorForJoint(const mjModel* model, int actuator_id, int joint_id) {
    const int transmission_type = model->actuator_trntype[actuator_id];
    const int transmission_id = model->actuator_trnid[2 * actuator_id];
    return (transmission_type == mjTRN_JOINT || transmission_type == mjTRN_JOINTINPARENT) &&
           transmission_id == joint_id;
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
           std::abs(bias[1]) > 1.0e-9 &&
           std::abs(gain[0] + bias[1]) <= std::max<mjtNum>(1.0, std::abs(gain[0])) * 1.0e-6;
}

bool LooksLikeVelocityActuator(const mjModel* model, int actuator_id) {
    const mjtNum* gain = model->actuator_gainprm + mjNGAIN * actuator_id;
    const mjtNum* bias = model->actuator_biasprm + mjNBIAS * actuator_id;
    return model->actuator_gaintype[actuator_id] == mjGAIN_FIXED &&
           model->actuator_biastype[actuator_id] == mjBIAS_AFFINE &&
           std::abs(bias[1]) <= 1.0e-9 &&
           std::abs(bias[2]) > 1.0e-9 &&
           std::abs(gain[0] + bias[2]) <= std::max<mjtNum>(1.0, std::abs(gain[0])) * 1.0e-6;
}

std::vector<RealType> ToRealVector(const mjtNum* values, int count) {
    std::vector<RealType> result;
    result.reserve(count);
    for (int index = 0; index < count; ++index) {
        result.push_back(static_cast<RealType>(values[index]));
    }
    return result;
}

int FindStandKeyframe(const mjModel* model) {
    for (int key_id = 0; key_id < model->nkey; ++key_id) {
        const char* name = mj_id2name(model, mjOBJ_KEY, key_id);
        if (name != nullptr && std::string_view(name) == "stand") {
            return key_id;
        }
    }
    return -1;
}

bool IsImportedIMUSensorType(int sensor_type) {
    return sensor_type == mjSENS_ACCELEROMETER ||
           sensor_type == mjSENS_GYRO ||
           sensor_type == mjSENS_VELOCIMETER ||
           sensor_type == mjSENS_FRAMEQUAT ||
           sensor_type == mjSENS_FRAMELINVEL ||
           sensor_type == mjSENS_FRAMEANGVEL ||
           sensor_type == mjSENS_FRAMELINACC;
}

bool IsImportedContactSensorType(int sensor_type) {
    return sensor_type == mjSENS_TOUCH;
}

bool IsImportedAngularMomentumSensorType(int sensor_type) {
    return sensor_type == mjSENS_SUBTREEANGMOM;
}

std::string TrimIMUSensorComponentSuffix(std::string sensor_name) {
    if (sensor_name.empty()) {
        return sensor_name;
    }

    const std::string lowered = ToLower(sensor_name);
    const std::array<std::string_view, 13> suffixes = {
            "_accelerometer",
            "_accel",
            "_linear_acceleration",
            "_lin_acc",
            "_gyro",
            "_angular_velocity",
            "_ang_vel",
            "_velocimeter",
            "_linear_velocity",
            "_lin_vel",
            "_framequat",
            "_orientation",
            "_quat"};

    for (std::string_view suffix : suffixes) {
        if (lowered.size() > suffix.size() &&
            lowered.ends_with(suffix)) {
            sensor_name.resize(sensor_name.size() - suffix.size());
            break;
        }
    }

    return sensor_name.empty() ? "imu" : sensor_name;
}

SceneState::NodeData MakeCommonSensorNode(const std::string& type,
                                          const std::string& name,
                                          int parent,
                                          const mjModel* model,
                                          int site_id,
                                          const std::vector<int>& sensor_ids) {
    SceneState::NodeData node_data;
    node_data.type = type;
    node_data.name = name;
    node_data.parent = parent;
    AddTransformProperties(node_data,
                           ToVector3(model->site_pos + 3 * site_id),
                           ToMatrix3FromMuJoCoQuat(model->site_quat + 4 * site_id));
    AddProperty(node_data, "enabled", true);

    RealType sensor_period = 0.0;
    RealType noise_stddev = 0.0;
    for (int sensor_id : sensor_ids) {
        if (sensor_id < 0 || sensor_id >= model->nsensor) {
            continue;
        }
        const auto period = static_cast<RealType>(model->sensor_interval[2 * sensor_id]);
        if (period > 0.0 && (sensor_period <= 0.0 || period < sensor_period)) {
            sensor_period = period;
        }
        noise_stddev = std::max(noise_stddev, static_cast<RealType>(model->sensor_noise[sensor_id]));
    }

    AddProperty(node_data, "sensor_period", sensor_period);
    AddProperty(node_data, "noise_stddev", noise_stddev);
    AddProperty(node_data, "visualize_debug", true);
    return node_data;
}

SceneState::NodeData MakeCommonSensorNode(const std::string& type,
                                          const std::string& name,
                                          int parent,
                                          const std::vector<int>& sensor_ids,
                                          const mjModel* model) {
    SceneState::NodeData node_data;
    node_data.type = type;
    node_data.name = name;
    node_data.parent = parent;
    AddTransformProperties(node_data, Vector3::Zero(), Matrix3::Identity());
    AddProperty(node_data, "enabled", true);

    RealType sensor_period = 0.0;
    RealType noise_stddev = 0.0;
    for (int sensor_id : sensor_ids) {
        if (sensor_id < 0 || sensor_id >= model->nsensor) {
            continue;
        }
        const auto period = static_cast<RealType>(model->sensor_interval[2 * sensor_id]);
        if (period > 0.0 && (sensor_period <= 0.0 || period < sensor_period)) {
            sensor_period = period;
        }
        noise_stddev = std::max(noise_stddev, static_cast<RealType>(model->sensor_noise[sensor_id]));
    }

    AddProperty(node_data, "sensor_period", sensor_period);
    AddProperty(node_data, "noise_stddev", noise_stddev);
    AddProperty(node_data, "visualize_debug", true);
    return node_data;
}

std::string MakeIMUSensorName(const mjModel* model, int site_id, const std::vector<int>& sensor_ids) {
    std::string common_component_name;
    bool has_component_name = false;
    bool component_names_match = true;
    for (int sensor_id : sensor_ids) {
        const std::string sensor_name = GetMuJoCoName(model, mjOBJ_SENSOR, sensor_id, "");
        if (sensor_name.empty()) {
            continue;
        }
        const std::string trimmed_name = TrimIMUSensorComponentSuffix(sensor_name);
        if (!has_component_name) {
            common_component_name = trimmed_name;
            has_component_name = true;
            continue;
        }
        if (trimmed_name != common_component_name) {
            component_names_match = false;
            break;
        }
    }
    if (has_component_name && component_names_match && !common_component_name.empty()) {
        return common_component_name;
    }

    const std::string site_name = GetMuJoCoName(model, mjOBJ_SITE, site_id, "");
    if (!site_name.empty()) {
        return site_name;
    }

    if (has_component_name && !common_component_name.empty()) {
        return common_component_name;
    }

    return fmt::format("site_{}_imu", site_id);
}

SceneState::NodeData MakeIMUSensorNode(const mjModel* model,
                                       int site_id,
                                       int parent,
                                       const std::vector<int>& sensor_ids) {
    return MakeCommonSensorNode("IMUSensor3D",
                                MakeIMUSensorName(model, site_id, sensor_ids),
                                parent,
                                model,
                                site_id,
                                sensor_ids);
}

SceneState::NodeData MakeContactSensorNode(const mjModel* model,
                                           int site_id,
                                           int parent,
                                           const std::vector<int>& sensor_ids) {
    std::string name;
    if (!sensor_ids.empty()) {
        name = GetMuJoCoName(model, mjOBJ_SENSOR, sensor_ids.front(), "");
    }
    if (name.empty()) {
        name = GetMuJoCoName(model, mjOBJ_SITE, site_id, fmt::format("site_{}", site_id)) + "_contact";
    }

    SceneState::NodeData node_data = MakeCommonSensorNode("ContactSensor3D", name, parent, model, site_id, sensor_ids);
    const RealType radius = static_cast<RealType>(std::max<mjtNum>(model->site_size[3 * site_id], 0.02));
    AddProperty(node_data, "radius", radius);
    AddProperty(node_data, "min_threshold", static_cast<RealType>(0.0));
    RealType max_threshold = 0.0;
    for (int sensor_id : sensor_ids) {
        if (sensor_id >= 0 && sensor_id < model->nsensor && model->sensor_cutoff[sensor_id] > 0.0) {
            max_threshold = std::max(max_threshold, static_cast<RealType>(model->sensor_cutoff[sensor_id]));
        }
    }
    AddProperty(node_data, "max_threshold", max_threshold);
    return node_data;
}

SceneState::NodeData MakeAngularMomentumSensorNode(const mjModel* model,
                                                   int body_id,
                                                   int parent,
                                                   const std::vector<int>& sensor_ids) {
    std::string name;
    if (!sensor_ids.empty()) {
        name = GetMuJoCoName(model, mjOBJ_SENSOR, sensor_ids.front(), "");
    }
    if (name.empty()) {
        name = GetMuJoCoName(model, mjOBJ_BODY, body_id, fmt::format("body_{}", body_id)) + "_angular_momentum";
    }

    return MakeCommonSensorNode("AngularMomentumSensor3D", name, parent, sensor_ids, model);
}

void AddSensorsToSceneState(const Ref<SceneState>& state,
                            const mjModel* model,
                            const std::unordered_map<int, int>& body_to_link_node) {
    if (!state.IsValid() || model == nullptr || model->nsensor <= 0) {
        return;
    }

    std::vector<std::vector<int>> imu_sensor_ids_by_site(static_cast<std::size_t>(model->nsite));
    std::vector<std::vector<int>> contact_sensor_ids_by_site(static_cast<std::size_t>(model->nsite));
    std::vector<std::vector<int>> angular_momentum_sensor_ids_by_body(static_cast<std::size_t>(model->nbody));
    std::unordered_set<int> warned_sensor_types;
    for (int sensor_id = 0; sensor_id < model->nsensor; ++sensor_id) {
        if (IsImportedAngularMomentumSensorType(model->sensor_type[sensor_id]) &&
            model->sensor_objtype[sensor_id] == mjOBJ_BODY &&
            model->sensor_objid[sensor_id] >= 0 &&
            model->sensor_objid[sensor_id] < model->nbody) {
            angular_momentum_sensor_ids_by_body[static_cast<std::size_t>(model->sensor_objid[sensor_id])].push_back(sensor_id);
            continue;
        }

        if (model->sensor_objtype[sensor_id] != mjOBJ_SITE ||
            model->sensor_objid[sensor_id] < 0 ||
            model->sensor_objid[sensor_id] >= model->nsite) {
            if (warned_sensor_types.insert(model->sensor_type[sensor_id]).second) {
                LOG_WARN("MJCF sensor type {} is not imported because it is not attached to a site.",
                         model->sensor_type[sensor_id]);
            }
            continue;
        }

        const int site_id = model->sensor_objid[sensor_id];
        if (IsImportedIMUSensorType(model->sensor_type[sensor_id])) {
            imu_sensor_ids_by_site[static_cast<std::size_t>(site_id)].push_back(sensor_id);
        } else if (IsImportedContactSensorType(model->sensor_type[sensor_id])) {
            contact_sensor_ids_by_site[static_cast<std::size_t>(site_id)].push_back(sensor_id);
        } else if (warned_sensor_types.insert(model->sensor_type[sensor_id]).second) {
            LOG_WARN("MJCF sensor type {} is not imported as a Gobot sensor node.",
                     model->sensor_type[sensor_id]);
        }
    }

    for (int body_id = 1; body_id < model->nbody; ++body_id) {
        const std::vector<int>& sensor_ids =
                angular_momentum_sensor_ids_by_body[static_cast<std::size_t>(body_id)];
        if (sensor_ids.empty()) {
            continue;
        }
        const auto parent_iter = body_to_link_node.find(body_id);
        if (parent_iter != body_to_link_node.end()) {
            state->AddNode(MakeAngularMomentumSensorNode(model, body_id, parent_iter->second, sensor_ids));
        }
    }

    for (int site_id = 0; site_id < model->nsite; ++site_id) {
        const std::vector<int>& sensor_ids = imu_sensor_ids_by_site[static_cast<std::size_t>(site_id)];
        if (sensor_ids.empty()) {
            continue;
        }
        const int body_id = model->site_bodyid[site_id];
        const auto parent_iter = body_to_link_node.find(body_id);
        if (parent_iter != body_to_link_node.end()) {
            state->AddNode(MakeIMUSensorNode(model, site_id, parent_iter->second, sensor_ids));
        }
    }

    for (int site_id = 0; site_id < model->nsite; ++site_id) {
        const std::vector<int>& sensor_ids = contact_sensor_ids_by_site[static_cast<std::size_t>(site_id)];
        if (sensor_ids.empty()) {
            continue;
        }
        const int body_id = model->site_bodyid[site_id];
        const auto parent_iter = body_to_link_node.find(body_id);
        if (parent_iter != body_to_link_node.end()) {
            state->AddNode(MakeContactSensorNode(model, site_id, parent_iter->second, sensor_ids));
        }
    }
}

void ApplyStandKeyframeToData(const mjModel* model, mjData* data) {
    const int stand_key_id = FindStandKeyframe(model);
    if (stand_key_id < 0) {
        return;
    }

    std::copy_n(model->key_qpos + stand_key_id * model->nq, model->nq, data->qpos);
    std::copy_n(model->key_ctrl + stand_key_id * model->nu, model->nu, data->ctrl);
    mj_forward(model, data);
}

std::string ReadMuJoCoString(const mjString* value) {
    const char* string_value = mjs_getString(value);
    return string_value != nullptr ? std::string(string_value) : std::string();
}

std::filesystem::path ResolveMJCFAssetPath(const std::filesystem::path& source_file,
                                           const std::string& mesh_dir,
                                           const std::string& asset_file) {
    const std::filesystem::path file(asset_file);
    if (file.is_absolute()) {
        const std::filesystem::path absolute_path = file.lexically_normal();
        if (std::filesystem::exists(absolute_path)) {
            return absolute_path;
        }
    }

    const std::filesystem::path relative_file = file.is_absolute() ? file.filename() : file;
    const std::filesystem::path source_dir = source_file.parent_path();
    const std::filesystem::path direct_path = (source_dir / relative_file).lexically_normal();
    if (std::filesystem::exists(direct_path)) {
        return direct_path;
    }

    if (source_dir.has_parent_path()) {
        const std::filesystem::path parent_direct_path = (source_dir.parent_path() / relative_file).lexically_normal();
        if (std::filesystem::exists(parent_direct_path)) {
            return parent_direct_path;
        }
    }

    if (!mesh_dir.empty()) {
        const std::filesystem::path mesh_root(mesh_dir);
        if (mesh_root.is_absolute()) {
            return (mesh_root / relative_file).lexically_normal();
        }

        const std::filesystem::path mesh_path = (source_dir / mesh_root / relative_file).lexically_normal();
        if (std::filesystem::exists(mesh_path)) {
            return mesh_path;
        }

        if (source_dir.has_parent_path()) {
            const std::filesystem::path parent_mesh_path =
                    (source_dir.parent_path() / mesh_root / relative_file).lexically_normal();
            if (std::filesystem::exists(parent_mesh_path)) {
                return parent_mesh_path;
            }
        }

        if (source_dir.has_parent_path() && source_dir.parent_path().has_parent_path()) {
            const std::filesystem::path grandparent_mesh_path =
                    (source_dir.parent_path().parent_path() / mesh_root / relative_file).lexically_normal();
            if (std::filesystem::exists(grandparent_mesh_path)) {
                return grandparent_mesh_path;
            }
        }

        return mesh_path;
    }

    return direct_path;
}

std::string LocalizePath(const std::filesystem::path& path) {
    ProjectSettings* settings = ProjectSettings::s_singleton;
    return settings != nullptr ? settings->LocalizePath(path.string()) : path.string();
}

std::string MakeUniqueVisualName(const mjModel* model, int geom_id) {
    const int mesh_id = model->geom_dataid[geom_id];
    const std::string geom_name = GetMuJoCoName(model, mjOBJ_GEOM, geom_id, "");
    if (!geom_name.empty()) {
        return geom_name;
    }

    return GetMuJoCoName(model, mjOBJ_MESH, mesh_id, fmt::format("mesh_{}", mesh_id)) + "_visual_" +
           std::to_string(geom_id);
}

Ref<Mesh> MakeMeshReference(const std::string& mesh_path) {
    if (mesh_path.empty()) {
        return {};
    }

    if (ResourceCache::Has(mesh_path)) {
        Ref<Mesh> cached_mesh = dynamic_pointer_cast<Mesh>(ResourceCache::GetRef(mesh_path));
        if (cached_mesh.IsValid()) {
            if (Ref<ArrayMesh> array_mesh = dynamic_pointer_cast<ArrayMesh>(cached_mesh); array_mesh.IsValid()) {
                array_mesh->SetMaterial({});
            }
            return cached_mesh;
        }
    }

    if (RenderServer::HasInstance() && ResourceLoader::Exists(mesh_path, "Mesh")) {
        Ref<Resource> loaded_resource = ResourceLoader::Load(mesh_path, "Mesh", ResourceFormatLoader::CacheMode::Reuse);
        Ref<Mesh> loaded_mesh = dynamic_pointer_cast<Mesh>(loaded_resource);
        if (loaded_mesh.IsValid()) {
            if (Ref<ArrayMesh> array_mesh = dynamic_pointer_cast<ArrayMesh>(loaded_mesh); array_mesh.IsValid()) {
                array_mesh->SetMaterial({});
            }
            return loaded_mesh;
        }
    }

    Ref<Mesh> mesh = MakeRef<Mesh>();
    mesh->SetPath(mesh_path);
    return mesh;
}

void NormalizeMJCFFileString(mjString* file,
                             const std::filesystem::path& source_file,
                             const std::string& asset_dir) {
    const std::string existing_path = ReadMuJoCoString(file);
    if (existing_path.empty()) {
        return;
    }

    const std::filesystem::path resolved_path = ResolveMJCFAssetPath(source_file, asset_dir, existing_path);
    if (!resolved_path.empty()) {
        mjs_setString(file, resolved_path.string().c_str());
    }
}

void NormalizeMJCFAssetPaths(mjSpec* spec, const std::filesystem::path& source_file) {
    if (spec == nullptr) {
        return;
    }

    const std::string mesh_dir = ReadMuJoCoString(spec->compiler.meshdir);
    const std::string texture_dir = ReadMuJoCoString(spec->compiler.texturedir);

    for (mjsElement* element = mjs_firstElement(spec, mjOBJ_MESH);
         element != nullptr;
         element = mjs_nextElement(spec, element)) {
        mjsMesh* mesh = mjs_asMesh(element);
        if (mesh != nullptr) {
            NormalizeMJCFFileString(mesh->file, source_file, mesh_dir);
        }
    }

    for (mjsElement* element = mjs_firstElement(spec, mjOBJ_HFIELD);
         element != nullptr;
         element = mjs_nextElement(spec, element)) {
        mjsHField* hfield = mjs_asHField(element);
        if (hfield != nullptr) {
            NormalizeMJCFFileString(hfield->file, source_file, mesh_dir);
        }
    }

    for (mjsElement* element = mjs_firstElement(spec, mjOBJ_TEXTURE);
         element != nullptr;
         element = mjs_nextElement(spec, element)) {
        mjsTexture* texture = mjs_asTexture(element);
        if (texture != nullptr) {
            NormalizeMJCFFileString(texture->file, source_file, texture_dir);
            if (texture->cubefiles != nullptr) {
                for (std::string& cube_file : *texture->cubefiles) {
                    if (!cube_file.empty()) {
                        cube_file = ResolveMJCFAssetPath(source_file, texture_dir, cube_file).string();
                    }
                }
            }
        }
    }

    for (mjsElement* element = mjs_firstElement(spec, mjOBJ_SKIN);
         element != nullptr;
         element = mjs_nextElement(spec, element)) {
        mjsSkin* skin = mjs_asSkin(element);
        if (skin != nullptr) {
            NormalizeMJCFFileString(skin->file, source_file, mesh_dir);
        }
    }

    mjs_setString(spec->compiler.meshdir, "");
    mjs_setString(spec->compiler.texturedir, "");
    mjs_setString(spec->modelfiledir, "");
}

std::unordered_map<int, std::string> BuildMeshAssetPaths(const std::string& input_path) {
    std::unordered_map<int, std::string> mesh_paths;
    char error[kMuJoCoErrorBufferSize] = {};
    std::unique_ptr<mjSpec, decltype(&mj_deleteSpec)> spec(
            mj_parseXML(input_path.c_str(), nullptr, error, sizeof(error)),
            mj_deleteSpec);
    if (!spec) {
        LOG_ERROR("MuJoCo failed to parse MJCF '{}' while resolving mesh assets: {}.", input_path, error);
        return mesh_paths;
    }

    const std::filesystem::path source_file(input_path);
    NormalizeMJCFAssetPaths(spec.get(), source_file);
    const std::string mesh_dir = ReadMuJoCoString(spec->compiler.meshdir);
    for (mjsElement* element = mjs_firstElement(spec.get(), mjOBJ_MESH);
         element != nullptr;
         element = mjs_nextElement(spec.get(), element)) {
        mjsMesh* mesh = mjs_asMesh(element);
        if (mesh == nullptr) {
            continue;
        }

        const int mesh_id = mjs_getId(element);
        const std::string file = ReadMuJoCoString(mesh->file);
        if (mesh_id < 0 || file.empty()) {
            continue;
        }

        mesh_paths[mesh_id] = LocalizePath(ResolveMJCFAssetPath(source_file, mesh_dir, file));
    }

    return mesh_paths;
}

SceneState::NodeData MakeRobotNode(const std::string& name, const std::string& source_path) {
    SceneState::NodeData node_data;
    node_data.type = "Robot3D";
    node_data.name = name.empty() ? "MuJoCoRobot" : name;
    AddProperty(node_data, "source_path", source_path);
    return node_data;
}

SceneState::NodeData MakeBodyLinkNode(const mjModel* model,
                                      int body_id,
                                      int parent,
                                      const Affine3& local_transform) {
    SceneState::NodeData node_data;
    node_data.type = "Link3D";
    node_data.name = GetMuJoCoName(model, mjOBJ_BODY, body_id, fmt::format("body_{}", body_id));
    node_data.parent = parent;
    AddTransformProperties(node_data, local_transform.translation(), local_transform.linear());
    AddProperty(node_data, "has_inertial", model->body_mass[body_id] > 0.0);
    AddProperty(node_data, "mass", static_cast<RealType>(model->body_mass[body_id]));
    AddProperty(node_data, "center_of_mass", ToVector3(model->body_ipos + 3 * body_id));
    AddProperty(node_data, "inertia_diagonal", ToVector3(model->body_inertia + 3 * body_id));
    AddProperty(node_data, "inertia_off_diagonal", Vector3{0.0, 0.0, 0.0});
    AddProperty(node_data, "role", LinkRole::Physical);
    return node_data;
}

JointType ToGobotJointType(int mujoco_joint_type) {
    switch (mujoco_joint_type) {
        case mjJNT_FREE:
            return JointType::Floating;
        case mjJNT_BALL:
            return JointType::Planar;
        case mjJNT_SLIDE:
            return JointType::Prismatic;
        case mjJNT_HINGE:
            return JointType::Revolute;
        default:
            return JointType::Fixed;
    }
}

SceneState::NodeData MakeBodyJointNode(const mjModel* model,
                                       int joint_id,
                                       int parent,
                                       const std::string& parent_link,
                                       const std::string& child_link,
                                       const Affine3& local_transform,
                                       const mjData* data) {
    SceneState::NodeData node_data;
    node_data.type = "Joint3D";
    node_data.name = GetMuJoCoName(model, mjOBJ_JOINT, joint_id, fmt::format("{}_joint", child_link));
    node_data.parent = parent;
    AddTransformProperties(node_data, local_transform.translation(), local_transform.linear());
    AddProperty(node_data, "joint_type", ToGobotJointType(model->jnt_type[joint_id]));
    AddProperty(node_data, "parent_link", parent_link);
    AddProperty(node_data, "child_link", child_link);
    AddProperty(node_data, "axis", ToVector3(model->jnt_axis + 3 * joint_id));

    RealType lower_limit = 0.0;
    RealType upper_limit = 0.0;
    if (model->jnt_limited[joint_id]) {
        lower_limit = model->jnt_range[2 * joint_id];
        upper_limit = model->jnt_range[2 * joint_id + 1];
    }
    AddProperty(node_data, "lower_limit", lower_limit);
    AddProperty(node_data, "upper_limit", upper_limit);

    RealType effort_limit = 0.0;
    if (model->jnt_actfrclimited[joint_id]) {
        effort_limit = std::max(std::abs(model->jnt_actfrcrange[2 * joint_id]),
                                std::abs(model->jnt_actfrcrange[2 * joint_id + 1]));
    }
    AddProperty(node_data, "effort_limit", effort_limit);

    const int dof_address = model->jnt_dofadr[joint_id];
    AddProperty(node_data, "velocity_limit", static_cast<RealType>(0.0));
    AddProperty(node_data, "damping",
                dof_address >= 0 && dof_address < model->nv
                        ? static_cast<RealType>(model->dof_damping[dof_address])
                        : static_cast<RealType>(0.0));
    RealType initial_joint_position = 0.0;
    if (model->jnt_type[joint_id] == mjJNT_HINGE || model->jnt_type[joint_id] == mjJNT_SLIDE) {
        const int qpos_address = model->jnt_qposadr[joint_id];
        if (qpos_address >= 0 && qpos_address < model->nq) {
            initial_joint_position = static_cast<RealType>(data->qpos[qpos_address]);
        }
    }
    AddProperty(node_data, "joint_position", initial_joint_position);
    AddProperty(node_data, "initial_position", initial_joint_position);
    (void)dof_address;
    return node_data;
}

Ref<Shape3D> MakeGeomShape(const mjModel* model, int geom_id) {
    const mjtNum* size = model->geom_size + 3 * geom_id;
    switch (model->geom_type[geom_id]) {
        case mjGEOM_BOX: {
            Ref<BoxShape3D> shape = MakeRef<BoxShape3D>();
            shape->SetSize(Vector3{
                    static_cast<RealType>(size[0] * 2.0),
                    static_cast<RealType>(size[1] * 2.0),
                    static_cast<RealType>(size[2] * 2.0)});
            return dynamic_pointer_cast<Shape3D>(shape);
        }
        case mjGEOM_SPHERE: {
            Ref<SphereShape3D> shape = MakeRef<SphereShape3D>();
            shape->SetRadius(static_cast<float>(size[0]));
            return dynamic_pointer_cast<Shape3D>(shape);
        }
        case mjGEOM_CYLINDER: {
            Ref<CylinderShape3D> shape = MakeRef<CylinderShape3D>();
            shape->SetRadius(static_cast<float>(size[0]));
            shape->SetHeight(static_cast<float>(size[1] * 2.0));
            return dynamic_pointer_cast<Shape3D>(shape);
        }
        case mjGEOM_CAPSULE: {
            Ref<CapsuleShape3D> shape = MakeRef<CapsuleShape3D>();
            shape->SetRadius(static_cast<float>(size[0]));
            shape->SetHeight(static_cast<float>(size[1] * 2.0));
            return dynamic_pointer_cast<Shape3D>(shape);
        }
        default:
            return {};
    }
}

SceneState::NodeData MakeGeomCollisionNode(const mjModel* model, int geom_id, int parent) {
    SceneState::NodeData node_data;
    node_data.type = "CollisionShape3D";
    node_data.name = GetMuJoCoName(model, mjOBJ_GEOM, geom_id, fmt::format("geom_{}", geom_id));
    node_data.parent = parent;
    AddTransformProperties(node_data,
                           ToVector3(model->geom_pos + 3 * geom_id),
                           ToMatrix3FromMuJoCoQuat(model->geom_quat + 4 * geom_id));
    AddProperty(node_data, "shape", MakeGeomShape(model, geom_id));
    AddProperty(node_data, "friction", ToVector3(model->geom_friction + 3 * geom_id));
    AddProperty(node_data, "contype", model->geom_contype[geom_id]);
    AddProperty(node_data, "conaffinity", model->geom_conaffinity[geom_id]);
    AddProperty(node_data, "condim", model->geom_condim[geom_id]);
    AddProperty(node_data, "solref", Vector2{
            static_cast<RealType>(model->geom_solref[2 * geom_id + 0]),
            static_cast<RealType>(model->geom_solref[2 * geom_id + 1])});
    AddProperty(node_data, "solimp", ToRealVector(model->geom_solimp + 5 * geom_id, 5));
    AddProperty(node_data, "margin", static_cast<RealType>(model->geom_margin[geom_id]));
    AddProperty(node_data, "gap", static_cast<RealType>(model->geom_gap[geom_id]));
    AddProperty(node_data, "visible", false);
    return node_data;
}

SceneState::NodeData MakeGeomVisualNode(const mjModel* model,
                                        int geom_id,
                                        int parent,
                                        const std::unordered_map<int, std::string>& mesh_paths) {
    SceneState::NodeData node_data;
    node_data.type = "MeshInstance3D";
    node_data.name = MakeUniqueVisualName(model, geom_id);
    node_data.parent = parent;

    const int mesh_id = model->geom_dataid[geom_id];
    mjtNum position[3] = {};
    mjtNum quaternion[4] = {};
    mju_copy3(position, model->geom_pos + 3 * geom_id);
    mju_copy4(quaternion, model->geom_quat + 4 * geom_id);
    InvertFrameAccumulation(position, quaternion, model->mesh_pos + 3 * mesh_id, model->mesh_quat + 4 * mesh_id);
    AddTransformProperties(node_data, ToVector3(position), ToMatrix3FromMuJoCoQuat(quaternion));

    auto mesh_path_iter = mesh_paths.find(mesh_id);
    if (mesh_path_iter != mesh_paths.end()) {
        AddProperty(node_data, "mesh", MakeMeshReference(mesh_path_iter->second));
    }

    const mjtNum* scale = model->mesh_scale + 3 * mesh_id;
    AddProperty(node_data, "scale", Vector3{
            static_cast<RealType>(scale[0]),
            static_cast<RealType>(scale[1]),
            static_cast<RealType>(scale[2])});
    const Color color = GetGeomColor(model, geom_id);
    AddProperty(node_data, "surface_color", color);

    Ref<PBRMaterial3D> material = MakeRef<PBRMaterial3D>();
    material->SetAlbedo(color);
    if (const int material_id = model->geom_matid[geom_id]; material_id >= 0 && material_id < model->nmat) {
        material->SetMetallic(static_cast<RealType>(model->mat_metallic[material_id]));
        material->SetRoughness(static_cast<RealType>(model->mat_roughness[material_id]));
        material->SetSpecular(static_cast<RealType>(model->mat_specular[material_id]));
    }
    AddProperty(node_data, "material", dynamic_pointer_cast<Material>(material));
    return node_data;
}

void ApplyActuatorsToJoints(const mjModel* model, std::unordered_map<int, SceneState::NodeData>* joint_nodes) {
    if (model == nullptr || joint_nodes == nullptr) {
        return;
    }

    for (int actuator_id = 0; actuator_id < model->nu; ++actuator_id) {
        if (model->actuator_trntype[actuator_id] != mjTRN_JOINT &&
            model->actuator_trntype[actuator_id] != mjTRN_JOINTINPARENT) {
            continue;
        }

        const int joint_id = model->actuator_trnid[2 * actuator_id];
        if (joint_id < 0 || joint_id >= model->njnt) {
            continue;
        }

        auto joint_iter = joint_nodes->find(joint_id);
        if (joint_iter == joint_nodes->end()) {
            continue;
        }

        SceneState::NodeData& joint_node = joint_iter->second;
        if (model->actuator_forcelimited[actuator_id]) {
            const RealType effort_limit = std::max(
                    std::abs(model->actuator_forcerange[2 * actuator_id]),
                    std::abs(model->actuator_forcerange[2 * actuator_id + 1]));
            SetProperty(joint_node, "effort_limit", effort_limit);
            SetProperty(joint_node, "force_lower_limit",
                        static_cast<RealType>(model->actuator_forcerange[2 * actuator_id]));
            SetProperty(joint_node, "force_upper_limit",
                        static_cast<RealType>(model->actuator_forcerange[2 * actuator_id + 1]));
        }

        if (model->actuator_ctrllimited[actuator_id]) {
            SetProperty(joint_node, "control_lower_limit",
                        static_cast<RealType>(model->actuator_ctrlrange[2 * actuator_id]));
            SetProperty(joint_node, "control_upper_limit",
                        static_cast<RealType>(model->actuator_ctrlrange[2 * actuator_id + 1]));
        }

        SetProperty(joint_node, "gear", ToRealVector(model->actuator_gear + 6 * actuator_id, 6));

        const mjtNum* gain = model->actuator_gainprm + mjNGAIN * actuator_id;
        if (LooksLikePositionActuator(model, actuator_id)) {
            SetProperty(joint_node, "drive_mode", JointDriveMode::Position);
            SetProperty(joint_node, "drive_stiffness", static_cast<RealType>(std::abs(gain[0])));
        } else if (LooksLikeVelocityActuator(model, actuator_id)) {
            SetProperty(joint_node, "drive_mode", JointDriveMode::Velocity);
            SetProperty(joint_node, "drive_damping", static_cast<RealType>(std::abs(gain[0])));
        } else if (LooksLikeMotorActuator(model, actuator_id) || IsActuatorForJoint(model, actuator_id, joint_id)) {
            SetProperty(joint_node, "drive_mode", JointDriveMode::Motor);
        }
    }
}
#endif

} // namespace

bool ResourceFormatLoaderMJCF::IsMuJoCoAvailable() {
#ifdef GOBOT_HAS_MUJOCO
    return true;
#else
    return false;
#endif
}

Ref<Resource> ResourceFormatLoaderMJCF::Load(const std::string& path,
                                             const std::string& original_path,
                                             CacheMode cache_mode) {
    (void)cache_mode;

    const std::string input_path = ResolveInputPath(path);
    const std::string xml = ReadTextFile(input_path);
    if (xml.empty() || !LooksLikeMJCF(xml)) {
        return {};
    }

#ifndef GOBOT_HAS_MUJOCO
    LOG_ERROR("MuJoCo support is disabled. Reconfigure with -DGOB_BUILD_MUJOCO=ON to load MJCF file {}.", path);
    return {};
#else
    char error[kMuJoCoErrorBufferSize] = {};
    std::unique_ptr<mjSpec, decltype(&mj_deleteSpec)> spec(
            mj_parseXML(input_path.c_str(), nullptr, error, sizeof(error)),
            mj_deleteSpec);
    if (!spec) {
        LOG_ERROR("MuJoCo failed to parse MJCF '{}': {}.", path, error);
        return {};
    }

    NormalizeMJCFAssetPaths(spec.get(), std::filesystem::path(input_path));

    std::unique_ptr<mjModel, decltype(&mj_deleteModel)> model(mj_compile(spec.get(), nullptr), mj_deleteModel);
    if (!model) {
        const char* compile_error = mjs_getError(spec.get());
        LOG_ERROR("MuJoCo failed to compile MJCF '{}': {}.", path, compile_error != nullptr ? compile_error : "unknown error");
        return {};
    }

    std::unique_ptr<mjData, decltype(&mj_deleteData)> data(mj_makeData(model.get()), mj_deleteData);
    if (!data) {
        LOG_ERROR("MuJoCo failed to allocate data while importing MJCF '{}'.", path);
        return {};
    }
    mj_forward(model.get(), data.get());
    ApplyStandKeyframeToData(model.get(), data.get());

    Ref<PackedScene> packed_scene = MakeRef<PackedScene>();
    Ref<SceneState> state = packed_scene->GetState();

    const std::string model_name = ReadMJCFModelName(xml);
    const std::unordered_map<int, std::string> mesh_paths = BuildMeshAssetPaths(input_path);
    const int robot_index = state->AddNode(MakeRobotNode(model_name, original_path.empty() ? path : original_path));

    std::unordered_map<int, int> body_to_link_node;
    std::unordered_map<int, SceneState::NodeData> joint_nodes;
    for (int body_id = 1; body_id < model->nbody; ++body_id) {
        const int parent_body_id = model->body_parentid[body_id];
        const int parent_link_index = parent_body_id > 0 && body_to_link_node.contains(parent_body_id)
                                              ? body_to_link_node[parent_body_id]
                                              : robot_index;
        const Affine3 parent_world_transform = parent_body_id > 0
                                                       ? GetBodyWorldTransform(data.get(), parent_body_id)
                                                       : Affine3::Identity();
        const Affine3 body_world_transform = GetBodyWorldTransform(data.get(), body_id);
        const std::string parent_link = parent_body_id > 0
                                                ? GetMuJoCoName(model.get(),
                                                                mjOBJ_BODY,
                                                                parent_body_id,
                                                                fmt::format("body_{}", parent_body_id))
                                                : "";
        const std::string child_link = GetMuJoCoName(model.get(),
                                                     mjOBJ_BODY,
                                                     body_id,
                                                     fmt::format("body_{}", body_id));
        int link_parent_index = parent_link_index;
        Affine3 link_parent_world_transform = parent_world_transform;
        Affine3 joint_parent_world_transform = parent_world_transform;
        for (int joint_offset = 0; joint_offset < model->body_jntnum[body_id]; ++joint_offset) {
            const int joint_id = model->body_jntadr[body_id] + joint_offset;
            const Affine3 joint_world_transform = GetJointWorldTransform(model.get(), data.get(), joint_id, body_id);
            const Affine3 joint_local_transform = joint_parent_world_transform.inverse() * joint_world_transform;
            joint_nodes.emplace(joint_id,
                                MakeBodyJointNode(model.get(), joint_id, parent_link_index, parent_link, child_link,
                                                  joint_local_transform,
                                                  data.get()));
            joint_parent_world_transform = joint_world_transform;
        }

        ApplyActuatorsToJoints(model.get(), &joint_nodes);
        for (int joint_offset = 0; joint_offset < model->body_jntnum[body_id]; ++joint_offset) {
            const int joint_id = model->body_jntadr[body_id] + joint_offset;
            auto joint_iter = joint_nodes.find(joint_id);
            if (joint_iter != joint_nodes.end()) {
                joint_iter->second.parent = link_parent_index;
                link_parent_index = state->AddNode(joint_iter->second);
                link_parent_world_transform =
                        GetJointWorldTransform(model.get(), data.get(), joint_id, body_id);
            }
        }

        const Affine3 link_local_transform = link_parent_world_transform.inverse() * body_world_transform;
        const int link_index =
                state->AddNode(MakeBodyLinkNode(model.get(), body_id, link_parent_index, link_local_transform));
        body_to_link_node[body_id] = link_index;

        for (int geom_offset = 0; geom_offset < model->body_geomnum[body_id]; ++geom_offset) {
            const int geom_id = model->body_geomadr[body_id] + geom_offset;
            if (model->geom_type[geom_id] == mjGEOM_MESH && model->geom_dataid[geom_id] >= 0) {
                state->AddNode(MakeGeomVisualNode(model.get(), geom_id, link_index, mesh_paths));
                continue;
            }
            if (!MakeGeomShape(model.get(), geom_id).IsValid()) {
                continue;
            }
            state->AddNode(MakeGeomCollisionNode(model.get(), geom_id, link_index));
        }
    }
    AddSensorsToSceneState(state, model.get(), body_to_link_node);

    LOG_INFO("MJCF '{}' generated PackedScene with {} nodes.", path, state->GetNodeCount());
    return packed_scene;
#endif
}

bool ResourceFormatLoaderMJCF::RecognizePath(const std::string& path, const std::string& type_hint) const {
    if (!type_hint.empty() && type_hint != "PackedScene") {
        return false;
    }

    const std::string extension = ToLower(GetFileExtension(path));
    if (extension != "xml" && extension != "mjcf") {
        return false;
    }

    const std::string input_path = ResolveInputPath(path);
    return LooksLikeMJCF(ReadTextFile(input_path));
}

void ResourceFormatLoaderMJCF::GetRecognizedExtensionsForType(const std::string& type,
                                                              std::vector<std::string>* extensions) const {
    if (type.empty() || HandlesType(type)) {
        GetRecognizedExtensions(extensions);
    }
}

void ResourceFormatLoaderMJCF::GetRecognizedExtensions(std::vector<std::string>* extensions) const {
    extensions->push_back("mjcf");
    extensions->push_back("xml");
}

bool ResourceFormatLoaderMJCF::HandlesType(const std::string& type) const {
    return type.empty() || type == "PackedScene";
}

} // namespace gobot

GOBOT_REGISTRATION {

    Class_<ResourceFormatLoaderMJCF>("ResourceFormatLoaderMJCF")
            .constructor()(CtorAsRawPtr)
            .method("is_mujoco_available", &ResourceFormatLoaderMJCF::IsMuJoCoAvailable);

};
