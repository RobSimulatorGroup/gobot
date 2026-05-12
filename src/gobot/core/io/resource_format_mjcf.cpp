/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/core/io/resource_format_mjcf.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
#include <regex>
#include <sstream>
#include <unordered_map>
#include <utility>

#include "gobot/core/config/project_setting.hpp"
#include "gobot/core/registration.hpp"
#include "gobot/core/string_utils.hpp"
#include "gobot/log.hpp"
#include "gobot/scene/collision_shape_3d.hpp"
#include "gobot/scene/joint_3d.hpp"
#include "gobot/scene/link_3d.hpp"
#include "gobot/scene/resources/box_shape_3d.hpp"
#include "gobot/scene/resources/cylinder_shape_3d.hpp"
#include "gobot/scene/resources/packed_scene.hpp"
#include "gobot/scene/resources/sphere_shape_3d.hpp"
#include "gobot/scene/robot_3d.hpp"

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

SceneState::NodeData MakeRobotNode(const std::string& name, const std::string& source_path) {
    SceneState::NodeData node_data;
    node_data.type = "Robot3D";
    node_data.name = name.empty() ? "MuJoCoRobot" : name;
    AddProperty(node_data, "source_path", source_path);
    return node_data;
}

SceneState::NodeData MakeBodyLinkNode(const mjModel* model, int body_id, int parent) {
    SceneState::NodeData node_data;
    node_data.type = "Link3D";
    node_data.name = GetMuJoCoName(model, mjOBJ_BODY, body_id, fmt::format("body_{}", body_id));
    node_data.parent = parent;
    AddTransformProperties(node_data,
                           ToVector3(model->body_pos + 3 * body_id),
                           ToMatrix3FromMuJoCoQuat(model->body_quat + 4 * body_id));
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
                                       const std::string& child_link) {
    SceneState::NodeData node_data;
    node_data.type = "Joint3D";
    node_data.name = GetMuJoCoName(model, mjOBJ_JOINT, joint_id, fmt::format("{}_joint", child_link));
    node_data.parent = parent;
    AddTransformProperties(node_data, ToVector3(model->jnt_pos + 3 * joint_id), Matrix3::Identity());
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
    AddProperty(node_data, "damping", static_cast<RealType>(model->dof_damping[dof_address]));
    AddProperty(node_data, "joint_position",
                model->jnt_type[joint_id] == mjJNT_HINGE || model->jnt_type[joint_id] == mjJNT_SLIDE
                        ? static_cast<RealType>(model->qpos0[model->jnt_qposadr[joint_id]])
                        : static_cast<RealType>(0.0));
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
            Ref<CylinderShape3D> shape = MakeRef<CylinderShape3D>();
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
    return node_data;
}

void ApplyActuatorLimitsToJoints(const mjModel* model, std::unordered_map<int, SceneState::NodeData>* joint_nodes) {
    if (model == nullptr || joint_nodes == nullptr) {
        return;
    }

    for (int actuator_id = 0; actuator_id < model->nu; ++actuator_id) {
        const int transmission_type = model->actuator_trntype[actuator_id];
        if (transmission_type != mjTRN_JOINT && transmission_type != mjTRN_JOINTINPARENT) {
            continue;
        }

        const int joint_id = model->actuator_trnid[2 * actuator_id];
        if (joint_id < 0 || joint_id >= model->njnt || !model->actuator_forcelimited[actuator_id]) {
            continue;
        }

        auto joint_iter = joint_nodes->find(joint_id);
        if (joint_iter == joint_nodes->end()) {
            continue;
        }

        const RealType effort_limit = std::max(
                std::abs(model->actuator_forcerange[2 * actuator_id]),
                std::abs(model->actuator_forcerange[2 * actuator_id + 1]));
        for (SceneState::PropertyData& property : joint_iter->second.properties) {
            if (property.name == "effort_limit") {
                property.value = Variant(effort_limit);
                break;
            }
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
    std::unique_ptr<mjModel, decltype(&mj_deleteModel)> model(
            mj_loadXML(input_path.c_str(), nullptr, error, sizeof(error)),
            mj_deleteModel);
    if (!model) {
        LOG_ERROR("MuJoCo failed to load MJCF '{}': {}.", path, error);
        return {};
    }

    std::unique_ptr<mjData, decltype(&mj_deleteData)> data(mj_makeData(model.get()), mj_deleteData);
    if (!data) {
        LOG_ERROR("MuJoCo failed to allocate data while importing MJCF '{}'.", path);
        return {};
    }
    mj_forward(model.get(), data.get());

    Ref<PackedScene> packed_scene = MakeRef<PackedScene>();
    Ref<SceneState> state = packed_scene->GetState();

    const std::string model_name = ReadMJCFModelName(xml);
    const int robot_index = state->AddNode(MakeRobotNode(model_name, original_path.empty() ? path : original_path));

    std::unordered_map<int, int> body_to_link_node;
    std::unordered_map<int, SceneState::NodeData> joint_nodes;
    for (int body_id = 1; body_id < model->nbody; ++body_id) {
        const int parent_body_id = model->body_parentid[body_id];
        const int parent_link_index = parent_body_id > 0 && body_to_link_node.contains(parent_body_id)
                                              ? body_to_link_node[parent_body_id]
                                              : robot_index;
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
        for (int joint_offset = 0; joint_offset < model->body_jntnum[body_id]; ++joint_offset) {
            const int joint_id = model->body_jntadr[body_id] + joint_offset;
            joint_nodes.emplace(joint_id,
                                MakeBodyJointNode(model.get(), joint_id, parent_link_index, parent_link, child_link));
        }

        ApplyActuatorLimitsToJoints(model.get(), &joint_nodes);
        for (int joint_offset = 0; joint_offset < model->body_jntnum[body_id]; ++joint_offset) {
            const int joint_id = model->body_jntadr[body_id] + joint_offset;
            auto joint_iter = joint_nodes.find(joint_id);
            if (joint_iter != joint_nodes.end()) {
                joint_iter->second.parent = link_parent_index;
                link_parent_index = state->AddNode(joint_iter->second);
            }
        }

        const int link_index = state->AddNode(MakeBodyLinkNode(model.get(), body_id, link_parent_index));
        body_to_link_node[body_id] = link_index;

        for (int geom_offset = 0; geom_offset < model->body_geomnum[body_id]; ++geom_offset) {
            const int geom_id = model->body_geomadr[body_id] + geom_offset;
            if (!MakeGeomShape(model.get(), geom_id).IsValid()) {
                continue;
            }
            state->AddNode(MakeGeomCollisionNode(model.get(), geom_id, link_index));
        }
    }

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
