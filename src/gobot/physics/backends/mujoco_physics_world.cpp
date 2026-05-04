/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
*/

#include "gobot/physics/backends/mujoco_physics_world.hpp"

#include <cctype>
#include <filesystem>
#include <memory>
#include <set>
#include <string_view>

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

int FindActuatorForJoint(const mjModel* model, int joint_id, const std::string& joint_name) {
    if (!model || joint_id < 0) {
        return -1;
    }

    const int joint_trnid = joint_id;
    const int dof_trnid = model->jnt_dofadr[joint_id];
    for (int actuator_id = 0; actuator_id < model->nu; ++actuator_id) {
        const int transmission_type = model->actuator_trntype[actuator_id];
        const int transmission_id = model->actuator_trnid[2 * actuator_id];
        if (transmission_type == mjTRN_JOINT && transmission_id == joint_trnid) {
            return actuator_id;
        }
        if (transmission_type == mjTRN_JOINTINPARENT && transmission_id == joint_trnid) {
            return actuator_id;
        }
        if (transmission_type == mjTRN_SITE && transmission_id == dof_trnid) {
            return actuator_id;
        }
    }

    const std::string motor_name = joint_name + "_motor";
    const int named_motor_id = mj_name2id(model, mjOBJ_ACTUATOR, motor_name.c_str());
    if (named_motor_id >= 0) {
        return named_motor_id;
    }

    return mj_name2id(model, mjOBJ_ACTUATOR, joint_name.c_str());
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

    if (!asset_dir.empty()) {
        const std::filesystem::path asset_path = (model_dir / asset_dir / file).lexically_normal();
        if (std::filesystem::exists(asset_path)) {
            return asset_path.string();
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

    SyncStateToMuJoCo();
    SyncStateFromMuJoCo();
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
        mj_resetData(static_cast<mjModel*>(model_), static_cast<mjData*>(data_));
        SyncStateToMuJoCo();
        SyncStateFromMuJoCo();
    }
#endif
}

bool MuJoCoPhysicsWorld::RestoreCompatibleState(const PhysicsSceneState& previous_state) {
    if (!PhysicsWorld::RestoreCompatibleState(previous_state)) {
        return false;
    }

#ifdef GOBOT_HAS_MUJOCO
    if (model_ && data_) {
        SyncStateToMuJoCo();
        SyncStateFromMuJoCo();
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

    model->opt.timestep = delta_time > 0.0 ? delta_time : settings_.fixed_time_step;
    ApplyControlsToMuJoCo();
    mj_step(model, data);
    SyncStateFromMuJoCo();
#else
    GOB_UNUSED(delta_time);
#endif
}

#ifdef GOBOT_HAS_MUJOCO
bool MuJoCoPhysicsWorld::LoadModelFromRobotSources() {
    FreeModel();

    std::vector<std::size_t> robot_indices;
    for (std::size_t robot_index = 0; robot_index < scene_snapshot_.robots.size(); ++robot_index) {
        if (!scene_snapshot_.robots[robot_index].source_path.empty()) {
            robot_indices.push_back(robot_index);
        }
    }

    if (robot_indices.empty()) {
        SetLastError("MuJoCo backend currently requires a Robot3D source_path pointing to a URDF or MJCF XML file.");
        return false;
    }

    mjSpec* parent_spec = mj_makeSpec();
    if (!parent_spec) {
        SetLastError("MuJoCo failed to allocate a parent spec.");
        return false;
    }
    std::unique_ptr<mjSpec, decltype(&mj_deleteSpec)> parent_spec_guard(parent_spec, mj_deleteSpec);

    AddLooseSceneGeomsToSpec(parent_spec);

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

        if (!AttachRobotModelToSpec(parent_spec, robot, robot_index, prefix)) {
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
        if (shape.disabled || shape.type != PhysicsShapeType::Box) {
            continue;
        }

        mjsGeom* geom = mjs_addGeom(world, nullptr);
        if (!geom) {
            continue;
        }

        const std::string name = fmt::format("gobot_loose_box_{}", shape_index);
        mjs_setName(geom->element, name.c_str());
        geom->type = mjGEOM_BOX;
        geom->pos[0] = shape.global_transform.translation().x();
        geom->pos[1] = shape.global_transform.translation().y();
        geom->pos[2] = shape.global_transform.translation().z();
        const Quaternion rotation(shape.global_transform.linear());
        geom->quat[0] = rotation.w();
        geom->quat[1] = rotation.x();
        geom->quat[2] = rotation.y();
        geom->quat[3] = rotation.z();
        geom->size[0] = shape.box_size.x() * 0.5;
        geom->size[1] = shape.box_size.y() * 0.5;
        geom->size[2] = shape.box_size.z() * 0.5;
        geom->contype = 1;
        geom->conaffinity = 1;
        geom->friction[0] = 1.0;
        geom->friction[1] = 0.005;
        geom->friction[2] = 0.0001;
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

        mjsJoint* free_joint = mjs_addFreeJoint(child_body);
        if (!free_joint) {
            LOG_WARN("Failed to add MuJoCo free joint for Gobot floating joint '{}'.", joint.name);
            continue;
        }

        const std::string free_joint_name = joint.name.empty()
                                                    ? fmt::format("{}_freejoint", joint.child_link)
                                                    : joint.name;
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
            binding.actuator_id = FindActuatorForJoint(model, joint_id, joint_name);
            binding.qpos_address = model->jnt_qposadr[joint_id];
            binding.dof_address = model->jnt_dofadr[joint_id];
            binding.joint_type = model->jnt_type[joint_id];
            joint_bindings_.emplace_back(binding);
        }
    }

    LOG_INFO("MuJoCo joint bindings built: {} of {} Gobot joints.",
             joint_bindings_.size(),
             scene_state_.total_joint_count);
}

std::string MuJoCoPhysicsWorld::GetRobotPrefix(std::size_t robot_index) const {
    for (const MuJoCoRobotBinding& binding : robot_bindings_) {
        if (binding.robot_index == robot_index) {
            return binding.prefix;
        }
    }

    return {};
}

void MuJoCoPhysicsWorld::ApplyControlsToMuJoCo() {
    auto* model = static_cast<mjModel*>(model_);
    auto* data = static_cast<mjData*>(data_);
    if (!model || !data) {
        return;
    }

    for (int velocity_index = 0; velocity_index < model->nv; ++velocity_index) {
        data->qfrc_applied[velocity_index] = 0.0;
    }

    for (const MuJoCoJointBinding& binding : joint_bindings_) {
        if (binding.robot_index >= scene_state_.robots.size()) {
            continue;
        }

        const PhysicsRobotState& robot_state = scene_state_.robots[binding.robot_index];
        if (binding.joint_index >= robot_state.joints.size()) {
            continue;
        }

        const PhysicsJointState& joint_state = robot_state.joints[binding.joint_index];
        if (binding.joint_type == mjJNT_FREE || binding.joint_type == mjJNT_BALL) {
            continue;
        }

        RealType control_value = 0.0;
        switch (joint_state.control_mode) {
            case PhysicsJointControlMode::Passive:
                control_value = 0.0;
                break;
            case PhysicsJointControlMode::Position:
                control_value = joint_state.target_position;
                break;
            case PhysicsJointControlMode::Velocity:
                control_value = joint_state.target_velocity;
                break;
            case PhysicsJointControlMode::Effort:
                control_value = joint_state.target_effort;
                break;
        }

        if (binding.actuator_id >= 0 && binding.actuator_id < model->nu) {
            data->ctrl[binding.actuator_id] = control_value;
            continue;
        }

        if (binding.dof_address < 0 || binding.dof_address >= model->nv) {
            continue;
        }

        JointController controller({
                settings_.default_position_stiffness,
                settings_.default_velocity_damping,
                0.0,
                0.0
        });

        JointControllerLimits limits;
        if (binding.robot_index < scene_snapshot_.robots.size() &&
            binding.joint_index < scene_snapshot_.robots[binding.robot_index].joints.size()) {
            limits = MakeJointControllerLimits(scene_snapshot_.robots[binding.robot_index].joints[binding.joint_index]);
        }

        data->qfrc_applied[binding.dof_address] =
                controller.ComputeEffort(MakeJointControllerState(joint_state),
                                         MakeJointControllerCommand(joint_state),
                                         limits,
                                         settings_.fixed_time_step);
    }
}

void MuJoCoPhysicsWorld::FreeModel() {
    link_bindings_.clear();
    joint_bindings_.clear();

    if (data_) {
        mj_deleteData(static_cast<mjData*>(data_));
        data_ = nullptr;
    }

    if (model_) {
        mj_deleteModel(static_cast<mjModel*>(model_));
        model_ = nullptr;
    }
}

void MuJoCoPhysicsWorld::SyncStateFromMuJoCo() {
    auto* model = static_cast<mjModel*>(model_);
    auto* data = static_cast<mjData*>(data_);
    if (!model || !data) {
        return;
    }

    for (const MuJoCoLinkBinding& binding : link_bindings_) {
        if (binding.robot_index >= scene_state_.robots.size()) {
            continue;
        }

        PhysicsRobotState& robot_state = scene_state_.robots[binding.robot_index];
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
        if (binding.robot_index >= scene_state_.robots.size()) {
            continue;
        }

        PhysicsRobotState& robot_state = scene_state_.robots[binding.robot_index];
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
}

void MuJoCoPhysicsWorld::SyncStateToMuJoCo() {
    auto* model = static_cast<mjModel*>(model_);
    auto* data = static_cast<mjData*>(data_);
    if (!model || !data) {
        return;
    }

    for (const MuJoCoJointBinding& binding : joint_bindings_) {
        if (binding.robot_index >= scene_state_.robots.size()) {
            continue;
        }

        const PhysicsRobotState& robot_state = scene_state_.robots[binding.robot_index];
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
#endif

} // namespace gobot

GOBOT_REGISTRATION {

    Class_<MuJoCoPhysicsWorld>("MuJoCoPhysicsWorld")
            .constructor()(CtorAsRawPtr)
            .method("is_backend_available", &MuJoCoPhysicsWorld::IsBackendAvailable);

    gobot::Type::register_wrapper_converter_for_base_classes<Ref<MuJoCoPhysicsWorld>, Ref<PhysicsWorld>>();

};
