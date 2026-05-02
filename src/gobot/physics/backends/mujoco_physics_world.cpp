/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
*/

#include "gobot/physics/backends/mujoco_physics_world.hpp"

#include <filesystem>
#include <string_view>

#include "gobot/core/config/project_setting.hpp"
#include "gobot/core/registration.hpp"
#include "gobot/log.hpp"

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

const PhysicsRobotSnapshot* FindFirstRobotWithSourcePath(const PhysicsSceneSnapshot& scene_snapshot) {
    for (const PhysicsRobotSnapshot& robot : scene_snapshot.robots) {
        if (!robot.source_path.empty()) {
            return &robot;
        }
    }

    return nullptr;
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
    if (!LoadModelFromRobotSource()) {
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
bool MuJoCoPhysicsWorld::LoadModelFromRobotSource() {
    FreeModel();

    const PhysicsRobotSnapshot* robot = FindFirstRobotWithSourcePath(scene_snapshot_);
    if (!robot) {
        SetLastError("MuJoCo backend currently requires a Robot3D source_path pointing to a URDF or MJCF XML file.");
        return false;
    }

    const std::string model_path = ResolvePhysicsSourcePath(robot->source_path);
    if (model_path.empty()) {
        SetLastError("MuJoCo robot source path is empty after path resolution.");
        return false;
    }

    if (!std::filesystem::exists(model_path)) {
        SetLastError(fmt::format("MuJoCo robot source file does not exist: {}", model_path));
        return false;
    }

    char error[kMuJoCoErrorBufferSize] = {};
    mjModel* model = nullptr;
    mjSpec* spec = mj_parseXML(model_path.c_str(), nullptr, error, sizeof(error));
    if (spec) {
        AddLooseSceneGeomsToSpec(spec);
        model = mj_compile(spec, nullptr);
        if (!model) {
            const std::string compile_error = mjs_getError(spec) ? mjs_getError(spec) : "unknown error";
            mj_deleteSpec(spec);
            SetLastError(fmt::format("MuJoCo failed to compile '{}': {}", model_path, compile_error));
            return false;
        }
        mj_deleteSpec(spec);
    } else {
        LOG_WARN("MuJoCo spec parse failed for '{}': {}. Falling back to mj_loadXML without Gobot scene geoms.",
                 model_path,
                 error);
        error[0] = '\0';
        model = mj_loadXML(model_path.c_str(), nullptr, error, sizeof(error));
    }

    if (!model) {
        SetLastError(fmt::format("MuJoCo failed to load '{}': {}", model_path, error));
        return false;
    }

    auto* data = mj_makeData(model);
    if (!data) {
        mj_deleteModel(model);
        SetLastError(fmt::format("MuJoCo failed to allocate runtime data for '{}'.", model_path));
        return false;
    }

    model_ = model;
    data_ = data;
    BuildLinkBindings();
    BuildJointBindings();

    LOG_INFO("MuJoCo physics model loaded from '{}': nq={}, nv={}, joints={}",
             model_path,
             model->nq,
             model->nv,
             model->njnt);
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
        geom->quat[0] = 1.0;
        geom->quat[1] = 0.0;
        geom->quat[2] = 0.0;
        geom->quat[3] = 0.0;
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

void MuJoCoPhysicsWorld::BuildLinkBindings() {
    link_bindings_.clear();

    auto* model = static_cast<mjModel*>(model_);
    if (!model) {
        return;
    }

    for (std::size_t robot_index = 0; robot_index < scene_state_.robots.size(); ++robot_index) {
        PhysicsRobotState& robot_state = scene_state_.robots[robot_index];
        for (std::size_t link_index = 0; link_index < robot_state.links.size(); ++link_index) {
            PhysicsLinkState& link_state = robot_state.links[link_index];
            const int body_id = mj_name2id(model, mjOBJ_BODY, link_state.link_name.c_str());
            if (body_id < 0) {
                LOG_WARN("MuJoCo model does not contain body '{}'. It will not be synchronized.",
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
        for (std::size_t joint_index = 0; joint_index < robot_state.joints.size(); ++joint_index) {
            PhysicsJointState& joint_state = robot_state.joints[joint_index];
            const int joint_id = mj_name2id(model, mjOBJ_JOINT, joint_state.joint_name.c_str());
            if (joint_id < 0) {
                LOG_WARN("MuJoCo model does not contain joint '{}'. It will not be synchronized.",
                         joint_state.joint_name);
                continue;
            }

            MuJoCoJointBinding binding;
            binding.robot_index = robot_index;
            binding.joint_index = joint_index;
            binding.mujoco_joint_id = joint_id;
            binding.actuator_id = FindActuatorForJoint(model, joint_id, joint_state.joint_name);
            binding.qpos_address = model->jnt_qposadr[joint_id];
            binding.dof_address = model->jnt_dofadr[joint_id];
            joint_bindings_.emplace_back(binding);
        }
    }

    LOG_INFO("MuJoCo joint bindings built: {} of {} Gobot joints.",
             joint_bindings_.size(),
             scene_state_.total_joint_count);
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

        switch (joint_state.control_mode) {
            case PhysicsJointControlMode::Passive:
                break;
            case PhysicsJointControlMode::Position:
                data->qfrc_applied[binding.dof_address] =
                        settings_.default_position_stiffness * (joint_state.target_position - joint_state.position) -
                        settings_.default_velocity_damping * joint_state.velocity;
                break;
            case PhysicsJointControlMode::Velocity:
                data->qfrc_applied[binding.dof_address] =
                        settings_.default_velocity_damping * (joint_state.target_velocity - joint_state.velocity);
                break;
            case PhysicsJointControlMode::Effort:
                data->qfrc_applied[binding.dof_address] = joint_state.target_effort;
                break;
        }
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
        if (binding.qpos_address >= 0 && binding.qpos_address < model->nq) {
            data->qpos[binding.qpos_address] = joint_state.position;
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
