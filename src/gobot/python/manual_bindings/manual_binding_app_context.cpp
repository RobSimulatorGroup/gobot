#include "manual_bindings_internal.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <random>
#include <cstring>
#include <thread>
#include <limits>
#include <unordered_set>

#ifdef GOBOT_HAS_MUJOCO
#include <mujoco/mujoco.h>
#endif

namespace gobot::python {

namespace {

Matrix3 PySensorRayAlignmentMatrix(const Affine3& transform, RayAlignmentMode alignment) {
    switch (alignment) {
        case RayAlignmentMode::World:
            return Matrix3::Identity();
        case RayAlignmentMode::Base:
            return transform.linear();
        case RayAlignmentMode::Yaw: {
            const Vector3 x_axis = transform.linear() * Vector3::UnitX();
            const RealType yaw = std::atan2(x_axis.y(), x_axis.x());
            return AngleAxis(yaw, Vector3::UnitZ()).toRotationMatrix();
        }
    }
    return Matrix3::Identity();
}

Vector3 PyResolveSensorRayDirection(const PhysicsSensorSnapshot& sensor_snapshot,
                                    const Matrix3& alignment_matrix,
                                    const Affine3& sensor_transform) {
    Vector3 ray_direction = sensor_snapshot.ray_direction;
    if (sensor_snapshot.ray_alignment == RayAlignmentMode::Base) {
        ray_direction = sensor_transform.linear() * ray_direction;
    } else if (sensor_snapshot.ray_alignment == RayAlignmentMode::Yaw) {
        ray_direction = alignment_matrix * ray_direction;
    } else if (!sensor_snapshot.ray_direction_world_space) {
        ray_direction = sensor_transform.linear() * ray_direction;
    }

    if (ray_direction.squaredNorm() <= CMP_EPSILON2) {
        ray_direction = Vector3{0.0, 0.0, -1.0};
    } else {
        ray_direction.normalize();
    }
    return ray_direction;
}

RealType PyReduceSensorRayValues(const std::vector<RealType>& values, RayReductionMode reduction_mode) {
    if (values.empty()) {
        return 0.0;
    }
    switch (reduction_mode) {
        case RayReductionMode::Min:
            return *std::min_element(values.begin(), values.end());
        case RayReductionMode::Max:
            return *std::max_element(values.begin(), values.end());
        case RayReductionMode::Mean: {
            RealType sum = 0.0;
            for (RealType value : values) {
                sum += value;
            }
            return sum / static_cast<RealType>(values.size());
        }
        case RayReductionMode::None:
            break;
    }
    return values.front();
}

std::vector<std::regex> CompileContactPatterns(const std::vector<std::string>& patterns) {
    std::vector<std::regex> compiled;
    compiled.reserve(patterns.size());
    for (const std::string& pattern : patterns) {
        compiled.emplace_back(pattern, std::regex::ECMAScript);
    }
    return compiled;
}

bool MatchesAnyPattern(const std::string& name, const std::vector<std::regex>& patterns) {
    for (const std::regex& pattern : patterns) {
        if (std::regex_match(name, pattern) || std::regex_search(name, pattern)) {
            return true;
        }
    }
    return false;
}

constexpr std::size_t kDefaultRewardTermCount = 15;
constexpr std::size_t kDefaultTaskParamCount = 11;
constexpr std::size_t kDefaultTaskFlagCount = 3;

enum LocomotionCommandRangeIndex : std::size_t {
    kCommandLinVelXMin = 0,
    kCommandLinVelXMax,
    kCommandLinVelYMin,
    kCommandLinVelYMax,
    kCommandAngVelZMin,
    kCommandAngVelZMax,
    kCommandHeadingMin,
    kCommandHeadingMax,
    kCommandRangeCount
};

enum LocomotionStepProfileIndex : std::size_t {
    kStepProfileTotal = 0,
    kStepProfilePrepareAction,
    kStepProfileApplyControl,
    kStepProfileMjStep,
    kStepProfileExtractState,
    kStepProfileCommand,
    kStepProfileReward,
    kStepProfileObservation,
    kStepProfileCount
};

template <typename T>
py::array_t<T> VectorArrayView(std::vector<T>& values,
                               std::initializer_list<py::ssize_t> shape,
                               const py::object& owner,
                               bool writable) {
    std::vector<py::ssize_t> shape_vector(shape.begin(), shape.end());
    std::vector<py::ssize_t> strides(shape_vector.size(), static_cast<py::ssize_t>(sizeof(T)));
    for (std::ptrdiff_t index = static_cast<std::ptrdiff_t>(shape_vector.size()) - 2; index >= 0; --index) {
        strides[static_cast<std::size_t>(index)] =
                strides[static_cast<std::size_t>(index) + 1] *
                shape_vector[static_cast<std::size_t>(index) + 1];
    }
    py::array_t<T> array(shape_vector, strides, values.data(), owner);
    if (!writable) {
        array.attr("setflags")(false);
    }
    return array;
}

} // namespace

class PyLocomotionBatchView {
public:
    PyLocomotionBatchView(Ref<MuJoCoPhysicsWorld> world,
                             std::string robot_name,
                             std::string base_link,
                             std::vector<std::string> joint_names,
                             std::vector<std::string> foot_link_names,
                             std::vector<std::string> foot_height_sensor_names,
                             std::vector<std::string> foot_contact_sensor_names,
                             std::string height_scan_sensor,
                             std::vector<std::string> thigh_link_patterns,
                             std::vector<std::string> shank_link_patterns,
                             std::vector<std::string> trunk_head_link_patterns,
                             bool terminate_on_thigh_contact,
                             double ground_force_threshold,
                             double self_collision_force_threshold,
                             std::size_t reward_term_count = 0,
                             std::size_t task_param_count = 0,
                             std::size_t task_flag_count = 0,
                             std::size_t actor_obs_dim = 0,
                             std::size_t critic_obs_dim = 0)
        : world_(std::move(world)),
          robot_name_(std::move(robot_name)),
          base_link_(std::move(base_link)),
          joint_names_(std::move(joint_names)),
          foot_link_names_(std::move(foot_link_names)),
          foot_height_sensor_names_(std::move(foot_height_sensor_names)),
          foot_contact_sensor_names_(std::move(foot_contact_sensor_names)),
          height_scan_sensor_(std::move(height_scan_sensor)),
          thigh_link_patterns_(CompileContactPatterns(thigh_link_patterns)),
          shank_link_patterns_(CompileContactPatterns(shank_link_patterns)),
          trunk_head_link_patterns_(CompileContactPatterns(trunk_head_link_patterns)),
          terminate_on_thigh_contact_(terminate_on_thigh_contact),
          ground_force_threshold_(static_cast<RealType>(ground_force_threshold)),
          self_collision_force_threshold_(static_cast<RealType>(self_collision_force_threshold)),
          reward_term_count_(reward_term_count),
          task_param_count_(task_param_count),
          task_flag_count_(task_flag_count),
          actor_obs_dim_(actor_obs_dim),
          critic_obs_dim_(critic_obs_dim) {
        Initialize();
        if (reward_term_count_ == 0) {
            reward_term_count_ = kDefaultRewardTermCount;
        }
        if (task_param_count_ == 0) {
            task_param_count_ = kDefaultTaskParamCount;
        }
        if (task_flag_count_ == 0) {
            task_flag_count_ = kDefaultTaskFlagCount;
        }
        if (actor_obs_dim_ == 0) {
            actor_obs_dim_ = DefaultActorObservationDim();
        }
        if (critic_obs_dim_ == 0) {
            critic_obs_dim_ = DefaultCriticObservationDim();
        }
        AllocateBuffers();
        Refresh();
    }

    ~PyLocomotionBatchView() {
        StopViewWorkers();
    }

    py::dict Arrays() {
        py::object owner = py::cast(this, py::return_value_policy::reference);
        py::dict arrays;
        arrays["target_position"] =
                VectorArrayView(target_position_, {EnvDim(), JointDim()}, owner, true);
        arrays["action"] =
                VectorArrayView(action_, {EnvDim(), JointDim()}, owner, true);
        arrays["submitted_action"] =
                VectorArrayView(submitted_action_, {EnvDim(), JointDim()}, owner, false);
        arrays["default_joint_position"] =
                VectorArrayView(default_joint_position_, {JointDim()}, owner, true);
        arrays["action_scale"] =
                VectorArrayView(action_scale_, {JointDim()}, owner, true);
        arrays["action_clip"] =
                VectorArrayView(action_clip_, {1}, owner, true);
        arrays["previous_action"] =
                VectorArrayView(previous_action_, {EnvDim(), JointDim()}, owner, true);
        arrays["last_action"] =
                VectorArrayView(last_action_, {EnvDim(), JointDim()}, owner, true);
        arrays["encoder_bias"] =
                VectorArrayView(encoder_bias_, {EnvDim(), JointDim()}, owner, true);
        arrays["command"] =
                VectorArrayView(command_, {EnvDim(), 3}, owner, true);
        arrays["commands"] =
                VectorArrayView(command_, {EnvDim(), 3}, owner, true);
        arrays["command_world"] =
                VectorArrayView(command_world_, {EnvDim(), 3}, owner, false);
        arrays["command_heading_target"] =
                VectorArrayView(command_heading_target_, {EnvDim()}, owner, false);
        arrays["heading_commands"] =
                VectorArrayView(command_heading_target_, {EnvDim()}, owner, false);
        arrays["command_heading_error"] =
                VectorArrayView(command_heading_error_, {EnvDim()}, owner, false);
        arrays["command_time_left"] =
                VectorArrayView(command_time_left_, {EnvDim()}, owner, false);
        arrays["command_is_heading_env"] =
                VectorArrayView(command_is_heading_env_, {EnvDim()}, owner, false);
        arrays["command_is_standing_env"] =
                VectorArrayView(command_is_standing_env_, {EnvDim()}, owner, false);
        arrays["command_is_world_env"] =
                VectorArrayView(command_is_world_env_, {EnvDim()}, owner, false);
        arrays["command_is_forward_env"] =
                VectorArrayView(command_is_forward_env_, {EnvDim()}, owner, false);
        arrays["command_ranges"] =
                VectorArrayView(command_ranges_, {CommandRangeDim()}, owner, true);
        arrays["gait_phase"] =
                VectorArrayView(gait_phase_, {EnvDim(), 2}, owner, true);
        arrays["feet_phase_height_target"] =
                VectorArrayView(feet_phase_height_target_, {EnvDim(), 2}, owner, true);
        arrays["pose_weights"] =
                VectorArrayView(pose_weights_, {JointDim()}, owner, true);
        arrays["step_profile_ms"] =
                VectorArrayView(step_profile_ms_, {StepProfileDim()}, owner, false);
        arrays["pose_std_standing"] =
                VectorArrayView(pose_std_standing_, {JointDim()}, owner, true);
        arrays["pose_std_walking"] =
                VectorArrayView(pose_std_walking_, {JointDim()}, owner, true);
        arrays["pose_std_running"] =
                VectorArrayView(pose_std_running_, {JointDim()}, owner, true);
        arrays["reward_weights"] =
                VectorArrayView(reward_weights_, {RewardTermDim()}, owner, true);
        arrays["task_params"] =
                VectorArrayView(task_params_, {TaskParamDim()}, owner, true);
        arrays["task_flags"] =
                VectorArrayView(task_flags_, {TaskFlagDim()}, owner, true);
        arrays["reset_base_position"] =
                VectorArrayView(reset_base_position_, {EnvDim(), 3}, owner, true);
        arrays["reset_base_quaternion"] =
                VectorArrayView(reset_base_quaternion_, {EnvDim(), 4}, owner, true);
        arrays["reset_base_linear_velocity"] =
                VectorArrayView(reset_base_linear_velocity_, {EnvDim(), 3}, owner, true);
        arrays["reset_base_angular_velocity"] =
                VectorArrayView(reset_base_angular_velocity_, {EnvDim(), 3}, owner, true);
        arrays["reset_joint_position"] =
                VectorArrayView(reset_joint_position_, {EnvDim(), JointDim()}, owner, true);
        arrays["reset_joint_velocity"] =
                VectorArrayView(reset_joint_velocity_, {EnvDim(), JointDim()}, owner, true);
        arrays["base_position"] =
                VectorArrayView(base_position_, {EnvDim(), 3}, owner, false);
        arrays["base_pos"] =
                VectorArrayView(base_position_, {EnvDim(), 3}, owner, false);
        arrays["base_quaternion"] =
                VectorArrayView(base_quaternion_, {EnvDim(), 4}, owner, false);
        arrays["base_quat"] =
                VectorArrayView(base_quaternion_, {EnvDim(), 4}, owner, false);
        arrays["base_linear_velocity"] =
                VectorArrayView(base_linear_velocity_, {EnvDim(), 3}, owner, false);
        arrays["base_angular_velocity"] =
                VectorArrayView(base_angular_velocity_, {EnvDim(), 3}, owner, false);
        arrays["base_linear_velocity_body"] =
                VectorArrayView(base_linear_velocity_body_, {EnvDim(), 3}, owner, false);
        arrays["linvel"] =
                VectorArrayView(base_linear_velocity_body_, {EnvDim(), 3}, owner, false);
        arrays["base_angular_velocity_body"] =
                VectorArrayView(base_angular_velocity_body_, {EnvDim(), 3}, owner, false);
        arrays["gyro"] =
                VectorArrayView(base_angular_velocity_body_, {EnvDim(), 3}, owner, false);
        arrays["projected_gravity"] =
                VectorArrayView(projected_gravity_, {EnvDim(), 3}, owner, false);
        arrays["gravity"] =
                VectorArrayView(projected_gravity_, {EnvDim(), 3}, owner, false);
        arrays["base_height"] =
                VectorArrayView(base_height_, {EnvDim()}, owner, false);
        arrays["joint_position"] =
                VectorArrayView(joint_position_, {EnvDim(), JointDim()}, owner, false);
        arrays["dof_pos"] =
                VectorArrayView(joint_position_, {EnvDim(), JointDim()}, owner, false);
        arrays["joint_velocity"] =
                VectorArrayView(joint_velocity_, {EnvDim(), JointDim()}, owner, false);
        arrays["dof_vel"] =
                VectorArrayView(joint_velocity_, {EnvDim(), JointDim()}, owner, false);
        arrays["qacc"] =
                VectorArrayView(joint_acceleration_, {EnvDim(), JointDim()}, owner, false);
        arrays["torques"] =
                VectorArrayView(joint_torque_, {EnvDim(), JointDim()}, owner, false);
        arrays["joint_lower_limit"] =
                VectorArrayView(joint_lower_limit_, {JointDim()}, owner, false);
        arrays["joint_upper_limit"] =
                VectorArrayView(joint_upper_limit_, {JointDim()}, owner, false);
        arrays["foot_position"] =
                VectorArrayView(foot_position_, {EnvDim(), FootDim(), 3}, owner, false);
        arrays["feet_pos"] =
                VectorArrayView(foot_position_, {EnvDim(), FootDim(), 3}, owner, false);
        arrays["feet_quat"] =
                VectorArrayView(foot_quaternion_, {EnvDim(), FootDim(), 4}, owner, false);
        arrays["foot_velocity"] =
                VectorArrayView(foot_velocity_, {EnvDim(), FootDim(), 3}, owner, false);
        arrays["foot_height"] =
                VectorArrayView(foot_height_, {EnvDim(), FootDim()}, owner, false);
        arrays["foot_contact"] =
                VectorArrayView(foot_contact_, {EnvDim(), FootDim()}, owner, false);
        arrays["feet_contact"] =
                VectorArrayView(foot_contact_, {EnvDim(), FootDim()}, owner, false);
        arrays["foot_contact_force"] =
                VectorArrayView(foot_contact_force_, {EnvDim(), FootDim(), 3}, owner, false);
        arrays["height_scan"] =
                VectorArrayView(height_scan_, {EnvDim(), HeightScanDim()}, owner, false);
        arrays["height_scan_hit"] =
                VectorArrayView(height_scan_hit_, {EnvDim(), HeightScanDim()}, owner, false);
        arrays["height_scan_point"] =
                VectorArrayView(height_scan_point_, {EnvDim(), HeightScanDim(), 3}, owner, false);
        arrays["height_scan_normal"] =
                VectorArrayView(height_scan_normal_, {EnvDim(), HeightScanDim(), 3}, owner, false);
        arrays["illegal_contact_count"] =
                VectorArrayView(illegal_contact_count_, {EnvDim()}, owner, false);
        arrays["self_collision_count"] =
                VectorArrayView(self_collision_count_, {EnvDim()}, owner, false);
        arrays["shank_collision_count"] =
                VectorArrayView(shank_collision_count_, {EnvDim()}, owner, false);
        arrays["trunk_head_collision_count"] =
                VectorArrayView(trunk_head_collision_count_, {EnvDim()}, owner, false);
        arrays["foot_air_time"] =
                VectorArrayView(foot_air_time_, {EnvDim(), FootDim()}, owner, true);
        arrays["foot_peak_height"] =
                VectorArrayView(foot_peak_height_, {EnvDim(), FootDim()}, owner, true);
        arrays["last_foot_contact"] =
                VectorArrayView(last_foot_contact_, {EnvDim(), FootDim()}, owner, true);
        arrays["first_contact"] =
                VectorArrayView(first_contact_, {EnvDim(), FootDim()}, owner, false);
        arrays["landing_force"] =
                VectorArrayView(landing_force_, {EnvDim(), FootDim()}, owner, false);
        arrays["previous_foot_position"] =
                VectorArrayView(previous_foot_position_, {EnvDim(), FootDim(), 3}, owner, true);
        arrays["reward"] =
                VectorArrayView(reward_, {EnvDim()}, owner, true);
        arrays["terminated"] =
                VectorArrayView(terminated_, {EnvDim()}, owner, true);
        arrays["base_clearance"] =
                VectorArrayView(base_clearance_, {EnvDim()}, owner, true);
        arrays["velocity_error"] =
                VectorArrayView(velocity_error_, {EnvDim()}, owner, true);
        arrays["foot_slip"] =
                VectorArrayView(foot_slip_, {EnvDim()}, owner, true);
        arrays["terrain_normal_error"] =
                VectorArrayView(terrain_normal_error_, {EnvDim()}, owner, true);
        arrays["reward_terms"] =
                VectorArrayView(reward_terms_, {EnvDim(), RewardTermDim()}, owner, true);
        arrays["actor_obs"] =
                VectorArrayView(actor_obs_, {EnvDim(), ActorObsDim()}, owner, true);
        arrays["critic_obs"] =
                VectorArrayView(critic_obs_, {EnvDim(), CriticObsDim()}, owner, true);
        return arrays;
    }

    void StepActions(std::uint64_t ticks, std::size_t workers, bool simulate_action_latency) {
#ifdef GOBOT_HAS_MUJOCO
        PrepareActionTargets(simulate_action_latency);
        Step(ticks, workers);
#else
        GOB_UNUSED(ticks);
        GOB_UNUSED(workers);
        GOB_UNUSED(simulate_action_latency);
        throw std::runtime_error("Gobot was built without MuJoCo support");
#endif
    }

    void Step(std::uint64_t ticks, std::size_t workers) {
#ifdef GOBOT_HAS_MUJOCO
        ApplyTargetPositions();
        const RealType fixed_time_step = world_->GetSettings().fixed_time_step;
        if (!world_->StepEnvironmentBatchInternal(fixed_time_step, ticks, workers, false, false)) {
            throw std::runtime_error(world_->GetLastError());
        }
        FillAllEnvironments(workers);
#else
        GOB_UNUSED(ticks);
        GOB_UNUSED(workers);
        throw std::runtime_error("Gobot was built without MuJoCo support");
#endif
    }

    void ConfigureCommand(double step_dt,
                          double resampling_time_min,
                          double resampling_time_max,
                          double lin_vel_x_min,
                          double lin_vel_x_max,
                          double lin_vel_y_min,
                          double lin_vel_y_max,
                          double ang_vel_z_min,
                          double ang_vel_z_max,
                          double heading_min,
                          double heading_max,
                          double rel_standing_envs,
                          double rel_heading_envs,
                          double rel_world_envs,
                          double rel_forward_envs,
                          bool heading_command,
                          double heading_control_stiffness,
                          std::uint64_t seed) {
        command_step_dt_ = static_cast<float>(std::max(step_dt, 1.0e-9));
        command_resampling_min_ = static_cast<float>(std::max(0.0, resampling_time_min));
        command_resampling_max_ = static_cast<float>(std::max(resampling_time_max, resampling_time_min));
        command_rel_standing_envs_ = static_cast<float>(std::clamp(rel_standing_envs, 0.0, 1.0));
        command_rel_heading_envs_ = static_cast<float>(std::clamp(rel_heading_envs, 0.0, 1.0));
        command_rel_world_envs_ = static_cast<float>(std::clamp(rel_world_envs, 0.0, 1.0));
        command_rel_forward_envs_ = static_cast<float>(std::clamp(rel_forward_envs, 0.0, 1.0));
        command_heading_enabled_ = heading_command;
        command_heading_stiffness_ = static_cast<float>(heading_control_stiffness);
        command_rng_.seed(seed);
        command_ranges_[kCommandLinVelXMin] = static_cast<float>(lin_vel_x_min);
        command_ranges_[kCommandLinVelXMax] = static_cast<float>(lin_vel_x_max);
        command_ranges_[kCommandLinVelYMin] = static_cast<float>(lin_vel_y_min);
        command_ranges_[kCommandLinVelYMax] = static_cast<float>(lin_vel_y_max);
        command_ranges_[kCommandAngVelZMin] = static_cast<float>(ang_vel_z_min);
        command_ranges_[kCommandAngVelZMax] = static_cast<float>(ang_vel_z_max);
        command_ranges_[kCommandHeadingMin] = static_cast<float>(heading_min);
        command_ranges_[kCommandHeadingMax] = static_cast<float>(heading_max);
        command_configured_ = true;
    }

    void SetCommandRanges(double lin_vel_x_min,
                          double lin_vel_x_max,
                          double lin_vel_y_min,
                          double lin_vel_y_max,
                          double ang_vel_z_min,
                          double ang_vel_z_max) {
        command_ranges_[kCommandLinVelXMin] = static_cast<float>(lin_vel_x_min);
        command_ranges_[kCommandLinVelXMax] = static_cast<float>(lin_vel_x_max);
        command_ranges_[kCommandLinVelYMin] = static_cast<float>(lin_vel_y_min);
        command_ranges_[kCommandLinVelYMax] = static_cast<float>(lin_vel_y_max);
        command_ranges_[kCommandAngVelZMin] = static_cast<float>(ang_vel_z_min);
        command_ranges_[kCommandAngVelZMax] = static_cast<float>(ang_vel_z_max);
    }

    void ResetCommands(const std::vector<std::size_t>& env_ids) {
        if (!command_configured_ || env_ids.empty()) {
            return;
        }
        for (std::size_t env_id : env_ids) {
            RequireEnvironmentIndex(env_id);
            ResampleCommand(env_id);
            command_time_left_[env_id] = Uniform(command_resampling_min_, command_resampling_max_);
        }
        UpdateCommands();
    }

    void StepTaskInputs(std::uint64_t ticks, std::size_t workers, bool simulate_action_latency) {
#ifdef GOBOT_HAS_MUJOCO
        ResetStepProfile();
        const TimePoint total_begin = Clock::now();
        TimePoint phase_begin = Clock::now();
        PrepareActionTargets(simulate_action_latency);
        SetProfileMs(kStepProfilePrepareAction, ElapsedMs(phase_begin));

        phase_begin = Clock::now();
        if (command_configured_) {
            AdvanceCommandTimersAndResample();
        }
        AddProfileMs(kStepProfileCommand, ElapsedMs(phase_begin));

        auto* model = static_cast<mjModel*>(world_->model_);
        if (model == nullptr) {
            throw std::runtime_error("MuJoCo model has not been built");
        }
        model->opt.timestep = world_->GetSettings().fixed_time_step;

        const std::size_t resolved_workers = ResolveViewWorkers(workers);
        std::vector<std::array<double, kStepProfileCount>> worker_profiles(resolved_workers);
        for (auto& profile : worker_profiles) {
            profile.fill(0.0);
        }

        ForEachEnvironmentWithWorker(resolved_workers, [&](std::size_t env_id, std::size_t worker_index) {
            auto& profile = worker_profiles[std::min(worker_index, worker_profiles.size() - 1)];
            TimePoint begin = Clock::now();
            ApplyTargetPositionsForEnvironment(*model, env_id);
            profile[kStepProfileApplyControl] += ElapsedMs(begin);

            auto* data = static_cast<mjData*>(world_->environment_data_[env_id]);
            if (data == nullptr) {
                throw std::runtime_error(fmt::format("MuJoCo data for environment {} is not available", env_id));
            }
            begin = Clock::now();
            for (std::uint64_t tick = 0; tick < ticks; ++tick) {
                mj_step(model, data);
            }
            profile[kStepProfileMjStep] += ElapsedMs(begin);

            begin = Clock::now();
            FillEnvironment(env_id);
            profile[kStepProfileExtractState] += ElapsedMs(begin);

            begin = Clock::now();
            if (command_configured_) {
                UpdateCommandForEnvironment(env_id);
            }
            UpdateFootHistory(env_id);
            profile[kStepProfileCommand] += ElapsedMs(begin);
        });

        for (std::size_t profile_index = kStepProfileApplyControl; profile_index < kStepProfileCount; ++profile_index) {
            double critical_path_ms = 0.0;
            for (const auto& worker_profile : worker_profiles) {
                critical_path_ms = std::max(critical_path_ms, worker_profile[profile_index]);
            }
            AddProfileMs(profile_index, critical_path_ms);
        }
        SetProfileMs(kStepProfileTotal, ElapsedMs(total_begin));
#else
        GOB_UNUSED(ticks);
        GOB_UNUSED(workers);
        GOB_UNUSED(simulate_action_latency);
        throw std::runtime_error("Gobot was built without MuJoCo support");
#endif
    }

    void Refresh() {
#ifdef GOBOT_HAS_MUJOCO
        FillAllEnvironments();
#else
        throw std::runtime_error("Gobot was built without MuJoCo support");
#endif
    }

    void Reset(const std::vector<std::size_t>& env_ids) {
        if (env_ids.empty()) {
            return;
        }
        std::vector<PhysicsEnvironmentRobotResetState> reset_states;
        reset_states.reserve(env_ids.size());
        for (std::size_t reset_index = 0; reset_index < env_ids.size(); ++reset_index) {
            const std::size_t env_id = env_ids[reset_index];
            RequireEnvironmentIndex(env_id);
            PhysicsEnvironmentRobotResetState reset_state;
            reset_state.environment_index = env_id;
            reset_state.robot_name = robot_name_;
            reset_state.base_link_name = base_link_;
            reset_state.base_position = Vector3(reset_base_position_[env_id * 3 + 0],
                                                reset_base_position_[env_id * 3 + 1],
                                                reset_base_position_[env_id * 3 + 2]);
            reset_state.base_orientation = Quaternion(reset_base_quaternion_[env_id * 4 + 0],
                                                      reset_base_quaternion_[env_id * 4 + 1],
                                                      reset_base_quaternion_[env_id * 4 + 2],
                                                      reset_base_quaternion_[env_id * 4 + 3]);
            reset_state.base_linear_velocity = Vector3(reset_base_linear_velocity_[env_id * 3 + 0],
                                                       reset_base_linear_velocity_[env_id * 3 + 1],
                                                       reset_base_linear_velocity_[env_id * 3 + 2]);
            reset_state.base_angular_velocity = Vector3(reset_base_angular_velocity_[env_id * 3 + 0],
                                                        reset_base_angular_velocity_[env_id * 3 + 1],
                                                        reset_base_angular_velocity_[env_id * 3 + 2]);
            reset_state.joint_names = joint_names_;
            const std::size_t joint_offset = env_id * joint_count_;
            reset_state.joint_positions.assign(reset_joint_position_.begin() + static_cast<std::ptrdiff_t>(joint_offset),
                                               reset_joint_position_.begin() +
                                                       static_cast<std::ptrdiff_t>(joint_offset + joint_count_));
            reset_state.joint_velocities.assign(reset_joint_velocity_.begin() + static_cast<std::ptrdiff_t>(joint_offset),
                                                reset_joint_velocity_.begin() +
                                                        static_cast<std::ptrdiff_t>(joint_offset + joint_count_));
            reset_state.joint_position_targets.assign(target_position_.begin() + static_cast<std::ptrdiff_t>(joint_offset),
                                                      target_position_.begin() +
                                                              static_cast<std::ptrdiff_t>(joint_offset + joint_count_));
            reset_states.emplace_back(std::move(reset_state));
        }
        if (!world_->ResetEnvironmentRobotStates(reset_states)) {
            throw std::runtime_error(world_->GetLastError());
        }
        for (std::size_t env_id : env_ids) {
            FillEnvironment(env_id);
        }
    }

    void SetBaseVelocity(std::size_t env_id,
                         py::array_t<float, py::array::c_style | py::array::forcecast> linear_velocity,
                         py::array_t<float, py::array::c_style | py::array::forcecast> angular_velocity) {
#ifdef GOBOT_HAS_MUJOCO
        RequireEnvironmentIndex(env_id);
        if (base_free_joint_binding_index_ >= world_->joint_bindings_.size()) {
            throw std::runtime_error("Go1 base free joint binding is not available");
        }
        auto linear_buffer = linear_velocity.request();
        auto angular_buffer = angular_velocity.request();
        if (linear_buffer.ndim != 1 || linear_buffer.shape[0] != 3 ||
            angular_buffer.ndim != 1 || angular_buffer.shape[0] != 3) {
            throw std::invalid_argument("linear_velocity and angular_velocity must have shape [3]");
        }
        const auto* linear = static_cast<const float*>(linear_buffer.ptr);
        const auto* angular = static_cast<const float*>(angular_buffer.ptr);
        SetBaseVelocityForEnvironment(env_id, linear, angular);
        FillEnvironment(env_id);
#else
        GOB_UNUSED(env_id);
        GOB_UNUSED(linear_velocity);
        GOB_UNUSED(angular_velocity);
        throw std::runtime_error("Gobot was built without MuJoCo support");
#endif
    }

    void SetBaseVelocities(const std::vector<std::size_t>& env_ids,
                           py::array_t<float, py::array::c_style | py::array::forcecast> linear_velocities,
                           py::array_t<float, py::array::c_style | py::array::forcecast> angular_velocities) {
#ifdef GOBOT_HAS_MUJOCO
        if (base_free_joint_binding_index_ >= world_->joint_bindings_.size()) {
            throw std::runtime_error("Go1 base free joint binding is not available");
        }
        auto linear_buffer = linear_velocities.request();
        auto angular_buffer = angular_velocities.request();
        const std::size_t count = env_ids.size();
        if (linear_buffer.ndim != 2 || angular_buffer.ndim != 2 ||
            linear_buffer.shape[0] != static_cast<py::ssize_t>(count) ||
            angular_buffer.shape[0] != static_cast<py::ssize_t>(count) ||
            linear_buffer.shape[1] != 3 || angular_buffer.shape[1] != 3) {
            throw std::invalid_argument("linear_velocities and angular_velocities must have shape [len(env_ids), 3]");
        }
        const auto* linear = static_cast<const float*>(linear_buffer.ptr);
        const auto* angular = static_cast<const float*>(angular_buffer.ptr);
        for (std::size_t row = 0; row < count; ++row) {
            const std::size_t env_id = env_ids[row];
            RequireEnvironmentIndex(env_id);
            SetBaseVelocityForEnvironment(env_id, linear + row * 3, angular + row * 3);
            FillEnvironment(env_id);
        }
#else
        GOB_UNUSED(env_ids);
        GOB_UNUSED(linear_velocities);
        GOB_UNUSED(angular_velocities);
        throw std::runtime_error("Gobot was built without MuJoCo support");
#endif
    }

private:
    struct SensorComponentView {
        int address{-1};
        int dimension{0};
        std::size_t value_offset{0};
    };

    struct SensorView {
        const PhysicsSensorSnapshot* snapshot{nullptr};
        std::size_t link_binding_index{0};
        std::vector<SensorComponentView> components;
    };

    py::ssize_t EnvDim() const {
        return static_cast<py::ssize_t>(environment_count_);
    }

    py::ssize_t JointDim() const {
        return static_cast<py::ssize_t>(joint_count_);
    }

    py::ssize_t FootDim() const {
        return static_cast<py::ssize_t>(foot_count_);
    }

    py::ssize_t HeightScanDim() const {
        return static_cast<py::ssize_t>(height_scan_count_);
    }

    py::ssize_t RewardTermDim() const {
        return static_cast<py::ssize_t>(reward_term_count_);
    }

    py::ssize_t TaskParamDim() const {
        return static_cast<py::ssize_t>(task_param_count_);
    }

    py::ssize_t TaskFlagDim() const {
        return static_cast<py::ssize_t>(task_flag_count_);
    }

    py::ssize_t CommandRangeDim() const {
        return static_cast<py::ssize_t>(kCommandRangeCount);
    }

    py::ssize_t StepProfileDim() const {
        return static_cast<py::ssize_t>(kStepProfileCount);
    }

    py::ssize_t ActorObsDim() const {
        return static_cast<py::ssize_t>(actor_obs_dim_);
    }

    py::ssize_t CriticObsDim() const {
        return static_cast<py::ssize_t>(critic_obs_dim_);
    }

    void RequireEnvironmentIndex(std::size_t env_id) const {
        if (env_id >= environment_count_) {
            throw std::out_of_range(fmt::format("environment index {} is out of range for {} environments",
                                                env_id,
                                                environment_count_));
        }
    }

    void Initialize() {
#ifndef GOBOT_HAS_MUJOCO
        throw std::runtime_error("Gobot was built without MuJoCo support");
#else
        if (!world_.IsValid()) {
            throw std::runtime_error("MuJoCo world is not available");
        }
        environment_count_ = world_->GetEnvironmentCount();
        if (environment_count_ == 0) {
            throw std::runtime_error("MuJoCo environment batch has not been configured");
        }
        auto* model = static_cast<mjModel*>(world_->model_);
        if (model == nullptr) {
            throw std::runtime_error("MuJoCo model has not been built");
        }
        const PhysicsSceneSnapshot& snapshot = world_->GetSceneSnapshot();
        const PhysicsRobotSnapshot* robot_snapshot = nullptr;
        for (std::size_t robot_index = 0; robot_index < snapshot.robots.size(); ++robot_index) {
            if (snapshot.robots[robot_index].name == robot_name_) {
                robot_index_ = robot_index;
                robot_snapshot = &snapshot.robots[robot_index];
                break;
            }
        }
        if (robot_snapshot == nullptr) {
            throw std::runtime_error(fmt::format("robot '{}' is not available in MuJoCo scene", robot_name_));
        }

        base_link_binding_index_ = FindLinkBindingIndex(base_link_);
        joint_binding_indices_.reserve(joint_names_.size());
        for (const std::string& joint_name : joint_names_) {
            joint_binding_indices_.push_back(FindJointBindingIndex(joint_name));
        }
        joint_count_ = joint_binding_indices_.size();

        base_free_joint_binding_index_ = world_->joint_bindings_.size();
        for (std::size_t binding_index = 0; binding_index < world_->joint_bindings_.size(); ++binding_index) {
            const auto& binding = world_->joint_bindings_[binding_index];
            if (binding.robot_index == robot_index_ && binding.joint_type == mjJNT_FREE) {
                base_free_joint_binding_index_ = binding_index;
                break;
            }
        }

        foot_link_binding_indices_.reserve(foot_link_names_.size());
        for (const std::string& foot_link_name : foot_link_names_) {
            foot_link_binding_indices_.push_back(FindLinkBindingIndex(foot_link_name));
        }
        foot_count_ = foot_link_binding_indices_.size();

        foot_height_sensors_.reserve(foot_height_sensor_names_.size());
        for (const std::string& sensor_name : foot_height_sensor_names_) {
            foot_height_sensors_.push_back(FindSensorView(sensor_name));
        }
        foot_contact_sensors_.reserve(foot_contact_sensor_names_.size());
        for (const std::string& sensor_name : foot_contact_sensor_names_) {
            foot_contact_sensors_.push_back(FindSensorView(sensor_name));
        }
        if (!height_scan_sensor_.empty()) {
            height_scan_sensor_view_ = FindSensorView(height_scan_sensor_);
            height_scan_count_ = height_scan_sensor_view_.snapshot != nullptr
                                         ? height_scan_sensor_view_.snapshot->sample_offsets.size()
                                         : 0;
        }

        BuildBodyContactMaps(*robot_snapshot, *model);
#endif
    }

#ifdef GOBOT_HAS_MUJOCO
    void SetBaseVelocityForEnvironment(std::size_t env_id, const float* linear, const float* angular) {
        auto* model = static_cast<mjModel*>(world_->model_);
        auto* data = static_cast<mjData*>(world_->environment_data_[env_id]);
        const auto& binding = world_->joint_bindings_[base_free_joint_binding_index_];
        if (binding.dof_address < 0 || binding.dof_address + 5 >= model->nv) {
            throw std::runtime_error("Go1 base free joint velocity address is invalid");
        }
        data->qvel[binding.dof_address + 0] = linear[0];
        data->qvel[binding.dof_address + 1] = linear[1];
        data->qvel[binding.dof_address + 2] = linear[2];
        data->qvel[binding.dof_address + 3] = angular[0];
        data->qvel[binding.dof_address + 4] = angular[1];
        data->qvel[binding.dof_address + 5] = angular[2];
        mj_forward(model, data);
    }

    std::size_t FindJointBindingIndex(const std::string& joint_name) const {
        const PhysicsSceneSnapshot& snapshot = world_->GetSceneSnapshot();
        for (std::size_t binding_index = 0; binding_index < world_->joint_bindings_.size(); ++binding_index) {
            const auto& binding = world_->joint_bindings_[binding_index];
            if (binding.robot_index != robot_index_) {
                continue;
            }
            const PhysicsJointSnapshot& joint = snapshot.robots[binding.robot_index].joints[binding.joint_index];
            if (joint.name == joint_name) {
                return binding_index;
            }
        }
        throw std::runtime_error(fmt::format("joint '{}' is not bound in MuJoCo world", joint_name));
    }

    std::size_t FindLinkBindingIndex(const std::string& link_name) const {
        const PhysicsSceneSnapshot& snapshot = world_->GetSceneSnapshot();
        for (std::size_t binding_index = 0; binding_index < world_->link_bindings_.size(); ++binding_index) {
            const auto& binding = world_->link_bindings_[binding_index];
            if (binding.robot_index != robot_index_) {
                continue;
            }
            const PhysicsLinkSnapshot& link = snapshot.robots[binding.robot_index].links[binding.link_index];
            if (link.name == link_name) {
                return binding_index;
            }
        }
        throw std::runtime_error(fmt::format("link '{}' is not bound in MuJoCo world", link_name));
    }

    SensorView FindSensorView(const std::string& sensor_name) const {
        auto* model = static_cast<mjModel*>(world_->model_);
        const PhysicsSceneSnapshot& snapshot = world_->GetSceneSnapshot();
        const PhysicsRobotSnapshot& robot = snapshot.robots[robot_index_];
        std::size_t sensor_index = robot.sensors.size();
        for (std::size_t index = 0; index < robot.sensors.size(); ++index) {
            if (robot.sensors[index].name == sensor_name) {
                sensor_index = index;
                break;
            }
        }
        if (sensor_index >= robot.sensors.size()) {
            throw std::runtime_error(fmt::format("sensor '{}' is not available on robot '{}'",
                                                 sensor_name,
                                                 robot_name_));
        }
        SensorView view;
        view.snapshot = &robot.sensors[sensor_index];
        view.link_binding_index = FindLinkBindingIndex(view.snapshot->link_name);
        for (const auto& binding : world_->sensor_bindings_) {
            if (binding.robot_index != robot_index_ || binding.sensor_index != sensor_index) {
                continue;
            }
            for (const auto& component : binding.components) {
                if (component.sensor_id < 0 || component.sensor_id >= model->nsensor) {
                    continue;
                }
                SensorComponentView component_view;
                component_view.address = model->sensor_adr[component.sensor_id];
                component_view.dimension = model->sensor_dim[component.sensor_id];
                component_view.value_offset = component.value_offset;
                view.components.push_back(component_view);
            }
        }
        return view;
    }

    void BuildBodyContactMaps(const PhysicsRobotSnapshot& robot, const mjModel& model) {
        body_is_robot_.assign(static_cast<std::size_t>(model.nbody), 0);
        body_foot_index_.assign(static_cast<std::size_t>(model.nbody), -1);
        body_link_name_.assign(static_cast<std::size_t>(model.nbody), std::string());
        body_is_thigh_.assign(static_cast<std::size_t>(model.nbody), 0);
        body_is_shank_.assign(static_cast<std::size_t>(model.nbody), 0);
        body_is_trunk_head_.assign(static_cast<std::size_t>(model.nbody), 0);

        for (const auto& binding : world_->link_bindings_) {
            if (binding.robot_index != robot_index_ || binding.body_id < 0 || binding.body_id >= model.nbody) {
                continue;
            }
            const PhysicsLinkSnapshot& link = robot.links[binding.link_index];
            const auto body_index = static_cast<std::size_t>(binding.body_id);
            body_is_robot_[body_index] = 1;
            body_link_name_[body_index] = link.name;
            body_is_thigh_[body_index] = MatchesAnyPattern(link.name, thigh_link_patterns_) ? 1 : 0;
            body_is_shank_[body_index] = MatchesAnyPattern(link.name, shank_link_patterns_) ? 1 : 0;
            body_is_trunk_head_[body_index] = MatchesAnyPattern(link.name, trunk_head_link_patterns_) ? 1 : 0;
        }

        for (std::size_t foot_index = 0; foot_index < foot_link_binding_indices_.size(); ++foot_index) {
            const auto& binding = world_->link_bindings_[foot_link_binding_indices_[foot_index]];
            if (binding.body_id >= 0 && binding.body_id < model.nbody) {
                body_foot_index_[static_cast<std::size_t>(binding.body_id)] = static_cast<int>(foot_index);
            }
        }
    }

    Affine3 BodyTransform(const mjData& data, int body_id) const {
        Affine3 transform = Affine3::Identity();
        transform.translation() = Vector3(data.xpos[3 * body_id + 0],
                                          data.xpos[3 * body_id + 1],
                                          data.xpos[3 * body_id + 2]);
        transform.linear() << data.xmat[9 * body_id + 0], data.xmat[9 * body_id + 1], data.xmat[9 * body_id + 2],
                data.xmat[9 * body_id + 3], data.xmat[9 * body_id + 4], data.xmat[9 * body_id + 5],
                data.xmat[9 * body_id + 6], data.xmat[9 * body_id + 7], data.xmat[9 * body_id + 8];
        return transform;
    }

    Vector3 BodyLinearVelocity(const mjData& data, int body_id) const {
        return Vector3(data.cvel[6 * body_id + 3], data.cvel[6 * body_id + 4], data.cvel[6 * body_id + 5]);
    }

    std::size_t ResolveViewWorkers(std::size_t workers) const {
        const std::size_t resolved = world_.IsValid()
                                             ? world_->ResolveEnvironmentBatchWorkerCount(workers)
                                             : workers;
        return std::max<std::size_t>(1, std::min(resolved, environment_count_));
    }

    using Clock = std::chrono::steady_clock;
    using TimePoint = Clock::time_point;
    using EnvironmentTask = std::function<void(std::size_t, std::size_t)>;

    static double ElapsedMs(TimePoint begin) {
        return std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
    }

    template <typename Func>
    void ForEachEnvironmentWithWorker(std::size_t workers, Func&& func) {
        const std::size_t resolved_workers = ResolveViewWorkers(workers);
        if (resolved_workers <= 1 || environment_count_ <= 1) {
            StopViewWorkers();
            for (std::size_t env_id = 0; env_id < environment_count_; ++env_id) {
                func(env_id, 0);
            }
            return;
        }
        EnsureViewWorkers(resolved_workers);
        auto task = EnvironmentTask([&func](std::size_t env_id, std::size_t worker_index) {
            func(env_id, worker_index);
        });
        RunViewWorkers(std::move(task), resolved_workers);
    }

    void EnsureViewWorkers(std::size_t worker_count) {
        if (worker_count <= 1) {
            StopViewWorkers();
            return;
        }
        if (view_workers_.size() == worker_count) {
            return;
        }
        StopViewWorkers();
        view_workers_.reserve(worker_count);
        std::size_t initial_generation = 0;
        {
            std::lock_guard<std::mutex> lock(view_worker_mutex_);
            initial_generation = view_worker_generation_;
        }
        for (std::size_t worker_index = 0; worker_index < worker_count; ++worker_index) {
            view_workers_.emplace_back(&PyLocomotionBatchView::ViewWorkerLoop, this, worker_index, initial_generation);
        }
    }

    void RunViewWorkers(EnvironmentTask task, std::size_t worker_count) {
        {
            std::lock_guard<std::mutex> lock(view_worker_mutex_);
            view_worker_task_ = std::move(task);
            view_worker_count_ = worker_count;
            view_worker_chunk_ = std::max<std::size_t>(1, environment_count_ / (worker_count * 8));
            view_worker_next_env_.store(0, std::memory_order_release);
            view_worker_completed_ = 0;
            view_worker_error_ = nullptr;
            ++view_worker_generation_;
        }
        view_worker_cv_.notify_all();
        std::unique_lock<std::mutex> lock(view_worker_mutex_);
        view_worker_done_cv_.wait(lock, [&]() {
            return view_worker_completed_ >= worker_count;
        });
        view_worker_task_ = nullptr;
        if (view_worker_error_ != nullptr) {
            std::rethrow_exception(view_worker_error_);
        }
    }

    void ViewWorkerLoop(std::size_t worker_index, std::size_t observed_generation) {
        while (true) {
            EnvironmentTask task;
            std::size_t generation = 0;
            {
                std::unique_lock<std::mutex> lock(view_worker_mutex_);
                view_worker_cv_.wait(lock, [&]() {
                    return view_worker_stop_ || view_worker_generation_ != observed_generation;
                });
                if (view_worker_stop_) {
                    return;
                }
                task = view_worker_task_;
                generation = view_worker_generation_;
            }

            try {
                while (true) {
                    {
                        std::lock_guard<std::mutex> lock(view_worker_mutex_);
                        if (view_worker_error_ != nullptr) {
                            break;
                        }
                    }
                    const std::size_t begin = view_worker_next_env_.fetch_add(view_worker_chunk_, std::memory_order_relaxed);
                    if (begin >= environment_count_) {
                        break;
                    }
                    const std::size_t end = std::min(begin + view_worker_chunk_, environment_count_);
                    for (std::size_t env_id = begin; env_id < end; ++env_id) {
                        task(env_id, worker_index);
                    }
                }
            } catch (...) {
                std::lock_guard<std::mutex> lock(view_worker_mutex_);
                if (view_worker_error_ == nullptr) {
                    view_worker_error_ = std::current_exception();
                }
            }

            {
                std::lock_guard<std::mutex> lock(view_worker_mutex_);
                observed_generation = generation;
                view_worker_completed_++;
                if (view_worker_completed_ >= view_worker_count_) {
                    view_worker_done_cv_.notify_one();
                }
            }
        }
    }

    void StopViewWorkers() {
        {
            std::lock_guard<std::mutex> lock(view_worker_mutex_);
            view_worker_stop_ = true;
            ++view_worker_generation_;
        }
        view_worker_cv_.notify_all();
        for (std::thread& worker : view_workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        view_workers_.clear();
        {
            std::lock_guard<std::mutex> lock(view_worker_mutex_);
            view_worker_stop_ = false;
            view_worker_task_ = nullptr;
            view_worker_count_ = 0;
            view_worker_completed_ = 0;
            view_worker_error_ = nullptr;
            view_worker_next_env_.store(0, std::memory_order_release);
        }
    }

    template <typename Func>
    void ForEachEnvironment(std::size_t workers, Func&& func) {
        ForEachEnvironmentWithWorker(workers, [&](std::size_t env_id, std::size_t) {
            func(env_id);
        });
    }

    void AllocateBuffers() {
        action_.assign(environment_count_ * joint_count_, 0.0f);
        submitted_action_.assign(environment_count_ * joint_count_, 0.0f);
        default_joint_position_.assign(joint_count_, 0.0f);
        action_scale_.assign(joint_count_, 0.0f);
        previous_action_.assign(environment_count_ * joint_count_, 0.0f);
        last_action_.assign(environment_count_ * joint_count_, 0.0f);
        encoder_bias_.assign(environment_count_ * joint_count_, 0.0f);
        command_.assign(environment_count_ * 3, 0.0f);
        command_world_.assign(environment_count_ * 3, 0.0f);
        command_heading_target_.assign(environment_count_, 0.0f);
        command_heading_error_.assign(environment_count_, 0.0f);
        command_time_left_.assign(environment_count_, 0.0f);
        command_is_heading_env_.assign(environment_count_, 0);
        command_is_standing_env_.assign(environment_count_, 0);
        command_is_world_env_.assign(environment_count_, 0);
        command_is_forward_env_.assign(environment_count_, 0);
        command_ranges_.assign(kCommandRangeCount, 0.0f);
        gait_phase_.assign(environment_count_ * 2, 0.0f);
        feet_phase_height_target_.assign(environment_count_ * 2, 0.0f);
        pose_weights_.assign(joint_count_, 1.0f);
        step_profile_ms_.assign(kStepProfileCount, 0.0f);
        pose_std_standing_.assign(joint_count_, 0.3f);
        pose_std_walking_.assign(joint_count_, 0.3f);
        pose_std_running_.assign(joint_count_, 0.3f);
        reward_weights_.assign(reward_term_count_, 0.0f);
        task_params_.assign(task_param_count_, 0.0f);
        task_flags_.assign(task_flag_count_, 0.0f);
        target_position_.assign(environment_count_ * joint_count_, 0.0f);
        reset_base_position_.assign(environment_count_ * 3, 0.0f);
        reset_base_quaternion_.assign(environment_count_ * 4, 0.0f);
        reset_base_linear_velocity_.assign(environment_count_ * 3, 0.0f);
        reset_base_angular_velocity_.assign(environment_count_ * 3, 0.0f);
        reset_joint_position_.assign(environment_count_ * joint_count_, 0.0f);
        reset_joint_velocity_.assign(environment_count_ * joint_count_, 0.0f);
        base_position_.assign(environment_count_ * 3, 0.0f);
        base_quaternion_.assign(environment_count_ * 4, 0.0f);
        base_linear_velocity_.assign(environment_count_ * 3, 0.0f);
        base_angular_velocity_.assign(environment_count_ * 3, 0.0f);
        base_linear_velocity_body_.assign(environment_count_ * 3, 0.0f);
        base_angular_velocity_body_.assign(environment_count_ * 3, 0.0f);
        projected_gravity_.assign(environment_count_ * 3, 0.0f);
        base_height_.assign(environment_count_, 0.0f);
        joint_position_.assign(environment_count_ * joint_count_, 0.0f);
        joint_velocity_.assign(environment_count_ * joint_count_, 0.0f);
        joint_acceleration_.assign(environment_count_ * joint_count_, 0.0f);
        joint_torque_.assign(environment_count_ * joint_count_, 0.0f);
        joint_lower_limit_.assign(joint_count_, 0.0f);
        joint_upper_limit_.assign(joint_count_, 0.0f);
        foot_position_.assign(environment_count_ * foot_count_ * 3, 0.0f);
        foot_quaternion_.assign(environment_count_ * foot_count_ * 4, 0.0f);
        foot_velocity_.assign(environment_count_ * foot_count_ * 3, 0.0f);
        foot_height_.assign(environment_count_ * foot_count_, 0.0f);
        foot_contact_.assign(environment_count_ * foot_count_, 0.0f);
        foot_contact_force_.assign(environment_count_ * foot_count_ * 3, 0.0f);
        height_scan_.assign(environment_count_ * height_scan_count_, 0.0f);
        height_scan_hit_.assign(environment_count_ * height_scan_count_, 0);
        height_scan_point_.assign(environment_count_ * height_scan_count_ * 3, 0.0f);
        height_scan_normal_.assign(environment_count_ * height_scan_count_ * 3, 0.0f);
        illegal_contact_count_.assign(environment_count_, 0.0f);
        self_collision_count_.assign(environment_count_, 0.0f);
        shank_collision_count_.assign(environment_count_, 0.0f);
        trunk_head_collision_count_.assign(environment_count_, 0.0f);
        foot_air_time_.assign(environment_count_ * foot_count_, 0.0f);
        foot_peak_height_.assign(environment_count_ * foot_count_, 0.0f);
        last_foot_contact_.assign(environment_count_ * foot_count_, 0.0f);
        first_contact_.assign(environment_count_ * foot_count_, 0.0f);
        landing_force_.assign(environment_count_ * foot_count_, 0.0f);
        previous_foot_position_.assign(environment_count_ * foot_count_ * 3, 0.0f);
        reward_.assign(environment_count_, 0.0f);
        terminated_.assign(environment_count_, 0);
        base_clearance_.assign(environment_count_, 0.0f);
        velocity_error_.assign(environment_count_, 0.0f);
        foot_slip_.assign(environment_count_, 0.0f);
        terrain_normal_error_.assign(environment_count_, 0.0f);
        reward_terms_.assign(environment_count_ * reward_term_count_, 0.0f);
        actor_obs_.assign(environment_count_ * actor_obs_dim_, 0.0f);
        critic_obs_.assign(environment_count_ * critic_obs_dim_, 0.0f);

        const PhysicsSceneSnapshot& snapshot = world_->GetSceneSnapshot();
        for (std::size_t joint_index = 0; joint_index < joint_binding_indices_.size(); ++joint_index) {
            const auto& binding = world_->joint_bindings_[joint_binding_indices_[joint_index]];
            const PhysicsJointSnapshot& joint = snapshot.robots[binding.robot_index].joints[binding.joint_index];
            joint_lower_limit_[joint_index] = static_cast<float>(joint.lower_limit);
            joint_upper_limit_[joint_index] = static_cast<float>(joint.upper_limit);
        }
    }

    void PrepareActionTargets(bool simulate_action_latency) {
        const float clip_limit = std::max(0.0f, action_clip_.empty() ? 1.0f : action_clip_[0]);
        for (std::size_t env_id = 0; env_id < environment_count_; ++env_id) {
            for (std::size_t joint_index = 0; joint_index < joint_count_; ++joint_index) {
                const std::size_t offset = env_id * joint_count_ + joint_index;
                const float clipped = std::clamp(action_[offset], -clip_limit, clip_limit);
                submitted_action_[offset] = clipped;
                const float control_action = simulate_action_latency ? last_action_[offset] : clipped;
                target_position_[offset] = default_joint_position_[joint_index] +
                                           action_scale_[joint_index] * control_action;
            }
        }
    }

    void ResetStepProfile() {
        std::fill(step_profile_ms_.begin(), step_profile_ms_.end(), 0.0f);
    }

    void AddProfileMs(std::size_t profile_index, double value_ms) {
        if (profile_index < step_profile_ms_.size()) {
            step_profile_ms_[profile_index] += static_cast<float>(value_ms);
        }
    }

    void SetProfileMs(std::size_t profile_index, double value_ms) {
        if (profile_index < step_profile_ms_.size()) {
            step_profile_ms_[profile_index] = static_cast<float>(value_ms);
        }
    }

    float Uniform(float lo, float hi) {
        if (hi < lo) {
            std::swap(lo, hi);
        }
        if (std::abs(hi - lo) <= 1.0e-9f) {
            return lo;
        }
        std::uniform_real_distribution<float> distribution(lo, hi);
        return distribution(command_rng_);
    }

    void ResampleCommand(std::size_t env_id) {
        const std::size_t env3 = env_id * 3;
        command_[env3 + 0] = Uniform(command_ranges_[kCommandLinVelXMin],
                                     command_ranges_[kCommandLinVelXMax]);
        command_[env3 + 1] = Uniform(command_ranges_[kCommandLinVelYMin],
                                     command_ranges_[kCommandLinVelYMax]);
        command_[env3 + 2] = Uniform(command_ranges_[kCommandAngVelZMin],
                                     command_ranges_[kCommandAngVelZMax]);
        command_heading_target_[env_id] = command_heading_enabled_
                                                  ? Uniform(command_ranges_[kCommandHeadingMin],
                                                            command_ranges_[kCommandHeadingMax])
                                                  : 0.0f;
        command_is_heading_env_[env_id] =
                command_heading_enabled_ && Uniform(0.0f, 1.0f) <= command_rel_heading_envs_ ? 1 : 0;
        command_is_standing_env_[env_id] = Uniform(0.0f, 1.0f) <= command_rel_standing_envs_ ? 1 : 0;
        command_is_world_env_[env_id] = Uniform(0.0f, 1.0f) <= command_rel_world_envs_ ? 1 : 0;
        command_world_[env3 + 0] = command_[env3 + 0];
        command_world_[env3 + 1] = command_[env3 + 1];
        command_world_[env3 + 2] = command_[env3 + 2];
        command_is_forward_env_[env_id] = Uniform(0.0f, 1.0f) <= command_rel_forward_envs_ ? 1 : 0;
        if (command_is_forward_env_[env_id] != 0) {
            command_[env3 + 0] = std::max(std::abs(command_[env3 + 0]), 0.3f);
            command_[env3 + 1] = 0.0f;
            command_[env3 + 2] = 0.0f;
        }
    }

    static float WrapToPi(float value) {
        constexpr float kPi = static_cast<float>(M_PI);
        constexpr float kTwoPi = static_cast<float>(2.0 * M_PI);
        value = std::fmod(value + kPi, kTwoPi);
        if (value < 0.0f) {
            value += kTwoPi;
        }
        return value - kPi;
    }

    void AdvanceCommandTimersAndResample() {
        for (std::size_t env_id = 0; env_id < environment_count_; ++env_id) {
            command_time_left_[env_id] -= command_step_dt_;
            if (command_time_left_[env_id] <= 0.0f) {
                ResampleCommand(env_id);
                command_time_left_[env_id] = Uniform(command_resampling_min_, command_resampling_max_);
            }
        }
    }

    void ComputeCommands() {
        AdvanceCommandTimersAndResample();
        UpdateCommands();
    }

    void UpdateCommands() {
        ForEachEnvironment(0, [this](std::size_t env_id) {
            UpdateCommandForEnvironment(env_id);
        });
    }

    void UpdateCommandForEnvironment(std::size_t env_id) {
        const std::size_t env3 = env_id * 3;
        const std::size_t env4 = env_id * 4;
        const float w = base_quaternion_[env4 + 0];
        const float x = base_quaternion_[env4 + 1];
        const float y = base_quaternion_[env4 + 2];
        const float z = base_quaternion_[env4 + 3];
        const float heading = std::atan2(2.0f * (w * z + x * y),
                                         1.0f - 2.0f * (y * y + z * z));
        if (command_heading_enabled_ && command_is_heading_env_[env_id] != 0) {
            const float error = WrapToPi(command_heading_target_[env_id] - heading);
            command_heading_error_[env_id] = error;
            command_[env3 + 2] = std::clamp(command_heading_stiffness_ * error,
                                            command_ranges_[kCommandAngVelZMin],
                                            command_ranges_[kCommandAngVelZMax]);
        }
        if (command_is_world_env_[env_id] != 0) {
            const float vx_w = command_world_[env3 + 0];
            const float vy_w = command_world_[env3 + 1];
            const float cos_h = std::cos(heading);
            const float sin_h = std::sin(heading);
            command_[env3 + 0] = cos_h * vx_w + sin_h * vy_w;
            command_[env3 + 1] = -sin_h * vx_w + cos_h * vy_w;
        }
        if (command_is_standing_env_[env_id] != 0) {
            command_[env3 + 0] = 0.0f;
            command_[env3 + 1] = 0.0f;
            command_[env3 + 2] = 0.0f;
            command_world_[env3 + 0] = 0.0f;
            command_world_[env3 + 1] = 0.0f;
            command_world_[env3 + 2] = 0.0f;
        }
    }

    void ApplyTargetPositions() {
        auto* model = static_cast<mjModel*>(world_->model_);
        if (model == nullptr) {
            throw std::runtime_error("MuJoCo model has not been built");
        }
        for (std::size_t env_id = 0; env_id < environment_count_; ++env_id) {
            ApplyTargetPositionsForEnvironment(*model, env_id);
        }
    }

    void ApplyTargetPositionsForEnvironment(const mjModel& model, std::size_t env_id) {
        auto* data = static_cast<mjData*>(world_->environment_data_[env_id]);
        if (data == nullptr) {
            throw std::runtime_error(fmt::format("MuJoCo data for environment {} is not available", env_id));
        }
        if (model.nv > 0) {
            mju_zero(data->qfrc_applied, model.nv);
        }
        if (model.nbody > 0) {
            mju_zero(data->xfrc_applied, 6 * model.nbody);
        }
        for (std::size_t joint_index = 0; joint_index < joint_count_; ++joint_index) {
            const auto& binding = world_->joint_bindings_[joint_binding_indices_[joint_index]];
            const int actuator_id = binding.position_actuator_id >= 0
                                            ? binding.position_actuator_id
                                            : binding.motor_actuator_id;
            if (actuator_id < 0 || actuator_id >= model.nu) {
                throw std::runtime_error(fmt::format("joint '{}' has no usable MuJoCo actuator",
                                                     joint_names_[joint_index]));
            }
            data->ctrl[actuator_id] =
                    static_cast<mjtNum>(target_position_[env_id * joint_count_ + joint_index]);
        }
    }

    void FillAllEnvironments(std::size_t workers = 0) {
        ForEachEnvironment(workers, [this](std::size_t env_id) {
            FillEnvironment(env_id);
        });
    }

    void FillEnvironment(std::size_t env_id) {
        auto* model = static_cast<mjModel*>(world_->model_);
        auto* data = static_cast<mjData*>(world_->environment_data_[env_id]);
        if (model == nullptr || data == nullptr) {
            throw std::runtime_error("MuJoCo model or data is not available");
        }

        const auto& base_binding = world_->link_bindings_[base_link_binding_index_];
        const int base_body = base_binding.body_id;
        const std::size_t env3 = env_id * 3;
        const std::size_t env4 = env_id * 4;
        base_position_[env3 + 0] = static_cast<float>(data->xpos[3 * base_body + 0]);
        base_position_[env3 + 1] = static_cast<float>(data->xpos[3 * base_body + 1]);
        base_position_[env3 + 2] = static_cast<float>(data->xpos[3 * base_body + 2]);
        base_height_[env_id] = base_position_[env3 + 2];
        Matrix3 rotation;
        rotation << data->xmat[9 * base_body + 0], data->xmat[9 * base_body + 1], data->xmat[9 * base_body + 2],
                data->xmat[9 * base_body + 3], data->xmat[9 * base_body + 4], data->xmat[9 * base_body + 5],
                data->xmat[9 * base_body + 6], data->xmat[9 * base_body + 7], data->xmat[9 * base_body + 8];
        const Quaternion base_orientation(rotation);
        base_quaternion_[env4 + 0] = static_cast<float>(base_orientation.w());
        base_quaternion_[env4 + 1] = static_cast<float>(base_orientation.x());
        base_quaternion_[env4 + 2] = static_cast<float>(base_orientation.y());
        base_quaternion_[env4 + 3] = static_cast<float>(base_orientation.z());
        base_angular_velocity_[env3 + 0] = static_cast<float>(data->cvel[6 * base_body + 0]);
        base_angular_velocity_[env3 + 1] = static_cast<float>(data->cvel[6 * base_body + 1]);
        base_angular_velocity_[env3 + 2] = static_cast<float>(data->cvel[6 * base_body + 2]);
        base_linear_velocity_[env3 + 0] = static_cast<float>(data->cvel[6 * base_body + 3]);
        base_linear_velocity_[env3 + 1] = static_cast<float>(data->cvel[6 * base_body + 4]);
        base_linear_velocity_[env3 + 2] = static_cast<float>(data->cvel[6 * base_body + 5]);
        const Vector3 base_linear_velocity_w(base_linear_velocity_[env3 + 0],
                                             base_linear_velocity_[env3 + 1],
                                             base_linear_velocity_[env3 + 2]);
        const Vector3 base_angular_velocity_w(base_angular_velocity_[env3 + 0],
                                              base_angular_velocity_[env3 + 1],
                                              base_angular_velocity_[env3 + 2]);
        const Vector3 base_linear_velocity_b = rotation.transpose() * base_linear_velocity_w;
        const Vector3 base_angular_velocity_b = rotation.transpose() * base_angular_velocity_w;
        const Vector3 gravity_b = rotation.transpose() * Vector3(0.0, 0.0, -1.0);
        base_linear_velocity_body_[env3 + 0] = static_cast<float>(base_linear_velocity_b.x());
        base_linear_velocity_body_[env3 + 1] = static_cast<float>(base_linear_velocity_b.y());
        base_linear_velocity_body_[env3 + 2] = static_cast<float>(base_linear_velocity_b.z());
        base_angular_velocity_body_[env3 + 0] = static_cast<float>(base_angular_velocity_b.x());
        base_angular_velocity_body_[env3 + 1] = static_cast<float>(base_angular_velocity_b.y());
        base_angular_velocity_body_[env3 + 2] = static_cast<float>(base_angular_velocity_b.z());
        projected_gravity_[env3 + 0] = static_cast<float>(gravity_b.x());
        projected_gravity_[env3 + 1] = static_cast<float>(gravity_b.y());
        projected_gravity_[env3 + 2] = static_cast<float>(gravity_b.z());

        for (std::size_t joint_index = 0; joint_index < joint_count_; ++joint_index) {
            const auto& binding = world_->joint_bindings_[joint_binding_indices_[joint_index]];
            const std::size_t offset = env_id * joint_count_ + joint_index;
            joint_position_[offset] = binding.qpos_address >= 0 && binding.qpos_address < model->nq
                                              ? static_cast<float>(data->qpos[binding.qpos_address])
                                              : 0.0f;
            joint_velocity_[offset] = binding.dof_address >= 0 && binding.dof_address < model->nv
                                              ? static_cast<float>(data->qvel[binding.dof_address])
                                              : 0.0f;
            joint_acceleration_[offset] = binding.dof_address >= 0 && binding.dof_address < model->nv
                                                  ? static_cast<float>(data->qacc[binding.dof_address])
                                                  : 0.0f;
            joint_torque_[offset] = binding.dof_address >= 0 && binding.dof_address < model->nv
                                            ? static_cast<float>(data->qfrc_actuator[binding.dof_address])
                                            : 0.0f;
        }

        for (std::size_t foot_index = 0; foot_index < foot_count_; ++foot_index) {
            const auto& binding = world_->link_bindings_[foot_link_binding_indices_[foot_index]];
            const int body_id = binding.body_id;
            const std::size_t foot3 = (env_id * foot_count_ + foot_index) * 3;
            foot_position_[foot3 + 0] = static_cast<float>(data->xpos[3 * body_id + 0]);
            foot_position_[foot3 + 1] = static_cast<float>(data->xpos[3 * body_id + 1]);
            foot_position_[foot3 + 2] = static_cast<float>(data->xpos[3 * body_id + 2]);
            Matrix3 foot_rotation;
            foot_rotation << data->xmat[9 * body_id + 0], data->xmat[9 * body_id + 1], data->xmat[9 * body_id + 2],
                    data->xmat[9 * body_id + 3], data->xmat[9 * body_id + 4], data->xmat[9 * body_id + 5],
                    data->xmat[9 * body_id + 6], data->xmat[9 * body_id + 7], data->xmat[9 * body_id + 8];
            const Quaternion foot_orientation(foot_rotation);
            const std::size_t foot4 = (env_id * foot_count_ + foot_index) * 4;
            foot_quaternion_[foot4 + 0] = static_cast<float>(foot_orientation.w());
            foot_quaternion_[foot4 + 1] = static_cast<float>(foot_orientation.x());
            foot_quaternion_[foot4 + 2] = static_cast<float>(foot_orientation.y());
            foot_quaternion_[foot4 + 3] = static_cast<float>(foot_orientation.z());
            const Vector3 foot_linear_velocity = BodyLinearVelocity(*data, body_id);
            foot_velocity_[foot3 + 0] = static_cast<float>(foot_linear_velocity.x());
            foot_velocity_[foot3 + 1] = static_cast<float>(foot_linear_velocity.y());
            foot_velocity_[foot3 + 2] = static_cast<float>(foot_linear_velocity.z());
            foot_contact_[env_id * foot_count_ + foot_index] = 0.0f;
            foot_contact_force_[foot3 + 0] = 0.0f;
            foot_contact_force_[foot3 + 1] = 0.0f;
            foot_contact_force_[foot3 + 2] = 0.0f;
        }

        FillFootSensors(env_id, *data);
        FillHeightScan(env_id, *data);
        FillContactSummary(env_id, *model, *data);
    }

    std::size_t DefaultActorObservationDim() const {
        return 12 + joint_count_ * 3 + height_scan_count_;
    }

    std::size_t DefaultCriticObservationDim() const {
        return DefaultActorObservationDim() + foot_count_ * 6;
    }

    void UpdateFootHistory(std::size_t env_id) {
        const float step_dt = std::max(command_step_dt_, 1.0e-6f);
        bool was_initialized = false;
        for (std::size_t foot_index = 0; foot_index < foot_count_; ++foot_index) {
            const std::size_t foot3 = (env_id * foot_count_ + foot_index) * 3;
            if (std::abs(previous_foot_position_[foot3 + 0]) > 0.0f ||
                std::abs(previous_foot_position_[foot3 + 1]) > 0.0f ||
                std::abs(previous_foot_position_[foot3 + 2]) > 0.0f) {
                was_initialized = true;
                break;
            }
        }
        for (std::size_t foot_index = 0; foot_index < foot_count_; ++foot_index) {
            const std::size_t foot = env_id * foot_count_ + foot_index;
            const std::size_t foot3 = foot * 3;
            if (!was_initialized) {
                previous_foot_position_[foot3 + 0] = foot_position_[foot3 + 0];
                previous_foot_position_[foot3 + 1] = foot_position_[foot3 + 1];
                previous_foot_position_[foot3 + 2] = foot_position_[foot3 + 2];
            }
            foot_velocity_[foot3 + 0] = (foot_position_[foot3 + 0] - previous_foot_position_[foot3 + 0]) / step_dt;
            foot_velocity_[foot3 + 1] = (foot_position_[foot3 + 1] - previous_foot_position_[foot3 + 1]) / step_dt;
            foot_velocity_[foot3 + 2] = (foot_position_[foot3 + 2] - previous_foot_position_[foot3 + 2]) / step_dt;
            previous_foot_position_[foot3 + 0] = foot_position_[foot3 + 0];
            previous_foot_position_[foot3 + 1] = foot_position_[foot3 + 1];
            previous_foot_position_[foot3 + 2] = foot_position_[foot3 + 2];

            const bool contact = foot_contact_[foot] > 0.0f;
            first_contact_[foot] = contact && last_foot_contact_[foot] <= 0.0f ? 1.0f : 0.0f;
            const float force_x = foot_contact_force_[foot3 + 0];
            const float force_y = foot_contact_force_[foot3 + 1];
            const float force_z = foot_contact_force_[foot3 + 2];
            landing_force_[foot] = std::sqrt(force_x * force_x + force_y * force_y + force_z * force_z) *
                                   first_contact_[foot];
            if (contact) {
                foot_air_time_[foot] = 0.0f;
            } else {
                foot_air_time_[foot] += step_dt;
            }
            if (!contact) {
                foot_peak_height_[foot] = std::max(foot_peak_height_[foot], foot_height_[foot]);
            }
        }
    }

    void FillFootSensors(std::size_t env_id, const mjData& data) {
        const std::size_t height_count = std::min(foot_count_, foot_height_sensors_.size());
        for (std::size_t foot_index = 0; foot_index < height_count; ++foot_index) {
            const std::size_t offset = env_id * foot_count_ + foot_index;
            foot_height_[offset] =
                    static_cast<float>(SampleRaySensor(env_id, data, foot_height_sensors_[foot_index], nullptr));
        }
        const std::size_t contact_count = std::min(foot_count_, foot_contact_sensors_.size());
        for (std::size_t foot_index = 0; foot_index < contact_count; ++foot_index) {
            const std::size_t offset = env_id * foot_count_ + foot_index;
            foot_contact_[offset] = ReadSensorValue(data, foot_contact_sensors_[foot_index]) > 1.0e-5 ? 1.0f : 0.0f;
        }
    }

    RealType ReadSensorValue(const mjData& data, const SensorView& sensor_view) const {
        for (const SensorComponentView& component : sensor_view.components) {
            if (component.address >= 0 && component.dimension > 0) {
                return static_cast<RealType>(data.sensordata[component.address]);
            }
        }
        return 0.0;
    }

    struct RaySensorOutputs {
        std::vector<float>* values{nullptr};
        std::vector<std::uint8_t>* hits{nullptr};
        std::vector<float>* points{nullptr};
        std::vector<float>* normals{nullptr};
        std::size_t base_index{0};
        std::size_t value_stride{1};
    };

    RealType SampleRaySensor(std::size_t env_id,
                             const mjData& data,
                             const SensorView& sensor_view,
                             RaySensorOutputs* outputs) {
        if (sensor_view.snapshot == nullptr || sensor_view.snapshot->sample_offsets.empty()) {
            return 0.0;
        }
        const auto& link_binding = world_->link_bindings_[sensor_view.link_binding_index];
        const Affine3 sensor_transform = BodyTransform(data, link_binding.body_id) *
                                         sensor_view.snapshot->local_transform;
        const Vector3 sensor_position = sensor_transform.translation();
        const Matrix3 alignment = PySensorRayAlignmentMatrix(sensor_transform,
                                                             sensor_view.snapshot->ray_alignment);
        const Vector3 ray_direction = PyResolveSensorRayDirection(*sensor_view.snapshot,
                                                                  alignment,
                                                                  sensor_transform);
        const bool reduce_values =
                (sensor_view.snapshot->type == PhysicsSensorType::TerrainHeight ||
                 sensor_view.snapshot->type == PhysicsSensorType::HeightScanner) &&
                sensor_view.snapshot->reduction_mode != RayReductionMode::None;
        std::vector<RealType> ray_values;
        if (reduce_values) {
            ray_values.reserve(sensor_view.snapshot->sample_offsets.size());
        }
        RealType first_value = 0.0;
        for (std::size_t sample_index = 0; sample_index < sensor_view.snapshot->sample_offsets.size(); ++sample_index) {
            const Vector3 origin = sensor_position + alignment * sensor_view.snapshot->sample_offsets[sample_index];
            const PhysicsRaycastHit hit = world_->RaycastTerrainWithMuJoCo({origin,
                                                                            ray_direction,
                                                                            sensor_view.snapshot->max_distance},
                                                                           env_id);
            const RealType value = (sensor_view.snapshot->type == PhysicsSensorType::TerrainHeight ||
                                    sensor_view.snapshot->type == PhysicsSensorType::HeightScanner)
                                           ? (hit.hit ? origin.z() - hit.point.z()
                                                      : sensor_view.snapshot->max_distance)
                                           : hit.distance;
            if (sample_index == 0) {
                first_value = value;
            }
            if (reduce_values) {
                ray_values.push_back(value);
            }
            if (outputs != nullptr && outputs->values != nullptr) {
                const std::size_t value_index = outputs->base_index + sample_index * outputs->value_stride;
                (*outputs->values)[value_index] = static_cast<float>(value);
            }
            if (outputs != nullptr && outputs->hits != nullptr) {
                (*outputs->hits)[outputs->base_index + sample_index] = hit.hit ? 1 : 0;
            }
            if (outputs != nullptr && outputs->points != nullptr) {
                const std::size_t point_index = (outputs->base_index + sample_index) * 3;
                (*outputs->points)[point_index + 0] = static_cast<float>(hit.point.x());
                (*outputs->points)[point_index + 1] = static_cast<float>(hit.point.y());
                (*outputs->points)[point_index + 2] = static_cast<float>(hit.point.z());
            }
            if (outputs != nullptr && outputs->normals != nullptr) {
                const std::size_t normal_index = (outputs->base_index + sample_index) * 3;
                (*outputs->normals)[normal_index + 0] = static_cast<float>(hit.normal.x());
                (*outputs->normals)[normal_index + 1] = static_cast<float>(hit.normal.y());
                (*outputs->normals)[normal_index + 2] = static_cast<float>(hit.normal.z());
            }
        }
        if (reduce_values) {
            return PyReduceSensorRayValues(ray_values, sensor_view.snapshot->reduction_mode);
        }
        return first_value;
    }

    void FillHeightScan(std::size_t env_id, const mjData& data) {
        if (height_scan_count_ == 0 || height_scan_sensor_view_.snapshot == nullptr) {
            return;
        }
        RaySensorOutputs outputs;
        outputs.values = &height_scan_;
        outputs.hits = &height_scan_hit_;
        outputs.points = &height_scan_point_;
        outputs.normals = &height_scan_normal_;
        outputs.base_index = env_id * height_scan_count_;
        SampleRaySensor(env_id, data, height_scan_sensor_view_, &outputs);
    }

    void FillContactSummary(std::size_t env_id, const mjModel& model, const mjData& data) {
        illegal_contact_count_[env_id] = 0.0f;
        self_collision_count_[env_id] = 0.0f;
        shank_collision_count_[env_id] = 0.0f;
        trunk_head_collision_count_[env_id] = 0.0f;

        for (int contact_index = 0; contact_index < data.ncon; ++contact_index) {
            const mjContact& contact = data.contact[contact_index];
            const int geom_a = contact.geom[0];
            const int geom_b = contact.geom[1];
            if (geom_a < 0 || geom_a >= model.ngeom || geom_b < 0 || geom_b >= model.ngeom) {
                continue;
            }
            const int body_a = model.geom_bodyid[geom_a];
            const int body_b = model.geom_bodyid[geom_b];
            const bool robot_a = IsRobotBody(body_a);
            const bool robot_b = IsRobotBody(body_b);
            if (!robot_a && !robot_b) {
                continue;
            }

            mjtNum force6[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
            mj_contactForce(&model, &data, contact_index, force6);
            const RealType normal_force = std::abs(static_cast<RealType>(force6[0]));
            const RealType force_threshold = robot_a && robot_b
                                                     ? self_collision_force_threshold_
                                                     : ground_force_threshold_;
            if (normal_force < force_threshold) {
                continue;
            }

            Vector3 force_local(force6[0], force6[1], force6[2]);
            Matrix3 contact_frame;
            contact_frame << contact.frame[0], contact.frame[1], contact.frame[2],
                    contact.frame[3], contact.frame[4], contact.frame[5],
                    contact.frame[6], contact.frame[7], contact.frame[8];
            const Vector3 force_world = contact_frame.transpose() * force_local;
            const Vector3 contact_position(contact.pos[0], contact.pos[1], contact.pos[2]);
            const int nearest_foot = NearestFootIndex(env_id, contact_position);
            if (nearest_foot >= 0) {
                const std::size_t foot3 = (env_id * foot_count_ + static_cast<std::size_t>(nearest_foot)) * 3;
                foot_contact_force_[foot3 + 0] += static_cast<float>(force_world.x());
                foot_contact_force_[foot3 + 1] += static_cast<float>(force_world.y());
                foot_contact_force_[foot3 + 2] += static_cast<float>(force_world.z());
                if (foot_contact_[env_id * foot_count_ + static_cast<std::size_t>(nearest_foot)] <= 0.0f) {
                    foot_contact_[env_id * foot_count_ + static_cast<std::size_t>(nearest_foot)] = 1.0f;
                }
            }

            if (robot_a && robot_b) {
                self_collision_count_[env_id] += 1.0f;
                continue;
            }

            const int robot_body = robot_a ? body_a : body_b;
            if (IsThighBody(robot_body) && terminate_on_thigh_contact_) {
                illegal_contact_count_[env_id] += 1.0f;
            }
            if (IsShankBody(robot_body) && nearest_foot < 0) {
                shank_collision_count_[env_id] += 1.0f;
            }
            if (IsTrunkHeadBody(robot_body)) {
                trunk_head_collision_count_[env_id] += 1.0f;
            }
        }
    }

    bool IsRobotBody(int body_id) const {
        return body_id >= 0 && static_cast<std::size_t>(body_id) < body_is_robot_.size() &&
               body_is_robot_[static_cast<std::size_t>(body_id)] != 0;
    }

    bool IsThighBody(int body_id) const {
        return body_id >= 0 && static_cast<std::size_t>(body_id) < body_is_thigh_.size() &&
               body_is_thigh_[static_cast<std::size_t>(body_id)] != 0;
    }

    bool IsShankBody(int body_id) const {
        return body_id >= 0 && static_cast<std::size_t>(body_id) < body_is_shank_.size() &&
               body_is_shank_[static_cast<std::size_t>(body_id)] != 0;
    }

    bool IsTrunkHeadBody(int body_id) const {
        return body_id >= 0 && static_cast<std::size_t>(body_id) < body_is_trunk_head_.size() &&
               body_is_trunk_head_[static_cast<std::size_t>(body_id)] != 0;
    }

    int NearestFootIndex(std::size_t env_id, const Vector3& contact_position) const {
        constexpr RealType kNearFootDistance = 0.18;
        RealType best_distance = std::numeric_limits<RealType>::max();
        int best_index = -1;
        for (std::size_t foot_index = 0; foot_index < foot_count_; ++foot_index) {
            const std::size_t foot3 = (env_id * foot_count_ + foot_index) * 3;
            const Vector3 foot_position(foot_position_[foot3 + 0],
                                        foot_position_[foot3 + 1],
                                        foot_position_[foot3 + 2]);
            const RealType distance = (foot_position - contact_position).norm();
            if (distance < best_distance) {
                best_distance = distance;
                best_index = static_cast<int>(foot_index);
            }
        }
        return best_distance <= kNearFootDistance ? best_index : -1;
    }
#endif

    Ref<MuJoCoPhysicsWorld> world_;
    std::string robot_name_;
    std::string base_link_;
    std::vector<std::string> joint_names_;
    std::vector<std::string> foot_link_names_;
    std::vector<std::string> foot_height_sensor_names_;
    std::vector<std::string> foot_contact_sensor_names_;
    std::string height_scan_sensor_;
    std::vector<std::regex> thigh_link_patterns_;
    std::vector<std::regex> shank_link_patterns_;
    std::vector<std::regex> trunk_head_link_patterns_;
    bool terminate_on_thigh_contact_{true};
    RealType ground_force_threshold_{50.0};
    RealType self_collision_force_threshold_{20.0};

    std::size_t robot_index_{0};
    std::size_t environment_count_{0};
    std::size_t joint_count_{0};
    std::size_t foot_count_{0};
    std::size_t height_scan_count_{0};
    std::size_t reward_term_count_{0};
    std::size_t task_param_count_{0};
    std::size_t task_flag_count_{0};
    std::size_t actor_obs_dim_{0};
    std::size_t critic_obs_dim_{0};
    std::size_t base_link_binding_index_{0};
    std::size_t base_free_joint_binding_index_{0};
    std::vector<std::size_t> joint_binding_indices_;
    std::vector<std::size_t> foot_link_binding_indices_;
    std::vector<SensorView> foot_height_sensors_;
    std::vector<SensorView> foot_contact_sensors_;
    SensorView height_scan_sensor_view_;
    std::vector<std::uint8_t> body_is_robot_;
    std::vector<int> body_foot_index_;
    std::vector<std::string> body_link_name_;
    std::vector<std::uint8_t> body_is_thigh_;
    std::vector<std::uint8_t> body_is_shank_;
    std::vector<std::uint8_t> body_is_trunk_head_;
    std::vector<float> action_;
    std::vector<float> submitted_action_;
    std::vector<float> default_joint_position_;
    std::vector<float> action_scale_;
    std::vector<float> action_clip_{1.0f};
    std::vector<float> previous_action_;
    std::vector<float> last_action_;
    std::vector<float> encoder_bias_;
    std::vector<float> command_;
    std::vector<float> command_world_;
    std::vector<float> command_heading_target_;
    std::vector<float> command_heading_error_;
    std::vector<float> command_time_left_;
    std::vector<std::uint8_t> command_is_heading_env_;
    std::vector<std::uint8_t> command_is_standing_env_;
    std::vector<std::uint8_t> command_is_world_env_;
    std::vector<std::uint8_t> command_is_forward_env_;
    std::vector<float> command_ranges_;
    std::vector<float> gait_phase_;
    std::vector<float> feet_phase_height_target_;
    std::vector<float> pose_weights_;
    std::vector<float> step_profile_ms_;
    std::mt19937 command_rng_{1};
    bool command_configured_{false};
    bool command_heading_enabled_{true};
    float command_step_dt_{0.02f};
    float command_resampling_min_{3.0f};
    float command_resampling_max_{8.0f};
    float command_rel_standing_envs_{0.1f};
    float command_rel_heading_envs_{0.3f};
    float command_rel_world_envs_{0.0f};
    float command_rel_forward_envs_{0.2f};
    float command_heading_stiffness_{0.5f};
    std::vector<float> pose_std_standing_;
    std::vector<float> pose_std_walking_;
    std::vector<float> pose_std_running_;
    std::vector<float> reward_weights_;
    std::vector<float> task_params_;
    std::vector<float> task_flags_;
    std::vector<float> target_position_;
    std::vector<float> reset_base_position_;
    std::vector<float> reset_base_quaternion_;
    std::vector<float> reset_base_linear_velocity_;
    std::vector<float> reset_base_angular_velocity_;
    std::vector<float> reset_joint_position_;
    std::vector<float> reset_joint_velocity_;
    std::vector<float> base_position_;
    std::vector<float> base_quaternion_;
    std::vector<float> base_linear_velocity_;
    std::vector<float> base_angular_velocity_;
    std::vector<float> base_linear_velocity_body_;
    std::vector<float> base_angular_velocity_body_;
    std::vector<float> projected_gravity_;
    std::vector<float> base_height_;
    std::vector<float> joint_position_;
    std::vector<float> joint_velocity_;
    std::vector<float> joint_acceleration_;
    std::vector<float> joint_torque_;
    std::vector<float> joint_lower_limit_;
    std::vector<float> joint_upper_limit_;
    std::vector<float> foot_position_;
    std::vector<float> foot_quaternion_;
    std::vector<float> foot_velocity_;
    std::vector<float> foot_height_;
    std::vector<float> foot_contact_;
    std::vector<float> foot_contact_force_;
    std::vector<float> height_scan_;
    std::vector<std::uint8_t> height_scan_hit_;
    std::vector<float> height_scan_point_;
    std::vector<float> height_scan_normal_;
    std::vector<float> illegal_contact_count_;
    std::vector<float> self_collision_count_;
    std::vector<float> shank_collision_count_;
    std::vector<float> trunk_head_collision_count_;
    std::vector<float> foot_air_time_;
    std::vector<float> foot_peak_height_;
    std::vector<float> last_foot_contact_;
    std::vector<float> first_contact_;
    std::vector<float> landing_force_;
    std::vector<float> previous_foot_position_;
    std::vector<float> reward_;
    std::vector<std::uint8_t> terminated_;
    std::vector<float> base_clearance_;
    std::vector<float> velocity_error_;
    std::vector<float> foot_slip_;
    std::vector<float> terrain_normal_error_;
    std::vector<float> reward_terms_;
    std::vector<float> actor_obs_;
    std::vector<float> critic_obs_;
    std::vector<std::thread> view_workers_;
    std::mutex view_worker_mutex_;
    std::condition_variable view_worker_cv_;
    std::condition_variable view_worker_done_cv_;
    EnvironmentTask view_worker_task_;
    std::atomic<std::size_t> view_worker_next_env_{0};
    std::size_t view_worker_chunk_{1};
    std::size_t view_worker_generation_{0};
    std::size_t view_worker_count_{0};
    std::size_t view_worker_completed_{0};
    bool view_worker_stop_{false};
    std::exception_ptr view_worker_error_;
};

void RegisterManualAppContextBindings(py::module_& module) {
    py::class_<PyLocomotionBatchView, std::shared_ptr<PyLocomotionBatchView>>(
            module,
            "_LocomotionBatchView")
            .def("arrays", &PyLocomotionBatchView::Arrays)
            .def("step",
                 &PyLocomotionBatchView::Step,
                 py::arg("ticks") = 1,
                 py::arg("workers") = 0,
                 py::call_guard<py::gil_scoped_release>())
            .def("step_actions",
                 &PyLocomotionBatchView::StepActions,
                 py::arg("ticks") = 1,
                 py::arg("workers") = 0,
                 py::arg("simulate_action_latency") = false,
                 py::call_guard<py::gil_scoped_release>())
            .def("step_task_inputs",
                 &PyLocomotionBatchView::StepTaskInputs,
                 py::arg("ticks") = 1,
                 py::arg("workers") = 0,
                 py::arg("simulate_action_latency") = false,
                 py::call_guard<py::gil_scoped_release>())
            .def("configure_command",
                 &PyLocomotionBatchView::ConfigureCommand,
                 py::arg("step_dt"),
                 py::arg("resampling_time_min"),
                 py::arg("resampling_time_max"),
                 py::arg("lin_vel_x_min"),
                 py::arg("lin_vel_x_max"),
                 py::arg("lin_vel_y_min"),
                 py::arg("lin_vel_y_max"),
                 py::arg("ang_vel_z_min"),
                 py::arg("ang_vel_z_max"),
                 py::arg("heading_min"),
                 py::arg("heading_max"),
                 py::arg("rel_standing_envs"),
                 py::arg("rel_heading_envs"),
                 py::arg("rel_world_envs"),
                 py::arg("rel_forward_envs"),
                 py::arg("heading_command"),
                 py::arg("heading_control_stiffness"),
                 py::arg("seed"))
            .def("set_command_ranges",
                 &PyLocomotionBatchView::SetCommandRanges,
                 py::arg("lin_vel_x_min"),
                 py::arg("lin_vel_x_max"),
                 py::arg("lin_vel_y_min"),
                 py::arg("lin_vel_y_max"),
                 py::arg("ang_vel_z_min"),
                 py::arg("ang_vel_z_max"))
            .def("reset_commands",
                 &PyLocomotionBatchView::ResetCommands,
                 py::arg("env_ids"))
            .def("refresh",
                 &PyLocomotionBatchView::Refresh,
                 py::call_guard<py::gil_scoped_release>())
            .def("reset",
                 &PyLocomotionBatchView::Reset,
                 py::arg("env_ids"),
                 py::call_guard<py::gil_scoped_release>())
            .def("set_base_velocity",
                 &PyLocomotionBatchView::SetBaseVelocity,
                 py::arg("env_id"),
                 py::arg("linear_velocity"),
                 py::arg("angular_velocity"))
            .def("set_base_velocities",
                 &PyLocomotionBatchView::SetBaseVelocities,
                 py::arg("env_ids"),
                 py::arg("linear_velocities"),
                 py::arg("angular_velocities"));

    py::class_<EngineContext, std::shared_ptr<EngineContext>>(module, "AppContext")
            .def_property_readonly("project_path", &EngineContext::GetProjectPath)
            .def_property_readonly("scene_path", &EngineContext::GetScenePath)
            .def_property_readonly("scene_epoch", &EngineContext::GetSceneEpoch)
            .def_property_readonly("scene_dirty", &EngineContext::IsSceneDirty)
            .def_property_readonly("can_undo", &EngineContext::CanUndoSceneCommand)
            .def_property_readonly("can_redo", &EngineContext::CanRedoSceneCommand)
            .def_property_readonly("undo_name", &EngineContext::GetUndoSceneCommandName)
            .def_property_readonly("redo_name", &EngineContext::GetRedoSceneCommandName)
            .def_property_readonly("root", [](EngineContext& context) -> py::object {
                Node* root = SceneRootForContext(context);
                if (root == nullptr) {
                    return py::none();
                }
                return MakeTypedNodeObject(root,
                                           PyNodeOwnership::Borrowed,
                                           &context,
                                           SceneEpochForContext(&context));
            })
            .def_property_readonly("input", [](EngineContext&) -> Input* {
                return Input::GetInstanceOrNull();
            }, py::return_value_policy::reference)
            .def_property("backend_type",
                          &EngineContext::GetBackendType,
                          &EngineContext::SetBackendType)
            .def_property_readonly("has_scene", &EngineContext::HasScene)
            .def_property_readonly("has_world", &EngineContext::HasWorld)
            .def_property_readonly("simulation_time", &EngineContext::GetSimulationTime)
            .def_property_readonly("frame_count", &EngineContext::GetFrameCount)
            .def_property("fixed_time_step",
                          &EngineContext::GetFixedTimeStep,
                          [](EngineContext& context, RealType fixed_time_step) {
                              if (!context.SetFixedTimeStep(fixed_time_step)) {
                                  throw std::runtime_error(context.GetLastError());
                              }
                          })
            .def_property("max_sub_steps",
                          &EngineContext::GetMaxSubSteps,
                          [](EngineContext& context, int max_sub_steps) {
                              if (!context.SetMaxSubSteps(max_sub_steps)) {
                                  throw std::runtime_error(context.GetLastError());
                              }
                          })
            .def_property_readonly("gravity", [](const EngineContext& context) {
                return Vector3ToPython(context.GetGravity());
            })
            .def("set_project_path", [](EngineContext& context, const std::string& project_path) {
                if (!context.SetProjectPath(project_path)) {
                    throw std::runtime_error(context.GetLastError());
                }
            }, py::arg("project_path"))
            .def("load_scene", [](EngineContext& context, const std::string& scene_path) {
                if (!context.LoadScene(scene_path)) {
                    throw std::runtime_error(context.GetLastError());
                }
                return MakeTypedNodeObject(context.GetSceneRoot(),
                                           PyNodeOwnership::Borrowed,
                                           &context,
                                           context.GetSceneEpoch());
            }, py::arg("scene_path"))
            .def("clear_scene", &EngineContext::ClearScene)
            .def("notify_scene_changed", &EngineContext::NotifySceneChanged)
            .def("mark_scene_clean", &EngineContext::MarkSceneClean)
            .def("undo", [](EngineContext& context) {
                return context.UndoSceneCommand();
            })
            .def("redo", [](EngineContext& context) {
                return context.RedoSceneCommand();
            })
            .def("begin_transaction", [](EngineContext& context, const std::string& name) {
                if (!context.BeginSceneTransaction(name)) {
                    throw std::runtime_error("failed to begin Gobot scene transaction '" + name + "'");
                }
            }, py::arg("name") = "Scene Transaction")
            .def("commit_transaction", [](EngineContext& context) {
                if (!context.CommitSceneTransaction()) {
                    throw std::runtime_error("failed to commit Gobot scene transaction");
                }
            })
            .def("cancel_transaction", [](EngineContext& context) {
                if (!context.CancelSceneTransaction()) {
                    throw std::runtime_error("failed to cancel Gobot scene transaction");
                }
            })
            .def("transaction", [](EngineContext&, const std::string& name) {
                return PySceneTransaction(name);
            }, py::arg("name") = "Scene Transaction")
            .def("build_world", [](EngineContext& context, PhysicsBackendType backend_type) {
                context.SetBackendType(backend_type);
                if (!context.BuildWorld()) {
                    throw std::runtime_error(context.GetLastError());
                }
            }, py::arg("backend_type") = PhysicsBackendType::Null)
            .def("rebuild_world", [](EngineContext& context, bool preserve_state) {
                if (!context.RebuildWorld(preserve_state)) {
                    throw std::runtime_error(context.GetLastError());
                }
            }, py::arg("preserve_state") = true)
            .def("clear_world", &EngineContext::ClearWorld)
            .def("reset_simulation", [](EngineContext& context) {
                if (!context.ResetSimulation()) {
                    throw std::runtime_error(context.GetLastError());
                }
            })
            .def("step_once", [](EngineContext& context) {
                if (!context.StepOnce()) {
                    throw std::runtime_error(context.GetLastError());
                }
            })
            .def("step", [](EngineContext& context, std::uint64_t ticks) {
                if (!context.StepTicks(ticks)) {
                    throw std::runtime_error(context.GetLastError());
                }
            }, py::arg("ticks") = 1)
            .def("configure_batch_world", [](EngineContext& context, std::size_t num_envs) {
                SimulationServer* simulation = context.GetSimulationServer();
                if (simulation == nullptr) {
                    throw std::runtime_error("active Gobot app context has no SimulationServer");
                }
                if (!simulation->ConfigureEnvironmentBatch(num_envs)) {
                    throw std::runtime_error(simulation->GetLastError());
                }
            }, py::arg("num_envs"))
            .def("create_locomotion_batch_view", [](EngineContext& context,
                                                    const std::string& robot,
                                                    const std::string& base_link,
                                                    const std::vector<std::string>& joint_names,
                                                    const std::vector<std::string>& foot_link_names,
                                                    const std::vector<std::string>& foot_height_sensor_names,
                                                    const std::vector<std::string>& foot_contact_sensor_names,
                                                    const std::string& height_scan_sensor,
                                                    const std::vector<std::string>& thigh_link_patterns,
                                                    const std::vector<std::string>& shank_link_patterns,
                                                    const std::vector<std::string>& trunk_head_link_patterns,
                                                    bool terminate_on_thigh_contact,
                                                    double ground_force_threshold,
                                                    double self_collision_force_threshold,
                                                    std::size_t reward_term_count,
                                                    std::size_t task_param_count,
                                                    std::size_t task_flag_count,
                                                    std::size_t actor_obs_dim,
                                                    std::size_t critic_obs_dim) {
                SimulationServer* simulation = context.GetSimulationServer();
                if (simulation == nullptr) {
                    throw std::runtime_error("active Gobot app context has no SimulationServer");
                }
                Ref<PhysicsWorld> world = simulation->GetWorld();
                auto mujoco_world = world.DynamicPointerCast<MuJoCoPhysicsWorld>();
                if (!mujoco_world.IsValid()) {
                    throw std::runtime_error("active batch world is not a MuJoCo world");
                }
                return std::make_shared<PyLocomotionBatchView>(mujoco_world,
                                                               robot,
                                                               base_link,
                                                               joint_names,
                                                               foot_link_names,
                                                               foot_height_sensor_names,
                                                               foot_contact_sensor_names,
                                                               height_scan_sensor,
                                                               thigh_link_patterns,
                                                               shank_link_patterns,
                                                               trunk_head_link_patterns,
                                                               terminate_on_thigh_contact,
                                                               ground_force_threshold,
                                                               self_collision_force_threshold,
                                                               reward_term_count,
                                                               task_param_count,
                                                               task_flag_count,
                                                               actor_obs_dim,
                                                               critic_obs_dim);
            }, py::arg("robot"),
               py::arg("base_link"),
               py::arg("joint_names"),
               py::arg("foot_link_names"),
               py::arg("foot_height_sensor_names"),
               py::arg("foot_contact_sensor_names"),
               py::arg("height_scan_sensor") = "",
               py::arg("thigh_link_patterns") = std::vector<std::string>{},
               py::arg("shank_link_patterns") = std::vector<std::string>{},
               py::arg("trunk_head_link_patterns") = std::vector<std::string>{},
               py::arg("terminate_on_thigh_contact") = true,
               py::arg("ground_force_threshold") = 50.0,
               py::arg("self_collision_force_threshold") = 20.0,
               py::arg("reward_term_count") = 0,
               py::arg("task_param_count") = 0,
               py::arg("task_flag_count") = 0,
               py::arg("actor_obs_dim") = 0,
               py::arg("critic_obs_dim") = 0)
            .def_property_readonly("batch_env_count", [](EngineContext& context) {
                SimulationServer* simulation = context.GetSimulationServer();
                if (simulation == nullptr) {
                    return static_cast<std::size_t>(0);
                }
                return simulation->GetEnvironmentCount();
            })
            .def("reset_batch_env", [](EngineContext& context, std::size_t env_id) {
                SimulationServer* simulation = context.GetSimulationServer();
                if (simulation == nullptr) {
                    throw std::runtime_error("active Gobot app context has no SimulationServer");
                }
                if (!simulation->ResetEnvironment(env_id)) {
                    throw std::runtime_error(simulation->GetLastError());
                }
            }, py::arg("env_id"))
            .def("step_batch_env", [](EngineContext& context, std::size_t env_id, std::uint64_t ticks) {
                SimulationServer* simulation = context.GetSimulationServer();
                if (simulation == nullptr) {
                    throw std::runtime_error("active Gobot app context has no SimulationServer");
                }
                if (!simulation->StepEnvironment(env_id, ticks)) {
                    throw std::runtime_error(simulation->GetLastError());
                }
            }, py::arg("env_id"), py::arg("ticks") = 1)
            .def("step_batch", [](EngineContext& context, std::uint64_t ticks, std::size_t workers) {
                SimulationServer* simulation = context.GetSimulationServer();
                if (simulation == nullptr) {
                    throw std::runtime_error("active Gobot app context has no SimulationServer");
                }
                py::gil_scoped_release release;
                if (!simulation->StepEnvironmentBatch(ticks, workers)) {
                    py::gil_scoped_acquire acquire;
                    throw std::runtime_error(simulation->GetLastError());
                }
            }, py::arg("ticks") = 1, py::arg("workers") = 0)
            .def("resolved_batch_workers", [](EngineContext& context, std::size_t workers) {
                SimulationServer* simulation = context.GetSimulationServer();
                if (simulation == nullptr) {
                    throw std::runtime_error("active Gobot app context has no SimulationServer");
                }
                return simulation->ResolveEnvironmentBatchWorkerCount(workers);
            }, py::arg("workers") = 0)
            .def("set_batch_joint_position_target", [](EngineContext& context,
                                                       std::size_t env_id,
                                                       const std::string& robot,
                                                       const std::string& joint,
                                                       RealType target_position) {
                SimulationServer* simulation = context.GetSimulationServer();
                if (simulation == nullptr) {
                    throw std::runtime_error("active Gobot app context has no SimulationServer");
                }
                SimulationScene* runtime_scene = simulation->GetRuntimeScene();
                if (runtime_scene == nullptr) {
                    throw std::runtime_error("simulation runtime scene has not been built");
                }
                if (!runtime_scene->SetEnvironmentJointPositionTarget(env_id, robot, joint, target_position)) {
                    throw std::runtime_error(runtime_scene->GetLastError());
                }
            }, py::arg("env_id"), py::arg("robot"), py::arg("joint"), py::arg("target_position"))
            .def("set_batch_joint_position_targets", [](EngineContext& context,
                                                        const std::string& robot,
                                                        const std::vector<std::string>& joint_names,
                                                        py::array_t<RealType, py::array::c_style | py::array::forcecast> target_positions) {
                SimulationServer* simulation = context.GetSimulationServer();
                if (simulation == nullptr) {
                    throw std::runtime_error("active Gobot app context has no SimulationServer");
                }
                SimulationScene* runtime_scene = simulation->GetRuntimeScene();
                if (runtime_scene == nullptr) {
                    throw std::runtime_error("simulation runtime scene has not been built");
                }
                const py::buffer_info buffer = target_positions.request();
                if (buffer.ndim != 2) {
                    throw std::invalid_argument("target_positions must be a 2D array with shape [num_envs, num_joints]");
                }
                const auto environment_count = static_cast<std::size_t>(buffer.shape[0]);
                const auto joint_count = static_cast<std::size_t>(buffer.shape[1]);
                if (joint_count != joint_names.size()) {
                    throw std::invalid_argument(fmt::format("target_positions has {} joint column(s), expected {}",
                                                            joint_count,
                                                            joint_names.size()));
                }
                const auto* data = static_cast<const RealType*>(buffer.ptr);
                std::vector<RealType> targets(data, data + environment_count * joint_count);
                if (!runtime_scene->SetEnvironmentJointPositionTargets(robot,
                                                                       joint_names,
                                                                       targets,
                                                                       environment_count)) {
                    throw std::runtime_error(runtime_scene->GetLastError());
                }
            }, py::arg("robot"), py::arg("joint_names"), py::arg("target_positions"))
            .def("reset_batch_joint_state", [](EngineContext& context,
                                               std::size_t env_id,
                                               const std::string& robot,
                                               const std::string& joint,
                                               RealType position,
                                               RealType velocity) {
                SimulationServer* simulation = context.GetSimulationServer();
                if (simulation == nullptr) {
                    throw std::runtime_error("active Gobot app context has no SimulationServer");
                }
                SimulationScene* runtime_scene = simulation->GetRuntimeScene();
                if (runtime_scene == nullptr) {
                    throw std::runtime_error("simulation runtime scene has not been built");
                }
                if (!runtime_scene->ResetEnvironmentJointState(env_id, robot, joint, position, velocity)) {
                    throw std::runtime_error(runtime_scene->GetLastError());
                }
            }, py::arg("env_id"), py::arg("robot"), py::arg("joint"), py::arg("position"), py::arg("velocity") = 0.0)
            .def("reset_batch_link_state", [](EngineContext& context,
                                              std::size_t env_id,
                                              const std::string& robot,
                                              const std::string& link,
                                              const py::object& position,
                                              const py::object& orientation,
                                              const py::object& linear_velocity,
                                              const py::object& angular_velocity) {
                SimulationServer* simulation = context.GetSimulationServer();
                if (simulation == nullptr) {
                    throw std::runtime_error("active Gobot app context has no SimulationServer");
                }
                SimulationScene* runtime_scene = simulation->GetRuntimeScene();
                if (runtime_scene == nullptr) {
                    throw std::runtime_error("simulation runtime scene has not been built");
                }
                if (!runtime_scene->ResetEnvironmentLinkState(env_id,
                                                              robot,
                                                              link,
                                                              PythonToVector3(position),
                                                              PythonToQuaternionWxyz(orientation),
                                                              PythonToVector3(linear_velocity),
                                                              PythonToVector3(angular_velocity))) {
                    throw std::runtime_error(runtime_scene->GetLastError());
                }
            }, py::arg("env_id"),
               py::arg("robot"),
               py::arg("link"),
               py::arg("position"),
               py::arg("orientation") = py::make_tuple(1.0, 0.0, 0.0, 0.0),
               py::arg("linear_velocity") = py::make_tuple(0.0, 0.0, 0.0),
               py::arg("angular_velocity") = py::make_tuple(0.0, 0.0, 0.0))
            .def("set_default_joint_gains", [](EngineContext& context, py::dict gains) {
                SimulationServer* simulation = context.GetSimulationServer();
                if (simulation == nullptr) {
                    throw std::runtime_error("active Gobot app context has no SimulationServer");
                }
                simulation->SetDefaultJointGains(DictToReflected<JointControllerGains>(gains));
            }, py::arg("gains"))
            .def("get_default_joint_gains", [](EngineContext& context) {
                SimulationServer* simulation = context.GetSimulationServer();
                if (simulation == nullptr) {
                    throw std::runtime_error("active Gobot app context has no SimulationServer");
                }
                return ReflectedToPythonDict(simulation->GetDefaultJointGains());
            })
            .def("get_batch_runtime_state", [](EngineContext& context, std::size_t env_id) {
                SimulationServer* simulation = context.GetSimulationServer();
                if (simulation == nullptr) {
                    throw std::runtime_error("active Gobot app context has no SimulationServer");
                }
                const PhysicsSceneState* state = simulation->GetEnvironmentState(env_id);
                if (state == nullptr) {
                    throw std::runtime_error("simulation environment state is not available");
                }
                return RuntimeStateToPythonDict(*state);
            }, py::arg("env_id"))
            .def("get_batch_robot_state", [](EngineContext& context,
                                             const std::string& robot,
                                             const std::string& base_link,
                                             const std::vector<std::string>& joint_names,
                                             const std::vector<std::string>& link_names,
                                             const std::vector<std::string>& sensor_names) {
                SimulationServer* simulation = context.GetSimulationServer();
                if (simulation == nullptr) {
                    throw std::runtime_error("active Gobot app context has no SimulationServer");
                }
                return BatchRobotStateToPythonDict(*simulation,
                                                   robot,
                                                   base_link,
                                                   joint_names,
                                                   link_names,
                                                   sensor_names);
            }, py::arg("robot"),
               py::arg("base_link"),
               py::arg("joint_names"),
               py::arg("link_names") = std::vector<std::string>{},
               py::arg("sensor_names") = std::vector<std::string>{})
            .def("_step_batch_and_get_robot_state", [](EngineContext& context,
                                                       const std::string& robot,
                                                       const std::string& base_link,
                                                       const std::vector<std::string>& joint_names,
                                                       const std::vector<std::string>& link_names,
                                                       const std::vector<std::string>& sensor_names,
                                                       py::array_t<RealType, py::array::c_style | py::array::forcecast> target_positions,
                                                       std::uint64_t ticks,
                                                       std::size_t workers) {
                SimulationServer* simulation = context.GetSimulationServer();
                if (simulation == nullptr) {
                    throw std::runtime_error("active Gobot app context has no SimulationServer");
                }
                SimulationScene* runtime_scene = simulation->GetRuntimeScene();
                if (runtime_scene == nullptr) {
                    throw std::runtime_error("simulation runtime scene has not been built");
                }
                const py::buffer_info buffer = target_positions.request();
                if (buffer.ndim != 2) {
                    throw std::invalid_argument("target_positions must be a 2D array with shape [num_envs, num_joints]");
                }
                const auto environment_count = static_cast<std::size_t>(buffer.shape[0]);
                const auto joint_count = static_cast<std::size_t>(buffer.shape[1]);
                if (joint_count != joint_names.size()) {
                    throw std::invalid_argument(fmt::format("target_positions has {} joint column(s), expected {}",
                                                            joint_count,
                                                            joint_names.size()));
                }
                const auto* data = static_cast<const RealType*>(buffer.ptr);
                std::vector<RealType> targets(data, data + environment_count * joint_count);
                Ref<PhysicsWorld> world = simulation->GetWorld();
                if (auto mujoco_world = world.DynamicPointerCast<MuJoCoPhysicsWorld>(); mujoco_world.IsValid()) {
                    MuJoCoPhysicsWorld::BatchRobotStateRequest request;
                    request.robot_name = robot;
                    request.base_link = base_link;
                    request.joint_names = joint_names;
                    request.link_names = link_names;
                    request.sensor_names = sensor_names;
                    request.target_positions = std::move(targets);
                    request.ticks = ticks;
                    request.worker_count = workers;
                    MuJoCoPhysicsWorld::BatchRobotStateArrays arrays;
                    {
                        py::gil_scoped_release release;
                        if (!mujoco_world->StepEnvironmentBatchFastRobotState(request, arrays)) {
                            py::gil_scoped_acquire acquire;
                            throw std::runtime_error(mujoco_world->GetLastError());
                        }
                    }
                    return BatchRobotStateArraysToPythonDict(std::move(arrays));
                }
                {
                    py::gil_scoped_release release;
                    if (!runtime_scene->SetEnvironmentJointPositionTargets(robot,
                                                                           joint_names,
                                                                           targets,
                                                                           environment_count)) {
                        py::gil_scoped_acquire acquire;
                        throw std::runtime_error(runtime_scene->GetLastError());
                    }
                    if (!simulation->StepEnvironmentBatch(ticks, workers)) {
                        py::gil_scoped_acquire acquire;
                        throw std::runtime_error(simulation->GetLastError());
                    }
                }
                return BatchRobotStateToPythonDict(*simulation,
                                                   robot,
                                                   base_link,
                                                   joint_names,
                                                   link_names,
                                                   sensor_names);
            }, py::arg("robot"),
               py::arg("base_link"),
               py::arg("joint_names"),
               py::arg("link_names"),
               py::arg("sensor_names"),
               py::arg("target_positions"),
               py::arg("ticks") = 1,
               py::arg("workers") = 0)
            .def("_reset_batch_robot_states", [](EngineContext& context,
                                                 const std::vector<std::size_t>& env_ids,
                                                 const std::string& robot,
                                                 const std::string& base_link,
                                                 const py::object& base_positions_object,
                                                 const py::object& base_orientations_object,
                                                 const py::object& base_linear_velocities_object,
                                                 const py::object& base_angular_velocities_object,
                                                 const std::vector<std::string>& joint_names,
                                                 const py::object& joint_positions_object,
                                                 const py::object& joint_velocities_object,
                                                 const py::object& joint_position_targets_object) {
                SimulationServer* simulation = context.GetSimulationServer();
                if (simulation == nullptr) {
                    throw std::runtime_error("active Gobot app context has no SimulationServer");
                }
                SimulationScene* runtime_scene = simulation->GetRuntimeScene();
                if (runtime_scene == nullptr) {
                    throw std::runtime_error("simulation runtime scene has not been built");
                }
                const std::size_t reset_count = env_ids.size();
                const std::size_t joint_count = joint_names.size();
                if (reset_count == 0) {
                    return;
                }

                auto require_array = [](const py::object& object,
                                        int ndim,
                                        py::ssize_t first_dim,
                                        py::ssize_t second_dim,
                                        const std::string& name) {
                    py::array_t<RealType, py::array::c_style | py::array::forcecast> array =
                            py::array_t<RealType, py::array::c_style | py::array::forcecast>::ensure(object);
                    if (!array) {
                        throw std::invalid_argument(name + " must be a numeric array");
                    }
                    const py::buffer_info buffer = array.request();
                    if (buffer.ndim != ndim || buffer.shape[0] != first_dim ||
                        (ndim >= 2 && buffer.shape[1] != second_dim)) {
                        throw std::invalid_argument(fmt::format("{} must have shape [{}, {}]",
                                                                name,
                                                                first_dim,
                                                                second_dim));
                    }
                    return array;
                };

                auto base_positions = require_array(base_positions_object,
                                                    2,
                                                    static_cast<py::ssize_t>(reset_count),
                                                    3,
                                                    "base_positions");
                auto base_orientations = require_array(base_orientations_object,
                                                       2,
                                                       static_cast<py::ssize_t>(reset_count),
                                                       4,
                                                       "base_orientations");
                auto base_linear_velocities = require_array(base_linear_velocities_object,
                                                            2,
                                                            static_cast<py::ssize_t>(reset_count),
                                                            3,
                                                            "base_linear_velocities");
                auto base_angular_velocities = require_array(base_angular_velocities_object,
                                                             2,
                                                             static_cast<py::ssize_t>(reset_count),
                                                             3,
                                                             "base_angular_velocities");
                auto joint_positions = require_array(joint_positions_object,
                                                     2,
                                                     static_cast<py::ssize_t>(reset_count),
                                                     static_cast<py::ssize_t>(joint_count),
                                                     "joint_positions");
                auto joint_velocities = require_array(joint_velocities_object,
                                                      2,
                                                      static_cast<py::ssize_t>(reset_count),
                                                      static_cast<py::ssize_t>(joint_count),
                                                      "joint_velocities");
                auto joint_position_targets = require_array(joint_position_targets_object,
                                                            2,
                                                            static_cast<py::ssize_t>(reset_count),
                                                            static_cast<py::ssize_t>(joint_count),
                                                            "joint_position_targets");

                const auto* base_position_data = static_cast<const RealType*>(base_positions.request().ptr);
                const auto* base_orientation_data = static_cast<const RealType*>(base_orientations.request().ptr);
                const auto* base_linear_velocity_data =
                        static_cast<const RealType*>(base_linear_velocities.request().ptr);
                const auto* base_angular_velocity_data =
                        static_cast<const RealType*>(base_angular_velocities.request().ptr);
                const auto* joint_position_data = static_cast<const RealType*>(joint_positions.request().ptr);
                const auto* joint_velocity_data = static_cast<const RealType*>(joint_velocities.request().ptr);
                const auto* joint_position_target_data =
                        static_cast<const RealType*>(joint_position_targets.request().ptr);

                std::vector<PhysicsEnvironmentRobotResetState> reset_states;
                reset_states.reserve(reset_count);
                for (std::size_t reset_index = 0; reset_index < reset_count; ++reset_index) {
                    const std::size_t base3 = reset_index * 3;
                    const std::size_t base4 = reset_index * 4;
                    const std::size_t joint_offset = reset_index * joint_count;
                    PhysicsEnvironmentRobotResetState reset_state;
                    reset_state.environment_index = env_ids[reset_index];
                    reset_state.robot_name = robot;
                    reset_state.base_link_name = base_link;
                    reset_state.base_position = Vector3(base_position_data[base3 + 0],
                                                        base_position_data[base3 + 1],
                                                        base_position_data[base3 + 2]);
                    reset_state.base_orientation = Quaternion(base_orientation_data[base4 + 0],
                                                              base_orientation_data[base4 + 1],
                                                              base_orientation_data[base4 + 2],
                                                              base_orientation_data[base4 + 3]);
                    reset_state.base_linear_velocity = Vector3(base_linear_velocity_data[base3 + 0],
                                                               base_linear_velocity_data[base3 + 1],
                                                               base_linear_velocity_data[base3 + 2]);
                    reset_state.base_angular_velocity = Vector3(base_angular_velocity_data[base3 + 0],
                                                                base_angular_velocity_data[base3 + 1],
                                                                base_angular_velocity_data[base3 + 2]);
                    reset_state.joint_names = joint_names;
                    reset_state.joint_positions.assign(joint_position_data + joint_offset,
                                                       joint_position_data + joint_offset + joint_count);
                    reset_state.joint_velocities.assign(joint_velocity_data + joint_offset,
                                                        joint_velocity_data + joint_offset + joint_count);
                    reset_state.joint_position_targets.assign(joint_position_target_data + joint_offset,
                                                              joint_position_target_data + joint_offset + joint_count);
                    reset_states.emplace_back(std::move(reset_state));
                }

                py::gil_scoped_release release;
                if (!runtime_scene->ResetEnvironmentRobotStates(reset_states)) {
                    py::gil_scoped_acquire acquire;
                    throw std::runtime_error(runtime_scene->GetLastError());
                }
            }, py::arg("env_ids"),
               py::arg("robot"),
               py::arg("base_link"),
               py::arg("base_positions"),
               py::arg("base_orientations"),
               py::arg("base_linear_velocities"),
               py::arg("base_angular_velocities"),
               py::arg("joint_names"),
               py::arg("joint_positions"),
               py::arg("joint_velocities"),
               py::arg("joint_position_targets"))
            .def("get_last_error", &EngineContext::GetLastError);

    py::class_<PyScene, std::unique_ptr<PyScene>>(module, "Scene")
            .def_property_readonly("root", [](PyScene& scene) {
                return MakeTypedNodeObject(scene.root);
            })
            .def_property_readonly("scene_epoch", [](const PyScene& scene) {
                return scene.scene_epoch;
            });

    py::class_<PySceneTransaction>(module, "SceneTransaction")
            .def("__enter__", &PySceneTransaction::Enter, py::return_value_policy::reference_internal)
            .def("__exit__", &PySceneTransaction::Exit,
                 py::arg("exc_type"),
                 py::arg("exc_value"),
                 py::arg("traceback"));
}

} // namespace gobot::python
