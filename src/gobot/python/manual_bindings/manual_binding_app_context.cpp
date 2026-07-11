#include "manual_bindings_internal.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <random>
#include <limits>
#include <string_view>

namespace gobot::python {

namespace {

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

bool ContainsCaseInsensitive(const std::string& value, const std::string& needle) {
    auto value_it = value.begin();
    return std::search(value_it,
                       value.end(),
                       needle.begin(),
                       needle.end(),
                       [](char a, char b) {
                           return std::tolower(static_cast<unsigned char>(a)) ==
                                  std::tolower(static_cast<unsigned char>(b));
                       }) != value.end();
}

bool EndsWith(std::string_view value, std::string_view suffix) {
    return value.size() >= suffix.size() &&
           value.substr(value.size() - suffix.size()) == suffix;
}

std::string FootGeomNameFromSensorOrLinkName(std::string_view sensor_name, std::string_view link_name) {
    constexpr std::string_view kFootContactSuffix = "_foot_contact";
    if (EndsWith(sensor_name, kFootContactSuffix) && sensor_name.size() > kFootContactSuffix.size()) {
        return fmt::format("{}_foot_collision",
                           sensor_name.substr(0, sensor_name.size() - kFootContactSuffix.size()));
    }
    const std::size_t separator = link_name.find('_');
    if (separator != std::string_view::npos && separator > 0) {
        return std::string(link_name.substr(0, separator));
    }
    return std::string(link_name);
}

std::string FootContactGroupName(std::size_t foot_index) {
    return fmt::format("foot_{}", foot_index);
}

constexpr std::size_t kDefaultRewardTermCount = 15;
constexpr std::size_t kDefaultTaskParamCount = 11;
constexpr std::size_t kDefaultTaskFlagCount = 3;
constexpr std::string_view kThighContactGroup = "thigh";
constexpr std::string_view kShankContactGroup = "shank";
constexpr std::string_view kTrunkHeadContactGroup = "trunk_head";

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
    PyLocomotionBatchView(Ref<PhysicsWorld> world,
                             std::string robot_name,
                             std::string base_link,
                             std::vector<std::string> joint_names,
                             std::vector<std::string> link_names,
                             std::vector<std::string> foot_link_names,
                             std::vector<std::string> foot_height_sensor_names,
                             std::vector<std::string> foot_contact_sensor_names,
                             std::string height_scan_sensor,
                             std::vector<std::string> thigh_shape_patterns,
                             std::vector<std::string> shank_shape_patterns,
                             std::vector<std::string> trunk_head_shape_patterns,
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
          link_names_(std::move(link_names)),
          foot_link_names_(std::move(foot_link_names)),
          foot_height_sensor_names_(std::move(foot_height_sensor_names)),
          foot_contact_sensor_names_(std::move(foot_contact_sensor_names)),
          height_scan_sensor_(std::move(height_scan_sensor)),
          thigh_shape_patterns_(CompileContactPatterns(thigh_shape_patterns)),
          shank_shape_patterns_(CompileContactPatterns(shank_shape_patterns)),
          trunk_head_shape_patterns_(CompileContactPatterns(trunk_head_shape_patterns)),
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

    ~PyLocomotionBatchView() = default;

    void Close() {
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
        arrays["push_force"] =
                VectorArrayView(push_force_, {EnvDim(), 3}, owner, true);
        arrays["push_torque"] =
                VectorArrayView(push_torque_, {EnvDim(), 3}, owner, true);
        arrays["base_mass_delta"] =
                VectorArrayView(base_mass_delta_, {EnvDim()}, owner, true);
        arrays["base_com_offset"] =
                VectorArrayView(base_com_offset_, {EnvDim(), 3}, owner, true);
        arrays["joint_kp"] =
                VectorArrayView(joint_kp_, {EnvDim(), JointDim()}, owner, true);
        arrays["joint_kd"] =
                VectorArrayView(joint_kd_, {EnvDim(), JointDim()}, owner, true);
        arrays["foot_friction"] =
                VectorArrayView(foot_friction_, {EnvDim(), 3}, owner, true);
        arrays["foot_friction_enabled"] =
                VectorArrayView(foot_friction_enabled_, {EnvDim()}, owner, true);
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
        arrays["command_step"] =
                VectorArrayView(command_step_, {EnvDim()}, owner, false);
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
        arrays["local_linvel"] =
                VectorArrayView(base_linear_velocity_body_, {EnvDim(), 3}, owner, false);
        arrays["base_angular_velocity_body"] =
                VectorArrayView(base_angular_velocity_body_, {EnvDim(), 3}, owner, false);
        arrays["gyro"] =
                VectorArrayView(base_angular_velocity_body_, {EnvDim(), 3}, owner, false);
        arrays["projected_gravity"] =
                VectorArrayView(projected_gravity_, {EnvDim(), 3}, owner, false);
        arrays["gravity"] =
                VectorArrayView(projected_gravity_, {EnvDim(), 3}, owner, false);
        arrays["upvector"] =
                VectorArrayView(upvector_, {EnvDim(), 3}, owner, false);
        arrays["framezaxis"] =
                VectorArrayView(upvector_, {EnvDim(), 3}, owner, false);
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
        arrays["link_position"] =
                VectorArrayView(link_position_, {EnvDim(), LinkDim(), 3}, owner, false);
        arrays["link_quaternion"] =
                VectorArrayView(link_quaternion_, {EnvDim(), LinkDim(), 4}, owner, false);
        arrays["link_linear_velocity"] =
                VectorArrayView(link_linear_velocity_, {EnvDim(), LinkDim(), 3}, owner, false);
        arrays["link_angular_velocity"] =
                VectorArrayView(link_angular_velocity_, {EnvDim(), LinkDim(), 3}, owner, false);
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
        arrays["undesired_contact_count"] =
                VectorArrayView(undesired_contact_count_, {EnvDim()}, owner, false);
        arrays["undesired_base_contact_count"] =
                VectorArrayView(undesired_base_contact_count_, {EnvDim()}, owner, false);
        arrays["undesired_hip_contact_count"] =
                VectorArrayView(undesired_hip_contact_count_, {EnvDim()}, owner, false);
        arrays["undesired_thigh_contact_count"] =
                VectorArrayView(undesired_thigh_contact_count_, {EnvDim()}, owner, false);
        arrays["undesired_calf_contact_count"] =
                VectorArrayView(undesired_calf_contact_count_, {EnvDim()}, owner, false);
        arrays["self_collision_count"] =
                VectorArrayView(self_collision_count_, {EnvDim()}, owner, false);
        arrays["shank_collision_count"] =
                VectorArrayView(shank_collision_count_, {EnvDim()}, owner, false);
        arrays["trunk_head_collision_count"] =
                VectorArrayView(trunk_head_collision_count_, {EnvDim()}, owner, false);
        arrays["base_collision_count"] =
                VectorArrayView(base_collision_count_, {EnvDim()}, owner, false);
        arrays["hip_collision_count"] =
                VectorArrayView(hip_collision_count_, {EnvDim()}, owner, false);
        arrays["thigh_collision_count"] =
                VectorArrayView(thigh_collision_count_, {EnvDim()}, owner, false);
        arrays["calf_collision_count"] =
                VectorArrayView(calf_collision_count_, {EnvDim()}, owner, false);
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

    py::array_t<float> TerrainHeights(py::array_t<float, py::array::c_style | py::array::forcecast> points,
                                      std::size_t env_id = 0) const {
        RequireEnvironmentIndex(env_id);
        py::buffer_info points_info = points.request();
        if (points_info.ndim != 2 || points_info.shape[1] < 2) {
            throw std::runtime_error("terrain height query points must have shape (N, 2) or (N, 3)");
        }

        const auto point_count = static_cast<std::size_t>(points_info.shape[0]);
        py::array_t<float> heights(static_cast<py::ssize_t>(point_count));
        py::buffer_info heights_info = heights.request();
        const float* point_values = static_cast<const float*>(points_info.ptr);
        float* height_values = static_cast<float*>(heights_info.ptr);

        constexpr RealType kRayHeight = 10.0;
        constexpr RealType kRayDistance = 100.0;
        const auto stride = static_cast<std::size_t>(points_info.shape[1]);
        for (std::size_t index = 0; index < point_count; ++index) {
            const float* point = point_values + index * stride;
            const Vector3 origin{
                    static_cast<RealType>(point[0]),
                    static_cast<RealType>(point[1]),
                    static_cast<RealType>(stride >= 3 ? point[2] : 0.0f) + kRayHeight};
            const PhysicsRaycastHit hit = world_->RaycastEnvironmentTerrain(
                    {origin, {0.0, 0.0, -1.0}, kRayDistance},
                    env_id);
            if (!hit.hit) {
                height_values[index] = std::numeric_limits<float>::quiet_NaN();
                continue;
            }
            height_values[index] = static_cast<float>(hit.point.z());
        }
        return heights;
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
        RunPhysicsBatch(ticks, workers, false, false);
        CopyPhysicsState();
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
                          double zero_small_xy_threshold,
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
        command_zero_small_xy_threshold_ = static_cast<float>(std::max(0.0, zero_small_xy_threshold));
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
    }

    void SetCommands(const std::vector<std::size_t>& env_ids,
                     py::array_t<float, py::array::c_style | py::array::forcecast> commands,
                     py::array_t<float, py::array::c_style | py::array::forcecast> heading_targets,
                     py::array_t<float, py::array::c_style | py::array::forcecast> time_left) {
        const std::size_t count = env_ids.size();
        if (count == 0) {
            return;
        }
        auto command_buffer = commands.request();
        auto heading_buffer = heading_targets.request();
        auto time_left_buffer = time_left.request();
        if (command_buffer.ndim != 2 || command_buffer.shape[0] != static_cast<py::ssize_t>(count) ||
            command_buffer.shape[1] != 3) {
            throw std::invalid_argument("commands must have shape [len(env_ids), 3]");
        }
        if (heading_buffer.ndim != 1 || heading_buffer.shape[0] != static_cast<py::ssize_t>(count)) {
            throw std::invalid_argument("heading_targets must have shape [len(env_ids)]");
        }
        if (time_left_buffer.ndim != 1 || time_left_buffer.shape[0] != static_cast<py::ssize_t>(count)) {
            throw std::invalid_argument("time_left must have shape [len(env_ids)]");
        }

        const auto* command = static_cast<const float*>(command_buffer.ptr);
        const auto* heading = static_cast<const float*>(heading_buffer.ptr);
        const auto* remaining = static_cast<const float*>(time_left_buffer.ptr);
        for (std::size_t row = 0; row < count; ++row) {
            const std::size_t env_id = env_ids[row];
            RequireEnvironmentIndex(env_id);
            const std::size_t command_offset = env_id * 3;
            command_[command_offset + 0] = command[row * 3 + 0];
            command_[command_offset + 1] = command[row * 3 + 1];
            command_[command_offset + 2] = command[row * 3 + 2];
            command_world_[command_offset + 0] = command[row * 3 + 0];
            command_world_[command_offset + 1] = command[row * 3 + 1];
            command_world_[command_offset + 2] = command[row * 3 + 2];
            command_heading_target_[env_id] = heading[row];
            command_heading_error_[env_id] = 0.0f;
            command_time_left_[env_id] = remaining[row];
            command_is_heading_env_[env_id] = command_heading_enabled_;
            command_is_standing_env_[env_id] =
                    std::abs(command[row * 3 + 0]) <= 1.0e-6f &&
                    std::abs(command[row * 3 + 1]) <= 1.0e-6f &&
                    std::abs(command[row * 3 + 2]) <= 1.0e-6f;
            command_is_world_env_[env_id] = false;
            command_is_forward_env_[env_id] = false;
        }
    }

    void SetCommandSteps(const std::vector<std::size_t>& env_ids,
                         py::array_t<std::uint32_t, py::array::c_style | py::array::forcecast> steps) {
        const std::size_t count = env_ids.size();
        if (count == 0) {
            return;
        }
        auto step_buffer = steps.request();
        if (step_buffer.ndim != 1 || step_buffer.shape[0] != static_cast<py::ssize_t>(count)) {
            throw std::invalid_argument("steps must have shape [len(env_ids)]");
        }
        const auto* step_values = static_cast<const std::uint32_t*>(step_buffer.ptr);
        for (std::size_t row = 0; row < count; ++row) {
            const std::size_t env_id = env_ids[row];
            RequireEnvironmentIndex(env_id);
            command_step_[env_id] = step_values[row];
        }
    }

    void SetCommandStepResampling(bool enabled) {
        command_step_resampling_enabled_ = enabled;
    }

    void StepTaskInputs(std::uint64_t ticks, std::size_t workers, bool simulate_action_latency) {
#ifdef GOBOT_HAS_MUJOCO
        ResetStepProfile();
        const TimePoint total_begin = Clock::now();
        TimePoint phase_begin = Clock::now();
        PrepareActionTargets(simulate_action_latency);
        SetProfileMs(kStepProfilePrepareAction, ElapsedMs(phase_begin));
        phase_begin = Clock::now();
        RunPhysicsBatch(ticks, workers, true, true);
        SetProfileMs(kStepProfileMjStep, ElapsedMs(phase_begin));
        phase_begin = Clock::now();
        CopyPhysicsState();
        SetProfileMs(kStepProfileExtractState, ElapsedMs(phase_begin));
        phase_begin = Clock::now();
        for (std::size_t env_id = 0; env_id < environment_count_; ++env_id) {
            UpdateFootHistory(env_id);
        }
        SetProfileMs(kStepProfileCommand, ElapsedMs(phase_begin));
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
        RunPhysicsBatch(0, 0, false, false);
        CopyPhysicsState();
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
        RunPhysicsBatch(0, 0, false, false);
        CopyPhysicsState();
        for (std::size_t env_id : env_ids) {
            ClearContactBuffersForEnvironment(env_id);
            const std::size_t foot_begin = env_id * foot_count_;
            std::fill(foot_contact_time_.begin() + static_cast<std::ptrdiff_t>(foot_begin),
                      foot_contact_time_.begin() + static_cast<std::ptrdiff_t>(foot_begin + foot_count_),
                      0.0f);
        }
    }

    void ClearResetContacts(const std::vector<std::size_t>& env_ids) {
        for (std::size_t env_id : env_ids) {
            RequireEnvironmentIndex(env_id);
            ClearContactBuffersForEnvironment(env_id);
            const std::size_t foot_begin = env_id * foot_count_;
            std::fill(foot_contact_time_.begin() + static_cast<std::ptrdiff_t>(foot_begin),
                      foot_contact_time_.begin() + static_cast<std::ptrdiff_t>(foot_begin + foot_count_),
                      0.0f);
        }
    }

    void SetBaseVelocity(std::size_t env_id,
                         py::array_t<float, py::array::c_style | py::array::forcecast> linear_velocity,
                         py::array_t<float, py::array::c_style | py::array::forcecast> angular_velocity) {
        RequireEnvironmentIndex(env_id);
        auto linear_buffer = linear_velocity.request();
        auto angular_buffer = angular_velocity.request();
        if (linear_buffer.ndim != 1 || linear_buffer.shape[0] != 3 ||
            angular_buffer.ndim != 1 || angular_buffer.shape[0] != 3) {
            throw std::invalid_argument("linear_velocity and angular_velocity must have shape [3]");
        }
        const auto* linear = static_cast<const float*>(linear_buffer.ptr);
        const auto* angular = static_cast<const float*>(angular_buffer.ptr);
        if (!world_->WriteEnvironmentLinkVelocity(
                    env_id,
                    robot_name_,
                    base_link_,
                    {linear[0], linear[1], linear[2]},
                    {angular[0], angular[1], angular[2]})) {
            throw std::runtime_error(world_->GetLastError());
        }
        RunPhysicsBatch(0, 0, false, false);
        CopyPhysicsState();
    }

    void SetBaseVelocities(const std::vector<std::size_t>& env_ids,
                           py::array_t<float, py::array::c_style | py::array::forcecast> linear_velocities,
                           py::array_t<float, py::array::c_style | py::array::forcecast> angular_velocities,
                           bool refresh) {
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
            const float* linear_row = linear + row * 3;
            const float* angular_row = angular + row * 3;
            if (!world_->WriteEnvironmentLinkVelocity(
                        env_id,
                        robot_name_,
                        base_link_,
                        {linear_row[0], linear_row[1], linear_row[2]},
                        {angular_row[0], angular_row[1], angular_row[2]})) {
                throw std::runtime_error(world_->GetLastError());
            }
        }
        if (refresh) {
            RunPhysicsBatch(0, 0, false, false);
            CopyPhysicsState();
        }
    }

    void AdvanceCommands() {
        if (command_configured_) {
            ComputeCommands();
        }
    }

    void UpdateCommandFrames() {
        if (command_configured_) {
            UpdateCommands();
        }
    }

private:
    py::ssize_t EnvDim() const {
        return static_cast<py::ssize_t>(environment_count_);
    }

    py::ssize_t JointDim() const {
        return static_cast<py::ssize_t>(joint_count_);
    }

    py::ssize_t FootDim() const {
        return static_cast<py::ssize_t>(foot_count_);
    }

    py::ssize_t LinkDim() const {
        return static_cast<py::ssize_t>(link_count_);
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

    using Clock = std::chrono::steady_clock;
    using TimePoint = Clock::time_point;

    static double ElapsedMs(TimePoint begin) {
        return std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
    }

    void Initialize() {
        if (!world_.IsValid()) {
            throw std::runtime_error("Physics world is not available");
        }
        environment_count_ = world_->GetEnvironmentCount();
        if (environment_count_ == 0) {
            throw std::runtime_error("Physics environment batch has not been configured");
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
            throw std::runtime_error(fmt::format("robot '{}' is not available in the physics scene", robot_name_));
        }

        const auto find_link = [&](const std::string& name) -> std::size_t {
            for (std::size_t index = 0; index < robot_snapshot->links.size(); ++index) {
                if (robot_snapshot->links[index].name == name) {
                    return index;
                }
            }
            throw std::runtime_error(fmt::format("link '{}' is not available on robot '{}'",
                                                 name,
                                                 robot_name_));
        };
        const auto find_joint = [&](const std::string& name) -> std::size_t {
            for (std::size_t index = 0; index < robot_snapshot->joints.size(); ++index) {
                if (robot_snapshot->joints[index].name == name) {
                    return index;
                }
            }
            throw std::runtime_error(fmt::format("joint '{}' is not available on robot '{}'",
                                                 name,
                                                 robot_name_));
        };
        const auto find_sensor = [&](const std::string& name) -> std::size_t {
            for (std::size_t index = 0; index < robot_snapshot->sensors.size(); ++index) {
                if (robot_snapshot->sensors[index].name == name) {
                    return index;
                }
            }
            throw std::runtime_error(fmt::format("sensor '{}' is not available on robot '{}'",
                                                 name,
                                                 robot_name_));
        };

        find_link(base_link_);
        for (const std::string& joint_name : joint_names_) {
            find_joint(joint_name);
        }
        joint_count_ = joint_names_.size();

        if (link_names_.empty()) {
            link_names_.reserve(robot_snapshot->links.size());
            for (const PhysicsLinkSnapshot& link : robot_snapshot->links) {
                if (link.role == PhysicsLinkRole::Physical) {
                    link_names_.push_back(link.name);
                }
            }
        }
        link_index_by_name_.clear();
        for (std::size_t index = 0; index < link_names_.size(); ++index) {
            find_link(link_names_[index]);
            link_index_by_name_[link_names_[index]] = index;
        }
        link_count_ = link_names_.size();

        foot_link_indices_.reserve(foot_link_names_.size());
        foot_shape_names_.reserve(foot_link_names_.size());
        for (std::size_t foot_index = 0; foot_index < foot_link_names_.size(); ++foot_index) {
            const std::string& link_name = foot_link_names_[foot_index];
            find_link(link_name);
            const auto link = link_index_by_name_.find(link_name);
            if (link == link_index_by_name_.end()) {
                throw std::runtime_error(fmt::format("foot link '{}' is missing from the batch link list",
                                                     link_name));
            }
            foot_link_indices_.push_back(link->second);
            const std::string_view sensor_name = foot_index < foot_contact_sensor_names_.size()
                                                         ? std::string_view(foot_contact_sensor_names_[foot_index])
                                                         : std::string_view();
            foot_shape_names_.push_back(FootGeomNameFromSensorOrLinkName(sensor_name, link_name));
        }
        foot_count_ = foot_link_indices_.size();

        const auto matching_shape_names = [&](const std::vector<std::regex>& patterns,
                                               std::string_view group_name) {
            std::vector<std::string> names;
            for (const PhysicsLinkSnapshot& link : robot_snapshot->links) {
                for (const PhysicsShapeSnapshot& shape : link.collision_shapes) {
                    if (!shape.name.empty() && MatchesAnyPattern(shape.name, patterns)) {
                        names.push_back(shape.name);
                    }
                }
            }
            if (!patterns.empty() && names.empty()) {
                throw std::runtime_error(fmt::format(
                        "contact shape group '{}' does not match any collision shape on robot '{}'",
                        group_name,
                        robot_name_));
            }
            return names;
        };
        thigh_shape_names_ = matching_shape_names(thigh_shape_patterns_, kThighContactGroup);
        shank_shape_names_ = matching_shape_names(shank_shape_patterns_, kShankContactGroup);
        trunk_head_shape_names_ = matching_shape_names(trunk_head_shape_patterns_, kTrunkHeadContactGroup);

        const std::size_t invalid_index = std::numeric_limits<std::size_t>::max();
        const auto add_sensor = [&](const std::string& name) -> std::size_t {
            find_sensor(name);
            const auto existing = sensor_index_by_name_.find(name);
            if (existing != sensor_index_by_name_.end()) {
                return existing->second;
            }
            const std::size_t index = sensor_names_.size();
            sensor_names_.push_back(name);
            sensor_index_by_name_[name] = index;
            return index;
        };
        foot_height_sensor_indices_.reserve(foot_height_sensor_names_.size());
        for (const std::string& sensor_name : foot_height_sensor_names_) {
            foot_height_sensor_indices_.push_back(add_sensor(sensor_name));
        }
        foot_contact_sensor_indices_.reserve(foot_contact_sensor_names_.size());
        for (const std::string& sensor_name : foot_contact_sensor_names_) {
            foot_contact_sensor_indices_.push_back(add_sensor(sensor_name));
        }
        height_scan_sensor_index_ = invalid_index;
        if (!height_scan_sensor_.empty()) {
            const std::size_t snapshot_index = find_sensor(height_scan_sensor_);
            height_scan_sensor_index_ = add_sensor(height_scan_sensor_);
            height_scan_count_ = robot_snapshot->sensors[snapshot_index].sample_offsets.size();
        }
        imu_sensor_index_ = invalid_index;
        std::size_t fallback_imu = invalid_index;
        for (std::size_t index = 0; index < robot_snapshot->sensors.size(); ++index) {
            const PhysicsSensorSnapshot& sensor = robot_snapshot->sensors[index];
            if (sensor.type != PhysicsSensorType::IMU) {
                continue;
            }
            if (fallback_imu == invalid_index) {
                fallback_imu = index;
            }
            if (sensor.name == "imu") {
                fallback_imu = index;
                break;
            }
        }
        if (fallback_imu != invalid_index) {
            imu_sensor_index_ = add_sensor(robot_snapshot->sensors[fallback_imu].name);
        }
    }

    void AllocateBuffers() {
        action_.assign(environment_count_ * joint_count_, 0.0f);
        submitted_action_.assign(environment_count_ * joint_count_, 0.0f);
        default_joint_position_.assign(joint_count_, 0.0f);
        action_scale_.assign(joint_count_, 0.0f);
        action_clip_.assign(1, 1.0f);
        push_force_.assign(environment_count_ * 3, 0.0f);
        push_torque_.assign(environment_count_ * 3, 0.0f);
        base_mass_delta_.assign(environment_count_, 0.0f);
        base_com_offset_.assign(environment_count_ * 3, 0.0f);
        joint_kp_.assign(environment_count_ * joint_count_, 0.0f);
        joint_kd_.assign(environment_count_ * joint_count_, 0.0f);
        foot_friction_.assign(environment_count_ * 3, 0.0f);
        foot_friction_enabled_.assign(environment_count_, 0.0f);
        previous_action_.assign(environment_count_ * joint_count_, 0.0f);
        last_action_.assign(environment_count_ * joint_count_, 0.0f);
        encoder_bias_.assign(environment_count_ * joint_count_, 0.0f);
        command_.assign(environment_count_ * 3, 0.0f);
        command_world_.assign(environment_count_ * 3, 0.0f);
        command_heading_target_.assign(environment_count_, 0.0f);
        command_heading_error_.assign(environment_count_, 0.0f);
        command_time_left_.assign(environment_count_, 0.0f);
        command_step_.assign(environment_count_, 0);
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
        upvector_.assign(environment_count_ * 3, 0.0f);
        base_height_.assign(environment_count_, 0.0f);
        joint_position_.assign(environment_count_ * joint_count_, 0.0f);
        joint_velocity_.assign(environment_count_ * joint_count_, 0.0f);
        joint_acceleration_.assign(environment_count_ * joint_count_, 0.0f);
        joint_torque_.assign(environment_count_ * joint_count_, 0.0f);
        joint_lower_limit_.assign(joint_count_, 0.0f);
        joint_upper_limit_.assign(joint_count_, 0.0f);
        link_position_.assign(environment_count_ * link_count_ * 3, 0.0f);
        link_quaternion_.assign(environment_count_ * link_count_ * 4, 0.0f);
        link_linear_velocity_.assign(environment_count_ * link_count_ * 3, 0.0f);
        link_angular_velocity_.assign(environment_count_ * link_count_ * 3, 0.0f);
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
        undesired_contact_count_.assign(environment_count_, 0.0f);
        undesired_base_contact_count_.assign(environment_count_, 0.0f);
        undesired_hip_contact_count_.assign(environment_count_, 0.0f);
        undesired_thigh_contact_count_.assign(environment_count_, 0.0f);
        undesired_calf_contact_count_.assign(environment_count_, 0.0f);
        self_collision_count_.assign(environment_count_, 0.0f);
        shank_collision_count_.assign(environment_count_, 0.0f);
        trunk_head_collision_count_.assign(environment_count_, 0.0f);
        base_collision_count_.assign(environment_count_, 0.0f);
        hip_collision_count_.assign(environment_count_, 0.0f);
        thigh_collision_count_.assign(environment_count_, 0.0f);
        calf_collision_count_.assign(environment_count_, 0.0f);
        foot_air_time_.assign(environment_count_ * foot_count_, 0.0f);
        foot_contact_time_.assign(environment_count_ * foot_count_, 0.0f);
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
        const PhysicsRobotSnapshot& robot = snapshot.robots[robot_index_];
        for (std::size_t joint_index = 0; joint_index < joint_names_.size(); ++joint_index) {
            const auto joint = std::find_if(
                    robot.joints.begin(),
                    robot.joints.end(),
                    [&](const PhysicsJointSnapshot& candidate) {
                        return candidate.name == joint_names_[joint_index];
                    });
            if (joint == robot.joints.end()) {
                throw std::runtime_error(fmt::format("joint '{}' disappeared from the physics snapshot",
                                                     joint_names_[joint_index]));
            }
            joint_lower_limit_[joint_index] = static_cast<float>(joint->lower_limit);
            joint_upper_limit_[joint_index] = static_cast<float>(joint->upper_limit);
        }
    }

    void PrepareActionTargets(bool simulate_action_latency) {
        const float configured_clip = action_clip_.empty() ? 1.0f : action_clip_[0];
        const bool clip_enabled = configured_clip >= 0.0f;
        const float clip_limit = std::max(0.0f, configured_clip);
        for (std::size_t env_id = 0; env_id < environment_count_; ++env_id) {
            for (std::size_t joint_index = 0; joint_index < joint_count_; ++joint_index) {
                const std::size_t offset = env_id * joint_count_ + joint_index;
                const float clipped = clip_enabled ? std::clamp(action_[offset], -clip_limit, clip_limit)
                                                   : action_[offset];
                previous_action_[offset] = last_action_[offset];
                submitted_action_[offset] = clipped;
                const float control_action = simulate_action_latency ? previous_action_[offset] : clipped;
                last_action_[offset] = clipped;
                target_position_[offset] = default_joint_position_[joint_index] +
                                           action_scale_[joint_index] * control_action -
                                           encoder_bias_[offset];
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
        if (command_zero_small_xy_threshold_ > 0.0f) {
            const float speed_xy = std::hypot(command_[env3 + 0], command_[env3 + 1]);
            if (speed_xy <= command_zero_small_xy_threshold_) {
                command_[env3 + 0] = 0.0f;
                command_[env3 + 1] = 0.0f;
            }
        }
        command_heading_target_[env_id] = Uniform(command_ranges_[kCommandHeadingMin],
                                                  command_ranges_[kCommandHeadingMax]);
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
        if (command_is_standing_env_[env_id] != 0) {
            command_[env3 + 0] = 0.0f;
            command_[env3 + 1] = 0.0f;
            command_[env3 + 2] = 0.0f;
        }
        command_heading_error_[env_id] = 0.0f;
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

    void AdvanceCommandStepAndResample(std::size_t env_id) {
        const std::uint32_t interval_steps =
                std::max<std::uint32_t>(
                        static_cast<std::uint32_t>(
                                std::max(1.0f, std::round(command_resampling_min_ / std::max(command_step_dt_, 1.0e-9f)))),
                        1);
        const std::uint32_t step = command_step_[env_id];
        if (step > 0 && step % interval_steps == 0) {
            ResampleCommand(env_id);
        }
        command_step_[env_id] = step + 1;
    }

    void ComputeCommands() {
        if (!command_step_resampling_enabled_) {
            AdvanceCommandTimersAndResample();
        } else {
            for (std::size_t env_id = 0; env_id < environment_count_; ++env_id) {
                AdvanceCommandStepAndResample(env_id);
            }
        }
        UpdateCommands();
    }

    void UpdateCommands() {
        for (std::size_t env_id = 0; env_id < environment_count_; ++env_id) {
            UpdateCommandForEnvironment(env_id);
        }
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

    PhysicsRobotBatchStepRequest MakePhysicsRequest(std::uint64_t ticks,
                                                    std::size_t workers,
                                                    bool include_task_inputs,
                                                    bool collect_contact_history) const {
        PhysicsRobotBatchStepRequest request;
        request.robot_name = robot_name_;
        request.base_link = base_link_;
        request.joint_names = joint_names_;
        request.link_names = link_names_;
        request.sensor_names = sensor_names_;
        request.target_positions.assign(target_position_.begin(), target_position_.end());
        request.ticks = ticks;
        request.worker_count = workers;
        request.collect_contact_history = collect_contact_history;
        request.ground_contact_force_threshold = ground_force_threshold_;
        request.self_contact_force_threshold = self_collision_force_threshold_;
        if (collect_contact_history) {
            const auto add_contact_group = [&](std::string_view name,
                                               const std::vector<std::string>& shape_names,
                                               RealType force_threshold) {
                if (!shape_names.empty()) {
                    request.contact_shape_groups.push_back({
                            std::string(name),
                            shape_names,
                            force_threshold});
                }
            };
            add_contact_group(kThighContactGroup, thigh_shape_names_, ground_force_threshold_);
            add_contact_group(kShankContactGroup, shank_shape_names_, ground_force_threshold_);
            add_contact_group(kTrunkHeadContactGroup, trunk_head_shape_names_, ground_force_threshold_);
            for (std::size_t foot_index = 0; foot_index < foot_shape_names_.size(); ++foot_index) {
                add_contact_group(FootContactGroupName(foot_index),
                                  {foot_shape_names_[foot_index]},
                                  0.0);
            }
        }
        if (!include_task_inputs) {
            return request;
        }

        request.joint_position_stiffness.assign(joint_kp_.begin(), joint_kp_.end());
        request.joint_velocity_damping.assign(joint_kd_.begin(), joint_kd_.end());
        request.override_link_names = {base_link_};
        request.link_mass_delta.assign(environment_count_, 0.0);
        request.link_center_of_mass_offset.assign(environment_count_ * 3, 0.0);
        request.override_shape_names = foot_shape_names_;
        const std::size_t foot_shape_count = request.override_shape_names.size();
        request.shape_friction.assign(environment_count_ * foot_shape_count * 3, 0.0);
        request.shape_friction_enabled.assign(environment_count_ * foot_shape_count, 0);
        for (std::size_t env_id = 0; env_id < environment_count_; ++env_id) {
            request.link_mass_delta[env_id] = base_mass_delta_[env_id];
            for (std::size_t axis = 0; axis < 3; ++axis) {
                request.link_center_of_mass_offset[env_id * 3 + axis] =
                        base_com_offset_[env_id * 3 + axis];
            }
            for (std::size_t foot_index = 0; foot_index < foot_shape_count; ++foot_index) {
                const std::size_t foot_scalar = env_id * foot_shape_count + foot_index;
                request.shape_friction_enabled[foot_scalar] = foot_friction_enabled_[env_id] > 0.5f ? 1 : 0;
                for (std::size_t axis = 0; axis < 3; ++axis) {
                    request.shape_friction[foot_scalar * 3 + axis] =
                            foot_friction_[env_id * 3 + axis];
                }
            }
        }

        request.external_wrench_link = base_link_;
        request.external_force.assign(push_force_.begin(), push_force_.end());
        request.external_torque.assign(push_torque_.begin(), push_torque_.end());
        return request;
    }

    void RunPhysicsBatch(std::uint64_t ticks,
                         std::size_t workers,
                         bool include_task_inputs,
                         bool collect_contact_history) {
        PhysicsRobotBatchStepRequest request = MakePhysicsRequest(
                ticks,
                workers,
                include_task_inputs,
                collect_contact_history);
        PhysicsRobotBatchStepResult result;
        if (!world_->StepRobotBatch(request, result)) {
            throw std::runtime_error(world_->GetLastError());
        }
        physics_state_ = std::move(result);
        if (include_task_inputs) {
            std::fill(push_force_.begin(), push_force_.end(), 0.0f);
            std::fill(push_torque_.begin(), push_torque_.end(), 0.0f);
        }
    }

    int FootIndexForShape(std::string_view shape_name) const {
        for (std::size_t index = 0; index < foot_shape_names_.size(); ++index) {
            if (shape_name == foot_shape_names_[index]) {
                return static_cast<int>(index);
            }
        }
        return -1;
    }

    void CopyPhysicsState() {
        const PhysicsRobotBatchStepResult& state = physics_state_;
        if (state.environment_count != environment_count_ ||
            state.joint_names != joint_names_ ||
            state.link_names != link_names_ ||
            state.sensor_names != sensor_names_) {
            throw std::runtime_error("Physics batch result does not match the locomotion view contract");
        }
        std::copy(state.joint_lower_limit.begin(),
                  state.joint_lower_limit.end(),
                  joint_lower_limit_.begin());
        std::copy(state.joint_upper_limit.begin(),
                  state.joint_upper_limit.end(),
                  joint_upper_limit_.begin());

        const std::size_t invalid_index = std::numeric_limits<std::size_t>::max();
        for (std::size_t env_id = 0; env_id < environment_count_; ++env_id) {
            ClearContactBuffersForEnvironment(env_id);
            const std::size_t env3 = env_id * 3;
            const std::size_t env4 = env_id * 4;
            for (std::size_t axis = 0; axis < 3; ++axis) {
                base_position_[env3 + axis] = static_cast<float>(state.base_position[env3 + axis]);
                base_linear_velocity_[env3 + axis] =
                        static_cast<float>(state.base_linear_velocity[env3 + axis]);
                base_angular_velocity_[env3 + axis] =
                        static_cast<float>(state.base_angular_velocity[env3 + axis]);
            }
            for (std::size_t axis = 0; axis < 4; ++axis) {
                base_quaternion_[env4 + axis] = static_cast<float>(state.base_quaternion[env4 + axis]);
            }
            base_height_[env_id] = base_position_[env3 + 2];

            const Quaternion base_orientation(state.base_quaternion[env4 + 0],
                                               state.base_quaternion[env4 + 1],
                                               state.base_quaternion[env4 + 2],
                                               state.base_quaternion[env4 + 3]);
            const Matrix3 base_rotation = base_orientation.normalized().toRotationMatrix();
            const Vector3 linear_world(state.base_linear_velocity[env3 + 0],
                                       state.base_linear_velocity[env3 + 1],
                                       state.base_linear_velocity[env3 + 2]);
            const Vector3 angular_world(state.base_angular_velocity[env3 + 0],
                                        state.base_angular_velocity[env3 + 1],
                                        state.base_angular_velocity[env3 + 2]);
            const Vector3 linear_body = base_rotation.transpose() * linear_world;
            const Vector3 angular_body = base_rotation.transpose() * angular_world;
            const Vector3 gravity_body = base_rotation.transpose() * Vector3(0.0, 0.0, -1.0);
            for (std::size_t axis = 0; axis < 3; ++axis) {
                base_linear_velocity_body_[env3 + axis] = static_cast<float>(linear_body[axis]);
                base_angular_velocity_body_[env3 + axis] = static_cast<float>(angular_body[axis]);
                projected_gravity_[env3 + axis] = static_cast<float>(gravity_body[axis]);
                upvector_[env3 + axis] = static_cast<float>(base_rotation(axis, 2));
            }

            if (imu_sensor_index_ != invalid_index &&
                imu_sensor_index_ < sensor_names_.size()) {
                const std::size_t sensor3 = (env_id * sensor_names_.size() + imu_sensor_index_) * 3;
                const std::size_t sensor4 = (env_id * sensor_names_.size() + imu_sensor_index_) * 4;
                const Quaternion imu_orientation(state.sensor_quaternion[sensor4 + 0],
                                                 state.sensor_quaternion[sensor4 + 1],
                                                 state.sensor_quaternion[sensor4 + 2],
                                                 state.sensor_quaternion[sensor4 + 3]);
                const Matrix3 imu_rotation = imu_orientation.normalized().toRotationMatrix();
                for (std::size_t axis = 0; axis < 3; ++axis) {
                    upvector_[env3 + axis] = static_cast<float>(imu_rotation(axis, 2));
                }
                const std::size_t sensor_values =
                        (env_id * sensor_names_.size() + imu_sensor_index_) * state.max_sensor_values;
                if (imu_sensor_index_ < state.sensor_value_count.size() &&
                    state.sensor_value_count[imu_sensor_index_] >= 10) {
                    for (std::size_t axis = 0; axis < 3; ++axis) {
                        base_angular_velocity_body_[env3 + axis] =
                                static_cast<float>(state.sensor_values[sensor_values + 4 + axis]);
                        base_linear_velocity_body_[env3 + axis] =
                                static_cast<float>(state.sensor_values[sensor_values + 7 + axis]);
                    }
                }
                GOB_UNUSED(sensor3);
            }

            for (std::size_t joint_index = 0; joint_index < joint_count_; ++joint_index) {
                const std::size_t offset = env_id * joint_count_ + joint_index;
                joint_position_[offset] = static_cast<float>(state.joint_position[offset]);
                joint_velocity_[offset] = static_cast<float>(state.joint_velocity[offset]);
                joint_acceleration_[offset] = static_cast<float>(state.joint_acceleration[offset]);
                joint_torque_[offset] = static_cast<float>(state.joint_effort[offset]);
            }

            for (std::size_t link_index = 0; link_index < link_count_; ++link_index) {
                const std::size_t link3 = (env_id * link_count_ + link_index) * 3;
                const std::size_t link4 = (env_id * link_count_ + link_index) * 4;
                for (std::size_t axis = 0; axis < 3; ++axis) {
                    link_position_[link3 + axis] = static_cast<float>(state.link_position[link3 + axis]);
                    link_linear_velocity_[link3 + axis] =
                            static_cast<float>(state.link_linear_velocity[link3 + axis]);
                    link_angular_velocity_[link3 + axis] =
                            static_cast<float>(state.link_angular_velocity[link3 + axis]);
                }
                for (std::size_t axis = 0; axis < 4; ++axis) {
                    link_quaternion_[link4 + axis] = static_cast<float>(state.link_quaternion[link4 + axis]);
                }
            }

            for (std::size_t foot_index = 0; foot_index < foot_count_; ++foot_index) {
                const std::size_t foot = env_id * foot_count_ + foot_index;
                const std::size_t foot3 = foot * 3;
                const std::size_t foot4 = foot * 4;
                const std::size_t link_index = foot_link_indices_[foot_index];
                const std::size_t link3 = (env_id * link_count_ + link_index) * 3;
                const std::size_t link4 = (env_id * link_count_ + link_index) * 4;
                for (std::size_t axis = 0; axis < 3; ++axis) {
                    foot_position_[foot3 + axis] = link_position_[link3 + axis];
                    foot_velocity_[foot3 + axis] = link_linear_velocity_[link3 + axis];
                }
                for (std::size_t axis = 0; axis < 4; ++axis) {
                    foot_quaternion_[foot4 + axis] = link_quaternion_[link4 + axis];
                }

                if (foot_index < foot_contact_sensor_indices_.size()) {
                    const std::size_t sensor_index = foot_contact_sensor_indices_[foot_index];
                    const std::size_t sensor3 = (env_id * sensor_names_.size() + sensor_index) * 3;
                    for (std::size_t axis = 0; axis < 3; ++axis) {
                        foot_position_[foot3 + axis] = static_cast<float>(state.sensor_position[sensor3 + axis]);
                        foot_velocity_[foot3 + axis] =
                                static_cast<float>(state.sensor_linear_velocity[sensor3 + axis]);
                    }
                }
                if (foot_index < foot_height_sensor_indices_.size()) {
                    const std::size_t sensor_index = foot_height_sensor_indices_[foot_index];
                    const std::size_t value_offset =
                            (env_id * sensor_names_.size() + sensor_index) * state.max_sensor_values;
                    if (sensor_index < state.sensor_value_count.size() &&
                        state.sensor_value_count[sensor_index] > 0) {
                        foot_height_[foot] = static_cast<float>(state.sensor_values[value_offset]);
                    }
                }
            }

            if (height_scan_sensor_index_ != invalid_index && height_scan_count_ > 0) {
                const std::size_t sensor_index = height_scan_sensor_index_;
                const std::size_t value_offset =
                        (env_id * sensor_names_.size() + sensor_index) * state.max_sensor_values;
                const std::size_t hit_offset =
                        (env_id * sensor_names_.size() + sensor_index) * state.max_sensor_hits;
                for (std::size_t sample = 0; sample < height_scan_count_; ++sample) {
                    const std::size_t output = env_id * height_scan_count_ + sample;
                    height_scan_[output] = static_cast<float>(state.sensor_values[value_offset + sample]);
                    height_scan_hit_[output] = state.sensor_hit[hit_offset + sample];
                    for (std::size_t axis = 0; axis < 3; ++axis) {
                        height_scan_point_[output * 3 + axis] =
                                static_cast<float>(state.sensor_hit_point[(hit_offset + sample) * 3 + axis]);
                        height_scan_normal_[output * 3 + axis] =
                                static_cast<float>(state.sensor_hit_normal[(hit_offset + sample) * 3 + axis]);
                    }
                }
            }

            const std::size_t contact_count = env_id < state.contact_count.size()
                                                      ? static_cast<std::size_t>(std::max(state.contact_count[env_id], 0))
                                                      : 0;
            for (std::size_t contact_index = 0;
                 contact_index < std::min(contact_count, state.max_contact_count);
                 ++contact_index) {
                const std::size_t contact = env_id * state.max_contact_count + contact_index;
                const int link_a = state.contact_link_index[contact * 2 + 0];
                const int link_b = state.contact_link_index[contact * 2 + 1];
                const int shape_a = state.contact_shape_index[contact * 2 + 0];
                const int shape_b = state.contact_shape_index[contact * 2 + 1];
                const bool self_contact = link_a >= 0 && link_b >= 0;
                const int robot_link = link_a >= 0 ? link_a : link_b;
                const int robot_shape = shape_a >= 0 ? shape_a : shape_b;
                if (self_contact) {
                    self_collision_count_[env_id] += 1.0f;
                    continue;
                }
                if (robot_link < 0 || static_cast<std::size_t>(robot_link) >= link_names_.size()) {
                    continue;
                }
                const std::string& link_name = link_names_[static_cast<std::size_t>(robot_link)];
                const std::string_view shape_name =
                        robot_shape >= 0 && static_cast<std::size_t>(robot_shape) < state.shape_names.size()
                                ? std::string_view(state.shape_names[static_cast<std::size_t>(robot_shape)])
                                : std::string_view();
                const int foot_index = FootIndexForShape(shape_name);
                if (foot_index >= 0) {
                    const std::size_t foot = env_id * foot_count_ + static_cast<std::size_t>(foot_index);
                    const std::size_t foot3 = foot * 3;
                    foot_contact_[foot] = 1.0f;
                    for (std::size_t axis = 0; axis < 3; ++axis) {
                        foot_contact_force_[foot3 + axis] +=
                                static_cast<float>(state.contact_force[contact * 3 + axis]);
                    }
                }

                const RealType normal_force = state.contact_normal_force[contact];
                if (normal_force < ground_force_threshold_) {
                    continue;
                }
                const std::string match_name = shape_name.empty() ? link_name : std::string(shape_name);
                const bool is_base = link_name == base_link_ ||
                                     MatchesAnyPattern(match_name, trunk_head_shape_patterns_);
                const bool is_hip = ContainsCaseInsensitive(link_name, "hip");
                const bool is_thigh = MatchesAnyPattern(match_name, thigh_shape_patterns_);
                const bool is_shank = foot_index < 0 && MatchesAnyPattern(match_name, shank_shape_patterns_);
                base_collision_count_[env_id] += is_base ? 1.0f : 0.0f;
                hip_collision_count_[env_id] += is_hip ? 1.0f : 0.0f;
                thigh_collision_count_[env_id] += is_thigh ? 1.0f : 0.0f;
                calf_collision_count_[env_id] += is_shank ? 1.0f : 0.0f;
                if (is_base || is_hip || is_thigh || is_shank) {
                    undesired_contact_count_[env_id] += 1.0f;
                    undesired_base_contact_count_[env_id] += is_base ? 1.0f : 0.0f;
                    undesired_hip_contact_count_[env_id] += is_hip ? 1.0f : 0.0f;
                    undesired_thigh_contact_count_[env_id] += is_thigh ? 1.0f : 0.0f;
                    undesired_calf_contact_count_[env_id] += is_shank ? 1.0f : 0.0f;
                }
            }

            if (env_id < state.self_contact_tick_count.size()) {
                self_collision_count_[env_id] =
                        static_cast<float>(state.self_contact_tick_count[env_id]);
            }
            const auto group_tick_count = [&](std::string_view group_name) {
                const auto group = std::find(state.contact_shape_group_names.begin(),
                                             state.contact_shape_group_names.end(),
                                             group_name);
                if (group == state.contact_shape_group_names.end()) {
                    return 0.0f;
                }
                const std::size_t group_index = static_cast<std::size_t>(
                        std::distance(state.contact_shape_group_names.begin(), group));
                const std::size_t offset =
                        env_id * state.contact_shape_group_names.size() + group_index;
                return offset < state.contact_shape_group_tick_count.size()
                               ? static_cast<float>(state.contact_shape_group_tick_count[offset])
                               : 0.0f;
            };
            illegal_contact_count_[env_id] = terminate_on_thigh_contact_
                                                      ? group_tick_count(kThighContactGroup)
                                                      : 0.0f;
            shank_collision_count_[env_id] = group_tick_count(kShankContactGroup);
            trunk_head_collision_count_[env_id] = group_tick_count(kTrunkHeadContactGroup);
        }
    }

    std::size_t DefaultActorObservationDim() const {
        return 12 + joint_count_ * 3 + height_scan_count_;
    }

    std::size_t DefaultCriticObservationDim() const {
        return DefaultActorObservationDim() + foot_count_ * 6;
    }

    void UpdateFootHistory(std::size_t env_id) {
        const float step_dt = std::max(command_step_dt_, 1.0e-6f);
        const float physics_dt = static_cast<float>(
                std::max(world_->GetSettings().fixed_time_step, static_cast<RealType>(1.0e-6)));
        for (std::size_t foot_index = 0; foot_index < foot_count_; ++foot_index) {
            const std::size_t foot = env_id * foot_count_ + foot_index;
            const std::size_t foot3 = foot * 3;
            bool contact = foot_contact_[foot] > 0.0f;
            const std::string group_name = FootContactGroupName(foot_index);
            const auto group = std::find(physics_state_.contact_shape_group_names.begin(),
                                         physics_state_.contact_shape_group_names.end(),
                                         group_name);
            if (group != physics_state_.contact_shape_group_names.end() &&
                physics_state_.contact_history_tick_count > 0) {
                const std::size_t group_index = static_cast<std::size_t>(
                        std::distance(physics_state_.contact_shape_group_names.begin(), group));
                for (std::size_t tick = 0; tick < physics_state_.contact_history_tick_count; ++tick) {
                    const std::size_t history_offset =
                            (env_id * physics_state_.contact_shape_group_names.size() + group_index) *
                                    physics_state_.contact_history_tick_count +
                            tick;
                    const bool tick_contact =
                            history_offset < physics_state_.contact_shape_group_history.size() &&
                            physics_state_.contact_shape_group_history[history_offset] != 0;
                    if (tick_contact) {
                        foot_contact_time_[foot] += physics_dt;
                        foot_air_time_[foot] = 0.0f;
                    } else {
                        foot_contact_time_[foot] = 0.0f;
                        foot_air_time_[foot] += physics_dt;
                    }
                    contact = tick_contact;
                }
                foot_contact_[foot] = contact ? 1.0f : 0.0f;
            } else if (contact) {
                foot_contact_time_[foot] += step_dt;
                foot_air_time_[foot] = 0.0f;
            } else {
                foot_contact_time_[foot] = 0.0f;
                foot_air_time_[foot] += step_dt;
            }
            first_contact_[foot] = contact &&
                                                   foot_contact_time_[foot] > 0.0f &&
                                                   foot_contact_time_[foot] < step_dt + 1.0e-6f
                                           ? 1.0f
                                           : 0.0f;
            const float force_x = foot_contact_force_[foot3 + 0];
            const float force_y = foot_contact_force_[foot3 + 1];
            const float force_z = foot_contact_force_[foot3 + 2];
            landing_force_[foot] = std::sqrt(force_x * force_x + force_y * force_y + force_z * force_z) *
                                   first_contact_[foot];
            if (!contact) {
                foot_peak_height_[foot] = std::max(foot_peak_height_[foot], foot_height_[foot]);
            }
            last_foot_contact_[foot] = contact ? 1.0f : 0.0f;
        }
    }

    void ClearContactBuffersForEnvironment(std::size_t env_id) {
        illegal_contact_count_[env_id] = 0.0f;
        undesired_contact_count_[env_id] = 0.0f;
        undesired_base_contact_count_[env_id] = 0.0f;
        undesired_hip_contact_count_[env_id] = 0.0f;
        undesired_thigh_contact_count_[env_id] = 0.0f;
        undesired_calf_contact_count_[env_id] = 0.0f;
        self_collision_count_[env_id] = 0.0f;
        shank_collision_count_[env_id] = 0.0f;
        trunk_head_collision_count_[env_id] = 0.0f;
        base_collision_count_[env_id] = 0.0f;
        hip_collision_count_[env_id] = 0.0f;
        thigh_collision_count_[env_id] = 0.0f;
        calf_collision_count_[env_id] = 0.0f;
        const std::size_t foot_begin = env_id * foot_count_;
        for (std::size_t foot_index = 0; foot_index < foot_count_; ++foot_index) {
            const std::size_t foot = foot_begin + foot_index;
            const std::size_t foot3 = foot * 3;
            foot_contact_[foot] = 0.0f;
            first_contact_[foot] = 0.0f;
            landing_force_[foot] = 0.0f;
            foot_contact_force_[foot3 + 0] = 0.0f;
            foot_contact_force_[foot3 + 1] = 0.0f;
            foot_contact_force_[foot3 + 2] = 0.0f;
        }
    }

    Ref<PhysicsWorld> world_;
    std::string robot_name_;
    std::string base_link_;
    std::vector<std::string> joint_names_;
    std::vector<std::string> link_names_;
    std::unordered_map<std::string, std::size_t> link_index_by_name_;
    std::vector<std::string> foot_link_names_;
    std::vector<std::string> foot_height_sensor_names_;
    std::vector<std::string> foot_contact_sensor_names_;
    std::string height_scan_sensor_;
    std::vector<std::string> sensor_names_;
    std::unordered_map<std::string, std::size_t> sensor_index_by_name_;
    std::vector<std::regex> thigh_shape_patterns_;
    std::vector<std::regex> shank_shape_patterns_;
    std::vector<std::regex> trunk_head_shape_patterns_;
    bool terminate_on_thigh_contact_{true};
    RealType ground_force_threshold_{50.0};
    RealType self_collision_force_threshold_{20.0};

    std::size_t robot_index_{0};
    std::size_t environment_count_{0};
    std::size_t joint_count_{0};
    std::size_t link_count_{0};
    std::size_t foot_count_{0};
    std::size_t height_scan_count_{0};
    std::size_t reward_term_count_{0};
    std::size_t task_param_count_{0};
    std::size_t task_flag_count_{0};
    std::size_t actor_obs_dim_{0};
    std::size_t critic_obs_dim_{0};
    std::vector<std::size_t> foot_link_indices_;
    std::vector<std::string> foot_shape_names_;
    std::vector<std::string> thigh_shape_names_;
    std::vector<std::string> shank_shape_names_;
    std::vector<std::string> trunk_head_shape_names_;
    std::vector<std::size_t> foot_height_sensor_indices_;
    std::vector<std::size_t> foot_contact_sensor_indices_;
    std::size_t height_scan_sensor_index_{std::numeric_limits<std::size_t>::max()};
    std::size_t imu_sensor_index_{std::numeric_limits<std::size_t>::max()};
    PhysicsRobotBatchStepResult physics_state_;
    std::vector<float> action_;
    std::vector<float> submitted_action_;
    std::vector<float> default_joint_position_;
    std::vector<float> action_scale_;
    std::vector<float> action_clip_{1.0f};
    std::vector<float> push_force_;
    std::vector<float> push_torque_;
    std::vector<float> base_mass_delta_;
    std::vector<float> base_com_offset_;
    std::vector<float> joint_kp_;
    std::vector<float> joint_kd_;
    std::vector<float> foot_friction_;
    std::vector<float> foot_friction_enabled_;
    std::vector<float> previous_action_;
    std::vector<float> last_action_;
    std::vector<float> encoder_bias_;
    std::vector<float> command_;
    std::vector<float> command_world_;
    std::vector<float> command_heading_target_;
    std::vector<float> command_heading_error_;
    std::vector<float> command_time_left_;
    std::vector<std::uint32_t> command_step_;
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
    bool command_step_resampling_enabled_{false};
    float command_step_dt_{0.02f};
    float command_resampling_min_{3.0f};
    float command_resampling_max_{8.0f};
    float command_rel_standing_envs_{0.1f};
    float command_rel_heading_envs_{0.3f};
    float command_rel_world_envs_{0.0f};
    float command_rel_forward_envs_{0.2f};
    float command_heading_stiffness_{0.5f};
    float command_zero_small_xy_threshold_{0.0f};
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
    std::vector<float> upvector_;
    std::vector<float> base_height_;
    std::vector<float> joint_position_;
    std::vector<float> joint_velocity_;
    std::vector<float> joint_acceleration_;
    std::vector<float> joint_torque_;
    std::vector<float> joint_lower_limit_;
    std::vector<float> joint_upper_limit_;
    std::vector<float> link_position_;
    std::vector<float> link_quaternion_;
    std::vector<float> link_linear_velocity_;
    std::vector<float> link_angular_velocity_;
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
    std::vector<float> undesired_contact_count_;
    std::vector<float> undesired_base_contact_count_;
    std::vector<float> undesired_hip_contact_count_;
    std::vector<float> undesired_thigh_contact_count_;
    std::vector<float> undesired_calf_contact_count_;
    std::vector<float> self_collision_count_;
    std::vector<float> shank_collision_count_;
    std::vector<float> trunk_head_collision_count_;
    std::vector<float> base_collision_count_;
    std::vector<float> hip_collision_count_;
    std::vector<float> thigh_collision_count_;
    std::vector<float> calf_collision_count_;
    std::vector<float> foot_air_time_;
    std::vector<float> foot_contact_time_;
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
};

void RegisterManualAppContextBindings(py::module_& module) {
    py::class_<PyLocomotionBatchView, std::shared_ptr<PyLocomotionBatchView>>(
            module,
            "_LocomotionBatchView")
            .def("arrays", &PyLocomotionBatchView::Arrays)
            .def("terrain_heights",
                 &PyLocomotionBatchView::TerrainHeights,
                 py::arg("points"),
                 py::arg("env_id") = 0)
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
                 py::arg("zero_small_xy_threshold"),
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
            .def("set_commands",
                 &PyLocomotionBatchView::SetCommands,
                 py::arg("env_ids"),
                 py::arg("commands"),
                 py::arg("heading_targets"),
                 py::arg("time_left"))
            .def("set_command_steps",
                 &PyLocomotionBatchView::SetCommandSteps,
                 py::arg("env_ids"),
                 py::arg("steps"))
            .def("set_command_step_resampling",
                 &PyLocomotionBatchView::SetCommandStepResampling,
                 py::arg("enabled"))
            .def("advance_commands", &PyLocomotionBatchView::AdvanceCommands)
            .def("update_command_frames", &PyLocomotionBatchView::UpdateCommandFrames)
            .def("refresh",
                 &PyLocomotionBatchView::Refresh,
                 py::call_guard<py::gil_scoped_release>())
            .def("close",
                 &PyLocomotionBatchView::Close)
            .def("reset",
                 &PyLocomotionBatchView::Reset,
                 py::arg("env_ids"),
                 py::call_guard<py::gil_scoped_release>())
            .def("clear_reset_contacts",
                 &PyLocomotionBatchView::ClearResetContacts,
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
                 py::arg("angular_velocities"),
                 py::arg("refresh") = true);

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
                                                    const std::vector<std::string>& thigh_shape_patterns,
                                                    const std::vector<std::string>& shank_shape_patterns,
                                                    const std::vector<std::string>& trunk_head_shape_patterns,
                                                    bool terminate_on_thigh_contact,
                                                    double ground_force_threshold,
                                                    double self_collision_force_threshold,
                                                    std::size_t reward_term_count,
                                                    std::size_t task_param_count,
                                                    std::size_t task_flag_count,
                                                    std::size_t actor_obs_dim,
                                                    std::size_t critic_obs_dim,
                                                    const std::vector<std::string>& link_names) {
                SimulationServer* simulation = context.GetSimulationServer();
                if (simulation == nullptr) {
                    throw std::runtime_error("active Gobot app context has no SimulationServer");
                }
                Ref<PhysicsWorld> world = simulation->GetWorld();
                if (!world.IsValid()) {
                    throw std::runtime_error("active batch physics world is unavailable");
                }
                return std::make_shared<PyLocomotionBatchView>(world,
                                                               robot,
                                                               base_link,
                                                               joint_names,
                                                               link_names,
                                                               foot_link_names,
                                                               foot_height_sensor_names,
                                                               foot_contact_sensor_names,
                                                               height_scan_sensor,
                                                               thigh_shape_patterns,
                                                               shank_shape_patterns,
                                                               trunk_head_shape_patterns,
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
               py::arg("thigh_shape_patterns") = std::vector<std::string>{},
               py::arg("shank_shape_patterns") = std::vector<std::string>{},
               py::arg("trunk_head_shape_patterns") = std::vector<std::string>{},
               py::arg("terminate_on_thigh_contact") = true,
               py::arg("ground_force_threshold") = 50.0,
               py::arg("self_collision_force_threshold") = 20.0,
               py::arg("reward_term_count") = 0,
               py::arg("task_param_count") = 0,
               py::arg("task_flag_count") = 0,
               py::arg("actor_obs_dim") = 0,
               py::arg("critic_obs_dim") = 0,
               py::arg("link_names") = std::vector<std::string>{})
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
            .def("set_mujoco_solver_settings", [](EngineContext& context, py::dict settings) {
                SimulationServer* simulation = context.GetSimulationServer();
                if (simulation == nullptr) {
                    throw std::runtime_error("active Gobot app context has no SimulationServer");
                }
                PhysicsWorldSettings world_settings = simulation->GetPhysicsWorldSettings();
                world_settings.mujoco_solver = DictToReflected<MuJoCoSolverSettings>(settings);
                simulation->SetPhysicsWorldSettings(world_settings);
            }, py::arg("settings"))
            .def("get_mujoco_solver_settings", [](EngineContext& context) {
                SimulationServer* simulation = context.GetSimulationServer();
                if (simulation == nullptr) {
                    throw std::runtime_error("active Gobot app context has no SimulationServer");
                }
                return ReflectedToPythonDict(simulation->GetPhysicsWorldSettings().mujoco_solver);
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
                PhysicsRobotBatchStepRequest request;
                request.robot_name = robot;
                request.base_link = base_link;
                request.joint_names = joint_names;
                request.link_names = link_names;
                request.sensor_names = sensor_names;
                request.target_positions = std::move(targets);
                request.ticks = ticks;
                request.worker_count = workers;
                PhysicsRobotBatchStepResult arrays;
                {
                    py::gil_scoped_release release;
                    if (!world->StepRobotBatch(request, arrays)) {
                        py::gil_scoped_acquire acquire;
                        throw std::runtime_error(world->GetLastError());
                    }
                }
                return RobotBatchStepResultToPythonDict(std::move(arrays));
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
