#include "manual_bindings_internal.hpp"

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

class PyGo1LocomotionBatchView {
public:
    PyGo1LocomotionBatchView(Ref<MuJoCoPhysicsWorld> world,
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
                             double self_collision_force_threshold)
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
          self_collision_force_threshold_(static_cast<RealType>(self_collision_force_threshold)) {
        Initialize();
        AllocateBuffers();
        Refresh();
    }

    py::dict Arrays() {
        py::object owner = py::cast(this, py::return_value_policy::reference);
        py::dict arrays;
        arrays["target_position"] =
                VectorArrayView(target_position_, {EnvDim(), JointDim()}, owner, true);
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
        arrays["base_quaternion"] =
                VectorArrayView(base_quaternion_, {EnvDim(), 4}, owner, false);
        arrays["base_linear_velocity"] =
                VectorArrayView(base_linear_velocity_, {EnvDim(), 3}, owner, false);
        arrays["base_angular_velocity"] =
                VectorArrayView(base_angular_velocity_, {EnvDim(), 3}, owner, false);
        arrays["joint_position"] =
                VectorArrayView(joint_position_, {EnvDim(), JointDim()}, owner, false);
        arrays["joint_velocity"] =
                VectorArrayView(joint_velocity_, {EnvDim(), JointDim()}, owner, false);
        arrays["joint_lower_limit"] =
                VectorArrayView(joint_lower_limit_, {JointDim()}, owner, false);
        arrays["joint_upper_limit"] =
                VectorArrayView(joint_upper_limit_, {JointDim()}, owner, false);
        arrays["foot_position"] =
                VectorArrayView(foot_position_, {EnvDim(), FootDim(), 3}, owner, false);
        arrays["foot_velocity"] =
                VectorArrayView(foot_velocity_, {EnvDim(), FootDim(), 3}, owner, false);
        arrays["foot_height"] =
                VectorArrayView(foot_height_, {EnvDim(), FootDim()}, owner, false);
        arrays["foot_contact"] =
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
        return arrays;
    }

    void Step(std::uint64_t ticks, std::size_t workers) {
#ifdef GOBOT_HAS_MUJOCO
        ApplyTargetPositions();
        const RealType fixed_time_step = world_->GetSettings().fixed_time_step;
        if (!world_->StepEnvironmentBatchInternal(fixed_time_step, ticks, workers, false, false)) {
            throw std::runtime_error(world_->GetLastError());
        }
        FillAllEnvironments();
#else
        GOB_UNUSED(ticks);
        GOB_UNUSED(workers);
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
        FillAllEnvironments();
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
        FillEnvironment(env_id);
#else
        GOB_UNUSED(env_id);
        GOB_UNUSED(linear_velocity);
        GOB_UNUSED(angular_velocity);
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

    void AllocateBuffers() {
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
        joint_position_.assign(environment_count_ * joint_count_, 0.0f);
        joint_velocity_.assign(environment_count_ * joint_count_, 0.0f);
        joint_lower_limit_.assign(joint_count_, 0.0f);
        joint_upper_limit_.assign(joint_count_, 0.0f);
        foot_position_.assign(environment_count_ * foot_count_ * 3, 0.0f);
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

        const PhysicsSceneSnapshot& snapshot = world_->GetSceneSnapshot();
        for (std::size_t joint_index = 0; joint_index < joint_binding_indices_.size(); ++joint_index) {
            const auto& binding = world_->joint_bindings_[joint_binding_indices_[joint_index]];
            const PhysicsJointSnapshot& joint = snapshot.robots[binding.robot_index].joints[binding.joint_index];
            joint_lower_limit_[joint_index] = static_cast<float>(joint.lower_limit);
            joint_upper_limit_[joint_index] = static_cast<float>(joint.upper_limit);
        }
    }

    void ApplyTargetPositions() {
        auto* model = static_cast<mjModel*>(world_->model_);
        if (model == nullptr) {
            throw std::runtime_error("MuJoCo model has not been built");
        }
        for (std::size_t env_id = 0; env_id < environment_count_; ++env_id) {
            auto* data = static_cast<mjData*>(world_->environment_data_[env_id]);
            if (data == nullptr) {
                throw std::runtime_error(fmt::format("MuJoCo data for environment {} is not available", env_id));
            }
            for (std::size_t joint_index = 0; joint_index < joint_count_; ++joint_index) {
                const auto& binding = world_->joint_bindings_[joint_binding_indices_[joint_index]];
                const int actuator_id = binding.position_actuator_id >= 0
                                                ? binding.position_actuator_id
                                                : binding.motor_actuator_id;
                if (actuator_id < 0 || actuator_id >= model->nu) {
                    throw std::runtime_error(fmt::format("joint '{}' has no usable MuJoCo actuator",
                                                         joint_names_[joint_index]));
                }
                data->ctrl[actuator_id] =
                        static_cast<mjtNum>(target_position_[env_id * joint_count_ + joint_index]);
            }
        }
    }

    void FillAllEnvironments() {
        for (std::size_t env_id = 0; env_id < environment_count_; ++env_id) {
            FillEnvironment(env_id);
        }
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

        for (std::size_t joint_index = 0; joint_index < joint_count_; ++joint_index) {
            const auto& binding = world_->joint_bindings_[joint_binding_indices_[joint_index]];
            const std::size_t offset = env_id * joint_count_ + joint_index;
            joint_position_[offset] = binding.qpos_address >= 0 && binding.qpos_address < model->nq
                                              ? static_cast<float>(data->qpos[binding.qpos_address])
                                              : 0.0f;
            joint_velocity_[offset] = binding.dof_address >= 0 && binding.dof_address < model->nv
                                              ? static_cast<float>(data->qvel[binding.dof_address])
                                              : 0.0f;
        }

        for (std::size_t foot_index = 0; foot_index < foot_count_; ++foot_index) {
            const auto& binding = world_->link_bindings_[foot_link_binding_indices_[foot_index]];
            const int body_id = binding.body_id;
            const std::size_t foot3 = (env_id * foot_count_ + foot_index) * 3;
            foot_position_[foot3 + 0] = static_cast<float>(data->xpos[3 * body_id + 0]);
            foot_position_[foot3 + 1] = static_cast<float>(data->xpos[3 * body_id + 1]);
            foot_position_[foot3 + 2] = static_cast<float>(data->xpos[3 * body_id + 2]);
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
    std::vector<float> joint_position_;
    std::vector<float> joint_velocity_;
    std::vector<float> joint_lower_limit_;
    std::vector<float> joint_upper_limit_;
    std::vector<float> foot_position_;
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
};

void RegisterManualAppContextBindings(py::module_& module) {
    py::class_<PyGo1LocomotionBatchView, std::shared_ptr<PyGo1LocomotionBatchView>>(
            module,
            "_Go1LocomotionBatchView")
            .def("arrays", &PyGo1LocomotionBatchView::Arrays)
            .def("step", &PyGo1LocomotionBatchView::Step, py::arg("ticks") = 1, py::arg("workers") = 0)
            .def("refresh", &PyGo1LocomotionBatchView::Refresh)
            .def("reset", &PyGo1LocomotionBatchView::Reset, py::arg("env_ids"))
            .def("set_base_velocity",
                 &PyGo1LocomotionBatchView::SetBaseVelocity,
                 py::arg("env_id"),
                 py::arg("linear_velocity"),
                 py::arg("angular_velocity"));

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
            .def("create_go1_locomotion_batch_view", [](EngineContext& context,
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
                                                        double self_collision_force_threshold) {
                SimulationServer* simulation = context.GetSimulationServer();
                if (simulation == nullptr) {
                    throw std::runtime_error("active Gobot app context has no SimulationServer");
                }
                Ref<PhysicsWorld> world = simulation->GetWorld();
                auto mujoco_world = world.DynamicPointerCast<MuJoCoPhysicsWorld>();
                if (!mujoco_world.IsValid()) {
                    throw std::runtime_error("active batch world is not a MuJoCo world");
                }
                return std::make_shared<PyGo1LocomotionBatchView>(mujoco_world,
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
                                                                  self_collision_force_threshold);
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
               py::arg("self_collision_force_threshold") = 20.0)
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
