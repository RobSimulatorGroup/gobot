#include "gobot/python/native_vector_env.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include <nlohmann/json.hpp>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "gobot/core/io/resource_loader.hpp"
#include "gobot/physics/joint_controller.hpp"
#include "gobot/physics/physics_server.hpp"
#include "gobot/physics/physics_world.hpp"
#include "gobot/scene/joint_3d.hpp"
#include "gobot/scene/node.hpp"
#include "gobot/scene/resources/packed_scene.hpp"

namespace gobot::python {
namespace {

using json = nlohmann::json;

constexpr RealType kInf = std::numeric_limits<RealType>::infinity();

RealType JsonReal(const json& value, const std::string& key, RealType fallback) {
    if (!value.is_object() || !value.contains(key) || value.at(key).is_null()) {
        return fallback;
    }
    return value.at(key).get<RealType>();
}

std::string JsonString(const json& value, const std::string& key, const std::string& fallback = {}) {
    if (!value.is_object() || !value.contains(key) || value.at(key).is_null()) {
        return fallback;
    }
    return value.at(key).get<std::string>();
}

bool JsonBool(const json& value, const std::string& key, bool fallback = false) {
    if (!value.is_object() || !value.contains(key) || value.at(key).is_null()) {
        return fallback;
    }
    return value.at(key).get<bool>();
}

std::vector<json> TermList(const json& section) {
    if (section.is_null()) {
        return {};
    }
    if (section.is_object() && section.contains("terms")) {
        return TermList(section.at("terms"));
    }
    if (section.is_array()) {
        std::vector<json> result;
        result.reserve(section.size());
        for (const json& item : section) {
            if (!item.is_object() || !item.contains("enabled") || item.at("enabled").get<bool>()) {
                result.push_back(item);
            }
        }
        return result;
    }
    if (section.is_object()) {
        std::vector<json> result;
        result.reserve(section.size());
        for (auto it = section.begin(); it != section.end(); ++it) {
            if (!it.value().is_object()) {
                continue;
            }
            json item = it.value();
            if (!item.contains("name")) {
                item["name"] = it.key();
            }
            if (!item.contains("enabled") || item.at("enabled").get<bool>()) {
                result.push_back(std::move(item));
            }
        }
        return result;
    }
    return {};
}

RealType WrapAngle(RealType value) {
    return std::atan2(std::sin(value), std::cos(value));
}

const PhysicsRobotSnapshot* FindRobotSnapshot(const PhysicsSceneSnapshot& snapshot, const std::string& robot_name) {
    for (const PhysicsRobotSnapshot& robot : snapshot.robots) {
        if (robot.name == robot_name) {
            return &robot;
        }
    }
    return nullptr;
}

const PhysicsRobotState* FindRobotState(const PhysicsSceneState& state, const std::string& robot_name) {
    for (const PhysicsRobotState& robot : state.robots) {
        if (robot.name == robot_name) {
            return &robot;
        }
    }
    return nullptr;
}

const PhysicsJointSnapshot* FindJointSnapshot(const PhysicsRobotSnapshot& robot, const std::string& joint_name) {
    for (const PhysicsJointSnapshot& joint : robot.joints) {
        if (joint.name == joint_name) {
            return &joint;
        }
    }
    return nullptr;
}

const PhysicsJointState* FindJointState(const PhysicsRobotState& robot, const std::string& joint_name) {
    for (const PhysicsJointState& joint : robot.joints) {
        if (joint.joint_name == joint_name) {
            return &joint;
        }
    }
    return nullptr;
}

bool IsControllableJoint(const PhysicsJointSnapshot& joint) {
    const auto joint_type = static_cast<JointType>(joint.joint_type);
    return joint_type == JointType::Revolute ||
           joint_type == JointType::Continuous ||
           joint_type == JointType::Prismatic;
}

NativeVectorActionMode ParseNativeVectorActionMode(const std::string& value) {
    if (value == "normalized_position" ||
        value == "normalized_joint_position" ||
        value == "normalized" ||
        value == "joint_normalized_position") {
        return NativeVectorActionMode::NormalizedPosition;
    }
    if (value == "position" || value == "target_position" || value == "joint_position") {
        return NativeVectorActionMode::Position;
    }
    if (value == "velocity" || value == "target_velocity" || value == "joint_velocity") {
        return NativeVectorActionMode::Velocity;
    }
    if (value == "effort" ||
        value == "force" ||
        value == "torque" ||
        value == "target_effort" ||
        value == "joint_effort") {
        return NativeVectorActionMode::Effort;
    }
    throw std::invalid_argument("unsupported NativeVectorEnv action mode '" + value + "'");
}

struct NativeActionTerm {
    std::string name;
    std::string joint;
    NativeVectorActionMode mode{NativeVectorActionMode::NormalizedPosition};
    RealType scale{1.0};
    RealType offset{0.0};
    RealType lower{-1.0};
    RealType upper{1.0};
    std::string unit{"normalized"};
    std::vector<std::string> passive_joints;
};

enum class ObservationTermType {
    Time,
    EpisodeProgress,
    JointPosition,
    JointVelocity,
    JointEffort,
    Command,
    CommandError,
    PreviousAction,
    Constant
};

struct NativeObservationTerm {
    std::string name;
    ObservationTermType type{ObservationTermType::JointPosition};
    std::string joint;
    std::string command;
    int action_index{-1};
    bool wrap{false};
    RealType value{0.0};
    RealType lower{-kInf};
    RealType upper{kInf};
    std::string unit;
};

enum class RewardTermType {
    Constant,
    JointPositionL2,
    JointVelocityL2,
    JointEffortL2,
    ActionL2,
    SquaredCommandError,
    OvershootL2,
    CommandErrorProgress,
    JointCommandSettleBonus,
    FirstJointCommandReachBonus,
    JointCommandOverspeed,
    JointCommandFastCrossing,
    Terminated
};

struct NativeRewardTerm {
    std::string name;
    RewardTermType type{RewardTermType::Constant};
    RealType weight{1.0};
    std::string joint;
    std::string condition_joint;
    std::string command;
    int action_index{-1};
    bool wrap{false};
    bool condition_wrap{false};
    bool scale_by_dt{false};
    RealType value{1.0};
    RealType target{0.0};
    RealType target_tolerance{0.05};
    RealType velocity_tolerance{0.2};
    RealType condition_position_tolerance{kInf};
    RealType velocity_limit{kInf};
    RealType clip_lower{-kInf};
    RealType clip_upper{kInf};
    bool has_clip{false};
};

enum class TerminationTermType {
    JointPositionAbsGt,
    JointPositionGt,
    JointPositionLt,
    NonFiniteObservation
};

struct NativeTerminationTerm {
    std::string name;
    TerminationTermType type{TerminationTermType::JointPositionAbsGt};
    std::string joint;
    bool wrap{false};
    RealType limit{kInf};
};

enum class CommandTermType {
    Constant,
    Uniform,
    Choice
};

struct NativeCommandTerm {
    std::string name;
    CommandTermType type{CommandTermType::Constant};
    RealType value{0.0};
    RealType lower{-1.0};
    RealType upper{1.0};
    std::vector<RealType> choices;
    std::string unit;
};

struct NativeEventTerm {
    std::string name;
    std::string type;
    struct JointReset {
        std::string joint;
        RealType position{0.0};
        RealType velocity{0.0};
        bool has_position_range{false};
        RealType position_lower{0.0};
        RealType position_upper{0.0};
    };
    std::vector<JointReset> joint_resets;
};

struct NativeVectorWorld {
    Ref<PhysicsWorld> world;
    std::vector<RealType> previous_action;
    std::vector<RealType> commands;
    std::unordered_map<std::string, RealType> previous_target_errors;
    std::unordered_map<std::string, RealType> next_target_errors;
    std::unordered_set<std::string> flags;
    std::int64_t episode_length{0};
    RealType episode_return{0.0};
    std::mt19937_64 rng;
};

struct NativeVectorStepResult {
    std::vector<int> env_ids;
    std::vector<RealType> observation;
    std::vector<RealType> reward;
    std::vector<std::uint8_t> terminated;
    std::vector<std::uint8_t> truncated;
    std::vector<RealType> terminal_observation;
    std::vector<std::uint8_t> has_terminal_observation;
    std::vector<std::int64_t> episode_length;
    std::vector<RealType> episode_return;
    std::vector<RealType> critic_observation;
};

class NativeVectorEnv {
public:
    explicit NativeVectorEnv(NativeVectorEnvConfig config)
        : NativeVectorEnv(std::move(config), {}) {
    }

    NativeVectorEnv(NativeVectorEnvConfig config, std::vector<NativeVectorActionConfig> action_configs)
        : cfg_(std::move(config)),
          action_configs_(std::move(action_configs)) {
        try {
            ValidateConfig();
            ParseTaskConfig();
            LoadScene();
            BuildWorlds();
            ConfigureRobotAndSpecs();
            StartWorkers();
            ResetAll(std::nullopt);
        } catch (...) {
            ShutdownWorkers();
            ClearWorldsAndRoots();
            throw;
        }
    }

    ~NativeVectorEnv() {
        ShutdownWorkers();
        ClearWorldsAndRoots();
    }

    int GetNumEnvs() const {
        return cfg_.num_envs;
    }

    int GetBatchSize() const {
        return cfg_.batch_size;
    }

    int GetNumWorkers() const {
        return cfg_.num_workers;
    }

    int GetObservationSize() const {
        return observation_size_;
    }

    int GetCriticObservationSize() const {
        return critic_observation_size_;
    }

    int GetActionSize() const {
        return action_size_;
    }

    RealType GetEnvDt() const {
        return cfg_.physics_dt * cfg_.decimation;
    }

    py::dict GetObservationSpec() const {
        return MakeSpec(observation_terms_, "gobot.native_vector.observation.policy.v1");
    }

    py::dict GetCriticObservationSpec() const {
        return MakeSpec(critic_observation_terms_, "gobot.native_vector.observation.critic.v1");
    }

    py::dict GetActionSpec() const {
        py::dict spec;
        py::list names;
        py::list lower;
        py::list upper;
        py::list units;
        for (const NativeActionTerm& term : action_terms_) {
            names.append(term.name);
            lower.append(term.lower);
            upper.append(term.upper);
            units.append(term.unit);
        }
        spec["version"] = "gobot.native_vector.action.v1";
        spec["names"] = names;
        spec["lower_bounds"] = lower;
        spec["upper_bounds"] = upper;
        spec["units"] = units;
        return spec;
    }

    py::tuple Reset(py::object seed = py::none(), py::object env_ids = py::none()) {
        const std::optional<std::uint64_t> parsed_seed = ParseOptionalSeed(seed);
        const std::vector<int> ids = ParseEnvIds(env_ids);
        std::vector<int> output_ids;
        std::vector<RealType> observations;
        std::vector<RealType> critic_observations;
        {
            py::gil_scoped_release release;
            std::unique_lock<std::mutex> api_lock(api_mutex_);
            if (ids.empty()) {
                output_ids = AllEnvIds();
                ResetAll(parsed_seed);
            } else {
                output_ids = ids;
                ResetIds(output_ids, parsed_seed);
            }
            observations = Observe(output_ids, observation_terms_, observation_size_);
            critic_observations = Observe(output_ids, critic_observation_terms_, critic_observation_size_);
        }
        return MakeResetReturn(output_ids, observations, critic_observations, parsed_seed);
    }

    py::tuple Step(py::object action, py::object env_ids = py::none()) {
        const std::vector<int> ids = ParseEnvIds(env_ids);
        const std::vector<int> active_ids = ids.empty() ? AllEnvIds() : ids;
        const std::vector<RealType> actions = ParseActionArray(action, static_cast<int>(active_ids.size()));
        NativeVectorStepResult result;
        {
            py::gil_scoped_release release;
            std::unique_lock<std::mutex> api_lock(api_mutex_);
            result = StepIds(active_ids, actions);
        }
        return MakeStepReturn(result);
    }

    void AsyncReset() {
        {
            py::gil_scoped_release release;
            std::unique_lock<std::mutex> api_lock(api_mutex_);
            async_queue_.clear();
            NativeVectorStepResult result;
            result.env_ids = AllEnvIds();
            result.reward.assign(result.env_ids.size(), 0.0);
            result.terminated.assign(result.env_ids.size(), 0);
            result.truncated.assign(result.env_ids.size(), 0);
            result.terminal_observation.assign(result.env_ids.size() * observation_size_, 0.0);
            result.has_terminal_observation.assign(result.env_ids.size(), 0);
            ResetAll(std::nullopt);
            result.observation = Observe(result.env_ids, observation_terms_, observation_size_);
            result.critic_observation = Observe(result.env_ids, critic_observation_terms_, critic_observation_size_);
            result.episode_length.reserve(result.env_ids.size());
            result.episode_return.reserve(result.env_ids.size());
            for (int env_id : result.env_ids) {
                result.episode_length.push_back(worlds_[env_id].episode_length);
                result.episode_return.push_back(worlds_[env_id].episode_return);
            }
            EnqueueBatches(result);
        }
    }

    void Send(py::object action, py::object env_ids = py::none()) {
        const std::vector<int> ids = ParseEnvIds(env_ids);
        const std::vector<int> active_ids = ids.empty() ? AllEnvIds() : ids;
        const std::vector<RealType> actions = ParseActionArray(action, static_cast<int>(active_ids.size()));
        {
            py::gil_scoped_release release;
            std::unique_lock<std::mutex> api_lock(api_mutex_);
            NativeVectorStepResult result = StepIds(active_ids, actions);
            EnqueueBatches(result);
        }
    }

    py::tuple Recv() {
        NativeVectorStepResult result;
        {
            py::gil_scoped_release release;
            std::unique_lock<std::mutex> api_lock(api_mutex_);
            if (async_queue_.empty()) {
                throw std::runtime_error("NativeVectorEnv.recv() has no ready batch; call async_reset() or send() first");
            }
            result = std::move(async_queue_.front());
            async_queue_.pop_front();
        }
        return MakeStepReturn(result);
    }

private:
    void ValidateConfig() {
        if (cfg_.num_envs < 1) {
            throw std::invalid_argument("NativeVectorEnv num_envs must be at least 1");
        }
        if (cfg_.decimation < 1) {
            throw std::invalid_argument("NativeVectorEnv decimation must be at least 1");
        }
        if (cfg_.physics_dt <= 0.0) {
            throw std::invalid_argument("NativeVectorEnv physics_dt must be greater than zero");
        }
        if (cfg_.max_episode_steps < 1) {
            throw std::invalid_argument("NativeVectorEnv max_episode_steps must be at least 1");
        }
        if (cfg_.batch_size <= 0) {
            cfg_.batch_size = cfg_.num_envs;
        }
        if (cfg_.batch_size < 1 || cfg_.batch_size > cfg_.num_envs) {
            throw std::invalid_argument("NativeVectorEnv batch_size must be in [1, num_envs]");
        }
        if (cfg_.num_workers <= 0) {
            const unsigned hardware_threads = std::max(1U, std::thread::hardware_concurrency());
            cfg_.num_workers = std::min<int>(cfg_.num_envs, static_cast<int>(hardware_threads));
        }
        cfg_.num_workers = std::clamp(cfg_.num_workers, 1, cfg_.num_envs);
        if (cfg_.scene.empty()) {
            throw std::invalid_argument("NativeVectorEnv requires a scene path");
        }
    }

    void ParseTaskConfig() {
        task_ = json::object();
        if (!cfg_.task_json.empty()) {
            task_ = json::parse(cfg_.task_json);
        }
        actions_section_ = task_.value("actions", json::object());
        observations_section_ = task_.value("observations", json::object());
        rewards_section_ = task_.value("rewards", json::object());
        terminations_section_ = task_.value("terminations", json::object());
        events_section_ = task_.value("events", json::object());
        commands_section_ = task_.value("commands", json::object());
    }

    void LoadScene() {
        Ref<Resource> resource =
                ResourceLoader::Load(cfg_.scene, "PackedScene", ResourceFormatLoader::CacheMode::Ignore);
        packed_scene_ = dynamic_pointer_cast<PackedScene>(resource);
        if (!packed_scene_.IsValid()) {
            throw std::runtime_error("NativeVectorEnv failed to load PackedScene from '" + cfg_.scene + "'");
        }
    }

    void BuildWorlds() {
        worlds_.clear();
        worlds_.resize(static_cast<std::size_t>(cfg_.num_envs));
        roots_.clear();
        roots_.resize(static_cast<std::size_t>(cfg_.num_envs), nullptr);

        PhysicsWorldSettings settings;
        settings.fixed_time_step = cfg_.physics_dt;
        for (int env_id = 0; env_id < cfg_.num_envs; ++env_id) {
            Node* root = packed_scene_->Instantiate();
            if (root == nullptr) {
                throw std::runtime_error("NativeVectorEnv failed to instantiate PackedScene for env " +
                                         std::to_string(env_id));
            }
            roots_[env_id] = root;

            Ref<PhysicsWorld> world = PhysicsServer::CreateWorldForBackend(cfg_.backend, settings);
            if (!world.IsValid()) {
                throw std::runtime_error("NativeVectorEnv failed to create physics world");
            }
            if (world->GetBackendType() != cfg_.backend) {
                throw std::runtime_error("NativeVectorEnv requested physics backend is unavailable");
            }
            if (!world->BuildFromScene(root)) {
                throw std::runtime_error("NativeVectorEnv failed to build physics world: " + world->GetLastError());
            }
            world->Reset();
            worlds_[env_id].world = world;
            worlds_[env_id].rng.seed(cfg_.seed + static_cast<std::uint64_t>(env_id) * 9973ULL);
        }
    }

    void ClearWorldsAndRoots() {
        for (NativeVectorWorld& world : worlds_) {
            world.world.Reset();
        }
        worlds_.clear();
        for (Node*& root : roots_) {
            if (root != nullptr) {
                Object::Delete(root);
                root = nullptr;
            }
        }
        roots_.clear();
        packed_scene_.Reset();
    }

    void ConfigureRobotAndSpecs() {
        const PhysicsSceneSnapshot& snapshot = worlds_.front().world->GetSceneSnapshot();
        const PhysicsRobotSnapshot* robot = nullptr;
        if (!cfg_.robot.empty()) {
            robot = FindRobotSnapshot(snapshot, cfg_.robot);
        } else if (!snapshot.robots.empty()) {
            robot = &snapshot.robots.front();
            cfg_.robot = robot->name;
        }
        if (robot == nullptr) {
            throw std::runtime_error("NativeVectorEnv scene has no robot named '" + cfg_.robot + "'");
        }

        ConfigureActions(*robot);
        ConfigureCommands();
        ConfigureEvents();
        ConfigureObservationGroups(*robot);
        ConfigureRewards();
        ConfigureTerminations();

        for (NativeVectorWorld& world : worlds_) {
            world.previous_action.assign(static_cast<std::size_t>(action_size_), 0.0);
            world.commands.assign(command_terms_.size(), 0.0);
        }
    }

    void ConfigureActions(const PhysicsRobotSnapshot& robot) {
        if (action_configs_.empty()) {
            for (const json& item : TermList(actions_section_)) {
                NativeVectorActionConfig cfg;
                cfg.name = JsonString(item, "name", "");
                cfg.joint = JsonString(item, "joint", cfg.name);
                cfg.mode = ParseNativeVectorActionMode(JsonString(item, "mode", JsonString(item, "type", "normalized_position")));
                std::string type = JsonString(item, "type", "");
                if (!type.empty() && !item.contains("mode")) {
                    cfg.mode = ParseNativeVectorActionMode(type);
                }
                cfg.scale = JsonReal(item, "scale", 1.0);
                cfg.offset = JsonReal(item, "offset", 0.0);
                if (item.contains("clip") && item.at("clip").is_array() && item.at("clip").size() == 2) {
                    cfg.lower = item.at("clip")[0].get<RealType>();
                    cfg.upper = item.at("clip")[1].get<RealType>();
                } else {
                    cfg.lower = JsonReal(item, "lower", -1.0);
                    cfg.upper = JsonReal(item, "upper", 1.0);
                }
                cfg.unit = JsonString(item, "unit", "normalized");
                if (item.contains("passive_joints") && item.at("passive_joints").is_array()) {
                    for (const json& passive : item.at("passive_joints")) {
                        cfg.passive_joints.push_back(passive.get<std::string>());
                    }
                }
                action_configs_.push_back(std::move(cfg));
            }
        }

        if (action_configs_.empty() && cfg_.controlled_joints.empty()) {
            for (const PhysicsJointSnapshot& joint : robot.joints) {
                if (IsControllableJoint(joint)) {
                    cfg_.controlled_joints.push_back(joint.name);
                }
            }
        }
        if (action_configs_.empty()) {
            for (const std::string& joint : cfg_.controlled_joints) {
                action_configs_.push_back({
                        joint + "/target_position_normalized",
                        joint,
                        NativeVectorActionMode::NormalizedPosition,
                        1.0,
                        0.0,
                        -1.0,
                        1.0,
                        "normalized",
                        {}
                });
            }
        }
        if (action_configs_.empty()) {
            throw std::runtime_error("NativeVectorEnv robot '" + cfg_.robot + "' has no controllable joints");
        }

        std::unordered_set<std::string> robot_joint_names;
        for (const PhysicsJointSnapshot& joint : robot.joints) {
            robot_joint_names.insert(joint.name);
        }
        for (const NativeVectorActionConfig& config : action_configs_) {
            if (!robot_joint_names.contains(config.joint)) {
                throw std::runtime_error("NativeVectorEnv robot '" + cfg_.robot + "' has no joint named '" + config.joint + "'");
            }
            action_terms_.push_back({
                    config.name,
                    config.joint,
                    config.mode,
                    config.scale,
                    config.offset,
                    config.lower,
                    config.upper,
                    config.unit,
                    config.passive_joints
            });
        }
        action_size_ = static_cast<int>(action_terms_.size());
    }

    void ConfigureCommands() {
        for (const json& item : TermList(commands_section_)) {
            NativeCommandTerm term;
            term.name = JsonString(item, "name", "");
            const std::string type = JsonString(item, "type", "constant");
            if (type == "constant") {
                term.type = CommandTermType::Constant;
                term.value = JsonReal(item, "value", 0.0);
            } else if (type == "uniform") {
                term.type = CommandTermType::Uniform;
                if (item.contains("range") && item.at("range").is_array() && item.at("range").size() == 2) {
                    term.lower = item.at("range")[0].get<RealType>();
                    term.upper = item.at("range")[1].get<RealType>();
                }
                term.value = JsonReal(item, "value", 0.0);
            } else if (type == "choice") {
                term.type = CommandTermType::Choice;
                if (item.contains("choices") && item.at("choices").is_array()) {
                    for (const json& choice : item.at("choices")) {
                        term.choices.push_back(choice.get<RealType>());
                    }
                }
            } else {
                throw std::runtime_error("unsupported NativeVectorEnv command term type '" + type + "'");
            }
            term.unit = JsonString(item, "unit", "");
            command_indices_[term.name] = static_cast<int>(command_terms_.size());
            command_terms_.push_back(std::move(term));
        }
    }

    void ConfigureEvents() {
        for (const json& item : TermList(events_section_)) {
            NativeEventTerm event;
            event.name = JsonString(item, "name", "");
            event.type = JsonString(item, "type", "");
            if (event.type == "reset_joint_state" && item.contains("joints") && item.at("joints").is_object()) {
                for (auto it = item.at("joints").begin(); it != item.at("joints").end(); ++it) {
                    NativeEventTerm::JointReset reset;
                    reset.joint = it.key();
                    const json& values = it.value();
                    reset.position = JsonReal(values, "position", 0.0);
                    reset.velocity = JsonReal(values, "velocity", 0.0);
                    if (values.contains("position_range") && values.at("position_range").is_array() &&
                        values.at("position_range").size() == 2 && !values.at("position_range")[0].is_null()) {
                        reset.has_position_range = true;
                        reset.position_lower = values.at("position_range")[0].get<RealType>();
                        reset.position_upper = values.at("position_range")[1].get<RealType>();
                    }
                    event.joint_resets.push_back(reset);
                }
            }
            event_terms_.push_back(std::move(event));
        }
    }

    void ConfigureObservationGroups(const PhysicsRobotSnapshot& robot) {
        const json groups = observations_section_.value("groups", json::object());
        if (groups.is_object() && groups.contains("policy")) {
            observation_terms_ = BuildObservationGroup("policy", groups, {});
        }
        if (groups.is_object() && groups.contains("critic")) {
            critic_observation_terms_ = BuildObservationGroup("critic", groups, {});
        }
        if (observation_terms_.empty()) {
            observation_terms_ = DefaultObservationTerms(robot);
        }
        if (critic_observation_terms_.empty()) {
            critic_observation_terms_ = observation_terms_;
        }
        observation_size_ = static_cast<int>(observation_terms_.size());
        critic_observation_size_ = static_cast<int>(critic_observation_terms_.size());
    }

    std::vector<NativeObservationTerm> BuildObservationGroup(const std::string& group,
                                                            const json& groups,
                                                            std::vector<std::string> stack) const {
        if (std::find(stack.begin(), stack.end(), group) != stack.end()) {
            throw std::runtime_error("recursive NativeVectorEnv observation group reference: " + group);
        }
        stack.push_back(group);
        std::vector<NativeObservationTerm> terms;
        if (!groups.is_object() || !groups.contains(group) || !groups.at(group).is_array()) {
            return terms;
        }
        for (const json& item : groups.at(group)) {
            if (item.is_object() && JsonString(item, "type", "") == "group") {
                std::vector<NativeObservationTerm> nested =
                        BuildObservationGroup(JsonString(item, "group", ""), groups, stack);
                terms.insert(terms.end(), nested.begin(), nested.end());
                continue;
            }
            terms.push_back(ParseObservationTerm(item));
        }
        return terms;
    }

    NativeObservationTerm ParseObservationTerm(const json& item) const {
        NativeObservationTerm term;
        term.name = JsonString(item, "name", "");
        const std::string type = JsonString(item, "type", term.name);
        if (type == "time") {
            term.type = ObservationTermType::Time;
        } else if (type == "episode_progress") {
            term.type = ObservationTermType::EpisodeProgress;
        } else if (type == "joint_position") {
            term.type = ObservationTermType::JointPosition;
        } else if (type == "joint_velocity") {
            term.type = ObservationTermType::JointVelocity;
        } else if (type == "joint_effort") {
            term.type = ObservationTermType::JointEffort;
        } else if (type == "command") {
            term.type = ObservationTermType::Command;
        } else if (type == "command_error") {
            term.type = ObservationTermType::CommandError;
        } else if (type == "previous_action") {
            term.type = ObservationTermType::PreviousAction;
            term.action_index = ActionIndex(JsonString(item, "action", term.name));
        } else if (type == "constant") {
            term.type = ObservationTermType::Constant;
            term.value = JsonReal(item, "value", 0.0);
        } else {
            throw std::runtime_error("unsupported NativeVectorEnv observation term type '" + type + "'");
        }
        term.joint = JsonString(item, "joint", "");
        term.command = JsonString(item, "command", term.name);
        term.wrap = JsonBool(item, "wrap", false);
        term.lower = JsonReal(item, "lower", JsonReal(item, "lower_bound", -kInf));
        term.upper = JsonReal(item, "upper", JsonReal(item, "upper_bound", kInf));
        term.unit = JsonString(item, "unit", JsonString(item, "units", ""));
        return term;
    }

    std::vector<NativeObservationTerm> DefaultObservationTerms(const PhysicsRobotSnapshot& robot) const {
        std::vector<NativeObservationTerm> terms;
        terms.push_back({"time", ObservationTermType::Time, "", "", -1, false, 0.0, 0.0, kInf, "s"});
        terms.push_back({"episode_progress", ObservationTermType::EpisodeProgress, "", "", -1, false, 0.0, 0.0, 1.0, "ratio"});
        for (const PhysicsJointSnapshot& joint : robot.joints) {
            if (static_cast<JointType>(joint.joint_type) == JointType::Fixed) {
                continue;
            }
            RealType lower = -kInf;
            RealType upper = kInf;
            if (joint.upper_limit > joint.lower_limit &&
                static_cast<JointType>(joint.joint_type) != JointType::Continuous) {
                lower = joint.lower_limit;
                upper = joint.upper_limit;
            }
            terms.push_back({joint.name + "/position", ObservationTermType::JointPosition, joint.name, "", -1, false, 0.0, lower, upper, "rad_or_m"});
            terms.push_back({joint.name + "/velocity", ObservationTermType::JointVelocity, joint.name, "", -1, false, 0.0, -kInf, kInf, "rad_or_m/s"});
        }
        for (int action_index = 0; action_index < action_size_; ++action_index) {
            terms.push_back({action_terms_[action_index].name + "/previous_action",
                             ObservationTermType::PreviousAction,
                             "",
                             "",
                             action_index,
                             false,
                             0.0,
                             -1.0,
                             1.0,
                             "normalized"});
        }
        return terms;
    }

    void ConfigureRewards() {
        for (const json& item : TermList(rewards_section_)) {
            NativeRewardTerm term;
            term.name = JsonString(item, "name", "");
            const std::string type = JsonString(item, "type", "constant");
            if (type == "constant") {
                term.type = RewardTermType::Constant;
            } else if (type == "joint_position_l2") {
                term.type = RewardTermType::JointPositionL2;
            } else if (type == "joint_velocity_l2") {
                term.type = RewardTermType::JointVelocityL2;
            } else if (type == "joint_effort_l2") {
                term.type = RewardTermType::JointEffortL2;
            } else if (type == "action_l2") {
                term.type = RewardTermType::ActionL2;
            } else if (type == "squared_command_error") {
                term.type = RewardTermType::SquaredCommandError;
            } else if (type == "overshoot_l2") {
                term.type = RewardTermType::OvershootL2;
            } else if (type == "command_error_progress" || type == "joint_command_progress") {
                term.type = RewardTermType::CommandErrorProgress;
            } else if (type == "joint_command_settle_bonus") {
                term.type = RewardTermType::JointCommandSettleBonus;
            } else if (type == "first_joint_command_reach_bonus") {
                term.type = RewardTermType::FirstJointCommandReachBonus;
            } else if (type == "joint_command_overspeed") {
                term.type = RewardTermType::JointCommandOverspeed;
            } else if (type == "joint_command_fast_crossing") {
                term.type = RewardTermType::JointCommandFastCrossing;
            } else if (type == "terminated") {
                term.type = RewardTermType::Terminated;
            } else {
                throw std::runtime_error("unsupported NativeVectorEnv reward term type '" + type + "'");
            }
            term.weight = JsonReal(item, "weight", 1.0);
            term.joint = JsonString(item, "joint", "");
            term.condition_joint = JsonString(item, "condition_joint", "");
            term.command = JsonString(item, "command", "");
            term.wrap = JsonBool(item, "wrap", false);
            term.condition_wrap = JsonBool(item, "condition_wrap", false);
            term.scale_by_dt = JsonBool(item, "scale_by_dt", false);
            term.value = JsonReal(item, "value", 1.0);
            term.target = JsonReal(item, "target", 0.0);
            term.target_tolerance = JsonReal(item, "target_tolerance", 0.05);
            term.velocity_tolerance = JsonReal(item, "velocity_tolerance", 0.2);
            term.condition_position_tolerance = JsonReal(item, "condition_position_tolerance", kInf);
            term.velocity_limit = JsonReal(item, "velocity_limit", kInf);
            if (item.contains("clip") && item.at("clip").is_array() && item.at("clip").size() == 2) {
                term.has_clip = true;
                term.clip_lower = item.at("clip")[0].get<RealType>();
                term.clip_upper = item.at("clip")[1].get<RealType>();
            }
            const std::string action_name = JsonString(item, "action", "");
            if (!action_name.empty()) {
                term.action_index = ActionIndex(action_name);
            }
            reward_terms_.push_back(std::move(term));
        }
    }

    void ConfigureTerminations() {
        for (const json& item : TermList(terminations_section_)) {
            NativeTerminationTerm term;
            term.name = JsonString(item, "name", "");
            const std::string type = JsonString(item, "type", "joint_position_abs_gt");
            if (type == "joint_position_abs_gt") {
                term.type = TerminationTermType::JointPositionAbsGt;
            } else if (type == "joint_position_gt") {
                term.type = TerminationTermType::JointPositionGt;
            } else if (type == "joint_position_lt") {
                term.type = TerminationTermType::JointPositionLt;
            } else if (type == "non_finite_observation") {
                term.type = TerminationTermType::NonFiniteObservation;
            } else {
                throw std::runtime_error("unsupported NativeVectorEnv termination term type '" + type + "'");
            }
            term.joint = JsonString(item, "joint", "");
            term.wrap = JsonBool(item, "wrap", false);
            term.limit = JsonReal(item, "limit", kInf);
            termination_terms_.push_back(std::move(term));
        }
    }

    int ActionIndex(const std::string& action_name) const {
        for (int index = 0; index < static_cast<int>(action_terms_.size()); ++index) {
            if (action_terms_[index].name == action_name) {
                return index;
            }
        }
        throw std::runtime_error("NativeVectorEnv has no action named '" + action_name + "'");
    }

    int CommandIndex(const std::string& command_name) const {
        auto iter = command_indices_.find(command_name);
        if (iter == command_indices_.end()) {
            return -1;
        }
        return iter->second;
    }

    RealType CommandValue(const NativeVectorWorld& slot, const std::string& command_name) const {
        const int index = CommandIndex(command_name);
        if (index < 0 || index >= static_cast<int>(slot.commands.size())) {
            return 0.0;
        }
        return slot.commands[index];
    }

    py::dict MakeSpec(const std::vector<NativeObservationTerm>& terms, const std::string& version) const {
        py::dict spec;
        py::list names;
        py::list lower;
        py::list upper;
        py::list units;
        for (const NativeObservationTerm& term : terms) {
            names.append(term.name);
            lower.append(term.lower);
            upper.append(term.upper);
            units.append(term.unit);
        }
        spec["version"] = version;
        spec["names"] = names;
        spec["lower_bounds"] = lower;
        spec["upper_bounds"] = upper;
        spec["units"] = units;
        return spec;
    }

    std::optional<std::uint64_t> ParseOptionalSeed(const py::object& seed) const {
        if (seed.is_none()) {
            return std::nullopt;
        }
        return static_cast<std::uint64_t>(py::cast<std::uint64_t>(seed));
    }

    std::vector<int> ParseEnvIds(const py::object& env_ids) const {
        std::vector<int> ids;
        if (env_ids.is_none()) {
            return ids;
        }
        if (py::isinstance<py::int_>(env_ids)) {
            ids.push_back(py::cast<int>(env_ids));
        } else {
            for (const py::handle item : env_ids) {
                ids.push_back(py::cast<int>(item));
            }
        }
        for (int id : ids) {
            if (id < 0 || id >= cfg_.num_envs) {
                throw std::out_of_range("NativeVectorEnv env_id out of range: " + std::to_string(id));
            }
        }
        return ids;
    }

    std::vector<int> AllEnvIds() const {
        std::vector<int> ids(static_cast<std::size_t>(cfg_.num_envs));
        for (int index = 0; index < cfg_.num_envs; ++index) {
            ids[index] = index;
        }
        return ids;
    }

    std::vector<RealType> ParseActionArray(const py::object& action, int rows) const {
        py::array_t<RealType, py::array::c_style | py::array::forcecast> array(action);
        const py::buffer_info info = array.request();
        if (info.ndim == 1) {
            if (action_size_ != 1 && info.shape[0] != action_size_) {
                throw std::invalid_argument("NativeVectorEnv action has wrong shape");
            }
            if (rows == 1 && info.shape[0] == action_size_) {
                const auto* data = static_cast<const RealType*>(info.ptr);
                return std::vector<RealType>(data, data + action_size_);
            }
            if (action_size_ == 1 && info.shape[0] == rows) {
                const auto* data = static_cast<const RealType*>(info.ptr);
                return std::vector<RealType>(data, data + rows);
            }
            throw std::invalid_argument("NativeVectorEnv action first dimension must match env_ids");
        }
        if (info.ndim != 2 ||
            info.shape[0] != rows ||
            info.shape[1] != action_size_) {
            throw std::invalid_argument("NativeVectorEnv action must have shape (batch, action_dim)");
        }
        const auto* data = static_cast<const RealType*>(info.ptr);
        return std::vector<RealType>(data, data + rows * action_size_);
    }

    void ResetAll(std::optional<std::uint64_t> seed) {
        std::vector<int> ids = AllEnvIds();
        ResetIds(ids, seed);
    }

    void ResetOne(int env_id, std::optional<std::uint64_t> seed) {
        NativeVectorWorld& slot = worlds_[env_id];
        if (seed.has_value()) {
            slot.rng.seed(*seed + static_cast<std::uint64_t>(env_id) * 9973ULL);
        }
        slot.world->Reset();
        slot.previous_action.assign(static_cast<std::size_t>(action_size_), 0.0);
        slot.episode_length = 0;
        slot.episode_return = 0.0;
        slot.previous_target_errors.clear();
        slot.next_target_errors.clear();
        slot.flags.clear();
        ResetCommands(slot);
        ResetEvents(slot);
    }

    void ResetIds(const std::vector<int>& ids, std::optional<std::uint64_t> seed) {
        ParallelForRows(ids.size(), [&](std::size_t row) {
            ResetOne(ids[row], seed);
        });
    }

    void ResetCommands(NativeVectorWorld& slot) {
        if (slot.commands.size() != command_terms_.size()) {
            slot.commands.assign(command_terms_.size(), 0.0);
        }
        for (std::size_t index = 0; index < command_terms_.size(); ++index) {
            const NativeCommandTerm& term = command_terms_[index];
            RealType value = term.value;
            switch (term.type) {
                case CommandTermType::Constant:
                    value = term.value;
                    break;
                case CommandTermType::Uniform: {
                    std::uniform_real_distribution<RealType> dist(term.lower, term.upper);
                    value = dist(slot.rng);
                    break;
                }
                case CommandTermType::Choice:
                    if (!term.choices.empty()) {
                        std::uniform_int_distribution<std::size_t> dist(0, term.choices.size() - 1);
                        value = term.choices[dist(slot.rng)];
                    }
                    break;
            }
            slot.commands[index] = value;
        }
    }

    void ResetEvents(NativeVectorWorld& slot) {
        for (const NativeEventTerm& event : event_terms_) {
            if (event.type != "reset_joint_state") {
                continue;
            }
            for (const NativeEventTerm::JointReset& reset : event.joint_resets) {
                RealType position = reset.position;
                if (reset.has_position_range) {
                    std::uniform_real_distribution<RealType> dist(reset.position_lower, reset.position_upper);
                    position = dist(slot.rng);
                }
                if (!slot.world->ResetJointState(cfg_.robot, reset.joint, position, reset.velocity)) {
                    throw std::runtime_error(slot.world->GetLastError());
                }
            }
        }
    }

    std::vector<RealType> Observe(const std::vector<int>& ids,
                                  const std::vector<NativeObservationTerm>& terms,
                                  int term_count) const {
        std::vector<RealType> observations(ids.size() * static_cast<std::size_t>(term_count), 0.0);
        ParallelForRows(ids.size(), [&](std::size_t row) {
            const std::vector<RealType> row_observation = ObserveOne(ids[row], terms);
            std::copy(row_observation.begin(),
                      row_observation.end(),
                      observations.begin() + static_cast<std::ptrdiff_t>(row * term_count));
        });
        return observations;
    }

    std::vector<RealType> ObserveOne(int env_id, const std::vector<NativeObservationTerm>& terms) const {
        std::vector<RealType> observation(terms.size(), 0.0);
        const NativeVectorWorld& slot = worlds_[env_id];
        const PhysicsRobotState* robot = FindRobotState(slot.world->GetSceneState(), cfg_.robot);
        if (robot == nullptr) {
            throw std::runtime_error("NativeVectorEnv runtime state has no robot named '" + cfg_.robot + "'");
        }
        for (std::size_t col = 0; col < terms.size(); ++col) {
            observation[col] = ComputeObservationTerm(slot, *robot, terms[col]);
        }
        return observation;
    }

    RealType ComputeObservationTerm(const NativeVectorWorld& slot,
                                    const PhysicsRobotState& robot,
                                    const NativeObservationTerm& term) const {
        switch (term.type) {
            case ObservationTermType::Time:
                return static_cast<RealType>(slot.episode_length) * GetEnvDt();
            case ObservationTermType::EpisodeProgress:
                return std::min<RealType>(static_cast<RealType>(slot.episode_length) /
                                                  static_cast<RealType>(cfg_.max_episode_steps),
                                          1.0);
            case ObservationTermType::JointPosition: {
                const PhysicsJointState* joint = FindJointState(robot, term.joint);
                const RealType value = joint == nullptr ? 0.0 : joint->position;
                return term.wrap ? WrapAngle(value) : value;
            }
            case ObservationTermType::JointVelocity: {
                const PhysicsJointState* joint = FindJointState(robot, term.joint);
                return joint == nullptr ? 0.0 : joint->velocity;
            }
            case ObservationTermType::JointEffort: {
                const PhysicsJointState* joint = FindJointState(robot, term.joint);
                return joint == nullptr ? 0.0 : joint->effort;
            }
            case ObservationTermType::Command:
                return CommandValue(slot, term.command);
            case ObservationTermType::CommandError: {
                const PhysicsJointState* joint = FindJointState(robot, term.joint);
                const RealType value = joint == nullptr ? 0.0 : joint->position;
                return CommandValue(slot, term.command) - (term.wrap ? WrapAngle(value) : value);
            }
            case ObservationTermType::PreviousAction:
                if (term.action_index >= 0 && term.action_index < static_cast<int>(slot.previous_action.size())) {
                    return slot.previous_action[term.action_index];
                }
                return 0.0;
            case ObservationTermType::Constant:
                return term.value;
        }
        return 0.0;
    }

    void StartWorkers() {
        const int helper_count = std::max(0, cfg_.num_workers - 1);
        if (helper_count == 0) {
            return;
        }
        workers_.reserve(static_cast<std::size_t>(helper_count));
        for (int worker_index = 0; worker_index < helper_count; ++worker_index) {
            workers_.emplace_back([this]() { WorkerLoop(); });
        }
    }

    void ShutdownWorkers() {
        {
            std::lock_guard<std::mutex> lock(worker_mutex_);
            workers_stop_ = true;
            active_task_ = nullptr;
            active_row_count_ = 0;
            ++task_epoch_;
        }
        worker_cv_.notify_all();
        for (std::thread& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        workers_.clear();
    }

    void WorkerLoop() {
        std::size_t seen_epoch = 0;
        while (true) {
            std::function<void(std::size_t)> task;
            std::size_t row_count = 0;
            {
                std::unique_lock<std::mutex> lock(worker_mutex_);
                worker_cv_.wait(lock, [&]() { return workers_stop_ || task_epoch_ != seen_epoch; });
                if (workers_stop_) {
                    return;
                }
                seen_epoch = task_epoch_;
                task = active_task_;
                row_count = active_row_count_;
            }

            RunWorkerTaskRows(task, row_count);

            {
                std::lock_guard<std::mutex> lock(worker_mutex_);
                if (active_helper_count_ > 0) {
                    --active_helper_count_;
                }
                if (active_helper_count_ == 0) {
                    worker_done_cv_.notify_one();
                }
            }
        }
    }

    void RunWorkerTaskRows(const std::function<void(std::size_t)>& task, std::size_t row_count) const {
        while (true) {
            const std::size_t row = next_worker_row_.fetch_add(1, std::memory_order_relaxed);
            if (row >= row_count) {
                break;
            }
            {
                std::lock_guard<std::mutex> lock(worker_mutex_);
                if (worker_error_) {
                    break;
                }
            }
            try {
                task(row);
            } catch (...) {
                std::lock_guard<std::mutex> lock(worker_mutex_);
                if (!worker_error_) {
                    worker_error_ = std::current_exception();
                }
                break;
            }
        }
    }

    template <typename Func>
    void ParallelForRows(std::size_t row_count, Func&& func) const {
        if (row_count == 0) {
            return;
        }
        if (row_count == 1 || workers_.empty()) {
            for (std::size_t row = 0; row < row_count; ++row) {
                func(row);
            }
            return;
        }

        std::function<void(std::size_t)> task(std::forward<Func>(func));
        {
            std::lock_guard<std::mutex> lock(worker_mutex_);
            active_task_ = task;
            active_row_count_ = row_count;
            next_worker_row_.store(0, std::memory_order_relaxed);
            active_helper_count_ = static_cast<int>(workers_.size());
            worker_error_ = nullptr;
            ++task_epoch_;
        }
        worker_cv_.notify_all();
        RunWorkerTaskRows(task, row_count);

        std::exception_ptr error;
        {
            std::unique_lock<std::mutex> lock(worker_mutex_);
            worker_done_cv_.wait(lock, [&]() { return active_helper_count_ == 0; });
            error = worker_error_;
            active_task_ = nullptr;
            active_row_count_ = 0;
            worker_error_ = nullptr;
        }
        if (error) {
            std::rethrow_exception(error);
        }
    }

    NativeVectorStepResult StepIds(const std::vector<int>& ids, const std::vector<RealType>& actions) {
        NativeVectorStepResult result;
        result.env_ids = ids;
        result.reward.assign(ids.size(), 0.0);
        result.terminated.assign(ids.size(), 0);
        result.truncated.assign(ids.size(), 0);
        result.has_terminal_observation.assign(ids.size(), 0);
        result.terminal_observation.assign(ids.size() * static_cast<std::size_t>(observation_size_), 0.0);
        result.episode_length.assign(ids.size(), 0);
        result.episode_return.assign(ids.size(), 0.0);

        ParallelForRows(ids.size(), [&](std::size_t row) {
            StepOne(ids[row],
                    std::vector<RealType>(actions.begin() + static_cast<std::ptrdiff_t>(row * action_size_),
                                          actions.begin() + static_cast<std::ptrdiff_t>((row + 1) * action_size_)),
                    result,
                    row);
        });
        result.observation = Observe(ids, observation_terms_, observation_size_);
        result.critic_observation = Observe(ids, critic_observation_terms_, critic_observation_size_);
        return result;
    }

    void StepOne(int env_id,
                 const std::vector<RealType>& action,
                 NativeVectorStepResult& result,
                 std::size_t row) {
        NativeVectorWorld& slot = worlds_[env_id];
        const PhysicsSceneSnapshot& snapshot = slot.world->GetSceneSnapshot();
        const PhysicsRobotSnapshot* robot_snapshot = FindRobotSnapshot(snapshot, cfg_.robot);
        if (robot_snapshot == nullptr) {
            throw std::runtime_error("NativeVectorEnv scene snapshot has no robot named '" + cfg_.robot + "'");
        }

        ApplyAction(slot, *robot_snapshot, action);
        for (int substep = 0; substep < cfg_.decimation; ++substep) {
            slot.world->Step(cfg_.physics_dt);
        }
        slot.episode_length += 1;

        const PhysicsRobotState* robot_state = FindRobotState(slot.world->GetSceneState(), cfg_.robot);
        if (robot_state == nullptr) {
            throw std::runtime_error("NativeVectorEnv runtime state has no robot named '" + cfg_.robot + "'");
        }
        const bool terminated = ComputeTerminated(env_id, slot, *robot_state);
        const bool truncated = slot.episode_length >= cfg_.max_episode_steps;
        result.terminated[row] = terminated ? 1 : 0;
        result.truncated[row] = truncated ? 1 : 0;
        result.reward[row] = ComputeReward(slot, *robot_state, terminated);
        slot.episode_return += result.reward[row];
        result.episode_length[row] = slot.episode_length;
        result.episode_return[row] = slot.episode_return;

        if (terminated || truncated) {
            std::vector<RealType> terminal = ObserveOne(env_id, observation_terms_);
            std::copy(terminal.begin(),
                      terminal.end(),
                      result.terminal_observation.begin() +
                              static_cast<std::ptrdiff_t>(row * observation_size_));
            result.has_terminal_observation[row] = 1;
            if (cfg_.auto_reset) {
                ResetOne(env_id, std::nullopt);
            }
        }
    }

    void ApplyAction(NativeVectorWorld& slot,
                     const PhysicsRobotSnapshot& robot_snapshot,
                     const std::vector<RealType>& action) {
        for (int action_index = 0; action_index < action_size_; ++action_index) {
            const NativeActionTerm& term = action_terms_[action_index];
            const PhysicsJointSnapshot* joint_snapshot = FindJointSnapshot(robot_snapshot, term.joint);
            const PhysicsRobotState* robot_state = FindRobotState(slot.world->GetSceneState(), cfg_.robot);
            const PhysicsJointState* joint_state =
                    robot_state == nullptr ? nullptr : FindJointState(*robot_state, term.joint);
            if (joint_snapshot == nullptr || joint_state == nullptr) {
                throw std::runtime_error("NativeVectorEnv cannot find controlled joint '" + term.joint + "'");
            }
            const RealType clipped = std::clamp(action[action_index], term.lower, term.upper);
            for (const std::string& passive_joint : term.passive_joints) {
                if (!slot.world->SetJointControl(cfg_.robot, passive_joint, PhysicsJointControlMode::Passive, 0.0)) {
                    throw std::runtime_error(slot.world->GetLastError());
                }
            }

            PhysicsJointControlMode control_mode = PhysicsJointControlMode::Position;
            RealType target = term.offset + clipped * term.scale;
            switch (term.mode) {
                case NativeVectorActionMode::NormalizedPosition: {
                    JointControllerLimits limits = MakeJointControllerLimits(*joint_snapshot);
                    if (static_cast<JointType>(joint_snapshot->joint_type) == JointType::Continuous) {
                        limits.has_position_limits = false;
                    }
                    target = JointController::MapNormalizedActionToTargetPosition(clipped,
                                                                                  limits,
                                                                                  joint_state->position,
                                                                                  term.scale);
                    control_mode = PhysicsJointControlMode::Position;
                    break;
                }
                case NativeVectorActionMode::Position:
                    control_mode = PhysicsJointControlMode::Position;
                    break;
                case NativeVectorActionMode::Velocity:
                    control_mode = PhysicsJointControlMode::Velocity;
                    break;
                case NativeVectorActionMode::Effort:
                    control_mode = PhysicsJointControlMode::Effort;
                    break;
            }
            if (!slot.world->SetJointControl(cfg_.robot, term.joint, control_mode, target)) {
                throw std::runtime_error(slot.world->GetLastError());
            }
            slot.previous_action[action_index] = clipped;
        }
    }

    bool ComputeTerminated(int env_id, const NativeVectorWorld& slot, const PhysicsRobotState& robot) const {
        bool terminated = false;
        for (const NativeTerminationTerm& term : termination_terms_) {
            switch (term.type) {
                case TerminationTermType::JointPositionAbsGt: {
                    const RealType value = JointPosition(robot, term.joint, term.wrap);
                    terminated = terminated || std::abs(value) > term.limit;
                    break;
                }
                case TerminationTermType::JointPositionGt:
                    terminated = terminated || JointPosition(robot, term.joint, term.wrap) > term.limit;
                    break;
                case TerminationTermType::JointPositionLt:
                    terminated = terminated || JointPosition(robot, term.joint, term.wrap) < term.limit;
                    break;
                case TerminationTermType::NonFiniteObservation: {
                    const std::vector<RealType> obs = ObserveOne(env_id, observation_terms_);
                    terminated = terminated || !std::all_of(obs.begin(), obs.end(), [](RealType value) {
                        return std::isfinite(value);
                    });
                    break;
                }
            }
        }
        return terminated;
    }

    RealType ComputeReward(NativeVectorWorld& slot, const PhysicsRobotState& robot, bool terminated) {
        if (reward_terms_.empty()) {
            return GetEnvDt();
        }

        RealType reward = 0.0;
        slot.next_target_errors.clear();
        for (const NativeRewardTerm& term : reward_terms_) {
            RealType raw = ComputeRewardRaw(slot, robot, term, terminated);
            RealType weighted = raw * term.weight;
            if (term.scale_by_dt) {
                weighted *= GetEnvDt();
            }
            reward += weighted;
        }
        slot.previous_target_errors = slot.next_target_errors;
        slot.next_target_errors.clear();
        return reward;
    }

    RealType ComputeRewardRaw(NativeVectorWorld& slot,
                              const PhysicsRobotState& robot,
                              const NativeRewardTerm& term,
                              bool terminated) {
        switch (term.type) {
            case RewardTermType::Constant:
                return term.value;
            case RewardTermType::JointPositionL2: {
                const RealType value = JointPosition(robot, term.joint, term.wrap);
                return (value - term.target) * (value - term.target);
            }
            case RewardTermType::JointVelocityL2: {
                const RealType value = JointVelocity(robot, term.joint);
                return value * value;
            }
            case RewardTermType::JointEffortL2: {
                const RealType value = JointEffort(robot, term.joint);
                return value * value;
            }
            case RewardTermType::ActionL2:
                if (term.action_index >= 0) {
                    const RealType value = slot.previous_action[term.action_index];
                    return value * value;
                }
                return std::accumulate(slot.previous_action.begin(), slot.previous_action.end(), RealType{0.0},
                                       [](RealType total, RealType value) { return total + value * value; });
            case RewardTermType::SquaredCommandError: {
                const RealType error = CommandValue(slot, term.command) - JointPosition(robot, term.joint, term.wrap);
                return error * error;
            }
            case RewardTermType::OvershootL2: {
                const RealType command = std::abs(CommandValue(slot, term.command));
                const RealType value = std::abs(JointPosition(robot, term.joint, term.wrap));
                const RealType overshoot = std::max<RealType>(0.0, value - command);
                return overshoot * overshoot;
            }
            case RewardTermType::CommandErrorProgress: {
                const RealType current_error = CommandValue(slot, term.command) - JointPosition(robot, term.joint, term.wrap);
                const std::string key = term.joint + "|" + term.command;
                const auto previous_iter = slot.previous_target_errors.find(key);
                const RealType previous_error = previous_iter == slot.previous_target_errors.end()
                                                        ? current_error
                                                        : previous_iter->second;
                RealType progress = std::abs(previous_error) - std::abs(current_error);
                if (term.has_clip) {
                    progress = std::clamp(progress, term.clip_lower, term.clip_upper);
                }
                slot.next_target_errors[key] = current_error;
                return progress;
            }
            case RewardTermType::JointCommandSettleBonus: {
                const RealType distance = std::abs(CommandValue(slot, term.command) - JointPosition(robot, term.joint, term.wrap));
                const bool ok = distance <= term.target_tolerance &&
                                std::abs(JointVelocity(robot, term.joint)) <= term.velocity_tolerance &&
                                (term.condition_joint.empty() ||
                                 std::abs(JointPosition(robot, term.condition_joint, term.condition_wrap)) <=
                                         term.condition_position_tolerance);
                return ok ? 1.0 : 0.0;
            }
            case RewardTermType::FirstJointCommandReachBonus: {
                const RealType distance = std::abs(CommandValue(slot, term.command) - JointPosition(robot, term.joint, term.wrap));
                const bool ok = !slot.flags.contains(term.name) &&
                                distance <= term.target_tolerance &&
                                (term.condition_joint.empty() ||
                                 std::abs(JointPosition(robot, term.condition_joint, term.condition_wrap)) <=
                                         term.condition_position_tolerance);
                if (ok) {
                    slot.flags.insert(term.name);
                    return 1.0 - std::min<RealType>(static_cast<RealType>(slot.episode_length) /
                                                            static_cast<RealType>(cfg_.max_episode_steps),
                                                    1.0);
                }
                return 0.0;
            }
            case RewardTermType::JointCommandOverspeed: {
                const RealType distance = std::abs(CommandValue(slot, term.command) - JointPosition(robot, term.joint, term.wrap));
                return distance <= term.target_tolerance &&
                               std::abs(JointVelocity(robot, term.joint)) > term.velocity_limit
                       ? 1.0
                       : 0.0;
            }
            case RewardTermType::JointCommandFastCrossing: {
                const RealType current_error = CommandValue(slot, term.command) - JointPosition(robot, term.joint, term.wrap);
                const std::string key = term.joint + "|" + term.command;
                const auto previous_iter = slot.previous_target_errors.find(key);
                const RealType previous_error = previous_iter == slot.previous_target_errors.end()
                                                        ? current_error
                                                        : previous_iter->second;
                return previous_error * current_error < 0.0 &&
                               std::abs(JointVelocity(robot, term.joint)) > term.velocity_tolerance
                       ? 1.0
                       : 0.0;
            }
            case RewardTermType::Terminated:
                return terminated ? 1.0 : 0.0;
        }
        return 0.0;
    }

    RealType JointPosition(const PhysicsRobotState& robot, const std::string& joint_name, bool wrap) const {
        const PhysicsJointState* joint = FindJointState(robot, joint_name);
        const RealType value = joint == nullptr ? 0.0 : joint->position;
        return wrap ? WrapAngle(value) : value;
    }

    RealType JointVelocity(const PhysicsRobotState& robot, const std::string& joint_name) const {
        const PhysicsJointState* joint = FindJointState(robot, joint_name);
        return joint == nullptr ? 0.0 : joint->velocity;
    }

    RealType JointEffort(const PhysicsRobotState& robot, const std::string& joint_name) const {
        const PhysicsJointState* joint = FindJointState(robot, joint_name);
        return joint == nullptr ? 0.0 : joint->effort;
    }

    void EnqueueBatches(const NativeVectorStepResult& result) {
        for (std::size_t begin = 0; begin < result.env_ids.size(); begin += static_cast<std::size_t>(cfg_.batch_size)) {
            const std::size_t end = std::min(result.env_ids.size(), begin + static_cast<std::size_t>(cfg_.batch_size));
            NativeVectorStepResult batch;
            batch.env_ids.assign(result.env_ids.begin() + static_cast<std::ptrdiff_t>(begin),
                                 result.env_ids.begin() + static_cast<std::ptrdiff_t>(end));
            batch.reward.assign(result.reward.begin() + static_cast<std::ptrdiff_t>(begin),
                                result.reward.begin() + static_cast<std::ptrdiff_t>(end));
            batch.terminated.assign(result.terminated.begin() + static_cast<std::ptrdiff_t>(begin),
                                    result.terminated.begin() + static_cast<std::ptrdiff_t>(end));
            batch.truncated.assign(result.truncated.begin() + static_cast<std::ptrdiff_t>(begin),
                                   result.truncated.begin() + static_cast<std::ptrdiff_t>(end));
            batch.has_terminal_observation.assign(result.has_terminal_observation.begin() + static_cast<std::ptrdiff_t>(begin),
                                                  result.has_terminal_observation.begin() + static_cast<std::ptrdiff_t>(end));
            batch.episode_length.assign(result.episode_length.begin() + static_cast<std::ptrdiff_t>(begin),
                                        result.episode_length.begin() + static_cast<std::ptrdiff_t>(end));
            batch.episode_return.assign(result.episode_return.begin() + static_cast<std::ptrdiff_t>(begin),
                                        result.episode_return.begin() + static_cast<std::ptrdiff_t>(end));
            batch.observation = SliceMatrix(result.observation, begin, end, observation_size_);
            batch.critic_observation = SliceMatrix(result.critic_observation, begin, end, critic_observation_size_);
            batch.terminal_observation = SliceMatrix(result.terminal_observation, begin, end, observation_size_);
            async_queue_.push_back(std::move(batch));
        }
    }

    std::vector<RealType> SliceMatrix(const std::vector<RealType>& values,
                                      std::size_t begin,
                                      std::size_t end,
                                      int cols) const {
        std::vector<RealType> slice((end - begin) * static_cast<std::size_t>(cols), 0.0);
        for (std::size_t row = begin; row < end; ++row) {
            const std::size_t dst = row - begin;
            std::copy(values.begin() + static_cast<std::ptrdiff_t>(row * cols),
                      values.begin() + static_cast<std::ptrdiff_t>((row + 1) * cols),
                      slice.begin() + static_cast<std::ptrdiff_t>(dst * cols));
        }
        return slice;
    }

    py::array_t<RealType> MakeMatrix(const std::vector<RealType>& values, int rows, int cols) const {
        py::array_t<RealType> array({rows, cols});
        std::copy(values.begin(), values.end(), static_cast<RealType*>(array.request().ptr));
        return array;
    }

    py::array_t<RealType> MakeVector(const std::vector<RealType>& values) const {
        py::array_t<RealType> array({static_cast<py::ssize_t>(values.size())});
        std::copy(values.begin(), values.end(), static_cast<RealType*>(array.request().ptr));
        return array;
    }

    py::array_t<bool> MakeBoolVector(const std::vector<std::uint8_t>& values) const {
        py::array_t<bool> array({static_cast<py::ssize_t>(values.size())});
        auto* data = static_cast<bool*>(array.request().ptr);
        for (std::size_t index = 0; index < values.size(); ++index) {
            data[index] = values[index] != 0;
        }
        return array;
    }

    py::array_t<std::int64_t> MakeIntVector(const std::vector<std::int64_t>& values) const {
        py::array_t<std::int64_t> array({static_cast<py::ssize_t>(values.size())});
        std::copy(values.begin(), values.end(), static_cast<std::int64_t*>(array.request().ptr));
        return array;
    }

    py::array_t<std::int64_t> MakeEnvIdVector(const std::vector<int>& values) const {
        py::array_t<std::int64_t> array({static_cast<py::ssize_t>(values.size())});
        auto* data = static_cast<std::int64_t*>(array.request().ptr);
        for (std::size_t index = 0; index < values.size(); ++index) {
            data[index] = values[index];
        }
        return array;
    }

    py::tuple MakeResetReturn(const std::vector<int>& ids,
                              const std::vector<RealType>& observations,
                              const std::vector<RealType>& critic_observations,
                              std::optional<std::uint64_t> seed) const {
        py::dict info;
        info["env_id"] = MakeEnvIdVector(ids);
        info["observations"] = MakeObservationInfo(critic_observations, static_cast<int>(ids.size()));
        if (seed.has_value()) {
            info["seed"] = *seed;
        } else {
            info["seed"] = py::none();
        }
        return py::make_tuple(MakeMatrix(observations, static_cast<int>(ids.size()), observation_size_), info);
    }

    py::tuple MakeStepReturn(const NativeVectorStepResult& result) const {
        py::dict info;
        info["env_id"] = MakeEnvIdVector(result.env_ids);
        info["episode_length"] = MakeIntVector(result.episode_length);
        info["episode_return"] = MakeVector(result.episode_return);
        info["has_terminal_observation"] = MakeBoolVector(result.has_terminal_observation);
        info["observations"] = MakeObservationInfo(result.critic_observation,
                                                   static_cast<int>(result.env_ids.size()));
        if (std::any_of(result.has_terminal_observation.begin(),
                        result.has_terminal_observation.end(),
                        [](std::uint8_t value) { return value != 0; })) {
            info["terminal_observation"] = MakeMatrix(result.terminal_observation,
                                                       static_cast<int>(result.env_ids.size()),
                                                       observation_size_);
        }
        return py::make_tuple(MakeMatrix(result.observation, static_cast<int>(result.env_ids.size()), observation_size_),
                              MakeVector(result.reward),
                              MakeBoolVector(result.terminated),
                              MakeBoolVector(result.truncated),
                              info);
    }

    py::dict MakeObservationInfo(const std::vector<RealType>& critic_observations, int rows) const {
        py::dict observations;
        if (critic_observation_size_ > 0) {
            observations["critic"] = MakeMatrix(critic_observations, rows, critic_observation_size_);
        }
        return observations;
    }

    NativeVectorEnvConfig cfg_;
    json task_;
    json actions_section_;
    json observations_section_;
    json rewards_section_;
    json terminations_section_;
    json events_section_;
    json commands_section_;
    Ref<PackedScene> packed_scene_;
    std::vector<Node*> roots_;
    std::vector<NativeVectorWorld> worlds_;
    std::deque<NativeVectorStepResult> async_queue_;
    std::vector<NativeVectorActionConfig> action_configs_;
    std::vector<NativeActionTerm> action_terms_;
    std::vector<NativeObservationTerm> observation_terms_;
    std::vector<NativeObservationTerm> critic_observation_terms_;
    std::vector<NativeRewardTerm> reward_terms_;
    std::vector<NativeTerminationTerm> termination_terms_;
    std::vector<NativeCommandTerm> command_terms_;
    std::vector<NativeEventTerm> event_terms_;
    std::unordered_map<std::string, int> command_indices_;
    int action_size_{0};
    int observation_size_{0};
    int critic_observation_size_{0};

    mutable std::mutex api_mutex_;
    mutable std::mutex worker_mutex_;
    mutable std::condition_variable worker_cv_;
    mutable std::condition_variable worker_done_cv_;
    mutable std::vector<std::thread> workers_;
    mutable std::function<void(std::size_t)> active_task_;
    mutable std::atomic_size_t next_worker_row_{0};
    mutable std::size_t active_row_count_{0};
    mutable int active_helper_count_{0};
    mutable std::size_t task_epoch_{0};
    mutable bool workers_stop_{false};
    mutable std::exception_ptr worker_error_;
};

} // namespace

void RegisterNativeVectorEnv(py::module_& module) {
    py::class_<NativeVectorEnvConfig>(module, "NativeVectorEnvConfig")
            .def(py::init<>())
            .def_readwrite("scene", &NativeVectorEnvConfig::scene)
            .def_readwrite("robot", &NativeVectorEnvConfig::robot)
            .def_readwrite("backend", &NativeVectorEnvConfig::backend)
            .def_readwrite("num_envs", &NativeVectorEnvConfig::num_envs)
            .def_readwrite("batch_size", &NativeVectorEnvConfig::batch_size)
            .def_readwrite("num_workers", &NativeVectorEnvConfig::num_workers)
            .def_readwrite("physics_dt", &NativeVectorEnvConfig::physics_dt)
            .def_readwrite("decimation", &NativeVectorEnvConfig::decimation)
            .def_readwrite("max_episode_steps", &NativeVectorEnvConfig::max_episode_steps)
            .def_readwrite("auto_reset", &NativeVectorEnvConfig::auto_reset)
            .def_readwrite("controlled_joints", &NativeVectorEnvConfig::controlled_joints)
            .def_readwrite("seed", &NativeVectorEnvConfig::seed)
            .def_readwrite("task_json", &NativeVectorEnvConfig::task_json);

    py::enum_<NativeVectorActionMode>(module, "NativeVectorActionMode")
            .value("NormalizedPosition", NativeVectorActionMode::NormalizedPosition)
            .value("Position", NativeVectorActionMode::Position)
            .value("Velocity", NativeVectorActionMode::Velocity)
            .value("Effort", NativeVectorActionMode::Effort)
            .export_values();

    py::class_<NativeVectorActionConfig>(module, "NativeVectorActionConfig")
            .def(py::init<>())
            .def_readwrite("name", &NativeVectorActionConfig::name)
            .def_readwrite("joint", &NativeVectorActionConfig::joint)
            .def_readwrite("mode", &NativeVectorActionConfig::mode)
            .def_readwrite("scale", &NativeVectorActionConfig::scale)
            .def_readwrite("offset", &NativeVectorActionConfig::offset)
            .def_readwrite("lower", &NativeVectorActionConfig::lower)
            .def_readwrite("upper", &NativeVectorActionConfig::upper)
            .def_readwrite("unit", &NativeVectorActionConfig::unit)
            .def_readwrite("passive_joints", &NativeVectorActionConfig::passive_joints);

    py::class_<NativeVectorEnv>(module, "NativeVectorEnv")
            .def(py::init<NativeVectorEnvConfig, std::vector<NativeVectorActionConfig>>(),
                 py::arg("config"),
                 py::arg("actions") = std::vector<NativeVectorActionConfig>{})
            .def_property_readonly("num_envs", &NativeVectorEnv::GetNumEnvs)
            .def_property_readonly("batch_size", &NativeVectorEnv::GetBatchSize)
            .def_property_readonly("num_workers", &NativeVectorEnv::GetNumWorkers)
            .def_property_readonly("observation_size", &NativeVectorEnv::GetObservationSize)
            .def_property_readonly("critic_observation_size", &NativeVectorEnv::GetCriticObservationSize)
            .def_property_readonly("action_size", &NativeVectorEnv::GetActionSize)
            .def_property_readonly("env_dt", &NativeVectorEnv::GetEnvDt)
            .def_property_readonly("observation_spec", &NativeVectorEnv::GetObservationSpec)
            .def_property_readonly("critic_observation_spec", &NativeVectorEnv::GetCriticObservationSpec)
            .def_property_readonly("action_spec", &NativeVectorEnv::GetActionSpec)
            .def("reset", &NativeVectorEnv::Reset,
                 py::arg("seed") = py::none(),
                 py::arg("env_ids") = py::none())
            .def("step", &NativeVectorEnv::Step,
                 py::arg("action"),
                 py::arg("env_ids") = py::none())
            .def("async_reset", &NativeVectorEnv::AsyncReset)
            .def("send", &NativeVectorEnv::Send,
                 py::arg("action"),
                 py::arg("env_ids") = py::none())
            .def("recv", &NativeVectorEnv::Recv);
}

} // namespace gobot::python
