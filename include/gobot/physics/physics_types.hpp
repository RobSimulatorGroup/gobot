/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "gobot/core/color.hpp"
#include "gobot/core/math/geometry.hpp"
#include "gobot/core/ref_counted.hpp"
#include "gobot/core/robotics_types.hpp"
#include "gobot/physics/joint_controller.hpp"

namespace gobot {

enum class PhysicsBackendType {
    Null,
    MuJoCoCpu
};

enum class PhysicsShapeType {
    Unknown,
    Box,
    Sphere,
    Cylinder,
    Capsule,
    Mesh
};

enum class PhysicsLinkRole {
    Physical,
    VirtualRoot
};

enum class PhysicsSensorType {
    Unknown,
    IMU,
    AngularMomentum,
    Contact,
    RayCast,
    TerrainHeight,
    HeightScanner
};

enum class PhysicsSolverType : int {
    ProjectedGaussSeidel = 0,
    ConjugateGradient = 1,
    Newton = 2,
};

enum class PhysicsIntegratorType : int {
    Euler = 0,
    RungeKutta4 = 1,
    Implicit = 2,
    ImplicitFast = 3,
};

enum class PhysicsFrictionConeType : int {
    Pyramidal = 0,
    Elliptic = 1,
};

enum class PhysicsJacobianType : int {
    Dense = 0,
    Sparse = 1,
    Auto = 2,
};

struct PhysicsBackendInfo {
    PhysicsBackendType type{PhysicsBackendType::Null};
    std::string name;
    bool available{false};
    bool cpu{true};
    bool gpu{false};
    bool robotics_focused{false};
    std::string status;
};

struct MuJoCoSolverSettings {
    PhysicsSolverType solver{PhysicsSolverType::Newton};
    PhysicsIntegratorType integrator{PhysicsIntegratorType::Euler};
    PhysicsFrictionConeType cone{PhysicsFrictionConeType::Pyramidal};
    PhysicsJacobianType jacobian{PhysicsJacobianType::Auto};
    int iterations{100};
    int line_search_iterations{50};
    int no_slip_iterations{0};
    int convex_collision_iterations{35};
    RealType tolerance{1.0e-8};
    RealType line_search_tolerance{0.01};
    RealType no_slip_tolerance{1.0e-6};
    RealType convex_collision_tolerance{1.0e-6};
    RealType impedance_ratio{1.0};
};

struct PhysicsWorldSettings {
    Vector3 gravity{0.0, 0.0, -9.81};
    RealType fixed_time_step{0.002};
    JointControllerGains default_joint_gains{100.0, 10.0, 0.0, 0.0};
    MuJoCoSolverSettings mujoco_solver;
    bool debug_draw_contacts{false};
    bool debug_draw_contact_forces{true};
    RealType debug_contact_force_scale{0.08};
    RealType debug_contact_force_max_length{0.8};
};

struct PhysicsShapeSnapshot {
    std::string name;
    PhysicsShapeType type{PhysicsShapeType::Unknown};
    Affine3 global_transform{Affine3::Identity()};
    Vector3 box_size{Vector3::Ones()};
    RealType radius{0.0};
    RealType height{0.0};
    Vector3 friction{1.0, 0.005, 0.0001};
    int contype{1};
    int conaffinity{1};
    int condim{3};
    Vector2 solref{0.02, 1.0};
    std::vector<RealType> solimp{0.9, 0.95, 0.001, 0.5, 2.0};
    RealType margin{0.0};
    RealType gap{0.0};
    int priority{0};
    bool disabled{false};
};

struct PhysicsLinkSnapshot {
    std::string name;
    PhysicsLinkRole role{PhysicsLinkRole::Physical};
    Affine3 global_transform{Affine3::Identity()};
    RealType mass{0.0};
    Vector3 center_of_mass{Vector3::Zero()};
    Quaternion inertia_orientation{Quaternion::Identity()};
    Vector3 inertia_diagonal{Vector3::Zero()};
    Vector3 inertia_off_diagonal{Vector3::Zero()};
    std::vector<PhysicsShapeSnapshot> collision_shapes;
};

struct PhysicsSensorSnapshot {
    PhysicsSensorType type{PhysicsSensorType::Unknown};
    std::string name;
    std::string link_name;
    Affine3 global_transform{Affine3::Identity()};
    Affine3 local_transform{Affine3::Identity()};
    bool enabled{true};
    RealType sensor_period{0.0};
    RealType noise_stddev{0.0};
    bool visualize_debug{false};
    bool visible{true};
    RealType debug_marker_radius{0.0};
    RealType radius{0.0};
    RealType min_threshold{0.0};
    RealType max_threshold{0.0};
    std::vector<Vector3> sample_offsets;
    Vector3 ray_direction{0.0, 0.0, -1.0};
    bool ray_direction_world_space{true};
    RealType max_distance{3.0};
    RayReductionMode reduction_mode{RayReductionMode::None};
    RayPatternMode pattern_mode{RayPatternMode::Custom};
    Vector2 grid_size{1.0, 1.0};
    RealType grid_resolution{0.1};
    RayAlignmentMode ray_alignment{RayAlignmentMode::World};
    std::vector<std::string> channel_names;
};

struct PhysicsJointSnapshot {
    std::string name;
    std::string parent_link;
    std::string child_link;
    Affine3 global_transform{Affine3::Identity()};
    Vector3 axis{Vector3::UnitX()};
    RealType lower_limit{0.0};
    RealType upper_limit{0.0};
    RealType effort_limit{0.0};
    RealType velocity_limit{0.0};
    RealType damping{0.0};
    RealType armature{0.0};
    RealType friction_loss{0.0};
    RealType joint_position{0.0};
    RealType initial_position{0.0};
    int drive_mode{0};
    RealType drive_stiffness{0.0};
    RealType drive_damping{0.0};
    RealType control_lower_limit{0.0};
    RealType control_upper_limit{0.0};
    RealType force_lower_limit{0.0};
    RealType force_upper_limit{0.0};
    std::vector<RealType> gear{1.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    int joint_type{0};
};

struct PhysicsRobotSnapshot {
    std::string name;
    std::vector<PhysicsLinkSnapshot> links;
    std::vector<PhysicsJointSnapshot> joints;
    std::vector<PhysicsSensorSnapshot> sensors;
};

struct PhysicsTerrainBoxSnapshot {
    Affine3 global_transform{Affine3::Identity()};
    Vector3 size{Vector3::Ones()};
    Vector2 xy_min{Vector2::Zero()};
    Vector2 xy_max{Vector2::Zero()};
    bool has_xy_bounds{false};
};

struct PhysicsTerrainHeightFieldSnapshot {
    Affine3 global_transform{Affine3::Identity()};
    Vector2 size{1.0, 1.0};
    int rows{0};
    int cols{0};
    std::vector<RealType> heights;
    std::vector<RealType> normalized_elevation;
    RealType base_thickness{0.1};
    RealType z_offset{0.0};
    Vector2 xy_min{Vector2::Zero()};
    Vector2 xy_max{Vector2::Zero()};
    bool has_xy_bounds{false};
};

struct PhysicsTerrainMeshPatchSnapshot {
    Affine3 global_transform{Affine3::Identity()};
    std::vector<Vector3> vertices;
    std::vector<std::uint32_t> indices;
    Color color{1.0f, 1.0f, 1.0f, 1.0f};
    Vector2 xy_min{Vector2::Zero()};
    Vector2 xy_max{Vector2::Zero()};
    bool has_xy_bounds{false};
};

struct PhysicsTerrainSnapshot {
    std::string name;
    Color surface_color{0.48f, 0.56f, 0.50f, 1.0f};
    Vector3 friction{1.0, 0.005, 0.0001};
    int contype{1};
    int conaffinity{1};
    int condim{3};
    Vector2 solref{0.02, 1.0};
    std::vector<RealType> solimp{0.9, 0.95, 0.001, 0.5, 2.0};
    RealType margin{0.0};
    RealType gap{0.0};
    std::vector<PhysicsTerrainBoxSnapshot> boxes;
    std::vector<PhysicsTerrainHeightFieldSnapshot> heightfields;
    std::vector<PhysicsTerrainMeshPatchSnapshot> mesh_patches;
    std::vector<Vector3> spawn_origins;
};

struct PhysicsSceneSnapshot {
    std::vector<PhysicsRobotSnapshot> robots;
    std::vector<PhysicsTerrainSnapshot> terrains;
    std::vector<PhysicsShapeSnapshot> loose_collision_shapes;
    std::vector<PhysicsSensorSnapshot> loose_sensors;
    std::size_t total_link_count{0};
    std::size_t total_joint_count{0};
    std::size_t total_collision_shape_count{0};
    std::size_t total_sensor_count{0};
    std::size_t total_terrain_count{0};
};

struct PhysicsSceneArtifact {
    std::uint32_t schema_version{0};
    PhysicsBackendType backend{PhysicsBackendType::Null};
    std::string format;
    std::string content;
    std::string content_digest;
    std::string backend_version;
    std::size_t nq{0};
    std::size_t nv{0};
    std::size_t nu{0};
    std::size_t nbody{0};
    std::size_t njoint{0};
    std::size_t ngeom{0};
    std::size_t nsensor{0};
    std::size_t nhfield{0};
    std::vector<std::string> robot_names;
    std::vector<std::string> robot_prefixes;
};

struct PhysicsContactShapeGroup {
    std::string name;
    std::vector<std::string> shape_names;
    RealType force_threshold{0.0};
};

struct PhysicsRobotBatchStepRequest {
    std::string robot_name;
    std::string base_link;
    std::vector<std::string> joint_names;
    std::vector<std::string> link_names;
    std::vector<std::string> sensor_names;
    std::vector<RealType> target_positions;
    std::vector<RealType> joint_position_stiffness;
    std::vector<RealType> joint_velocity_damping;
    std::vector<std::string> override_link_names;
    std::vector<RealType> link_mass_delta;
    std::vector<RealType> link_center_of_mass_offset;
    std::vector<std::string> override_shape_names;
    std::vector<RealType> shape_friction;
    std::vector<std::uint8_t> shape_friction_enabled;
    std::string external_wrench_link;
    std::vector<RealType> external_force;
    std::vector<RealType> external_torque;
    RealType ground_contact_force_threshold{0.0};
    RealType self_contact_force_threshold{0.0};
    std::vector<PhysicsContactShapeGroup> contact_shape_groups;
    bool collect_contact_history{false};
    std::uint64_t ticks{1};
    std::size_t worker_count{0};
};

struct PhysicsRobotBatchStepResult {
    std::string robot_name;
    std::string base_link;
    std::vector<std::string> joint_names;
    std::vector<std::string> link_names;
    std::vector<std::string> sensor_names;
    std::vector<std::string> shape_names;
    std::size_t environment_count{0};
    std::size_t max_sensor_values{0};
    std::size_t max_sensor_hits{0};
    std::size_t max_contact_count{0};
    std::vector<RealType> base_position;
    std::vector<RealType> base_quaternion;
    std::vector<RealType> base_linear_velocity;
    std::vector<RealType> base_angular_velocity;
    std::vector<RealType> joint_position;
    std::vector<RealType> joint_velocity;
    std::vector<RealType> joint_acceleration;
    std::vector<RealType> joint_effort;
    std::vector<RealType> joint_target_position;
    std::vector<RealType> joint_target_velocity;
    std::vector<RealType> joint_target_effort;
    std::vector<RealType> joint_lower_limit;
    std::vector<RealType> joint_upper_limit;
    std::vector<RealType> link_position;
    std::vector<RealType> link_quaternion;
    std::vector<RealType> link_linear_velocity;
    std::vector<RealType> link_angular_velocity;
    std::vector<std::int32_t> sensor_value_count;
    std::vector<std::int32_t> sensor_hit_count;
    std::vector<RealType> sensor_position;
    std::vector<RealType> sensor_quaternion;
    std::vector<RealType> sensor_linear_velocity;
    std::vector<RealType> sensor_values;
    std::vector<std::uint8_t> sensor_hit;
    std::vector<RealType> sensor_hit_origin;
    std::vector<RealType> sensor_hit_point;
    std::vector<RealType> sensor_hit_normal;
    std::vector<RealType> sensor_hit_distance;
    std::vector<std::int32_t> contact_count;
    std::vector<std::int32_t> contact_link_index;
    std::vector<std::int32_t> contact_shape_index;
    std::vector<RealType> contact_position;
    std::vector<RealType> contact_normal;
    std::vector<RealType> contact_force;
    std::vector<RealType> contact_normal_force;
    std::vector<RealType> contact_distance;
    std::vector<std::int32_t> link_contact_tick_count;
    std::vector<std::int32_t> shape_contact_tick_count;
    std::vector<std::string> contact_shape_group_names;
    std::vector<std::int32_t> contact_shape_group_tick_count;
    std::size_t contact_history_tick_count{0};
    std::vector<std::uint8_t> contact_shape_group_history;
    std::vector<std::int32_t> self_contact_tick_count;
};

struct PhysicsJointState {
    std::string robot_name;
    std::string joint_name;
    int joint_type{0};
    RealType position{0.0};
    RealType velocity{0.0};
    RealType effort{0.0};
    PhysicsJointControlMode control_mode{PhysicsJointControlMode::Passive};
    RealType target_position{0.0};
    RealType target_velocity{0.0};
    RealType target_effort{0.0};
};

struct PhysicsLinkState {
    std::string robot_name;
    std::string link_name;
    PhysicsLinkRole role{PhysicsLinkRole::Physical};
    Affine3 global_transform{Affine3::Identity()};
    Vector3 linear_velocity{Vector3::Zero()}; // World frame.
    Vector3 angular_velocity{Vector3::Zero()}; // World frame.
};

struct PhysicsContactState {
    std::string robot_name;
    std::string link_name;
    std::string other_robot_name;
    std::string other_link_name;
    Vector3 position{Vector3::Zero()};
    Vector3 normal{Vector3::UnitZ()};
    Vector3 force{Vector3::Zero()};
    RealType normal_force{0.0};
    RealType distance{0.0};
};

struct PhysicsRaycastQuery {
    Vector3 origin{Vector3::Zero()};
    Vector3 direction{0.0, 0.0, -1.0};
    RealType max_distance{1.0};
};

struct PhysicsRaycastHit {
    bool hit{false};
    Vector3 origin{Vector3::Zero()};
    Vector3 point{Vector3::Zero()};
    Vector3 normal{Vector3::UnitZ()};
    RealType distance{0.0};
    std::string terrain_name;
};

struct PhysicsSensorRaycastHit {
    bool hit{false};
    Vector3 origin{Vector3::Zero()};
    Vector3 point{Vector3::Zero()};
    Vector3 normal{Vector3::UnitZ()};
    RealType distance{0.0};
    std::string terrain_name;
};

struct PhysicsSensorState {
    std::string robot_name;
    std::string link_name;
    std::string sensor_name;
    PhysicsSensorType type{PhysicsSensorType::Unknown};
    bool enabled{true};
    bool visualize_debug{false};
    bool visible{true};
    RealType debug_marker_radius{0.0};
    Affine3 global_transform{Affine3::Identity()};
    std::vector<RealType> values;
    std::vector<PhysicsSensorRaycastHit> hits;
    std::vector<std::string> channel_names;
    RealType timestamp{0.0};
};

struct PhysicsExternalForce {
    std::string robot_name;
    std::string link_name;
    Vector3 point{Vector3::Zero()};
    Vector3 local_point{Vector3::Zero()};
    Vector3 target_point{Vector3::Zero()};
    Vector3 force{Vector3::Zero()};
    bool use_spring{false};
};

struct PhysicsEnvironmentRobotResetState {
    std::size_t environment_index{0};
    std::string robot_name;
    std::string base_link_name;
    Vector3 base_position{Vector3::Zero()};
    Quaternion base_orientation{Quaternion::Identity()};
    Vector3 base_linear_velocity{Vector3::Zero()}; // World frame.
    Vector3 base_angular_velocity{Vector3::Zero()}; // World frame.
    std::vector<std::string> joint_names;
    std::vector<RealType> joint_positions;
    std::vector<RealType> joint_velocities;
    std::vector<RealType> joint_position_targets;
};

struct PhysicsRobotState {
    std::string name;
    std::vector<PhysicsLinkState> links;
    std::vector<PhysicsJointState> joints;
    std::vector<PhysicsSensorState> sensors;
};

struct PhysicsSceneState {
    std::vector<PhysicsRobotState> robots;
    std::vector<PhysicsContactState> contacts;
    std::vector<PhysicsSensorState> loose_sensors;
    std::size_t total_link_count{0};
    std::size_t total_joint_count{0};
    std::size_t total_sensor_count{0};
};

} // namespace gobot
