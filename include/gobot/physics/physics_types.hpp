/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
*/

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "gobot/core/math/geometry.hpp"
#include "gobot/core/ref_counted.hpp"
#include "gobot/physics/joint_controller.hpp"

namespace gobot {

class CollisionShape3D;
class Joint3D;
class Link3D;
class Node3D;
class Robot3D;

enum class PhysicsBackendType {
    Null,
    MuJoCoCpu,
    NewtonGpu,
    RigidIpcCpu,
    MuJoCoWarp
};

enum class PhysicsShapeType {
    Unknown,
    Box,
    Sphere,
    Cylinder,
    Mesh
};

enum class PhysicsLinkRole {
    Physical,
    VirtualRoot
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

struct PhysicsWorldSettings {
    Vector3 gravity{0.0, 0.0, -9.81};
    RealType fixed_time_step{1.0 / 240.0};
    JointControllerGains default_joint_gains{100.0, 10.0, 0.0, 0.0};
};

struct PhysicsShapeSnapshot {
    const CollisionShape3D* node{nullptr};
    PhysicsShapeType type{PhysicsShapeType::Unknown};
    Affine3 global_transform{Affine3::Identity()};
    Vector3 box_size{Vector3::Ones()};
    RealType radius{0.0};
    RealType height{0.0};
    bool disabled{false};
};

struct PhysicsLinkSnapshot {
    const Link3D* node{nullptr};
    std::string name;
    PhysicsLinkRole role{PhysicsLinkRole::Physical};
    Affine3 global_transform{Affine3::Identity()};
    RealType mass{0.0};
    Vector3 center_of_mass{Vector3::Zero()};
    Vector3 inertia_diagonal{Vector3::Zero()};
    Vector3 inertia_off_diagonal{Vector3::Zero()};
    std::vector<PhysicsShapeSnapshot> collision_shapes;
};

struct PhysicsJointSnapshot {
    const Joint3D* node{nullptr};
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
    RealType joint_position{0.0};
    int joint_type{0};
};

struct PhysicsRobotSnapshot {
    const Robot3D* node{nullptr};
    std::string name;
    std::string source_path;
    std::vector<PhysicsLinkSnapshot> links;
    std::vector<PhysicsJointSnapshot> joints;
};

struct PhysicsSceneSnapshot {
    std::vector<PhysicsRobotSnapshot> robots;
    std::vector<PhysicsShapeSnapshot> loose_collision_shapes;
    std::size_t total_link_count{0};
    std::size_t total_joint_count{0};
    std::size_t total_collision_shape_count{0};
};

struct PhysicsJointState {
    const Joint3D* node{nullptr};
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
    const Link3D* node{nullptr};
    std::string robot_name;
    std::string link_name;
    PhysicsLinkRole role{PhysicsLinkRole::Physical};
    Affine3 global_transform{Affine3::Identity()};
    Vector3 linear_velocity{Vector3::Zero()};
    Vector3 angular_velocity{Vector3::Zero()};
};

struct PhysicsContactState {
    std::string robot_name;
    std::string link_name;
    std::string other_robot_name;
    std::string other_link_name;
    Vector3 position{Vector3::Zero()};
    Vector3 normal{Vector3::UnitZ()};
    RealType distance{0.0};
};

struct PhysicsRobotState {
    const Robot3D* node{nullptr};
    std::string name;
    std::vector<PhysicsLinkState> links;
    std::vector<PhysicsJointState> joints;
};

struct PhysicsSceneState {
    std::vector<PhysicsRobotState> robots;
    std::vector<PhysicsContactState> contacts;
    std::size_t total_link_count{0};
    std::size_t total_joint_count{0};
};

} // namespace gobot
