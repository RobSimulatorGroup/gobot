/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/physics/physics_world.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <thread>
#include <utility>

#include "gobot/core/registration.hpp"
#include "gobot/core/robotics_types.hpp"
#include "gobot/physics/physics_sensor_utils.hpp"

namespace gobot {
namespace {

PhysicsJointControlMode JointDriveModeToControlMode(int drive_mode) {
    switch (static_cast<JointDriveMode>(drive_mode)) {
        case JointDriveMode::Position:
            return PhysicsJointControlMode::Position;
        case JointDriveMode::Velocity:
            return PhysicsJointControlMode::Velocity;
        case JointDriveMode::Motor:
            return PhysicsJointControlMode::Effort;
        case JointDriveMode::Passive:
            return PhysicsJointControlMode::Passive;
    }

    return PhysicsJointControlMode::Passive;
}

const PhysicsRobotState* FindRobotState(const PhysicsSceneState& state, const std::string& robot_name) {
    for (const PhysicsRobotState& robot_state : state.robots) {
        if (robot_state.name == robot_name) {
            return &robot_state;
        }
    }

    return nullptr;
}

const PhysicsLinkState* FindLinkState(const PhysicsRobotState& robot_state, const std::string& link_name) {
    for (const PhysicsLinkState& link_state : robot_state.links) {
        if (link_state.link_name == link_name) {
            return &link_state;
        }
    }

    return nullptr;
}

const PhysicsJointState* FindPreviousJointState(const PhysicsRobotState& robot_state, const std::string& joint_name) {
    for (const PhysicsJointState& joint_state : robot_state.joints) {
        if (joint_state.joint_name == joint_name) {
            return &joint_state;
        }
    }

    return nullptr;
}

const PhysicsSensorState* FindPreviousSensorState(const PhysicsRobotState& robot_state,
                                                  const std::string& sensor_name,
                                                  const std::string& link_name) {
    for (const PhysicsSensorState& sensor_state : robot_state.sensors) {
        if (sensor_state.sensor_name == sensor_name && sensor_state.link_name == link_name) {
            return &sensor_state;
        }
    }

    return nullptr;
}

PhysicsSensorState MakeSensorStateFromSnapshot(const PhysicsSensorSnapshot& sensor_snapshot,
                                               const std::string& robot_name = {}) {
    PhysicsSensorState sensor_state;
    sensor_state.robot_name = robot_name;
    sensor_state.link_name = sensor_snapshot.link_name;
    sensor_state.sensor_name = sensor_snapshot.name;
    sensor_state.type = sensor_snapshot.type;
    sensor_state.enabled = sensor_snapshot.enabled;
    sensor_state.visualize_debug = sensor_snapshot.visualize_debug;
    sensor_state.visible = sensor_snapshot.visible;
    sensor_state.debug_marker_radius = sensor_snapshot.debug_marker_radius;
    sensor_state.global_transform = sensor_snapshot.global_transform;
    sensor_state.channel_names = sensor_snapshot.channel_names;
    sensor_state.values.assign(sensor_state.channel_names.size(), 0.0);
    if (sensor_snapshot.type == PhysicsSensorType::IMU && sensor_state.values.size() >= 4) {
        const Quaternion orientation(sensor_snapshot.global_transform.linear());
        sensor_state.values[0] = orientation.w();
        sensor_state.values[1] = orientation.x();
        sensor_state.values[2] = orientation.y();
        sensor_state.values[3] = orientation.z();
    }
    return sensor_state;
}

bool IsRaycastSensorType(PhysicsSensorType type) {
    return type == PhysicsSensorType::RayCast ||
           type == PhysicsSensorType::TerrainHeight ||
           type == PhysicsSensorType::HeightScanner;
}

bool IsTerrainHeightSensorType(PhysicsSensorType type) {
    return type == PhysicsSensorType::TerrainHeight ||
           type == PhysicsSensorType::HeightScanner;
}

bool IsVerticalTerrainHeightRay(const PhysicsRaycastQuery& query);

struct InternalTerrainRaycastHit {
    bool hit{false};
    Vector3 point{Vector3::Zero()};
    Vector3 normal{Vector3::UnitZ()};
    RealType distance{0.0};
    const std::string* terrain_name{nullptr};
};

bool PointInXYBounds(const Vector3& point,
                     const Vector2& xy_min,
                     const Vector2& xy_max,
                     RealType margin = CMP_EPSILON) {
    return point.x() >= xy_min.x() - margin &&
           point.x() <= xy_max.x() + margin &&
           point.y() >= xy_min.y() - margin &&
           point.y() <= xy_max.y() + margin;
}

void ExpandXYBounds(const Vector3& point, Vector2* xy_min, Vector2* xy_max, bool* initialized) {
    if (xy_min == nullptr || xy_max == nullptr || initialized == nullptr) {
        return;
    }
    const Vector2 xy{point.x(), point.y()};
    if (!*initialized) {
        *xy_min = xy;
        *xy_max = xy;
        *initialized = true;
        return;
    }
    xy_min->x() = std::min(xy_min->x(), xy.x());
    xy_min->y() = std::min(xy_min->y(), xy.y());
    xy_max->x() = std::max(xy_max->x(), xy.x());
    xy_max->y() = std::max(xy_max->y(), xy.y());
}

void SetBoxXYBounds(PhysicsTerrainBoxSnapshot* box) {
    if (box == nullptr) {
        return;
    }
    const Vector3 half = box->size.cwiseMax(Vector3::Zero()) * 0.5;
    bool initialized = false;
    for (int x_sign : {-1, 1}) {
        for (int y_sign : {-1, 1}) {
            for (int z_sign : {-1, 1}) {
                ExpandXYBounds(box->global_transform *
                                       Vector3(static_cast<RealType>(x_sign) * half.x(),
                                               static_cast<RealType>(y_sign) * half.y(),
                                               static_cast<RealType>(z_sign) * half.z()),
                               &box->xy_min,
                               &box->xy_max,
                               &initialized);
            }
        }
    }
    box->has_xy_bounds = initialized;
}

void SetHeightFieldXYBounds(PhysicsTerrainHeightFieldSnapshot* heightfield) {
    if (heightfield == nullptr) {
        return;
    }
    const RealType half_x = heightfield->size.x() * 0.5;
    const RealType half_y = heightfield->size.y() * 0.5;
    bool initialized = false;
    for (int x_sign : {-1, 1}) {
        for (int y_sign : {-1, 1}) {
            ExpandXYBounds(heightfield->global_transform *
                                   Vector3(static_cast<RealType>(x_sign) * half_x,
                                           static_cast<RealType>(y_sign) * half_y,
                                           0.0),
                           &heightfield->xy_min,
                           &heightfield->xy_max,
                           &initialized);
        }
    }
    heightfield->has_xy_bounds = initialized;
}

void SetMeshPatchXYBounds(PhysicsTerrainMeshPatchSnapshot* mesh_patch) {
    if (mesh_patch == nullptr) {
        return;
    }
    bool initialized = false;
    for (const Vector3& vertex : mesh_patch->vertices) {
        ExpandXYBounds(mesh_patch->global_transform * vertex,
                       &mesh_patch->xy_min,
                       &mesh_patch->xy_max,
                       &initialized);
    }
    mesh_patch->has_xy_bounds = initialized;
}

std::optional<RealType> QueryTerrainBoxHeight(const PhysicsTerrainBoxSnapshot& box, const Vector3& world_position) {
    if (box.has_xy_bounds && !PointInXYBounds(world_position, box.xy_min, box.xy_max)) {
        return std::nullopt;
    }
    const Vector3 half = box.size.cwiseMax(Vector3::Zero()) * 0.5;
    if (half.x() <= CMP_EPSILON || half.y() <= CMP_EPSILON || half.z() <= CMP_EPSILON) {
        return std::nullopt;
    }

    const Affine3 inverse = box.global_transform.inverse();
    const Vector3 local = inverse * world_position;
    if (std::abs(local.x()) > half.x() + CMP_EPSILON ||
        std::abs(local.y()) > half.y() + CMP_EPSILON) {
        return std::nullopt;
    }

    const Vector3 top = box.global_transform * Vector3(local.x(), local.y(), half.z());
    return top.z();
}

std::optional<InternalTerrainRaycastHit> RaycastTerrainBox(const PhysicsTerrainBoxSnapshot& box,
                                                           const std::string& terrain_name,
                                                           const PhysicsRaycastQuery& query) {
    if (IsVerticalTerrainHeightRay(query) &&
        box.has_xy_bounds &&
        !PointInXYBounds(query.origin, box.xy_min, box.xy_max)) {
        return std::nullopt;
    }
    const Vector3 half = box.size.cwiseMax(Vector3::Zero()) * 0.5;
    if (half.x() <= CMP_EPSILON || half.y() <= CMP_EPSILON || half.z() <= CMP_EPSILON) {
        return std::nullopt;
    }

    const Affine3 inverse = box.global_transform.inverse();
    const Vector3 origin = inverse * query.origin;
    const Vector3 direction = inverse.linear() * query.direction;
    RealType t_min = 0.0;
    RealType t_max = query.max_distance;
    int hit_axis = -1;
    RealType hit_sign = 1.0;

    for (int axis = 0; axis < 3; ++axis) {
        const RealType min_value = -half[axis];
        const RealType max_value = half[axis];
        if (std::abs(direction[axis]) <= CMP_EPSILON) {
            if (origin[axis] < min_value || origin[axis] > max_value) {
                return std::nullopt;
            }
            continue;
        }

        RealType t0 = (min_value - origin[axis]) / direction[axis];
        RealType t1 = (max_value - origin[axis]) / direction[axis];
        RealType sign = -1.0;
        if (t0 > t1) {
            std::swap(t0, t1);
            sign = 1.0;
        }
        if (t0 > t_min) {
            t_min = t0;
            hit_axis = axis;
            hit_sign = sign;
        }
        t_max = std::min(t_max, t1);
        if (t_min > t_max) {
            return std::nullopt;
        }
    }

    const RealType distance = t_min >= 0.0 ? t_min : t_max;
    if (distance < 0.0 || distance > query.max_distance) {
        return std::nullopt;
    }

    Vector3 local_normal = Vector3::Zero();
    if (hit_axis >= 0) {
        local_normal[hit_axis] = hit_sign;
    } else {
        local_normal.z() = 1.0;
    }

    InternalTerrainRaycastHit hit;
    hit.hit = true;
    hit.point = box.global_transform * (origin + direction * distance);
    hit.normal = (box.global_transform.linear() * local_normal).normalized();
    hit.distance = distance;
    hit.terrain_name = &terrain_name;
    return hit;
}

std::optional<RealType> QueryTerrainHeightFieldHeight(const PhysicsTerrainHeightFieldSnapshot& heightfield,
                                                      const Vector3& world_position) {
    if (heightfield.has_xy_bounds && !PointInXYBounds(world_position, heightfield.xy_min, heightfield.xy_max)) {
        return std::nullopt;
    }
    if (heightfield.rows < 2 ||
        heightfield.cols < 2 ||
        heightfield.size.x() <= CMP_EPSILON ||
        heightfield.size.y() <= CMP_EPSILON ||
        heightfield.heights.size() < static_cast<std::size_t>(heightfield.rows * heightfield.cols)) {
        return std::nullopt;
    }

    const Vector3 center = heightfield.global_transform.translation();
    const Vector3 local = world_position - center;
    const RealType half_x = heightfield.size.x() * 0.5;
    const RealType half_y = heightfield.size.y() * 0.5;
    if (std::abs(local.x()) > half_x + CMP_EPSILON || std::abs(local.y()) > half_y + CMP_EPSILON) {
        return std::nullopt;
    }

    const RealType u = (local.x() / heightfield.size.x() + 0.5) * static_cast<RealType>(heightfield.cols - 1);
    const RealType v = (-local.y() / heightfield.size.y() + 0.5) * static_cast<RealType>(heightfield.rows - 1);
    const int c0 = std::clamp(static_cast<int>(std::floor(u)), 0, heightfield.cols - 1);
    const int r0 = std::clamp(static_cast<int>(std::floor(v)), 0, heightfield.rows - 1);
    const int c1 = std::min(c0 + 1, heightfield.cols - 1);
    const int r1 = std::min(r0 + 1, heightfield.rows - 1);
    const RealType fu = u - static_cast<RealType>(c0);
    const RealType fv = v - static_cast<RealType>(r0);

    auto height_at = [&heightfield](int row, int col) {
        const std::size_t index = static_cast<std::size_t>(row * heightfield.cols + col);
        return heightfield.heights[index];
    };

    const RealType h00 = height_at(r0, c0);
    const RealType h10 = height_at(r0, c1);
    const RealType h01 = height_at(r1, c0);
    const RealType h11 = height_at(r1, c1);
    const RealType h0 = h00 * (1.0 - fu) + h10 * fu;
    const RealType h1 = h01 * (1.0 - fu) + h11 * fu;
    const RealType local_height = heightfield.z_offset + h0 * (1.0 - fv) + h1 * fv;
    return center.z() + local_height;
}

std::optional<Vector3> QueryTerrainHeightFieldNormal(const PhysicsTerrainHeightFieldSnapshot& heightfield,
                                                     const Vector3& world_position) {
    if (heightfield.has_xy_bounds && !PointInXYBounds(world_position, heightfield.xy_min, heightfield.xy_max)) {
        return std::nullopt;
    }
    if (heightfield.rows < 2 ||
        heightfield.cols < 2 ||
        heightfield.size.x() <= CMP_EPSILON ||
        heightfield.size.y() <= CMP_EPSILON ||
        heightfield.heights.size() < static_cast<std::size_t>(heightfield.rows * heightfield.cols)) {
        return std::nullopt;
    }

    const Vector3 center = heightfield.global_transform.translation();
    const Vector3 local = world_position - center;
    const RealType half_x = heightfield.size.x() * 0.5;
    const RealType half_y = heightfield.size.y() * 0.5;
    if (std::abs(local.x()) > half_x + CMP_EPSILON || std::abs(local.y()) > half_y + CMP_EPSILON) {
        return std::nullopt;
    }

    const RealType cell_x = heightfield.size.x() / static_cast<RealType>(heightfield.cols - 1);
    const RealType cell_y = heightfield.size.y() / static_cast<RealType>(heightfield.rows - 1);
    const RealType u = (local.x() / heightfield.size.x() + 0.5) * static_cast<RealType>(heightfield.cols - 1);
    const RealType v = (-local.y() / heightfield.size.y() + 0.5) * static_cast<RealType>(heightfield.rows - 1);
    const int col = std::clamp(static_cast<int>(std::round(u)), 0, heightfield.cols - 1);
    const int row = std::clamp(static_cast<int>(std::round(v)), 0, heightfield.rows - 1);
    const int col_minus = std::max(col - 1, 0);
    const int col_plus = std::min(col + 1, heightfield.cols - 1);
    const int row_minus = std::max(row - 1, 0);
    const int row_plus = std::min(row + 1, heightfield.rows - 1);

    auto height_at = [&heightfield](int sample_row, int sample_col) {
        const std::size_t index = static_cast<std::size_t>(sample_row * heightfield.cols + sample_col);
        return heightfield.heights[index] + heightfield.z_offset;
    };

    const RealType dx = std::max<RealType>(cell_x * static_cast<RealType>(col_plus - col_minus), CMP_EPSILON);
    const RealType dy = std::max<RealType>(cell_y * static_cast<RealType>(row_plus - row_minus), CMP_EPSILON);
    const RealType dh_dx = (height_at(row, col_plus) - height_at(row, col_minus)) / dx;
    const RealType dh_dy = (height_at(row_minus, col) - height_at(row_plus, col)) / dy;

    Vector3 local_normal{-dh_dx, -dh_dy, 1.0};
    if (local_normal.squaredNorm() <= CMP_EPSILON2) {
        local_normal = Vector3::UnitZ();
    } else {
        local_normal.normalize();
    }
    Vector3 world_normal = heightfield.global_transform.linear() * local_normal;
    if (world_normal.squaredNorm() <= CMP_EPSILON2) {
        return Vector3::UnitZ();
    }
    world_normal.normalize();
    if (world_normal.z() < 0.0) {
        world_normal = -world_normal;
    }
    return world_normal;
}

bool IsVerticalTerrainHeightRay(const PhysicsRaycastQuery& query) {
    return std::abs(query.direction.x()) <= CMP_EPSILON &&
           std::abs(query.direction.y()) <= CMP_EPSILON &&
           std::abs(query.direction.z()) > CMP_EPSILON;
}

std::optional<InternalTerrainRaycastHit> RaycastTerrainHeightFieldVertical(
        const PhysicsTerrainHeightFieldSnapshot& heightfield,
        const std::string& terrain_name,
        const PhysicsRaycastQuery& query) {
    const std::optional<RealType> height = QueryTerrainHeightFieldHeight(heightfield, query.origin);
    if (!height.has_value()) {
        return std::nullopt;
    }

    const RealType distance = (height.value() - query.origin.z()) / query.direction.z();
    if (distance < 0.0 || distance > query.max_distance) {
        return std::nullopt;
    }

    InternalTerrainRaycastHit hit;
    hit.hit = true;
    hit.point = Vector3(query.origin.x(), query.origin.y(), height.value());
    hit.normal = QueryTerrainHeightFieldNormal(heightfield, hit.point).value_or(Vector3::UnitZ());
    hit.distance = distance;
    hit.terrain_name = &terrain_name;
    return hit;
}

std::optional<InternalTerrainRaycastHit> RaycastTerrainHeightField(const PhysicsTerrainHeightFieldSnapshot& heightfield,
                                                                   const std::string& terrain_name,
                                                                   const PhysicsRaycastQuery& query) {
    const RealType direction_z = query.direction.z();
    if (std::abs(direction_z) <= CMP_EPSILON) {
        return std::nullopt;
    }
    if (IsVerticalTerrainHeightRay(query)) {
        return RaycastTerrainHeightFieldVertical(heightfield, terrain_name, query);
    }

    std::optional<InternalTerrainRaycastHit> best;
    constexpr int kStepCount = 64;
    RealType previous_t = 0.0;
    Vector3 previous_point = query.origin;
    std::optional<RealType> previous_height = QueryTerrainHeightFieldHeight(heightfield, previous_point);
    RealType previous_delta = previous_height.has_value()
                                      ? previous_point.z() - previous_height.value()
                                      : std::numeric_limits<RealType>::quiet_NaN();

    for (int step = 1; step <= kStepCount; ++step) {
        const RealType t = query.max_distance * static_cast<RealType>(step) / static_cast<RealType>(kStepCount);
        const Vector3 point = query.origin + query.direction * t;
        const std::optional<RealType> height = QueryTerrainHeightFieldHeight(heightfield, point);
        if (!height.has_value()) {
            previous_t = t;
            previous_point = point;
            previous_height = std::nullopt;
            previous_delta = std::numeric_limits<RealType>::quiet_NaN();
            continue;
        }

        const RealType delta = point.z() - height.value();
        if (std::abs(delta) <= CMP_EPSILON ||
            (previous_height.has_value() && std::isfinite(previous_delta) &&
             ((previous_delta > 0.0 && delta <= 0.0) || (previous_delta < 0.0 && delta >= 0.0)))) {
            RealType hit_t = t;
            if (previous_height.has_value() && std::abs(previous_delta - delta) > CMP_EPSILON) {
                const RealType alpha = std::clamp(previous_delta / (previous_delta - delta), RealType{0.0}, RealType{1.0});
                hit_t = previous_t + (t - previous_t) * alpha;
            }
            const Vector3 hit_point = query.origin + query.direction * hit_t;
            const std::optional<RealType> hit_height = QueryTerrainHeightFieldHeight(heightfield, hit_point);
            if (!hit_height.has_value()) {
                continue;
            }

            InternalTerrainRaycastHit hit;
            hit.hit = true;
            hit.point = Vector3(hit_point.x(), hit_point.y(), hit_height.value());
            hit.normal = QueryTerrainHeightFieldNormal(heightfield, hit.point).value_or(Vector3::UnitZ());
            hit.distance = hit_t;
            hit.terrain_name = &terrain_name;
            best = hit;
            break;
        }

        previous_t = t;
        previous_point = point;
        previous_height = height;
        previous_delta = delta;
    }

    return best;
}

std::optional<RealType> IntersectVerticalRayWithTriangle(const Vector3& world_position,
                                                         const Vector3& a,
                                                         const Vector3& b,
                                                         const Vector3& c) {
    const Vector2 p{world_position.x(), world_position.y()};
    const Vector2 a2{a.x(), a.y()};
    const Vector2 b2{b.x(), b.y()};
    const Vector2 c2{c.x(), c.y()};
    const Vector2 v0 = b2 - a2;
    const Vector2 v1 = c2 - a2;
    const Vector2 v2 = p - a2;
    const RealType denominator = v0.x() * v1.y() - v1.x() * v0.y();
    if (std::abs(denominator) <= CMP_EPSILON) {
        return std::nullopt;
    }

    const RealType inv_denominator = 1.0 / denominator;
    const RealType u = (v2.x() * v1.y() - v1.x() * v2.y()) * inv_denominator;
    const RealType v = (v0.x() * v2.y() - v2.x() * v0.y()) * inv_denominator;
    const RealType w = 1.0 - u - v;
    if (u < -CMP_EPSILON || v < -CMP_EPSILON || w < -CMP_EPSILON) {
        return std::nullopt;
    }

    return w * a.z() + u * b.z() + v * c.z();
}

std::optional<InternalTerrainRaycastHit> RaycastTriangle(const Vector3& origin,
                                                         const Vector3& direction,
                                                         RealType max_distance,
                                                         const Vector3& a,
                                                         const Vector3& b,
                                                         const Vector3& c) {
    const Vector3 edge1 = b - a;
    const Vector3 edge2 = c - a;
    const Vector3 pvec = direction.cross(edge2);
    const RealType determinant = edge1.dot(pvec);
    if (std::abs(determinant) <= CMP_EPSILON) {
        return std::nullopt;
    }

    const RealType inv_det = 1.0 / determinant;
    const Vector3 tvec = origin - a;
    const RealType u = tvec.dot(pvec) * inv_det;
    if (u < -CMP_EPSILON || u > 1.0 + CMP_EPSILON) {
        return std::nullopt;
    }

    const Vector3 qvec = tvec.cross(edge1);
    const RealType v = direction.dot(qvec) * inv_det;
    if (v < -CMP_EPSILON || u + v > 1.0 + CMP_EPSILON) {
        return std::nullopt;
    }

    const RealType distance = edge2.dot(qvec) * inv_det;
    if (distance < 0.0 || distance > max_distance) {
        return std::nullopt;
    }

    InternalTerrainRaycastHit hit;
    hit.hit = true;
    hit.point = origin + direction * distance;
    Vector3 normal = edge1.cross(edge2);
    if (normal.squaredNorm() <= CMP_EPSILON2) {
        normal = Vector3::UnitZ();
    } else {
        normal.normalize();
        if (normal.dot(direction) > 0.0) {
            normal = -normal;
        }
    }
    hit.normal = normal;
    hit.distance = distance;
    return hit;
}

std::optional<RealType> QueryTerrainMeshPatchHeight(const PhysicsTerrainMeshPatchSnapshot& mesh_patch,
                                                    const Vector3& world_position) {
    if (mesh_patch.has_xy_bounds && !PointInXYBounds(world_position, mesh_patch.xy_min, mesh_patch.xy_max)) {
        return std::nullopt;
    }
    if (mesh_patch.vertices.empty() || mesh_patch.indices.size() < 3) {
        return std::nullopt;
    }

    std::optional<RealType> height;
    for (std::size_t index = 0; index + 2 < mesh_patch.indices.size(); index += 3) {
        const std::uint32_t i0 = mesh_patch.indices[index + 0];
        const std::uint32_t i1 = mesh_patch.indices[index + 1];
        const std::uint32_t i2 = mesh_patch.indices[index + 2];
        if (i0 >= mesh_patch.vertices.size() ||
            i1 >= mesh_patch.vertices.size() ||
            i2 >= mesh_patch.vertices.size()) {
            continue;
        }

        const Vector3 a = mesh_patch.global_transform * mesh_patch.vertices[i0];
        const Vector3 b = mesh_patch.global_transform * mesh_patch.vertices[i1];
        const Vector3 c = mesh_patch.global_transform * mesh_patch.vertices[i2];
        const std::optional<RealType> candidate = IntersectVerticalRayWithTriangle(world_position, a, b, c);
        if (!candidate.has_value()) {
            continue;
        }
        height = height.has_value()
                         ? std::optional<RealType>(std::max(height.value(), candidate.value()))
                         : candidate;
    }

    return height;
}

std::optional<InternalTerrainRaycastHit> RaycastTerrainMeshPatch(const PhysicsTerrainMeshPatchSnapshot& mesh_patch,
                                                                 const std::string& terrain_name,
                                                                 const PhysicsRaycastQuery& query) {
    if (IsVerticalTerrainHeightRay(query) &&
        mesh_patch.has_xy_bounds &&
        !PointInXYBounds(query.origin, mesh_patch.xy_min, mesh_patch.xy_max)) {
        return std::nullopt;
    }
    if (mesh_patch.vertices.empty() || mesh_patch.indices.size() < 3) {
        return std::nullopt;
    }

    std::optional<InternalTerrainRaycastHit> best;
    for (std::size_t index = 0; index + 2 < mesh_patch.indices.size(); index += 3) {
        const std::uint32_t i0 = mesh_patch.indices[index + 0];
        const std::uint32_t i1 = mesh_patch.indices[index + 1];
        const std::uint32_t i2 = mesh_patch.indices[index + 2];
        if (i0 >= mesh_patch.vertices.size() ||
            i1 >= mesh_patch.vertices.size() ||
            i2 >= mesh_patch.vertices.size()) {
            continue;
        }

        const Vector3 a = mesh_patch.global_transform * mesh_patch.vertices[i0];
        const Vector3 b = mesh_patch.global_transform * mesh_patch.vertices[i1];
        const Vector3 c = mesh_patch.global_transform * mesh_patch.vertices[i2];
        std::optional<InternalTerrainRaycastHit> candidate =
                RaycastTriangle(query.origin, query.direction, query.max_distance, a, b, c);
        if (!candidate.has_value()) {
            continue;
        }
        candidate->terrain_name = &terrain_name;
        if (!best.has_value() || candidate->distance < best->distance) {
            best = candidate;
        }
    }

    return best;
}

RealType QueryTerrainHeight(const PhysicsSceneSnapshot& snapshot, const Vector3& world_position) {
    RealType height = -std::numeric_limits<RealType>::infinity();
    for (const PhysicsTerrainSnapshot& terrain : snapshot.terrains) {
        for (const PhysicsTerrainBoxSnapshot& box : terrain.boxes) {
            if (const std::optional<RealType> candidate = QueryTerrainBoxHeight(box, world_position)) {
                height = std::max(height, candidate.value());
            }
        }
        for (const PhysicsTerrainHeightFieldSnapshot& heightfield : terrain.heightfields) {
            if (const std::optional<RealType> candidate = QueryTerrainHeightFieldHeight(heightfield, world_position)) {
                height = std::max(height, candidate.value());
            }
        }
        for (const PhysicsTerrainMeshPatchSnapshot& mesh_patch : terrain.mesh_patches) {
            if (const std::optional<RealType> candidate = QueryTerrainMeshPatchHeight(mesh_patch, world_position)) {
                height = std::max(height, candidate.value());
            }
        }
    }

    return std::isfinite(height) ? height : 0.0;
}

} // namespace

const PhysicsWorldSettings& PhysicsWorld::GetSettings() const {
    return settings_;
}

void PhysicsWorld::SetSettings(const PhysicsWorldSettings& settings) {
    settings_ = settings;
}

bool PhysicsWorld::Build(PhysicsSceneSnapshot scene_snapshot) {
    scene_snapshot_ = std::move(scene_snapshot);
    ResetSceneStateFromSnapshot();
    last_error_.clear();
    return true;
}

bool PhysicsWorld::RestoreCompatibleState(const PhysicsSceneState& previous_state) {
    for (PhysicsRobotState& robot_state : scene_state_.robots) {
        const PhysicsRobotState* previous_robot_state = FindRobotState(previous_state, robot_state.name);
        if (previous_robot_state == nullptr) {
            continue;
        }

        for (PhysicsLinkState& link_state : robot_state.links) {
            const PhysicsLinkState* previous_link_state = FindLinkState(*previous_robot_state, link_state.link_name);
            if (previous_link_state == nullptr || previous_link_state->role != link_state.role) {
                continue;
            }

            link_state.global_transform = previous_link_state->global_transform;
            link_state.linear_velocity = previous_link_state->linear_velocity;
            link_state.angular_velocity = previous_link_state->angular_velocity;
        }

        for (PhysicsJointState& joint_state : robot_state.joints) {
            const PhysicsJointState* previous_joint_state =
                    FindPreviousJointState(*previous_robot_state, joint_state.joint_name);
            if (previous_joint_state == nullptr || previous_joint_state->joint_type != joint_state.joint_type) {
                continue;
            }

            joint_state.position = previous_joint_state->position;
            joint_state.velocity = previous_joint_state->velocity;
            joint_state.effort = previous_joint_state->effort;
            joint_state.control_mode = previous_joint_state->control_mode;
            joint_state.target_position = previous_joint_state->target_position;
            joint_state.target_velocity = previous_joint_state->target_velocity;
            joint_state.target_effort = previous_joint_state->target_effort;
        }

        for (PhysicsSensorState& sensor_state : robot_state.sensors) {
            const PhysicsSensorState* previous_sensor_state =
                    FindPreviousSensorState(*previous_robot_state,
                                            sensor_state.sensor_name,
                                            sensor_state.link_name);
            if (previous_sensor_state == nullptr || previous_sensor_state->type != sensor_state.type) {
                continue;
            }

            sensor_state.values = previous_sensor_state->values;
            sensor_state.timestamp = previous_sensor_state->timestamp;
        }
    }

    UpdateSensorGlobalTransformsAndRaycastSensors(scene_state_, 0.0);
    last_error_.clear();
    return true;
}

void PhysicsWorld::Reset() {
    ResetSceneStateFromSnapshot();
    ClearExternalForces();
}

void PhysicsWorld::Step(RealType delta_time) {
    GOB_UNUSED(delta_time);
    UpdateSensorGlobalTransformsAndRaycastSensors(scene_state_, 0.0);
}

bool PhysicsWorld::ConfigureEnvironmentBatch(std::size_t environment_count) {
    if (environment_count == 0) {
        SetLastError("Physics environment batch size must be greater than zero.");
        return false;
    }

    if (environment_count != 1) {
        SetLastError("This physics backend does not support batched environments.");
        return false;
    }

    last_error_.clear();
    return true;
}

std::size_t PhysicsWorld::GetEnvironmentCount() const {
    return 1;
}

const PhysicsSceneState* PhysicsWorld::GetEnvironmentState(std::size_t environment_index) const {
    if (environment_index != 0) {
        return nullptr;
    }

    return &scene_state_;
}

bool PhysicsWorld::ResetEnvironment(std::size_t environment_index) {
    if (environment_index != 0) {
        SetLastError(fmt::format("Environment index {} is out of range.", environment_index));
        return false;
    }

    Reset();
    last_error_.clear();
    return true;
}

bool PhysicsWorld::StepEnvironment(std::size_t environment_index, RealType delta_time) {
    if (environment_index != 0) {
        SetLastError(fmt::format("Environment index {} is out of range.", environment_index));
        return false;
    }

    Step(delta_time);
    last_error_.clear();
    return true;
}

bool PhysicsWorld::StepEnvironmentBatch(RealType delta_time, std::uint64_t ticks, std::size_t worker_count) {
    GOB_UNUSED(worker_count);
    const std::size_t environment_count = GetEnvironmentCount();
    for (std::uint64_t tick = 0; tick < ticks; ++tick) {
        for (std::size_t environment_index = 0; environment_index < environment_count; ++environment_index) {
            if (!StepEnvironment(environment_index, delta_time)) {
                return false;
            }
        }
    }
    last_error_.clear();
    return true;
}

bool PhysicsWorld::StepRobotBatch(const PhysicsRobotBatchStepRequest& request,
                                  PhysicsRobotBatchStepResult& result) {
    GOB_UNUSED(request);
    GOB_UNUSED(result);
    SetLastError("The active physics backend does not provide typed batch robot state extraction.");
    return false;
}

std::size_t PhysicsWorld::ResolveEnvironmentBatchWorkerCount(std::size_t worker_count) const {
    const std::size_t environment_count = GetEnvironmentCount();
    if (environment_count == 0) {
        return 0;
    }
    if (worker_count == 0) {
        worker_count = static_cast<std::size_t>(std::thread::hardware_concurrency());
    }
    if (worker_count == 0) {
        worker_count = 1;
    }
    return std::min(environment_count, worker_count);
}

bool PhysicsWorld::ResetJointState(const std::string& robot_name,
                                   const std::string& joint_name,
                                   RealType position,
                                   RealType velocity) {
    return ResetJointStateIn(scene_state_, robot_name, joint_name, position, velocity);
}

bool PhysicsWorld::ResetEnvironmentJointState(std::size_t environment_index,
                                              const std::string& robot_name,
                                              const std::string& joint_name,
                                              RealType position,
                                              RealType velocity) {
    if (environment_index != 0) {
        SetLastError(fmt::format("Environment index {} is out of range.", environment_index));
        return false;
    }

    return ResetJointState(robot_name, joint_name, position, velocity);
}

bool PhysicsWorld::ResetLinkState(const std::string& robot_name,
                                  const std::string& link_name,
                                  const Vector3& position,
                                  const Quaternion& orientation,
                                  const Vector3& linear_velocity,
                                  const Vector3& angular_velocity) {
    const bool reset = ResetLinkStateIn(scene_state_,
                                        robot_name,
                                        link_name,
                                        position,
                                        orientation,
                                        linear_velocity,
                                        angular_velocity);
    if (reset) {
        UpdateSensorGlobalTransformsAndRaycastSensors(scene_state_, 0.0);
    }
    return reset;
}

bool PhysicsWorld::ResetEnvironmentLinkState(std::size_t environment_index,
                                             const std::string& robot_name,
                                             const std::string& link_name,
                                             const Vector3& position,
                                             const Quaternion& orientation,
                                             const Vector3& linear_velocity,
                                             const Vector3& angular_velocity) {
    if (environment_index != 0) {
        SetLastError(fmt::format("Environment index {} is out of range.", environment_index));
        return false;
    }

    return ResetLinkState(robot_name, link_name, position, orientation, linear_velocity, angular_velocity);
}

bool PhysicsWorld::WriteEnvironmentLinkVelocity(std::size_t environment_index,
                                                const std::string& robot_name,
                                                const std::string& link_name,
                                                const Vector3& linear_velocity,
                                                const Vector3& angular_velocity) {
    const PhysicsSceneState* state = GetEnvironmentState(environment_index);
    if (state == nullptr) {
        SetLastError(fmt::format("Environment index {} is out of range.", environment_index));
        return false;
    }

    for (const PhysicsRobotState& robot : state->robots) {
        if (robot.name != robot_name) {
            continue;
        }
        for (const PhysicsLinkState& link : robot.links) {
            if (link.link_name != link_name) {
                continue;
            }
            return ResetEnvironmentLinkState(environment_index,
                                             robot_name,
                                             link_name,
                                             link.global_transform.translation(),
                                             link.global_transform.GetQuaternion(),
                                             linear_velocity,
                                             angular_velocity);
        }
    }

    SetLastError(fmt::format("Cannot set velocity for missing link '{}::{}'.", robot_name, link_name));
    return false;
}

bool PhysicsWorld::ResetEnvironmentRobotStates(const std::vector<PhysicsEnvironmentRobotResetState>& reset_states) {
    for (const PhysicsEnvironmentRobotResetState& reset_state : reset_states) {
        if (reset_state.joint_positions.size() != reset_state.joint_names.size() ||
            reset_state.joint_velocities.size() != reset_state.joint_names.size()) {
            SetLastError(fmt::format("Expected {} joint reset value(s) for robot '{}', got {} position(s) and {} velocity value(s).",
                                     reset_state.joint_names.size(),
                                     reset_state.robot_name,
                                     reset_state.joint_positions.size(),
                                     reset_state.joint_velocities.size()));
            return false;
        }
        if (!reset_state.joint_position_targets.empty() &&
            reset_state.joint_position_targets.size() != reset_state.joint_names.size()) {
            SetLastError(fmt::format("Expected {} joint target value(s) for robot '{}', got {}.",
                                     reset_state.joint_names.size(),
                                     reset_state.robot_name,
                                     reset_state.joint_position_targets.size()));
            return false;
        }
        if (!ResetEnvironment(reset_state.environment_index)) {
            return false;
        }
        if (!ResetEnvironmentLinkState(reset_state.environment_index,
                                       reset_state.robot_name,
                                       reset_state.base_link_name,
                                       reset_state.base_position,
                                       reset_state.base_orientation,
                                       reset_state.base_linear_velocity,
                                       reset_state.base_angular_velocity)) {
            return false;
        }
        for (std::size_t joint_index = 0; joint_index < reset_state.joint_names.size(); ++joint_index) {
            if (!ResetEnvironmentJointState(reset_state.environment_index,
                                            reset_state.robot_name,
                                            reset_state.joint_names[joint_index],
                                            reset_state.joint_positions[joint_index],
                                            reset_state.joint_velocities[joint_index])) {
                return false;
            }
        }
        for (std::size_t joint_index = 0; joint_index < reset_state.joint_position_targets.size(); ++joint_index) {
            if (!SetEnvironmentJointControl(reset_state.environment_index,
                                            reset_state.robot_name,
                                            reset_state.joint_names[joint_index],
                                            PhysicsJointControlMode::Position,
                                            reset_state.joint_position_targets[joint_index])) {
                return false;
            }
        }
    }

    last_error_.clear();
    return true;
}

bool PhysicsWorld::SetJointControl(const std::string& robot_name,
                                   const std::string& joint_name,
                                   PhysicsJointControlMode control_mode,
                                   RealType target) {
    return SetJointControlIn(scene_state_, robot_name, joint_name, control_mode, target);
}

bool PhysicsWorld::SetEnvironmentJointControl(std::size_t environment_index,
                                              const std::string& robot_name,
                                              const std::string& joint_name,
                                              PhysicsJointControlMode control_mode,
                                              RealType target) {
    if (environment_index != 0) {
        SetLastError(fmt::format("Environment index {} is out of range.", environment_index));
        return false;
    }

    return SetJointControl(robot_name, joint_name, control_mode, target);
}

bool PhysicsWorld::SetEnvironmentJointControls(const std::string& robot_name,
                                               const std::vector<std::string>& joint_names,
                                               PhysicsJointControlMode control_mode,
                                               const std::vector<RealType>& targets,
                                               std::size_t environment_count) {
    if (environment_count != GetEnvironmentCount()) {
        SetLastError(fmt::format("Expected targets for {} environment(s), got {}.",
                                 GetEnvironmentCount(),
                                 environment_count));
        return false;
    }
    if (targets.size() != environment_count * joint_names.size()) {
        SetLastError(fmt::format("Expected {} batched joint target value(s), got {}.",
                                 environment_count * joint_names.size(),
                                 targets.size()));
        return false;
    }
    if (joint_names.empty()) {
        last_error_.clear();
        return true;
    }

    for (std::size_t environment_index = 0; environment_index < environment_count; ++environment_index) {
        const std::size_t row_offset = environment_index * joint_names.size();
        for (std::size_t joint_index = 0; joint_index < joint_names.size(); ++joint_index) {
            if (!SetEnvironmentJointControl(environment_index,
                                            robot_name,
                                            joint_names[joint_index],
                                            control_mode,
                                            targets[row_offset + joint_index])) {
                return false;
            }
        }
    }

    last_error_.clear();
    return true;
}

bool PhysicsWorld::SetLinkExternalForce(const std::string& robot_name,
                                        const std::string& link_name,
                                        const Vector3& point,
                                        const Vector3& force) {
    if (FindMutableLinkState(robot_name, link_name) == nullptr) {
        SetLastError(fmt::format("Cannot apply external force to missing link '{}::{}'.",
                                 robot_name,
                                 link_name));
        return false;
    }

    for (PhysicsExternalForce& external_force : external_forces_) {
        if (external_force.robot_name == robot_name && external_force.link_name == link_name) {
            external_force.point = point;
            external_force.local_point = Vector3::Zero();
            external_force.target_point = point;
            external_force.force = force;
            external_force.use_spring = false;
            last_error_.clear();
            return true;
        }
    }

    PhysicsExternalForce external_force;
    external_force.robot_name = robot_name;
    external_force.link_name = link_name;
    external_force.point = point;
    external_force.local_point = Vector3::Zero();
    external_force.target_point = point;
    external_force.force = force;
    external_force.use_spring = false;
    external_forces_.push_back(external_force);
    last_error_.clear();
    return true;
}

bool PhysicsWorld::SetLinkSpringForce(const std::string& robot_name,
                                      const std::string& link_name,
                                      const Vector3& local_point,
                                      const Vector3& target_point,
                                      const Vector3& force_hint) {
    PhysicsLinkState* link_state = FindMutableLinkState(robot_name, link_name);
    if (link_state == nullptr) {
        SetLastError(fmt::format("Cannot apply external force to missing link '{}::{}'.",
                                 robot_name,
                                 link_name));
        return false;
    }

    const Vector3 point = link_state->global_transform * local_point;
    for (PhysicsExternalForce& external_force : external_forces_) {
        if (external_force.robot_name == robot_name && external_force.link_name == link_name) {
            external_force.point = point;
            external_force.local_point = local_point;
            external_force.target_point = target_point;
            external_force.force = force_hint;
            external_force.use_spring = true;
            last_error_.clear();
            return true;
        }
    }

    PhysicsExternalForce external_force;
    external_force.robot_name = robot_name;
    external_force.link_name = link_name;
    external_force.point = point;
    external_force.local_point = local_point;
    external_force.target_point = target_point;
    external_force.force = force_hint;
    external_force.use_spring = true;
    external_forces_.push_back(external_force);
    last_error_.clear();
    return true;
}

void PhysicsWorld::ClearExternalForces() {
    external_forces_.clear();
}

const PhysicsSceneSnapshot& PhysicsWorld::GetSceneSnapshot() const {
    return scene_snapshot_;
}

const PhysicsSceneArtifact* PhysicsWorld::GetSceneArtifact() const {
    return nullptr;
}

const PhysicsSceneState& PhysicsWorld::GetSceneState() const {
    return scene_state_;
}

PhysicsRaycastHit PhysicsWorld::RaycastTerrainFallback(const PhysicsRaycastQuery& query,
                                                       bool include_terrain_name) const {
    PhysicsRaycastHit result;
    result.origin = query.origin;

    Vector3 direction = query.direction;
    if (direction.squaredNorm() <= CMP_EPSILON2 || query.max_distance <= 0.0) {
        result.point = query.origin;
        result.distance = 0.0;
        return result;
    }
    direction.normalize();

    PhysicsRaycastQuery normalized_query;
    normalized_query.origin = query.origin;
    normalized_query.direction = direction;
    normalized_query.max_distance = query.max_distance;

    result.point = query.origin + direction * query.max_distance;
    result.distance = query.max_distance;

    std::optional<InternalTerrainRaycastHit> best;
    for (const PhysicsTerrainSnapshot& terrain : scene_snapshot_.terrains) {
        for (const PhysicsTerrainBoxSnapshot& box : terrain.boxes) {
            std::optional<InternalTerrainRaycastHit> candidate =
                    RaycastTerrainBox(box, terrain.name, normalized_query);
            if (candidate.has_value() && (!best.has_value() || candidate->distance < best->distance)) {
                best = candidate;
            }
        }
        for (const PhysicsTerrainHeightFieldSnapshot& heightfield : terrain.heightfields) {
            std::optional<InternalTerrainRaycastHit> candidate =
                    RaycastTerrainHeightField(heightfield, terrain.name, normalized_query);
            if (candidate.has_value() && (!best.has_value() || candidate->distance < best->distance)) {
                best = candidate;
            }
        }
        for (const PhysicsTerrainMeshPatchSnapshot& mesh_patch : terrain.mesh_patches) {
            std::optional<InternalTerrainRaycastHit> candidate =
                    RaycastTerrainMeshPatch(mesh_patch, terrain.name, normalized_query);
            if (candidate.has_value() && (!best.has_value() || candidate->distance < best->distance)) {
                best = candidate;
            }
        }
    }

    if (best.has_value()) {
        result.hit = true;
        result.point = best->point;
        result.normal = best->normal;
        result.distance = best->distance;
        if (include_terrain_name && best->terrain_name != nullptr) {
            result.terrain_name = *best->terrain_name;
        }
    }
    return result;
}

PhysicsRaycastHit PhysicsWorld::RaycastTerrain(const PhysicsRaycastQuery& query) const {
    return RaycastTerrainFallback(query);
}

PhysicsRaycastHit PhysicsWorld::RaycastEnvironmentTerrain(const PhysicsRaycastQuery& query,
                                                          std::size_t environment_index) const {
    return RaycastTerrainForSensor(query, environment_index);
}

PhysicsRaycastHit PhysicsWorld::RaycastTerrainForSensor(const PhysicsRaycastQuery& query,
                                                        std::size_t environment_index) const {
    GOB_UNUSED(environment_index);
    return RaycastTerrainFallback(query, false);
}

void PhysicsWorld::UpdateRaycastSensorState(PhysicsSensorState& sensor_state,
                                            const PhysicsSensorSnapshot& sensor_snapshot,
                                            const Affine3& parent_transform,
                                            RealType timestamp,
                                            std::size_t environment_index) {
    sensor_state.global_transform = parent_transform * sensor_snapshot.local_transform;

    if (!IsRaycastSensorType(sensor_state.type) || !sensor_state.enabled) {
        return;
    }
    if (sensor_snapshot.sensor_period > 0.0 &&
        timestamp > 0.0 &&
        sensor_state.timestamp > 0.0) {
        const auto current_bucket = static_cast<std::int64_t>(
                std::floor((timestamp + CMP_EPSILON) / sensor_snapshot.sensor_period));
        const auto previous_bucket = static_cast<std::int64_t>(
                std::floor((sensor_state.timestamp + CMP_EPSILON) / sensor_snapshot.sensor_period));
        if (current_bucket <= previous_bucket) {
            return;
        }
    }

    const bool reduce_values = IsTerrainHeightSensorType(sensor_state.type) &&
                               sensor_snapshot.reduction_mode != RayReductionMode::None;
    const std::size_t value_count = reduce_values ? 1 : sensor_snapshot.sample_offsets.size();
    if (sensor_state.values.size() != value_count) {
        sensor_state.values.assign(value_count, 0.0);
    }
    if (sensor_state.hits.size() != sensor_snapshot.sample_offsets.size()) {
        sensor_state.hits.assign(sensor_snapshot.sample_offsets.size(), {});
    }
    const Matrix3 alignment_matrix =
            GetPhysicsRayAlignmentMatrix(sensor_state.global_transform, sensor_snapshot.ray_alignment);
    const Vector3 ray_direction =
            ResolvePhysicsRayDirection(sensor_snapshot, alignment_matrix, sensor_state.global_transform);
    const bool terrain_height_sensor = sensor_state.type == PhysicsSensorType::TerrainHeight;
    bool any_hit = false;
    std::vector<RealType> per_ray_values;
    if (reduce_values) {
        per_ray_values.reserve(sensor_snapshot.sample_offsets.size());
    }
    for (std::size_t sample_index = 0; sample_index < sensor_snapshot.sample_offsets.size(); ++sample_index) {
        const Vector3 origin = sensor_state.global_transform.translation() +
                               alignment_matrix * sensor_snapshot.sample_offsets[sample_index];
        const PhysicsRaycastHit hit = RaycastTerrainForSensor({
                origin,
                ray_direction,
                sensor_snapshot.max_distance
        }, environment_index);
        any_hit = any_hit || hit.hit;
        RealType value = hit.distance;
        if (terrain_height_sensor) {
            value = hit.hit
                            ? (hit.normal.z() < 0.0 ? 0.0 : origin.z() - hit.point.z())
                            : sensor_snapshot.max_distance;
        } else if (sensor_state.type == PhysicsSensorType::HeightScanner) {
            value = hit.hit ? origin.z() - hit.point.z() : sensor_snapshot.max_distance;
        }
        if (!reduce_values) {
            sensor_state.values[sample_index] = value;
        } else {
            per_ray_values.push_back(value);
        }
        PhysicsSensorRaycastHit sensor_hit;
        sensor_hit.hit = hit.hit;
        sensor_hit.origin = hit.origin;
        sensor_hit.point = hit.point;
        sensor_hit.normal = hit.normal;
        sensor_hit.distance = hit.distance;
        sensor_hit.terrain_name = hit.terrain_name;
        sensor_state.hits[sample_index] = std::move(sensor_hit);
    }
    if (terrain_height_sensor && !any_hit) {
        const RealType fallback = std::clamp(
                sensor_state.global_transform.translation().z(),
                static_cast<RealType>(0.0),
                sensor_snapshot.max_distance);
        if (reduce_values) {
            std::fill(per_ray_values.begin(), per_ray_values.end(), fallback);
        } else {
            std::fill(sensor_state.values.begin(), sensor_state.values.end(), fallback);
        }
    }
    if (reduce_values) {
        sensor_state.values[0] = ReducePhysicsRayValues(per_ray_values, sensor_snapshot.reduction_mode);
    }
    sensor_state.timestamp = timestamp;
}

void PhysicsWorld::UpdateSensorGlobalTransformsAndRaycastSensors(PhysicsSceneState& scene_state,
                                                                 RealType timestamp,
                                                                 std::size_t environment_index) {
    for (std::size_t robot_index = 0; robot_index < scene_state.robots.size(); ++robot_index) {
        if (robot_index >= scene_snapshot_.robots.size()) {
            continue;
        }

        PhysicsRobotState& robot_state = scene_state.robots[robot_index];
        const PhysicsRobotSnapshot& robot_snapshot = scene_snapshot_.robots[robot_index];
        for (std::size_t sensor_index = 0; sensor_index < robot_state.sensors.size(); ++sensor_index) {
            if (sensor_index >= robot_snapshot.sensors.size()) {
                continue;
            }

            PhysicsSensorState& sensor_state = robot_state.sensors[sensor_index];
            const PhysicsSensorSnapshot& sensor_snapshot = robot_snapshot.sensors[sensor_index];
            const PhysicsLinkState* link_state = FindLinkState(robot_state, sensor_snapshot.link_name);
            const Affine3 link_transform = link_state != nullptr
                                                   ? link_state->global_transform
                                                   : Affine3::Identity();
            UpdateRaycastSensorState(sensor_state, sensor_snapshot, link_transform, timestamp, environment_index);
        }
    }
    for (std::size_t sensor_index = 0; sensor_index < scene_state.loose_sensors.size(); ++sensor_index) {
        if (sensor_index >= scene_snapshot_.loose_sensors.size()) {
            continue;
        }
        UpdateRaycastSensorState(scene_state.loose_sensors[sensor_index],
                                 scene_snapshot_.loose_sensors[sensor_index],
                                 Affine3::Identity(),
                                 timestamp,
                                 environment_index);
    }
}

PhysicsJointState* PhysicsWorld::FindJointState(const std::string& robot_name,
                                                const std::string& joint_name) {
    return FindJointStateIn(scene_state_, robot_name, joint_name);
}

PhysicsJointState* PhysicsWorld::FindJointStateIn(PhysicsSceneState& scene_state,
                                                  const std::string& robot_name,
                                                  const std::string& joint_name) {
    for (PhysicsRobotState& robot_state : scene_state.robots) {
        if (robot_state.name != robot_name) {
            continue;
        }

        for (PhysicsJointState& joint_state : robot_state.joints) {
            if (joint_state.joint_name == joint_name) {
                return &joint_state;
            }
        }
    }

    return nullptr;
}

PhysicsLinkState* PhysicsWorld::FindMutableLinkState(const std::string& robot_name,
                                                     const std::string& link_name) {
    return FindMutableLinkStateIn(scene_state_, robot_name, link_name);
}

PhysicsLinkState* PhysicsWorld::FindMutableLinkStateIn(PhysicsSceneState& scene_state,
                                                       const std::string& robot_name,
                                                       const std::string& link_name) {
    for (PhysicsRobotState& robot_state : scene_state.robots) {
        if (robot_state.name != robot_name) {
            continue;
        }

        for (PhysicsLinkState& link_state : robot_state.links) {
            if (link_state.link_name == link_name) {
                return &link_state;
            }
        }
    }

    return nullptr;
}

bool PhysicsWorld::ResetJointStateIn(PhysicsSceneState& scene_state,
                                     const std::string& robot_name,
                                     const std::string& joint_name,
                                     RealType position,
                                     RealType velocity) {
    PhysicsJointState* joint_state = FindJointStateIn(scene_state, robot_name, joint_name);
    if (!joint_state) {
        SetLastError(fmt::format("Cannot reset state for missing joint '{}::{}'.", robot_name, joint_name));
        return false;
    }

    joint_state->position = position;
    joint_state->velocity = velocity;
    joint_state->effort = 0.0;
    joint_state->control_mode = PhysicsJointControlMode::Passive;
    joint_state->target_position = position;
    joint_state->target_velocity = 0.0;
    joint_state->target_effort = 0.0;
    last_error_.clear();
    return true;
}

bool PhysicsWorld::ResetLinkStateIn(PhysicsSceneState& scene_state,
                                    const std::string& robot_name,
                                    const std::string& link_name,
                                    const Vector3& position,
                                    const Quaternion& orientation,
                                    const Vector3& linear_velocity,
                                    const Vector3& angular_velocity) {
    PhysicsLinkState* link_state = FindMutableLinkStateIn(scene_state, robot_name, link_name);
    if (!link_state) {
        SetLastError(fmt::format("Cannot reset state for missing link '{}::{}'.", robot_name, link_name));
        return false;
    }

    Quaternion normalized_orientation = orientation;
    if (normalized_orientation.norm() <= CMP_EPSILON) {
        normalized_orientation = Quaternion::Identity();
    } else {
        normalized_orientation.normalize();
    }

    link_state->global_transform.translation() = position;
    link_state->global_transform.linear() = normalized_orientation.toRotationMatrix();
    link_state->linear_velocity = linear_velocity;
    link_state->angular_velocity = angular_velocity;
    last_error_.clear();
    return true;
}

bool PhysicsWorld::SetJointControlIn(PhysicsSceneState& scene_state,
                                     const std::string& robot_name,
                                     const std::string& joint_name,
                                     PhysicsJointControlMode control_mode,
                                     RealType target) {
    PhysicsJointState* joint_state = FindJointStateIn(scene_state, robot_name, joint_name);
    if (!joint_state) {
        SetLastError(fmt::format("Cannot set control for missing joint '{}::{}'.", robot_name, joint_name));
        return false;
    }

    joint_state->control_mode = control_mode;
    switch (control_mode) {
        case PhysicsJointControlMode::Passive:
            joint_state->target_position = joint_state->position;
            joint_state->target_velocity = 0.0;
            joint_state->target_effort = 0.0;
            break;
        case PhysicsJointControlMode::Position:
            joint_state->target_position = target;
            break;
        case PhysicsJointControlMode::Velocity:
            joint_state->target_velocity = target;
            break;
        case PhysicsJointControlMode::Effort:
            joint_state->target_effort = target;
            break;
    }

    last_error_.clear();
    return true;
}

PhysicsSceneState PhysicsWorld::MakeSceneStateFromSnapshot() const {
    PhysicsSceneState scene_state;
    scene_state.robots.reserve(scene_snapshot_.robots.size());

    for (const PhysicsRobotSnapshot& robot_snapshot : scene_snapshot_.robots) {
        PhysicsRobotState robot_state;
        robot_state.name = robot_snapshot.name;
        robot_state.links.reserve(robot_snapshot.links.size());
        robot_state.joints.reserve(robot_snapshot.joints.size());

        for (const PhysicsLinkSnapshot& link_snapshot : robot_snapshot.links) {
            PhysicsLinkState link_state;
            link_state.robot_name = robot_snapshot.name;
            link_state.link_name = link_snapshot.name;
            link_state.role = link_snapshot.role;
            link_state.global_transform = link_snapshot.global_transform;
            robot_state.links.emplace_back(std::move(link_state));
            ++scene_state.total_link_count;
        }

        for (const PhysicsJointSnapshot& joint_snapshot : robot_snapshot.joints) {
            PhysicsJointState joint_state;
            joint_state.robot_name = robot_snapshot.name;
            joint_state.joint_name = joint_snapshot.name;
            joint_state.joint_type = joint_snapshot.joint_type;
            joint_state.position = joint_snapshot.joint_position;
            joint_state.target_position = joint_snapshot.joint_position;
            joint_state.control_mode = JointDriveModeToControlMode(joint_snapshot.drive_mode);
            robot_state.joints.emplace_back(std::move(joint_state));
            ++scene_state.total_joint_count;
        }

        for (const PhysicsSensorSnapshot& sensor_snapshot : robot_snapshot.sensors) {
            robot_state.sensors.emplace_back(MakeSensorStateFromSnapshot(sensor_snapshot, robot_snapshot.name));
            ++scene_state.total_sensor_count;
        }

        scene_state.robots.emplace_back(std::move(robot_state));
    }

    for (const PhysicsSensorSnapshot& sensor_snapshot : scene_snapshot_.loose_sensors) {
        scene_state.loose_sensors.emplace_back(MakeSensorStateFromSnapshot(sensor_snapshot));
        ++scene_state.total_sensor_count;
    }

    return scene_state;
}

void PhysicsWorld::ResetSceneStateFromSnapshot() {
    scene_state_ = MakeSceneStateFromSnapshot();
    UpdateSensorGlobalTransformsAndRaycastSensors(scene_state_, 0.0);
}

void PhysicsWorld::SetLastError(std::string error) {
    last_error_ = std::move(error);
}

} // namespace gobot

GOBOT_REGISTRATION {

    QuickEnumeration_<PhysicsBackendType>("PhysicsBackendType");
    QuickEnumeration_<PhysicsShapeType>("PhysicsShapeType");
    QuickEnumeration_<PhysicsSensorType>("PhysicsSensorType");
    QuickEnumeration_<PhysicsJointControlMode>("PhysicsJointControlMode");
    QuickEnumeration_<PhysicsSolverType>("PhysicsSolverType");
    QuickEnumeration_<PhysicsIntegratorType>("PhysicsIntegratorType");
    QuickEnumeration_<PhysicsFrictionConeType>("PhysicsFrictionConeType");
    QuickEnumeration_<PhysicsJacobianType>("PhysicsJacobianType");

    Class_<PhysicsBackendInfo>("PhysicsBackendInfo")
            .constructor()
            .property("type", &PhysicsBackendInfo::type)
            .property("name", &PhysicsBackendInfo::name)
            .property("available", &PhysicsBackendInfo::available)
            .property("cpu", &PhysicsBackendInfo::cpu)
            .property("gpu", &PhysicsBackendInfo::gpu)
            .property("robotics_focused", &PhysicsBackendInfo::robotics_focused)
            .property("status", &PhysicsBackendInfo::status);

    Class_<MuJoCoSolverSettings>("MuJoCoSolverSettings")
            .constructor()
            .property("solver", &MuJoCoSolverSettings::solver)
            .property("integrator", &MuJoCoSolverSettings::integrator)
            .property("cone", &MuJoCoSolverSettings::cone)
            .property("jacobian", &MuJoCoSolverSettings::jacobian)
            .property("iterations", &MuJoCoSolverSettings::iterations)
            .property("line_search_iterations", &MuJoCoSolverSettings::line_search_iterations)
            .property("no_slip_iterations", &MuJoCoSolverSettings::no_slip_iterations)
            .property("convex_collision_iterations", &MuJoCoSolverSettings::convex_collision_iterations)
            .property("tolerance", &MuJoCoSolverSettings::tolerance)
            .property("line_search_tolerance", &MuJoCoSolverSettings::line_search_tolerance)
            .property("no_slip_tolerance", &MuJoCoSolverSettings::no_slip_tolerance)
            .property("convex_collision_tolerance", &MuJoCoSolverSettings::convex_collision_tolerance)
            .property("impedance_ratio", &MuJoCoSolverSettings::impedance_ratio);

    Class_<PhysicsWorldSettings>("PhysicsWorldSettings")
            .constructor()
            .property("gravity", &PhysicsWorldSettings::gravity)
            .property("fixed_time_step", &PhysicsWorldSettings::fixed_time_step)
            .property("default_joint_gains", &PhysicsWorldSettings::default_joint_gains)
            .property("mujoco_solver", &PhysicsWorldSettings::mujoco_solver)
            .property("debug_draw_contacts", &PhysicsWorldSettings::debug_draw_contacts)
            .property("debug_draw_contact_forces", &PhysicsWorldSettings::debug_draw_contact_forces)
            .property("debug_contact_force_scale", &PhysicsWorldSettings::debug_contact_force_scale)
            .property("debug_contact_force_max_length", &PhysicsWorldSettings::debug_contact_force_max_length);

    Class_<PhysicsWorld>("PhysicsWorld")
            .method("is_available", &PhysicsWorld::IsAvailable)
            .method("get_last_error", &PhysicsWorld::GetLastError)
            .method("reset", &PhysicsWorld::Reset)
            .method("step", &PhysicsWorld::Step)
            .method("reset_link_state", &PhysicsWorld::ResetLinkState)
            .method("set_joint_control", &PhysicsWorld::SetJointControl);

    gobot::Type::register_wrapper_converter_for_base_classes<Ref<PhysicsWorld>, Ref<RefCounted>>();

};
