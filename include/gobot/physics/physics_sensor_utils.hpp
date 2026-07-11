/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <vector>

#include "gobot/physics/physics_types.hpp"

namespace gobot {

inline Matrix3 GetPhysicsRayAlignmentMatrix(const Affine3& transform, RayAlignmentMode alignment) {
    switch (alignment) {
        case RayAlignmentMode::World:
            return Matrix3::Identity();
        case RayAlignmentMode::Base:
            return transform.linear();
        case RayAlignmentMode::Yaw: {
            Vector3 x_axis = transform.linear() * Vector3::UnitX();
            x_axis.z() = 0.0;
            if (x_axis.head<2>().norm() < 0.1) {
                const Vector3 y_axis = transform.linear() * Vector3::UnitY();
                x_axis = Vector3(y_axis.y(), -y_axis.x(), 0.0);
            }
            const RealType norm = std::max(x_axis.head<2>().norm(), static_cast<RealType>(1.0e-6));
            x_axis /= norm;
            Matrix3 yaw = Matrix3::Zero();
            yaw(0, 0) = x_axis.x();
            yaw(1, 0) = x_axis.y();
            yaw(0, 1) = -x_axis.y();
            yaw(1, 1) = x_axis.x();
            yaw(2, 2) = 1.0;
            return yaw;
        }
    }
    return Matrix3::Identity();
}

inline Vector3 ResolvePhysicsRayDirection(const PhysicsSensorSnapshot& sensor,
                                          const Matrix3& alignment,
                                          const Affine3& sensor_transform) {
    Vector3 direction = sensor.ray_direction;
    if (sensor.ray_alignment == RayAlignmentMode::Base) {
        direction = sensor_transform.linear() * direction;
    } else if (sensor.ray_alignment == RayAlignmentMode::Yaw) {
        direction = alignment * direction;
    } else if (!sensor.ray_direction_world_space) {
        direction = sensor_transform.linear() * direction;
    }

    if (direction.squaredNorm() <= CMP_EPSILON2) {
        return Vector3{0.0, 0.0, -1.0};
    }
    return direction.normalized();
}

inline RealType ReducePhysicsRayValues(const std::vector<RealType>& values,
                                       RayReductionMode reduction_mode) {
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
            return values.front();
    }
    return values.front();
}

} // namespace gobot
