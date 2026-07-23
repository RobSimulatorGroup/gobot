/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * This file is created by Qiqi Wu, 23-2-26
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "gobot/core/math/matrix.hpp"

#include <algorithm>
#include <limits>
#include <vector>

namespace gobot {

struct AABB {
    Vector3 position = Vector3::Zero();
    Vector3 size = Vector3::Zero();
    bool valid = false;

    AABB() = default;

    AABB(const Vector3& p_position, const Vector3& p_size)
        : position(p_position), size(p_size), valid(true) {}

    [[nodiscard]] static AABB FromMinMax(const Vector3& minimum, const Vector3& maximum) {
        if (!minimum.allFinite() || !maximum.allFinite() ||
            (maximum.array() < minimum.array()).any()) {
            return {};
        }
        return {minimum, maximum - minimum};
    }

    [[nodiscard]] static AABB FromPoints(const std::vector<Vector3>& points) {
        Vector3 minimum = Vector3::Constant(std::numeric_limits<RealType>::infinity());
        Vector3 maximum = Vector3::Constant(-std::numeric_limits<RealType>::infinity());
        bool found = false;
        for (const Vector3& point : points) {
            if (!point.allFinite()) {
                continue;
            }
            minimum = minimum.cwiseMin(point);
            maximum = maximum.cwiseMax(point);
            found = true;
        }
        return found ? FromMinMax(minimum, maximum) : AABB{};
    }

    [[nodiscard]] bool IsValid() const {
        return valid && position.allFinite() && size.allFinite() &&
               (size.array() >= static_cast<RealType>(0.0)).all();
    }

    [[nodiscard]] Vector3 GetMin() const { return position; }
    [[nodiscard]] Vector3 GetMax() const { return position + size; }
    [[nodiscard]] Vector3 GetCenter() const { return position + size * static_cast<RealType>(0.5); }
    [[nodiscard]] Vector3 GetExtents() const { return size * static_cast<RealType>(0.5); }

    [[nodiscard]] AABB Transformed(const Matrix4& transform) const {
        if (!IsValid() || !transform.allFinite()) {
            return {};
        }
        const Matrix3 linear = transform.template block<3, 3>(0, 0);
        const Vector3 translation = transform.template block<3, 1>(0, 3);
        const Vector3 center = linear * GetCenter() + translation;
        const Vector3 extents = linear.cwiseAbs() * GetExtents();
        return FromMinMax(center - extents, center + extents);
    }
};


}
