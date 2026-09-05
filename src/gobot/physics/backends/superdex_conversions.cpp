/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/physics/backends/superdex_conversions.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <unordered_set>

namespace gobot::superdex {

namespace {

std::vector<RealType> CollectShellEdgeLengths(
        const std::vector<Vector3>& vertices,
        const std::vector<std::uint32_t>& triangle_indices) {
    std::vector<RealType> edge_lengths;
    edge_lengths.reserve(triangle_indices.size());
    std::unordered_set<std::uint64_t> seen_edges;
    const auto add_edge = [&](std::uint32_t a, std::uint32_t b) {
        if (a >= vertices.size() || b >= vertices.size() || a == b) {
            return;
        }
        const std::uint32_t lower = std::min(a, b);
        const std::uint32_t upper = std::max(a, b);
        const std::uint64_t key = (static_cast<std::uint64_t>(lower) << 32U) | upper;
        if (!seen_edges.insert(key).second) {
            return;
        }
        const RealType length = (vertices[a] - vertices[b]).norm();
        if (std::isfinite(length) && length > 0.0) {
            edge_lengths.push_back(length);
        }
    };

    for (std::size_t index = 0; index + 2 < triangle_indices.size(); index += 3) {
        add_edge(triangle_indices[index], triangle_indices[index + 1]);
        add_edge(triangle_indices[index + 1], triangle_indices[index + 2]);
        add_edge(triangle_indices[index + 2], triangle_indices[index]);
    }
    return edge_lengths;
}

RealType MedianEdgeLength(std::vector<RealType> edge_lengths, RealType fallback) {
    if (edge_lengths.empty()) {
        return fallback;
    }
    const auto middle =
            edge_lengths.begin() + static_cast<std::ptrdiff_t>(edge_lengths.size() / 2);
    std::nth_element(edge_lengths.begin(), middle, edge_lengths.end());
    RealType median = *middle;
    if (edge_lengths.size() % 2 == 0) {
        const auto lower = std::max_element(edge_lengths.begin(), middle);
        median = (*lower + median) * 0.5;
    }
    return median;
}

} // namespace

Matrix3 GobotToMochiBasis() {
    Matrix3 basis;
    basis << 1.0, 0.0, 0.0,
             0.0, 0.0, 1.0,
             0.0, -1.0, 0.0;
    return basis;
}

Vector3 ToMochiVector(const Vector3& value) {
    return {value.x(), value.z(), -value.y()};
}

Vector3 FromMochiVector(const Vector3& value) {
    return {value.x(), -value.z(), value.y()};
}

Affine3 ToMochiFrame(const Affine3& value) {
    const Matrix3 basis = GobotToMochiBasis();
    Affine3 result = Affine3::Identity();
    result.linear() = basis * value.linear() * basis.transpose();
    result.translation() = basis * value.translation();
    return result;
}

Affine3 FromMochiFrame(const Affine3& value) {
    const Matrix3 basis = GobotToMochiBasis();
    Affine3 result = Affine3::Identity();
    result.linear() = basis.transpose() * value.linear() * basis;
    result.translation() = basis.transpose() * value.translation();
    return result;
}

Matrix3 ToMochiInertiaTensor(const PhysicsLinkSnapshot& link) {
    const RealType fallback = std::max<RealType>(link.mass, 1.0) * 0.01;
    Vector3 diagonal = link.inertia_diagonal;
    for (int axis = 0; axis < 3; ++axis) {
        if (!(diagonal[axis] > 0.0) || !std::isfinite(diagonal[axis])) {
            diagonal[axis] = fallback;
        }
    }

    const Vector3 off_diagonal = link.inertia_off_diagonal.allFinite()
            ? link.inertia_off_diagonal
            : Vector3::Zero();
    Matrix3 tensor;
    tensor << diagonal.x(), off_diagonal.x(), off_diagonal.y(),
              off_diagonal.x(), diagonal.y(), off_diagonal.z(),
              off_diagonal.y(), off_diagonal.z(), diagonal.z();

    Quaternion orientation = link.inertia_orientation;
    if (!orientation.coeffs().allFinite() || orientation.squaredNorm() <= CMP_EPSILON2) {
        orientation = Quaternion::Identity();
    } else {
        orientation.normalize();
    }
    const Matrix3 oriented =
            orientation.toRotationMatrix() * tensor * orientation.toRotationMatrix().transpose();
    const Matrix3 basis = GobotToMochiBasis();
    return basis * oriented * basis.transpose();
}

JointKind ToJointKind(JointType type) {
    switch (type) {
        case JointType::Fixed:
            return JointKind::Hard;
        case JointType::Revolute:
        case JointType::Continuous:
            return JointKind::Revolute;
        case JointType::Prismatic:
            return JointKind::Prismatic;
        case JointType::Floating:
            return JointKind::Free;
        case JointType::Planar:
            return JointKind::Unsupported;
    }
    return JointKind::Unsupported;
}

bool ShouldEnableContact(std::uint32_t layer_a,
                         std::uint32_t mask_a,
                         std::uint32_t layer_b,
                         std::uint32_t mask_b) {
    return (layer_a & mask_b) != 0U && (layer_b & mask_a) != 0U;
}

std::string MakeContactLayerName(std::uint32_t layer, std::uint32_t mask) {
    char name[32]{};
    std::snprintf(name, sizeof(name), "gobot_%08x_%08x", layer, mask);
    return name;
}

RealType ComputeShellContactRadius(const std::vector<Vector3>& vertices,
                                   const std::vector<std::uint32_t>& triangle_indices,
                                   RealType thickness) {
    const RealType median = MedianEdgeLength(
            CollectShellEdgeLengths(vertices, triangle_indices), RealType{0.01});

    const RealType minimum = std::min<RealType>(
            RealType{0.01}, std::max<RealType>(0.0, 2.0 * thickness));
    return std::clamp(median, minimum, RealType{0.01});
}

bool RequiresShellContactQuadrature(const std::vector<Vector3>& vertices,
                                    const std::vector<std::uint32_t>& triangle_indices,
                                    RealType contact_radius) {
    if (!std::isfinite(contact_radius) || contact_radius <= 0.0) {
        return false;
    }
    const std::vector<RealType> edge_lengths =
            CollectShellEdgeLengths(vertices, triangle_indices);
    if (edge_lengths.empty()) {
        return false;
    }
    return MedianEdgeLength(edge_lengths, 0.0) > 2.0 * contact_radius;
}

} // namespace gobot::superdex
