/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "gobot/physics/physics_types.hpp"

namespace gobot::superdex {

enum class JointKind {
    Hard,
    Revolute,
    Prismatic,
    Free,
    Unsupported,
};

Matrix3 GobotToMochiBasis();

Vector3 ToMochiVector(const Vector3& value);

Vector3 FromMochiVector(const Vector3& value);

Affine3 ToMochiFrame(const Affine3& value);

Affine3 FromMochiFrame(const Affine3& value);

Matrix3 ToMochiInertiaTensor(const PhysicsLinkSnapshot& link);

JointKind ToJointKind(JointType type);

bool ShouldEnableContact(std::uint32_t layer_a,
                         std::uint32_t mask_a,
                         std::uint32_t layer_b,
                         std::uint32_t mask_b);

std::string MakeContactLayerName(std::uint32_t layer, std::uint32_t mask);

RealType ComputeShellContactRadius(const std::vector<Vector3>& vertices,
                                   const std::vector<std::uint32_t>& triangle_indices,
                                   RealType thickness);

bool RequiresShellContactQuadrature(const std::vector<Vector3>& vertices,
                                    const std::vector<std::uint32_t>& triangle_indices,
                                    RealType contact_radius);

} // namespace gobot::superdex
