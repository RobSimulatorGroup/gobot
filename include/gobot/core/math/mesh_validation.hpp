/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>
#include "gobot/core/math/geometry.hpp"

namespace gobot {
GOBOT_EXPORT std::vector<std::uint32_t> ResolveTetrahedralSurface(
        std::span<const std::uint32_t> tetrahedra,
        std::span<const std::uint32_t> surface_triangles = {});
GOBOT_EXPORT bool ValidateTetrahedralMesh(
        std::span<const Vector3> vertices,
        std::span<const std::uint32_t> tetrahedra,
        std::span<const std::uint32_t> surface_triangles,
        std::string* error = nullptr);
GOBOT_EXPORT bool ValidateTriangleMesh(
        std::span<const Vector3> vertices,
        std::span<const std::uint32_t> triangles,
        std::string* error = nullptr,
        bool require_all_vertices = true);
} // namespace gobot
