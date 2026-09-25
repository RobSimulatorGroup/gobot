/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/core/math/mesh_validation.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <set>

namespace gobot {
namespace {
using Face = std::array<std::uint32_t, 3>;
using Tetrahedron = std::array<std::uint32_t, 4>;

struct FaceEntry {
    Face oriented{};
    std::size_t count{0};
};

Face SortedFace(Face face) {
    std::sort(face.begin(), face.end());
    return face;
}

bool HasSameOrientation(const Face& left, const Face& right) {
    return left == right ||
           Face{left[1], left[2], left[0]} == right ||
           Face{left[2], left[0], left[1]} == right;
}

bool HasOppositeOrientation(const Face& left, const Face& right) {
    return HasSameOrientation(Face{left[0], left[2], left[1]}, right);
}

bool SetValidationError(std::string* error, std::string message) {
    if (error != nullptr) {
        *error = std::move(message);
    }
    return false;
}

} // namespace

std::vector<std::uint32_t> ResolveTetrahedralSurface(std::span<const std::uint32_t> tetrahedra, std::span<const std::uint32_t> surface_triangles) {
    if (!surface_triangles.empty()) {
        return {surface_triangles.begin(), surface_triangles.end()};
    }

    std::map<Face, FaceEntry> faces;
    for (std::size_t index = 0; index + 3 < tetrahedra.size(); index += 4) {
        const std::uint32_t a = tetrahedra[index];
        const std::uint32_t b = tetrahedra[index + 1];
        const std::uint32_t c = tetrahedra[index + 2];
        const std::uint32_t d = tetrahedra[index + 3];
        const std::array<Face, 4> oriented_faces{
                Face{b, c, d}, Face{a, d, c}, Face{a, b, d}, Face{a, c, b}};
        for (const Face& face : oriented_faces) {
            FaceEntry& entry = faces[SortedFace(face)];
            if (entry.count == 0) {
                entry.oriented = face;
            }
            ++entry.count;
        }
    }

    std::vector<std::uint32_t> resolved;
    resolved.reserve(faces.size() * 3);
    for (const auto& [key, entry] : faces) {
        GOB_UNUSED(key);
        if (entry.count != 1) {
            continue;
        }
        resolved.insert(resolved.end(), entry.oriented.begin(), entry.oriented.end());
    }
    return resolved;
}

bool ValidateTetrahedralMesh(std::span<const Vector3> vertices, std::span<const std::uint32_t> tetrahedra, std::span<const std::uint32_t> surface_triangles, std::string* error) {
    if (vertices.size() < 4) {
        return SetValidationError(error, "tetrahedral mesh requires at least four vertices");
    }
    for (std::size_t index = 0; index < vertices.size(); ++index) {
        if (!vertices[index].allFinite()) {
            return SetValidationError(
                    error, "tetrahedral mesh vertex " + std::to_string(index) + " is not finite");
        }
    }
    if (tetrahedra.empty() || tetrahedra.size() % 4 != 0) {
        return SetValidationError(
                error, "tetrahedral mesh indices must contain one or more groups of four");
    }

    std::map<Face, FaceEntry> faces;
    std::map<Tetrahedron, bool> unique_tetrahedra;
    std::vector<bool> referenced_vertices(vertices.size(), false);
    for (std::size_t index = 0; index < tetrahedra.size(); index += 4) {
        std::array<std::uint32_t, 4> tet{
                tetrahedra[index], tetrahedra[index + 1],
                tetrahedra[index + 2], tetrahedra[index + 3]};
        for (const std::uint32_t vertex : tet) {
            if (vertex >= vertices.size()) {
                return SetValidationError(
                        error, "tetrahedral mesh references vertex outside its vertex table");
            }
            referenced_vertices[vertex] = true;
        }
        std::array<std::uint32_t, 4> unique = tet;
        std::sort(unique.begin(), unique.end());
        if (std::adjacent_find(unique.begin(), unique.end()) != unique.end()) {
            return SetValidationError(error, "tetrahedral mesh contains a repeated tet vertex");
        }
        if (!unique_tetrahedra.emplace(unique, true).second) {
            return SetValidationError(error, "tetrahedral mesh contains a duplicate tetrahedron");
        }

        const Vector3& a = vertices[tet[0]];
        const Vector3& b = vertices[tet[1]];
        const Vector3& c = vertices[tet[2]];
        const Vector3& d = vertices[tet[3]];
        const RealType signed_six_volume = (b - a).dot((c - a).cross(d - a));
        const RealType edge_scale = std::max({
                (b - a).norm(), (c - a).norm(), (d - a).norm(),
                (c - b).norm(), (d - b).norm(), (d - c).norm()});
        const RealType volume_tolerance =
                std::numeric_limits<RealType>::epsilon() * 128.0 *
                edge_scale * edge_scale * edge_scale;
        if (!std::isfinite(signed_six_volume) ||
            !std::isfinite(volume_tolerance) || signed_six_volume <= volume_tolerance) {
            return SetValidationError(
                    error,
                    "tetrahedral mesh requires finite, positively oriented, non-degenerate tetrahedra");
        }

        const std::array<Face, 4> tet_faces{
                Face{tet[1], tet[2], tet[3]}, Face{tet[0], tet[3], tet[2]},
                Face{tet[0], tet[1], tet[3]}, Face{tet[0], tet[2], tet[1]}};
        for (const Face& face : tet_faces) {
            FaceEntry& entry = faces[SortedFace(face)];
            if (entry.count == 0) {
                entry.oriented = face;
            } else if (entry.count == 1 &&
                       !HasOppositeOrientation(entry.oriented, face)) {
                return SetValidationError(
                        error,
                        "tetrahedral mesh shared faces must have opposite orientation");
            }
            if (++entry.count > 2) {
                return SetValidationError(
                        error, "tetrahedral mesh has a non-manifold face shared by more than two tetrahedra");
            }
        }
    }

    if (std::find(referenced_vertices.begin(), referenced_vertices.end(), false) !=
        referenced_vertices.end()) {
        return SetValidationError(
                error, "tetrahedral mesh contains a vertex not referenced by any tetrahedron");
    }

    if (!surface_triangles.empty()) {
        if (surface_triangles.size() % 3 != 0) {
            return SetValidationError(error, "tetrahedral mesh surface indices must be groups of three");
        }
        const std::size_t boundary_face_count = static_cast<std::size_t>(std::count_if(
                faces.begin(), faces.end(), [](const auto& value) {
                    return value.second.count == 1;
                }));
        if (surface_triangles.size() / 3 != boundary_face_count) {
            return SetValidationError(
                    error, "tetrahedral mesh surface must contain every boundary face exactly once");
        }
        std::map<Face, bool> visited;
        for (std::size_t index = 0; index < surface_triangles.size(); index += 3) {
            const Face face{
                    surface_triangles[index],
                    surface_triangles[index + 1],
                    surface_triangles[index + 2]};
            if (face[0] >= vertices.size() || face[1] >= vertices.size() ||
                face[2] >= vertices.size()) {
                return SetValidationError(
                        error, "tetrahedral mesh surface references an invalid vertex");
            }
            const Face sorted = SortedFace(face);
            if (sorted[0] == sorted[1] || sorted[1] == sorted[2]) {
                return SetValidationError(
                        error, "tetrahedral mesh surface contains a repeated triangle vertex");
            }
            const auto entry = faces.find(sorted);
            if (entry == faces.end() || entry->second.count != 1) {
                return SetValidationError(
                        error, "tetrahedral mesh surface contains a non-boundary face");
            }
            if (!visited.emplace(sorted, true).second) {
                return SetValidationError(
                        error, "tetrahedral mesh surface contains a duplicate boundary face");
            }
            if (!HasSameOrientation(entry->second.oriented, face)) {
                return SetValidationError(
                        error, "tetrahedral mesh surface triangle orientation must face outward");
            }
        }
    }
    if (error != nullptr) {
        error->clear();
    }
    return true;
}
bool ValidateTriangleMesh(std::span<const Vector3> vertices, std::span<const std::uint32_t> triangles, std::string* error, bool require_all_vertices) {
    if (vertices.size() < 3) {
        return SetValidationError(error, "surface mesh requires at least three vertices");
    }
    for (std::size_t index = 0; index < vertices.size(); ++index) {
        if (!vertices[index].allFinite()) {
            return SetValidationError(
                    error, "surface mesh vertex " + std::to_string(index) + " is not finite");
        }
    }
    if (triangles.empty() || triangles.size() % 3 != 0) {
        return SetValidationError(
                error, "surface mesh indices must contain one or more groups of three");
    }

    std::set<std::array<std::uint32_t, 3>> unique_triangles;
    std::vector<bool> referenced(vertices.size(), false);
    for (std::size_t index = 0; index < triangles.size(); index += 3) {
        const std::array<std::uint32_t, 3> triangle{
                triangles[index], triangles[index + 1], triangles[index + 2]};
        for (const std::uint32_t vertex : triangle) {
            if (vertex >= vertices.size()) {
                return SetValidationError(
                        error, "surface mesh references vertex outside its vertex table");
            }
            referenced[vertex] = true;
        }
        std::array<std::uint32_t, 3> sorted = triangle;
        std::sort(sorted.begin(), sorted.end());
        if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {
            return SetValidationError(error, "surface mesh contains a repeated triangle vertex");
        }
        if (!unique_triangles.insert(sorted).second) {
            return SetValidationError(error, "surface mesh contains a duplicate triangle");
        }
        const Vector3& a = vertices[triangle[0]];
        const Vector3& b = vertices[triangle[1]];
        const Vector3& c = vertices[triangle[2]];
        const RealType edge_scale = std::max({
                (b - a).norm(), (c - a).norm(), (c - b).norm()});
        const RealType area_tolerance =
                std::numeric_limits<RealType>::epsilon() * 128.0 *
                edge_scale * edge_scale;
        const RealType double_area = (b - a).cross(c - a).norm();
        if (!std::isfinite(double_area) || !std::isfinite(area_tolerance) ||
            edge_scale <= 0.0 || double_area <= area_tolerance) {
            return SetValidationError(
                    error, "surface mesh requires finite, non-degenerate triangles");
        }
    }
    if (require_all_vertices && std::find(referenced.begin(), referenced.end(), false) != referenced.end()) {
        return SetValidationError(
                error, "surface mesh contains a vertex not referenced by any triangle");
    }
    if (error != nullptr) {
        error->clear();
    }
    return true;
}
} // namespace gobot
