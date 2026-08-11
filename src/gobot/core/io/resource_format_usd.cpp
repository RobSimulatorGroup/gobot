/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/core/io/resource_format_usd.hpp"

#include "gobot/core/config/project_setting.hpp"
#include "gobot/core/math/geometry.hpp"
#include "gobot/core/registration.hpp"
#include "gobot/core/robotics_types.hpp"
#include "gobot/log.hpp"
#include "gobot/scene/link_3d.hpp"
#include "gobot/scene/resources/array_mesh.hpp"
#include "gobot/scene/resources/box_shape_3d.hpp"
#include "gobot/scene/resources/convex_mesh_shape_3d.hpp"
#include "gobot/scene/resources/cylinder_shape_3d.hpp"
#include "gobot/scene/resources/material.hpp"
#include "gobot/scene/resources/packed_scene.hpp"
#include "gobot/scene/resources/physics_material_3d.hpp"
#include "gobot/scene/resources/sphere_shape_3d.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#ifdef GOBOT_HAS_OPENUSD
#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/gf/quatf.h>
#include <pxr/base/gf/vec2d.h>
#include <pxr/base/gf/vec2f.h>
#include <pxr/base/gf/vec3d.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/tf/token.h>
#include <pxr/base/vt/array.h>
#include <pxr/base/vt/value.h>
#include <pxr/usd/usd/prim.h>
#include <pxr/usd/usd/primFlags.h>
#include <pxr/usd/usd/primRange.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdGeom/gprim.h>
#include <pxr/usd/usdGeom/cube.h>
#include <pxr/usd/usdGeom/cylinder.h>
#include <pxr/usd/usdGeom/imageable.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/metrics.h>
#include <pxr/usd/usdGeom/primvar.h>
#include <pxr/usd/usdGeom/primvarsAPI.h>
#include <pxr/usd/usdGeom/sphere.h>
#include <pxr/usd/usdGeom/tokens.h>
#include <pxr/usd/usdGeom/xformCache.h>
#include <pxr/usd/usdGeom/xformable.h>
#include <pxr/usd/usdShade/material.h>
#include <pxr/usd/usdShade/materialBindingAPI.h>
#include <pxr/usd/usdShade/nodeGraph.h>
#include <pxr/usd/usdShade/input.h>
#include <pxr/usd/usdShade/shader.h>
#include <pxr/usd/usdPhysics/articulationRootAPI.h>
#include <pxr/usd/usdPhysics/collisionAPI.h>
#include <pxr/usd/usdPhysics/driveAPI.h>
#include <pxr/usd/usdPhysics/joint.h>
#include <pxr/usd/usdPhysics/massAPI.h>
#include <pxr/usd/usdPhysics/materialAPI.h>
#include <pxr/usd/usdPhysics/metrics.h>
#include <pxr/usd/usdPhysics/prismaticJoint.h>
#include <pxr/usd/usdPhysics/revoluteJoint.h>
#include <pxr/usd/usdPhysics/rigidBodyAPI.h>
#endif

namespace gobot {
namespace {

#ifdef GOBOT_HAS_OPENUSD

template <typename T>
void AddProperty(SceneState::NodeData& node, std::string name, T value) {
    node.properties.push_back({std::move(name), Variant(std::move(value))});
}

Affine3 ToGobotTransform(const pxr::GfMatrix4d& usd_transform) {
    Matrix4 matrix;
    for (int row = 0; row < 4; ++row) {
        for (int column = 0; column < 4; ++column) {
            // Gf matrices transform row vectors; Gobot/Eigen transforms column vectors.
            matrix(row, column) = static_cast<RealType>(usd_transform[column][row]);
        }
    }
    return Affine3(matrix);
}

void AddTransformProperties(SceneState::NodeData& node,
                            const pxr::GfMatrix4d& usd_transform,
                            const std::string& prim_path) {
    const Affine3 transform = ToGobotTransform(usd_transform);
    Vector3 scale = transform.GetScale();
    Matrix3 rotation = Matrix3::Identity();
    bool degenerate = false;
    for (int column = 0; column < 3; ++column) {
        if (std::abs(scale[column]) <= CMP_EPSILON) {
            degenerate = true;
            continue;
        }
        rotation.col(column) = transform.linear().col(column) / scale[column];
    }

    const RealType shear = std::max({
            std::abs(rotation.col(0).dot(rotation.col(1))),
            std::abs(rotation.col(0).dot(rotation.col(2))),
            std::abs(rotation.col(1).dot(rotation.col(2)))});
    if (degenerate || shear > static_cast<RealType>(1e-4)) {
        LOG_WARN("USD prim '{}' contains a degenerate or sheared transform; Gobot stores the closest TRS transform.",
                 prim_path);
    }

    Affine3 rotation_transform = Affine3::Identity();
    rotation_transform.linear() = rotation;
    if (!degenerate) {
        rotation_transform.Orthonormalize();
    }
    const Vector3 euler = rotation_transform.GetEulerAngle(EulerOrder::SXYZ);

    AddProperty(node, "position", Vector3(transform.translation()));
    AddProperty(node, "rotation_degrees", Vector3{
            RAD_TO_DEG(euler.x()),
            RAD_TO_DEG(euler.y()),
            RAD_TO_DEG(euler.z())});
    AddProperty(node, "scale", scale);
}

bool ConvertVec3Array(const pxr::VtValue& value, std::vector<Vector3>* output) {
    output->clear();
    if (value.IsHolding<pxr::VtVec3fArray>()) {
        const pxr::VtVec3fArray& values = value.UncheckedGet<pxr::VtVec3fArray>();
        output->reserve(values.size());
        for (const pxr::GfVec3f& item : values) {
            output->emplace_back(item[0], item[1], item[2]);
        }
        return true;
    }
    if (value.IsHolding<pxr::VtVec3dArray>()) {
        const pxr::VtVec3dArray& values = value.UncheckedGet<pxr::VtVec3dArray>();
        output->reserve(values.size());
        for (const pxr::GfVec3d& item : values) {
            output->emplace_back(static_cast<RealType>(item[0]),
                                 static_cast<RealType>(item[1]),
                                 static_cast<RealType>(item[2]));
        }
        return true;
    }
    return false;
}

bool ConvertVec2Array(const pxr::VtValue& value, std::vector<Vector2>* output) {
    output->clear();
    if (value.IsHolding<pxr::VtVec2fArray>()) {
        const pxr::VtVec2fArray& values = value.UncheckedGet<pxr::VtVec2fArray>();
        output->reserve(values.size());
        for (const pxr::GfVec2f& item : values) {
            output->emplace_back(item[0], item[1]);
        }
        return true;
    }
    if (value.IsHolding<pxr::VtVec2dArray>()) {
        const pxr::VtVec2dArray& values = value.UncheckedGet<pxr::VtVec2dArray>();
        output->reserve(values.size());
        for (const pxr::GfVec2d& item : values) {
            output->emplace_back(static_cast<RealType>(item[0]), static_cast<RealType>(item[1]));
        }
        return true;
    }
    return false;
}

std::optional<std::size_t> ResolvePrimvarIndex(const pxr::TfToken& interpolation,
                                               std::size_t face_index,
                                               std::size_t point_index,
                                               std::size_t corner_index,
                                               std::size_t value_count) {
    std::size_t index = 0;
    if (interpolation == pxr::UsdGeomTokens->constant) {
        index = 0;
    } else if (interpolation == pxr::UsdGeomTokens->uniform) {
        index = face_index;
    } else if (interpolation == pxr::UsdGeomTokens->vertex ||
               interpolation == pxr::UsdGeomTokens->varying) {
        index = point_index;
    } else if (interpolation == pxr::UsdGeomTokens->faceVarying) {
        index = corner_index;
    } else {
        return std::nullopt;
    }
    return index < value_count ? std::optional<std::size_t>(index) : std::nullopt;
}

RealType Cross2D(const Vector2& a, const Vector2& b, const Vector2& c) {
    const Vector2 ab = b - a;
    const Vector2 ac = c - a;
    return ab.x() * ac.y() - ab.y() * ac.x();
}

std::optional<std::vector<std::array<std::size_t, 3>>> TriangulateFace(
        const pxr::VtVec3fArray& points,
        const pxr::VtIntArray& face_vertex_indices,
        std::size_t corner_offset,
        std::size_t corner_count) {
    if (corner_count == 3) {
        for (std::size_t local_corner = 0; local_corner < corner_count; ++local_corner) {
            const int point_index = face_vertex_indices[corner_offset + local_corner];
            if (point_index < 0 || static_cast<std::size_t>(point_index) >= points.size()) {
                return std::nullopt;
            }
        }
        return std::vector<std::array<std::size_t, 3>>{{{0, 1, 2}}};
    }

    std::vector<Vector3> vertices;
    vertices.reserve(corner_count);
    for (std::size_t local_corner = 0; local_corner < corner_count; ++local_corner) {
        const int point_index = face_vertex_indices[corner_offset + local_corner];
        if (point_index < 0 || static_cast<std::size_t>(point_index) >= points.size()) {
            return std::nullopt;
        }
        const pxr::GfVec3f& point = points[static_cast<std::size_t>(point_index)];
        vertices.emplace_back(point[0], point[1], point[2]);
    }

    Vector3 normal = Vector3::Zero();
    for (std::size_t index = 0; index < corner_count; ++index) {
        const Vector3& current = vertices[index];
        const Vector3& next = vertices[(index + 1) % corner_count];
        normal.x() += (current.y() - next.y()) * (current.z() + next.z());
        normal.y() += (current.z() - next.z()) * (current.x() + next.x());
        normal.z() += (current.x() - next.x()) * (current.y() + next.y());
    }
    if (!normal.allFinite() || normal.squaredNorm() <= CMP_EPSILON * CMP_EPSILON) {
        return std::nullopt;
    }

    int dropped_axis = 0;
    if (std::abs(normal.y()) > std::abs(normal.x())) {
        dropped_axis = 1;
    }
    if (std::abs(normal.z()) > std::abs(normal[dropped_axis])) {
        dropped_axis = 2;
    }

    std::vector<Vector2> projected;
    projected.reserve(corner_count);
    for (const Vector3& vertex : vertices) {
        if (dropped_axis == 0) {
            projected.emplace_back(vertex.y(), vertex.z());
        } else if (dropped_axis == 1) {
            projected.emplace_back(vertex.x(), vertex.z());
        } else {
            projected.emplace_back(vertex.x(), vertex.y());
        }
    }

    RealType signed_area_twice = 0.0f;
    for (std::size_t index = 0; index < corner_count; ++index) {
        const Vector2& current = projected[index];
        const Vector2& next = projected[(index + 1) % corner_count];
        signed_area_twice += current.x() * next.y() - next.x() * current.y();
    }
    if (std::abs(signed_area_twice) <= CMP_EPSILON) {
        return std::nullopt;
    }
    const RealType winding = signed_area_twice > 0.0f ? 1.0f : -1.0f;

    std::vector<std::size_t> polygon(corner_count);
    for (std::size_t index = 0; index < corner_count; ++index) {
        polygon[index] = index;
    }
    std::vector<std::array<std::size_t, 3>> triangles;
    triangles.reserve(corner_count - 2);
    while (polygon.size() > 3) {
        bool clipped = false;
        for (std::size_t index = 0; index < polygon.size(); ++index) {
            const std::size_t previous = polygon[(index + polygon.size() - 1) % polygon.size()];
            const std::size_t current = polygon[index];
            const std::size_t next = polygon[(index + 1) % polygon.size()];
            if (winding * Cross2D(projected[previous], projected[current], projected[next]) <=
                CMP_EPSILON) {
                continue;
            }

            bool contains_vertex = false;
            for (const std::size_t candidate : polygon) {
                if (candidate == previous || candidate == current || candidate == next) {
                    continue;
                }
                const Vector2& point = projected[candidate];
                const bool inside =
                        winding * Cross2D(projected[previous], projected[current], point) >= -CMP_EPSILON &&
                        winding * Cross2D(projected[current], projected[next], point) >= -CMP_EPSILON &&
                        winding * Cross2D(projected[next], projected[previous], point) >= -CMP_EPSILON;
                if (inside) {
                    contains_vertex = true;
                    break;
                }
            }
            if (contains_vertex) {
                continue;
            }

            triangles.push_back({previous, current, next});
            polygon.erase(polygon.begin() + static_cast<std::ptrdiff_t>(index));
            clipped = true;
            break;
        }
        if (!clipped) {
            return std::nullopt;
        }
    }
    triangles.push_back({polygon[0], polygon[1], polygon[2]});
    return triangles;
}

template <typename T>
bool ReadConstantInput(const pxr::UsdShadeShader& shader, const char* name, T* value) {
    const pxr::UsdShadeInput input = shader.GetInput(pxr::TfToken(name));
    return input && !input.HasConnectedSource() && input.Get(value);
}

Ref<PBRMaterial3D> ImportBoundMaterial(
        const pxr::UsdPrim& prim,
        std::unordered_map<std::string, Ref<PBRMaterial3D>>* material_cache) {
    bool double_sided = false;
    if (const pxr::UsdGeomGprim gprim(prim); gprim) {
        gprim.GetDoubleSidedAttr().Get(&double_sided);
    }

    const pxr::UsdShadeMaterial usd_material =
            pxr::UsdShadeMaterialBindingAPI(prim).ComputeBoundMaterial();
    const std::string material_path = usd_material
                                              ? usd_material.GetPrim().GetPath().GetString()
                                              : std::string();
    if (!usd_material && !double_sided) {
        return {};
    }

    const std::string cache_key = material_path + (double_sided ? "|double" : "|single");
    if (const auto cached = material_cache->find(cache_key); cached != material_cache->end()) {
        return cached->second;
    }

    Ref<PBRMaterial3D> material = MakeRef<PBRMaterial3D>();
    material->SetDoubleSided(double_sided);
    if (usd_material) {
        material->SetName(usd_material.GetPrim().GetName().GetString());
        const pxr::UsdShadeShader shader = usd_material.ComputeSurfaceSource();
        pxr::TfToken shader_id;
        if (shader && shader.GetIdAttr().Get(&shader_id) && shader_id == pxr::TfToken("UsdPreviewSurface")) {
            pxr::GfVec3f diffuse(0.8f);
            float opacity = 1.0f;
            ReadConstantInput(shader, "diffuseColor", &diffuse);
            ReadConstantInput(shader, "opacity", &opacity);
            material->SetAlbedo(Color(diffuse[0], diffuse[1], diffuse[2], opacity));

            float scalar = 0.0f;
            if (ReadConstantInput(shader, "metallic", &scalar)) {
                material->SetMetallic(scalar);
            }
            if (ReadConstantInput(shader, "roughness", &scalar)) {
                material->SetRoughness(scalar);
            }

            pxr::GfVec3f emissive(0.0f);
            if (ReadConstantInput(shader, "emissiveColor", &emissive)) {
                material->SetEmissive(Color(emissive[0], emissive[1], emissive[2], 1.0f));
            }

            float opacity_threshold = 0.0f;
            if (ReadConstantInput(shader, "opacityThreshold", &opacity_threshold) &&
                opacity_threshold > 0.0f) {
                material->SetAlphaMode(AlphaMode::Mask);
                material->SetAlphaCutoff(opacity_threshold);
            } else if (opacity < 1.0f - CMP_EPSILON) {
                material->SetAlphaMode(AlphaMode::Blend);
            }
        }
    }

    material_cache->emplace(cache_key, material);
    return material;
}

Ref<ArrayMesh> ImportMesh(
        const pxr::UsdGeomMesh& usd_mesh,
        std::unordered_map<std::string, Ref<PBRMaterial3D>>* material_cache) {
    pxr::VtVec3fArray points;
    pxr::VtIntArray face_vertex_counts;
    pxr::VtIntArray face_vertex_indices;
    if (!usd_mesh.GetPointsAttr().Get(&points) ||
        !usd_mesh.GetFaceVertexCountsAttr().Get(&face_vertex_counts) ||
        !usd_mesh.GetFaceVertexIndicesAttr().Get(&face_vertex_indices)) {
        LOG_WARN("USD mesh '{}' is missing points or topology and was skipped.",
                 usd_mesh.GetPrim().GetPath().GetString());
        return {};
    }

    pxr::TfToken subdivision_scheme;
    if (usd_mesh.GetSubdivisionSchemeAttr().Get(&subdivision_scheme) &&
        subdivision_scheme != pxr::UsdGeomTokens->none) {
        LOG_WARN("USD mesh '{}' uses subdivision scheme '{}'; Gobot currently imports its control cage.",
                 usd_mesh.GetPrim().GetPath().GetString(), subdivision_scheme.GetString());
    }

    std::unordered_set<int> hole_faces;
    pxr::VtIntArray holes;
    if (usd_mesh.GetHoleIndicesAttr().Get(&holes)) {
        hole_faces.insert(holes.begin(), holes.end());
    }

    std::vector<Vector3> normal_values;
    pxr::TfToken normal_interpolation;
    const pxr::UsdGeomPrimvar normal_primvar =
            pxr::UsdGeomPrimvarsAPI(usd_mesh.GetPrim()).FindPrimvarWithInheritance(pxr::TfToken("normals"));
    if (normal_primvar) {
        pxr::VtValue flattened;
        if (normal_primvar.ComputeFlattened(&flattened) && ConvertVec3Array(flattened, &normal_values)) {
            normal_interpolation = normal_primvar.GetInterpolation();
        }
    }
    if (normal_values.empty()) {
        pxr::VtVec3fArray authored_normals;
        if (usd_mesh.GetNormalsAttr().Get(&authored_normals)) {
            ConvertVec3Array(pxr::VtValue(authored_normals), &normal_values);
            normal_interpolation = usd_mesh.GetNormalsInterpolation();
        }
    }

    std::vector<Vector2> uv_values;
    pxr::TfToken uv_interpolation;
    const pxr::UsdGeomPrimvar st =
            pxr::UsdGeomPrimvarsAPI(usd_mesh.GetPrim()).FindPrimvarWithInheritance(pxr::TfToken("st"));
    if (st) {
        pxr::VtValue flattened;
        if (st.ComputeFlattened(&flattened) && ConvertVec2Array(flattened, &uv_values)) {
            uv_interpolation = st.GetInterpolation();
        }
    }

    MeshSurfaceData surface;
    const Ref<PBRMaterial3D> material = ImportBoundMaterial(usd_mesh.GetPrim(), material_cache);
    surface.material = dynamic_pointer_cast<Material>(material);
    bool normals_complete = !normal_values.empty();
    bool uvs_complete = !uv_values.empty();
    std::size_t corner_offset = 0;
    const bool left_handed = [&usd_mesh]() {
        pxr::TfToken orientation;
        return usd_mesh.GetOrientationAttr().Get(&orientation) && orientation == pxr::UsdGeomTokens->leftHanded;
    }();

    for (std::size_t face = 0; face < face_vertex_counts.size(); ++face) {
        const int count_value = face_vertex_counts[face];
        if (count_value < 0 || corner_offset + static_cast<std::size_t>(count_value) > face_vertex_indices.size()) {
            LOG_WARN("USD mesh '{}' has invalid face topology and was skipped.",
                     usd_mesh.GetPrim().GetPath().GetString());
            return {};
        }
        const std::size_t count = static_cast<std::size_t>(count_value);
        if (count >= 3 && !hole_faces.contains(static_cast<int>(face))) {
            const auto triangles = TriangulateFace(points, face_vertex_indices, corner_offset, count);
            if (!triangles.has_value()) {
                LOG_WARN("USD mesh '{}' contains a degenerate or non-simple polygon and was skipped.",
                         usd_mesh.GetPrim().GetPath().GetString());
                return {};
            }
            for (std::array<std::size_t, 3> local_corners : *triangles) {
                if (left_handed) {
                    std::swap(local_corners[1], local_corners[2]);
                }
                for (const std::size_t local_corner : local_corners) {
                    const std::size_t corner = corner_offset + local_corner;
                    const int point_value = face_vertex_indices[corner];
                    if (point_value < 0 || static_cast<std::size_t>(point_value) >= points.size()) {
                        LOG_WARN("USD mesh '{}' references an invalid point index and was skipped.",
                                 usd_mesh.GetPrim().GetPath().GetString());
                        return {};
                    }
                    const std::size_t point_index = static_cast<std::size_t>(point_value);
                    const pxr::GfVec3f& point = points[point_index];
                    surface.vertices.emplace_back(point[0], point[1], point[2]);
                    surface.indices.push_back(static_cast<std::uint32_t>(surface.indices.size()));

                    if (normals_complete) {
                        const auto normal_index = ResolvePrimvarIndex(
                                normal_interpolation, face, point_index, corner, normal_values.size());
                        if (!normal_index.has_value()) {
                            normals_complete = false;
                        } else {
                            Vector3 normal = normal_values[*normal_index];
                            if (!normal.allFinite() || normal.norm() <= CMP_EPSILON) {
                                normals_complete = false;
                            } else {
                                surface.normals.emplace_back(normal.normalized());
                            }
                        }
                    }
                    if (uvs_complete) {
                        const auto uv_index = ResolvePrimvarIndex(
                                uv_interpolation, face, point_index, corner, uv_values.size());
                        if (!uv_index.has_value()) {
                            uvs_complete = false;
                        } else {
                            surface.uv0.push_back(uv_values[*uv_index]);
                        }
                    }
                }
            }
        }
        corner_offset += count;
    }

    if (corner_offset != face_vertex_indices.size() || surface.indices.empty()) {
        LOG_WARN("USD mesh '{}' has inconsistent or empty topology and was skipped.",
                 usd_mesh.GetPrim().GetPath().GetString());
        return {};
    }
    if (!normals_complete || surface.normals.size() != surface.vertices.size()) {
        surface.normals.clear();
    }
    if (!uvs_complete || surface.uv0.size() != surface.vertices.size()) {
        surface.uv0.clear();
    }

    Ref<ArrayMesh> mesh = MakeRef<ArrayMesh>();
    mesh->SetName(usd_mesh.GetPrim().GetName().GetString());
    mesh->SetSurfaces({std::move(surface)});
    return mesh;
}

Ref<Material> DetachSingleSurfaceMaterial(const Ref<ArrayMesh>& mesh) {
    if (!mesh.IsValid()) {
        return {};
    }
    MeshSurfaceList surfaces = mesh->GetSurfaces();
    if (surfaces.size() != 1 || !surfaces.front().material.IsValid()) {
        return {};
    }
    Ref<Material> material = surfaces.front().material;
    surfaces.front().material.Reset();
    mesh->SetSurfaces(std::move(surfaces));
    return material;
}

void ClearSurfaceMaterials(const Ref<ArrayMesh>& mesh) {
    if (!mesh.IsValid()) {
        return;
    }
    MeshSurfaceList surfaces = mesh->GetSurfaces();
    bool changed = false;
    for (MeshSurfaceData& surface : surfaces) {
        if (surface.material.IsValid()) {
            surface.material.Reset();
            changed = true;
        }
    }
    if (changed) {
        mesh->SetSurfaces(std::move(surfaces));
    }
}

std::string MakeSceneName(const std::string& path) {
    const std::string stem = std::filesystem::path(path).stem().string();
    return stem.empty() ? "USDScene" : stem;
}

bool PrimContainsPhysicsSchema(const pxr::UsdPrim& prim) {
    if (prim.GetTypeName().GetString().starts_with("Physics")) {
        return true;
    }
    return std::ranges::any_of(prim.GetAppliedSchemas(), [](const pxr::TfToken& schema) {
        return schema.GetString().starts_with("Physics");
    });
}

bool IsSupportedVisualMesh(const pxr::UsdPrim& prim) {
    if (!pxr::UsdGeomMesh(prim)) {
        return false;
    }
    if (const pxr::UsdGeomImageable imageable(prim); imageable) {
        const pxr::TfToken purpose = imageable.ComputePurpose();
        if (purpose == pxr::UsdGeomTokens->proxy || purpose == pxr::UsdGeomTokens->guide) {
            return false;
        }
    }
    return true;
}

void IncludePrimAndAncestors(const pxr::UsdPrim& prim,
                             std::unordered_set<std::string>* included_paths) {
    for (pxr::UsdPrim current = prim; current && !current.IsPseudoRoot();
         current = current.GetParent()) {
        included_paths->insert(current.GetPath().GetString());
    }
}

Vector3 ToGobotVector3(const pxr::GfVec3f& value) {
    return {value[0], value[1], value[2]};
}

Quaternion ToGobotQuaternion(const pxr::GfQuatf& value) {
    const pxr::GfVec3f imaginary = value.GetImaginary();
    Quaternion result(value.GetReal(), imaginary[0], imaginary[1], imaginary[2]);
    if (!result.coeffs().allFinite() || result.norm() <= CMP_EPSILON) {
        return Quaternion::Identity();
    }
    return result.normalized();
}

Affine3 MakeFrame(const pxr::GfVec3f& position, const pxr::GfQuatf& rotation) {
    Affine3 result = Affine3::Identity();
    result.translation() = ToGobotVector3(position);
    result.linear() = ToGobotQuaternion(rotation).toRotationMatrix();
    return result;
}

void AddAffineTransformProperties(SceneState::NodeData& node,
                                  const Affine3& transform,
                                  const std::string& prim_path,
                                  bool preserve_scale) {
    Affine3 rigid = transform;
    Vector3 scale = rigid.GetScale();
    Matrix3 rotation = Matrix3::Identity();
    bool degenerate = false;
    for (int column = 0; column < 3; ++column) {
        if (std::abs(scale[column]) <= CMP_EPSILON) {
            degenerate = true;
            continue;
        }
        rotation.col(column) = rigid.linear().col(column) / scale[column];
    }
    if (degenerate) {
        LOG_WARN("USD physics prim '{}' has a degenerate transform; using identity rotation.", prim_path);
        rotation = Matrix3::Identity();
    } else {
        Affine3 rotation_transform = Affine3::Identity();
        rotation_transform.linear() = rotation;
        rotation_transform.Orthonormalize();
        rotation = rotation_transform.linear();
    }
    Affine3 rotation_transform = Affine3::Identity();
    rotation_transform.linear() = rotation;
    const Vector3 euler = rotation_transform.GetEulerAngle(EulerOrder::SXYZ);
    AddProperty(node, "position", Vector3(rigid.translation()));
    AddProperty(node, "rotation_degrees", Vector3{
            RAD_TO_DEG(euler.x()), RAD_TO_DEG(euler.y()), RAD_TO_DEG(euler.z())});
    AddProperty(node, "scale", preserve_scale ? scale : Vector3::Ones());
}

void AddRigidTransformProperties(SceneState::NodeData& node,
                                 const Affine3& transform,
                                 const std::string& prim_path) {
    AddAffineTransformProperties(node, transform, prim_path, false);
}

Affine3 ScaleTranslation(Affine3 transform, RealType meters_per_unit) {
    transform.translation() *= meters_per_unit;
    return transform;
}

template <typename T>
bool ReadAttribute(const pxr::UsdPrim& prim, const char* name, T* value) {
    const pxr::UsdAttribute attribute = prim.GetAttribute(pxr::TfToken(name));
    return attribute && attribute.Get(value);
}

bool ReadRealArrayAttribute(const pxr::UsdPrim& prim,
                            const char* name,
                            std::vector<RealType>* values) {
    pxr::VtDoubleArray source;
    if (!ReadAttribute(prim, name, &source)) {
        return false;
    }
    values->clear();
    values->reserve(source.size());
    for (double value : source) {
        values->push_back(static_cast<RealType>(value));
    }
    return true;
}

bool IsApproximatelyZero(RealType value) {
    return std::abs(value) <= static_cast<RealType>(1.0e-12);
}

bool AddAffineActuatorProperties(const pxr::UsdPrim& actuator,
                                 SceneState::NodeData& joint_node) {
    pxr::TfToken bias_type("none");
    pxr::TfToken gain_type("fixed");
    pxr::TfToken dynamics_type("none");
    ReadAttribute(actuator, "mjc:biasType", &bias_type);
    ReadAttribute(actuator, "mjc:gainType", &gain_type);
    ReadAttribute(actuator, "mjc:dynType", &dynamics_type);
    if (bias_type != pxr::TfToken("affine") ||
        gain_type != pxr::TfToken("fixed") ||
        dynamics_type != pxr::TfToken("none")) {
        return false;
    }

    std::vector<RealType> gain;
    std::vector<RealType> bias;
    if (!ReadRealArrayAttribute(actuator, "mjc:gainPrm", &gain) || gain.empty() ||
        !ReadRealArrayAttribute(actuator, "mjc:biasPrm", &bias) || bias.size() < 3) {
        return false;
    }
    if (!std::isfinite(gain[0]) ||
        !std::all_of(bias.begin(), bias.begin() + 3,
                     [](RealType value) { return std::isfinite(value); }) ||
        !std::all_of(gain.begin() + 1, gain.end(), IsApproximatelyZero) ||
        !std::all_of(bias.begin() + 3, bias.end(), IsApproximatelyZero)) {
        return false;
    }

    std::vector<RealType> gear;
    if (ReadRealArrayAttribute(actuator, "mjc:gear", &gear)) {
        if (gear.empty() || !std::isfinite(gear.front()) ||
            std::abs(gear.front() - static_cast<RealType>(1.0)) > static_cast<RealType>(1.0e-12) ||
            !std::all_of(gear.begin() + 1, gear.end(), IsApproximatelyZero)) {
            return false;
        }
    }

    double inherit_range = 0.0;
    ReadAttribute(actuator, "mjc:inheritRange", &inherit_range);
    if (!std::isfinite(inherit_range)) {
        return false;
    }
    AddProperty(joint_node, "affine_actuator_enabled", true);
    AddProperty(joint_node, "affine_actuator_control_gain", gain[0]);
    AddProperty(joint_node, "affine_actuator_force_offset", bias[0]);
    AddProperty(joint_node, "affine_actuator_position_gain", bias[1]);
    AddProperty(joint_node, "affine_actuator_velocity_gain", bias[2]);
    AddProperty(joint_node, "affine_actuator_inherit_range",
                static_cast<RealType>(inherit_range));
    return true;
}

std::optional<std::string> ReadRelationshipTarget(const pxr::UsdPrim& prim,
                                                  const char* name) {
    const pxr::UsdRelationship relationship = prim.GetRelationship(pxr::TfToken(name));
    pxr::SdfPathVector targets;
    if (!relationship || !relationship.GetTargets(&targets) || targets.size() != 1) {
        return std::nullopt;
    }
    return targets.front().GetString();
}

Vector3 AxisForToken(const pxr::TfToken& axis) {
    if (axis == pxr::TfToken("Y")) {
        return Vector3::UnitY();
    }
    if (axis == pxr::TfToken("Z")) {
        return Vector3::UnitZ();
    }
    return Vector3::UnitX();
}

struct PhysicsLinkImportData {
    pxr::UsdPrim prim;
    std::string path;
    std::string name;
    pxr::GfMatrix4d world;
    int node_index{-1};
};

struct PhysicsJointImportData {
    pxr::UsdPrim prim;
    std::string parent_path;
    std::string child_path;
    Affine3 frame0{Affine3::Identity()};
    Affine3 frame1{Affine3::Identity()};
};

bool IsPhysicsArticulationStage(const pxr::UsdStageRefPtr& stage) {
    std::vector<pxr::UsdPrim> articulation_roots;
    for (const pxr::UsdPrim& prim :
         pxr::UsdPrimRange::Stage(stage, pxr::UsdTraverseInstanceProxies())) {
        if (prim.IsActive() && prim.HasAPI<pxr::UsdPhysicsArticulationRootAPI>()) {
            articulation_roots.push_back(prim);
        }
    }
    if (articulation_roots.empty()) {
        return false;
    }
    return std::ranges::any_of(
            pxr::UsdPrimRange::Stage(stage, pxr::UsdTraverseInstanceProxies()),
            [&articulation_roots](const pxr::UsdPrim& prim) {
                return prim.IsActive() && prim.HasAPI<pxr::UsdPhysicsRigidBodyAPI>() &&
                       std::ranges::any_of(
                               articulation_roots,
                               [&prim](const pxr::UsdPrim& root) {
                                   if (prim.GetPath().HasPrefix(root.GetPath())) {
                                       return true;
                                   }
                                   if (!pxr::UsdPhysicsJoint(root)) {
                                       return false;
                                   }
                                   for (const char* relationship : {"physics:body0", "physics:body1"}) {
                                       if (const auto target = ReadRelationshipTarget(root, relationship);
                                           target.has_value() && *target == prim.GetPath().GetString()) {
                                           return true;
                                       }
                                   }
                                   return false;
                               });
            });
}

pxr::UsdPrim FindOwningRigidBody(pxr::UsdPrim prim) {
    for (pxr::UsdPrim current = prim; current && !current.IsPseudoRoot();
         current = current.GetParent()) {
        if (current.HasAPI<pxr::UsdPhysicsRigidBodyAPI>()) {
            return current;
        }
    }
    return {};
}

Ref<PhysicsMaterial3D> ReadPhysicsMaterial(const pxr::UsdPrim& collision_prim) {
    Ref<PhysicsMaterial3D> result = MakeRef<PhysicsMaterial3D>();
    const auto material_path = ReadRelationshipTarget(collision_prim, "material:binding:physics");
    if (!material_path.has_value()) {
        return result;
    }
    const pxr::UsdPrim material = collision_prim.GetStage()->GetPrimAtPath(pxr::SdfPath(*material_path));
    float value = 0.0f;
    if (material && pxr::UsdPhysicsMaterialAPI(material).GetDynamicFrictionAttr().Get(&value)) {
        result->SetSlidingFriction(value);
    }
    if (material && ReadAttribute(material, "newton:torsionalFriction", &value)) {
        result->SetTorsionalFriction(value);
    }
    if (material && ReadAttribute(material, "newton:rollingFriction", &value)) {
        result->SetRollingFriction(value);
    }
    if (material && pxr::UsdPhysicsMaterialAPI(material).GetRestitutionAttr().Get(&value)) {
        result->SetRestitution(value);
    }
    return result;
}

void AddPhysicsContactProperties(SceneState::NodeData& node, const pxr::UsdPrim& prim) {
    bool collision_enabled = true;
    pxr::UsdPhysicsCollisionAPI(prim).GetCollisionEnabledAttr().Get(&collision_enabled);
    AddProperty(node, "disabled", !collision_enabled);
    AddProperty(node, "physics_material", ReadPhysicsMaterial(prim));
    int contype = 1;
    int conaffinity = 1;
    ReadAttribute(prim, "mjc:contype", &contype);
    ReadAttribute(prim, "mjc:conaffinity", &conaffinity);
    AddProperty(node, "collision_layer", static_cast<std::uint32_t>(contype));
    AddProperty(node, "collision_mask", static_cast<std::uint32_t>(conaffinity));
    float gap = 0.0f;
    if (ReadAttribute(prim, "newton:contactGap", &gap)) {
        AddProperty(node, "contact_offset", static_cast<RealType>(gap));
        AddProperty(node, "rest_offset", static_cast<RealType>(0.0));
    }
    AddProperty(node, "visible", false);
}

Ref<ArrayMesh> ScaledMesh(const Ref<ArrayMesh>& source, const Vector3& scale) {
    if (!source.IsValid() || scale.isApprox(Vector3::Ones(), CMP_EPSILON)) {
        return source;
    }
    MeshSurfaceList surfaces = source->GetSurfaces();
    for (MeshSurfaceData& surface : surfaces) {
        for (Vector3& vertex : surface.vertices) {
            vertex = vertex.cwiseProduct(scale);
        }
        for (Vector3& normal : surface.normals) {
            normal = scale.cwiseInverse().cwiseProduct(normal).normalized();
        }
    }
    Ref<ArrayMesh> result = MakeRef<ArrayMesh>();
    result->SetName(source->GetName());
    result->SetSurfaces(std::move(surfaces));
    return result;
}

Ref<PackedScene> ImportPhysicsArticulation(const pxr::UsdStageRefPtr& stage,
                                           const std::string& path,
                                           const std::string& global_path) {
    pxr::UsdGeomXformCache xform_cache(pxr::UsdTimeCode::Default());
    pxr::UsdPrim articulation_root;
    for (const pxr::UsdPrim& prim :
         pxr::UsdPrimRange::Stage(stage, pxr::UsdTraverseInstanceProxies())) {
        if (!prim.IsActive() || !prim.HasAPI<pxr::UsdPhysicsArticulationRootAPI>()) {
            continue;
        }
        if (articulation_root) {
            LOG_WARN("USD stage '{}' contains multiple articulation roots; importing the first one at '{}'.",
                     path, articulation_root.GetPath().GetString());
            break;
        }
        articulation_root = prim;
    }
    if (!articulation_root) {
        return {};
    }

    pxr::UsdPrim model_prim = stage->GetDefaultPrim();
    if (!model_prim || !articulation_root.GetPath().HasPrefix(model_prim.GetPath())) {
        model_prim = articulation_root;
        while (model_prim.GetParent() && !model_prim.GetParent().IsPseudoRoot()) {
            model_prim = model_prim.GetParent();
        }
    }

    const RealType meters_per_unit = static_cast<RealType>(pxr::UsdGeomGetStageMetersPerUnit(stage));
    const RealType kilograms_per_unit = static_cast<RealType>(pxr::UsdPhysicsGetStageKilogramsPerUnit(stage));
    const RealType inertia_scale = kilograms_per_unit * meters_per_unit * meters_per_unit;
    const pxr::GfMatrix4d model_world = xform_cache.GetLocalToWorldTransform(model_prim);

    std::vector<PhysicsLinkImportData> links;
    std::unordered_map<std::string, std::size_t> link_by_path;
    std::vector<PhysicsJointImportData> joints;
    std::unordered_map<std::string, std::vector<std::size_t>> outgoing_joints;
    std::unordered_set<std::string> child_link_paths;
    std::unordered_set<std::string> world_anchored_link_paths;
    std::unordered_map<std::string, pxr::UsdPrim> actuator_by_joint_path;
    std::unordered_map<std::string, std::vector<pxr::UsdPrim>> owned_prims;

    for (const pxr::UsdPrim& prim :
         pxr::UsdPrimRange::Stage(stage, pxr::UsdTraverseInstanceProxies())) {
        if (!prim.IsActive() || !prim.GetPath().HasPrefix(model_prim.GetPath())) {
            continue;
        }
        const std::string prim_path = prim.GetPath().GetString();
        if (prim.HasAPI<pxr::UsdPhysicsRigidBodyAPI>()) {
            link_by_path.emplace(prim_path, links.size());
            links.push_back({prim, prim_path, prim.GetName().GetString(),
                             xform_cache.GetLocalToWorldTransform(prim)});
        }
        if (prim.GetTypeName() == pxr::TfToken("MjcActuator")) {
            if (const auto target = ReadRelationshipTarget(prim, "mjc:target")) {
                actuator_by_joint_path.emplace(*target, prim);
            }
        }
    }

    for (const pxr::UsdPrim& prim :
         pxr::UsdPrimRange::Stage(stage, pxr::UsdTraverseInstanceProxies())) {
        if (!prim.IsActive() || !prim.GetPath().HasPrefix(model_prim.GetPath())) {
            continue;
        }
        const pxr::UsdPhysicsJoint usd_joint(prim);
        if (usd_joint) {
            const auto body0 = ReadRelationshipTarget(prim, "physics:body0");
            const auto body1 = ReadRelationshipTarget(prim, "physics:body1");
            const bool known_body0 = body0.has_value() && link_by_path.contains(*body0);
            const bool known_body1 = body1.has_value() && link_by_path.contains(*body1);
            if (known_body0 && known_body1) {
                pxr::GfVec3f local_pos0(0.0f);
                pxr::GfVec3f local_pos1(0.0f);
                pxr::GfQuatf local_rot0(1.0f);
                pxr::GfQuatf local_rot1(1.0f);
                usd_joint.GetLocalPos0Attr().Get(&local_pos0);
                usd_joint.GetLocalPos1Attr().Get(&local_pos1);
                usd_joint.GetLocalRot0Attr().Get(&local_rot0);
                usd_joint.GetLocalRot1Attr().Get(&local_rot1);
                PhysicsJointImportData data{
                        prim, *body0, *body1,
                        MakeFrame(local_pos0, local_rot0),
                        MakeFrame(local_pos1, local_rot1)};
                outgoing_joints[*body0].push_back(joints.size());
                child_link_paths.insert(*body1);
                joints.push_back(std::move(data));
            } else if (prim.GetTypeName() == pxr::TfToken("PhysicsFixedJoint") &&
                       known_body0 != known_body1) {
                // A fixed joint with exactly one body target welds that body to
                // the world. Gobot represents the same constraint by placing
                // the root Link3D directly under Robot3D without a floating
                // Joint3D.
                world_anchored_link_paths.insert(known_body0 ? *body0 : *body1);
            }
        }

        if (const pxr::UsdPrim owner = FindOwningRigidBody(prim); owner) {
            const std::string owner_path = owner.GetPath().GetString();
            if (link_by_path.contains(owner_path)) {
                owned_prims[owner_path].push_back(prim);
            }
        }
    }

    if (links.empty()) {
        LOG_ERROR("USD articulation '{}' contains no rigid bodies.", path);
        return {};
    }

    pxr::UsdPrim articulation_body;
    const std::string articulation_root_path = articulation_root.GetPath().GetString();
    if (articulation_root.HasAPI<pxr::UsdPhysicsRigidBodyAPI>() &&
        link_by_path.contains(articulation_root_path)) {
        articulation_body = articulation_root;
    } else if (pxr::UsdPhysicsJoint(articulation_root)) {
        for (const char* relationship : {"physics:body1", "physics:body0"}) {
            const auto target = ReadRelationshipTarget(articulation_root, relationship);
            if (target.has_value()) {
                const auto link = link_by_path.find(*target);
                if (link != link_by_path.end()) {
                    articulation_body = links[link->second].prim;
                    break;
                }
            }
        }
    } else {
        const auto root_link = std::find_if(
                links.begin(), links.end(),
                [&articulation_root, &child_link_paths](const PhysicsLinkImportData& link) {
                    return link.prim.GetPath().HasPrefix(articulation_root.GetPath()) &&
                           !child_link_paths.contains(link.path);
                });
        if (root_link != links.end()) {
            articulation_body = root_link->prim;
        }
    }
    if (!articulation_body) {
        LOG_ERROR("USD articulation '{}' has no root rigid body below '{}'.",
                  path, articulation_root_path);
        return {};
    }

    Ref<PackedScene> packed_scene = MakeRef<PackedScene>();
    Ref<SceneState> state = packed_scene->GetState();
    SceneState::NodeData robot_node;
    robot_node.type = "Robot3D";
    robot_node.name = model_prim.GetName().GetString();
    AddProperty(robot_node, "source_path", path);
    Affine3 robot_transform = ScaleTranslation(ToGobotTransform(model_world), meters_per_unit);
    for (int column = 0; column < 3; ++column) {
        const RealType length = robot_transform.linear().col(column).norm();
        if (length > CMP_EPSILON) {
            robot_transform.linear().col(column) /= length;
        }
    }
    if (pxr::UsdGeomGetStageUpAxis(stage) == pxr::UsdGeomTokens->y) {
        Affine3 up_axis = Affine3::Identity();
        up_axis.SetEulerAngle({DEG_TO_RAD(90.0), 0.0, 0.0}, EulerOrder::SXYZ);
        robot_transform = up_axis * robot_transform;
    }
    AddRigidTransformProperties(robot_node, robot_transform, model_prim.GetPath().GetString());
    const int robot_index = state->AddNode(robot_node);

    std::unordered_map<std::string, Ref<PBRMaterial3D>> material_cache;
    auto emit_owned_prims = [&](PhysicsLinkImportData& link) {
        const auto owned = owned_prims.find(link.path);
        if (owned == owned_prims.end()) {
            return;
        }
        for (const pxr::UsdPrim& prim : owned->second) {
            const bool collision = prim.HasAPI<pxr::UsdPhysicsCollisionAPI>();
            const pxr::UsdGeomMesh usd_mesh(prim);
            if (IsSupportedVisualMesh(prim)) {
                SceneState::NodeData visual;
                visual.type = "MeshInstance3D";
                visual.name = prim.GetName().GetString();
                visual.parent = link.node_index;
                Affine3 relative = ToGobotTransform(
                        xform_cache.GetLocalToWorldTransform(prim) * link.world.GetInverse());
                relative.translation() *= meters_per_unit;
                relative.linear() *= meters_per_unit;
                AddAffineTransformProperties(visual, relative, prim.GetPath().GetString(), true);
                if (const pxr::UsdGeomImageable imageable(prim);
                    imageable && imageable.ComputeVisibility() == pxr::UsdGeomTokens->invisible) {
                    AddProperty(visual, "visible", false);
                }
                if (const Ref<ArrayMesh> mesh = ImportMesh(usd_mesh, &material_cache); mesh.IsValid()) {
                    const Ref<Material> material = DetachSingleSurfaceMaterial(mesh);
                    AddProperty(visual, "mesh", dynamic_pointer_cast<Mesh>(mesh));
                    if (material.IsValid()) {
                        AddProperty(visual, "material", material);
                    }
                }
                state->AddNode(visual);
            }
            if (!collision) {
                continue;
            }

            SceneState::NodeData collision_node;
            collision_node.type = "CollisionShape3D";
            collision_node.name = link.name + "_" + prim.GetName().GetString() + "_collision";
            collision_node.parent = link.node_index;
            Affine3 relative = ToGobotTransform(
                    xform_cache.GetLocalToWorldTransform(prim) * link.world.GetInverse());
            relative.translation() *= meters_per_unit;
            Vector3 scale = relative.GetScale().cwiseAbs();
            for (int column = 0; column < 3; ++column) {
                const RealType length = relative.linear().col(column).norm();
                if (length > CMP_EPSILON) {
                    relative.linear().col(column) /= length;
                }
            }
            Affine3 rigid = relative;
            rigid.Orthonormalize();

            Ref<Shape3D> shape;
            if (usd_mesh) {
                pxr::TfToken approximation;
                ReadAttribute(prim, "physics:approximation", &approximation);
                if (approximation == pxr::TfToken("convexDecomposition")) {
                    LOG_WARN("USD collision mesh '{}' requests convex decomposition; Gobot currently "
                             "imports it as one convex hull.",
                             prim.GetPath().GetString());
                } else if (!approximation.IsEmpty() && approximation != pxr::TfToken("convexHull")) {
                    LOG_WARN("USD collision mesh '{}' uses unsupported approximation '{}'; using convex hull.",
                             prim.GetPath().GetString(), approximation.GetString());
                }
                Ref<ArrayMesh> mesh = ImportMesh(usd_mesh, &material_cache);
                ClearSurfaceMaterials(mesh);
                mesh = ScaledMesh(mesh, scale * meters_per_unit);
                if (mesh.IsValid()) {
                    Ref<ConvexMeshShape3D> convex = MakeRef<ConvexMeshShape3D>();
                    convex->SetMesh(dynamic_pointer_cast<Mesh>(mesh));
                    shape = dynamic_pointer_cast<Shape3D>(convex);
                }
            } else if (const pxr::UsdGeomSphere sphere(prim); sphere) {
                double radius = 0.5;
                sphere.GetRadiusAttr().Get(&radius);
                Ref<SphereShape3D> sphere_shape = MakeRef<SphereShape3D>();
                sphere_shape->SetRadius(static_cast<float>(radius * scale.maxCoeff() * meters_per_unit));
                shape = dynamic_pointer_cast<Shape3D>(sphere_shape);
            } else if (const pxr::UsdGeomCylinder cylinder(prim); cylinder) {
                double radius = 0.5;
                double height = 1.0;
                pxr::TfToken axis("Z");
                cylinder.GetRadiusAttr().Get(&radius);
                cylinder.GetHeightAttr().Get(&height);
                cylinder.GetAxisAttr().Get(&axis);
                if (axis == pxr::TfToken("X")) {
                    Affine3 correction = Affine3::Identity();
                    correction.SetEulerAngle({0.0, DEG_TO_RAD(90.0), 0.0}, EulerOrder::SXYZ);
                    rigid.linear() *= correction.linear();
                    std::swap(scale.x(), scale.z());
                } else if (axis == pxr::TfToken("Y")) {
                    Affine3 correction = Affine3::Identity();
                    correction.SetEulerAngle({DEG_TO_RAD(-90.0), 0.0, 0.0}, EulerOrder::SXYZ);
                    rigid.linear() *= correction.linear();
                    std::swap(scale.y(), scale.z());
                }
                Ref<CylinderShape3D> cylinder_shape = MakeRef<CylinderShape3D>();
                cylinder_shape->SetRadius(static_cast<float>(radius * std::max(scale.x(), scale.y()) * meters_per_unit));
                cylinder_shape->SetHeight(static_cast<float>(height * scale.z() * meters_per_unit));
                shape = dynamic_pointer_cast<Shape3D>(cylinder_shape);
            } else if (const pxr::UsdGeomCube cube(prim); cube) {
                double size = 2.0;
                cube.GetSizeAttr().Get(&size);
                Ref<BoxShape3D> box = MakeRef<BoxShape3D>();
                box->SetSize(scale * static_cast<RealType>(size * meters_per_unit));
                shape = dynamic_pointer_cast<Shape3D>(box);
            }
            if (!shape.IsValid()) {
                LOG_WARN("USD collision prim '{}' has unsupported or empty geometry and was skipped.",
                         prim.GetPath().GetString());
                continue;
            }
            AddRigidTransformProperties(collision_node, rigid, prim.GetPath().GetString());
            AddProperty(collision_node, "shape", shape);
            AddPhysicsContactProperties(collision_node, prim);
            state->AddNode(collision_node);
        }
    };

    std::function<int(const std::string&, int, const Affine3&)> emit_link;
    emit_link = [&](const std::string& link_path, int parent, const Affine3& local_transform) -> int {
        const auto link_iter = link_by_path.find(link_path);
        if (link_iter == link_by_path.end()) {
            return -1;
        }
        PhysicsLinkImportData& link = links[link_iter->second];
        if (link.node_index >= 0) {
            return link.node_index;
        }
        SceneState::NodeData link_node;
        link_node.type = "Link3D";
        link_node.name = link.name;
        link_node.parent = parent;
        AddRigidTransformProperties(link_node, local_transform, link.path);
        const pxr::UsdPhysicsMassAPI mass_api(link.prim);
        float mass = 0.0f;
        pxr::GfVec3f center_of_mass(0.0f);
        pxr::GfVec3f diagonal_inertia(0.0f);
        pxr::GfQuatf principal_axes(1.0f);
        const bool has_mass = mass_api && mass_api.GetMassAttr().Get(&mass) &&
                              std::isfinite(mass) && mass > 0.0f;
        if (mass_api) {
            mass_api.GetCenterOfMassAttr().Get(&center_of_mass);
            mass_api.GetDiagonalInertiaAttr().Get(&diagonal_inertia);
            mass_api.GetPrincipalAxesAttr().Get(&principal_axes);
        }
        if (!std::isfinite(mass)) {
            mass = 0.0f;
        }
        if (!ToGobotVector3(center_of_mass).allFinite()) {
            center_of_mass = pxr::GfVec3f(0.0f);
        }
        const Vector3 imported_inertia = ToGobotVector3(diagonal_inertia);
        if (!imported_inertia.allFinite() || (imported_inertia.array() < 0.0).any()) {
            diagonal_inertia = pxr::GfVec3f(0.0f);
        }
        AddProperty(link_node, "has_inertial", has_mass);
        AddProperty(link_node, "mass", static_cast<RealType>(mass) * kilograms_per_unit);
        const Vector3 center_of_mass_meters = ToGobotVector3(center_of_mass) * meters_per_unit;
        const Vector3 inertia_si = ToGobotVector3(diagonal_inertia) * inertia_scale;
        AddProperty(link_node, "center_of_mass", center_of_mass_meters);
        AddProperty(link_node, "inertia_orientation", ToGobotQuaternion(principal_axes));
        AddProperty(link_node, "inertia_diagonal", inertia_si);
        AddProperty(link_node, "inertia_off_diagonal", Vector3{0.0, 0.0, 0.0});
        AddProperty(link_node, "role", LinkRole::Physical);
        link.node_index = state->AddNode(link_node);
        emit_owned_prims(link);

        const auto outgoing = outgoing_joints.find(link_path);
        if (outgoing == outgoing_joints.end()) {
            return link.node_index;
        }
        for (const std::size_t joint_index : outgoing->second) {
            PhysicsJointImportData& joint = joints[joint_index];
            SceneState::NodeData joint_node;
            joint_node.type = "Joint3D";
            joint_node.name = joint.prim.GetName().GetString();
            joint_node.parent = link.node_index;
            joint.frame0.translation() *= meters_per_unit;
            joint.frame1.translation() *= meters_per_unit;
            AddRigidTransformProperties(joint_node, joint.frame0, joint.prim.GetPath().GetString());
            const auto child_iter = link_by_path.find(joint.child_path);
            if (child_iter == link_by_path.end()) {
                continue;
            }
            AddProperty(joint_node, "parent_link", link.name);
            AddProperty(joint_node, "child_link", links[child_iter->second].name);

            JointType joint_type = JointType::Fixed;
            pxr::TfToken axis_token("X");
            float lower_limit = 0.0f;
            float upper_limit = 0.0f;
            if (const pxr::UsdPhysicsRevoluteJoint revolute(joint.prim); revolute) {
                joint_type = JointType::Revolute;
                revolute.GetAxisAttr().Get(&axis_token);
                revolute.GetLowerLimitAttr().Get(&lower_limit);
                revolute.GetUpperLimitAttr().Get(&upper_limit);
                lower_limit = static_cast<float>(DEG_TO_RAD(lower_limit));
                upper_limit = static_cast<float>(DEG_TO_RAD(upper_limit));
            } else if (const pxr::UsdPhysicsPrismaticJoint prismatic(joint.prim); prismatic) {
                joint_type = JointType::Prismatic;
                prismatic.GetAxisAttr().Get(&axis_token);
                prismatic.GetLowerLimitAttr().Get(&lower_limit);
                prismatic.GetUpperLimitAttr().Get(&upper_limit);
                lower_limit *= meters_per_unit;
                upper_limit *= meters_per_unit;
            }
            AddProperty(joint_node, "joint_type", joint_type);
            AddProperty(joint_node, "axis", AxisForToken(axis_token));
            AddProperty(joint_node, "lower_limit", static_cast<RealType>(lower_limit));
            AddProperty(joint_node, "upper_limit", static_cast<RealType>(upper_limit));

            double force_min = 0.0;
            double force_max = 0.0;
            ReadAttribute(joint.prim, "mjc:actuatorfrcrange:min", &force_min);
            ReadAttribute(joint.prim, "mjc:actuatorfrcrange:max", &force_max);
            AddProperty(joint_node, "force_lower_limit", static_cast<RealType>(force_min));
            AddProperty(joint_node, "force_upper_limit", static_cast<RealType>(force_max));
            AddProperty(joint_node, "effort_limit",
                        static_cast<RealType>(std::max(std::abs(force_min), std::abs(force_max))));
            double scalar = 0.0;
            if (ReadAttribute(joint.prim, "mjc:armature", &scalar)) {
                AddProperty(joint_node, "armature", static_cast<RealType>(scalar));
            }
            if (ReadAttribute(joint.prim, "mjc:frictionloss", &scalar)) {
                AddProperty(joint_node, "friction_loss", static_cast<RealType>(scalar));
            }
            if (ReadAttribute(joint.prim, "mjc:damping", &scalar)) {
                AddProperty(joint_node, "damping", static_cast<RealType>(scalar));
            }

            const pxr::TfToken drive_name = joint_type == JointType::Prismatic
                                                     ? pxr::TfToken("linear")
                                                     : pxr::TfToken("angular");
            const pxr::UsdPhysicsDriveAPI drive = pxr::UsdPhysicsDriveAPI::Get(joint.prim, drive_name);
            float stiffness = 0.0f;
            float damping = 0.0f;
            float max_force = 0.0f;
            float target_position = 0.0f;
            const auto actuator_iter = actuator_by_joint_path.find(joint.prim.GetPath().GetString());
            const bool has_affine_actuator =
                    actuator_iter != actuator_by_joint_path.end() &&
                    AddAffineActuatorProperties(actuator_iter->second, joint_node);
            if (actuator_iter != actuator_by_joint_path.end() && !has_affine_actuator) {
                LOG_WARN("USD actuator '{}' uses features outside Gobot's affine joint actuator contract; "
                         "importing it as a direct motor.",
                         actuator_iter->second.GetPath().GetString());
            }

            if (drive) {
                drive.GetStiffnessAttr().Get(&stiffness);
                drive.GetDampingAttr().Get(&damping);
                drive.GetMaxForceAttr().Get(&max_force);
                drive.GetTargetPositionAttr().Get(&target_position);
                AddProperty(joint_node, "drive_mode",
                            stiffness > 0.0f ? JointDriveMode::Position : JointDriveMode::Velocity);
                AddProperty(joint_node, "drive_stiffness", static_cast<RealType>(stiffness));
                AddProperty(joint_node, "drive_damping", static_cast<RealType>(damping));
                if (max_force > 0.0f && std::isfinite(max_force)) {
                    AddProperty(joint_node, "effort_limit", static_cast<RealType>(max_force));
                    AddProperty(joint_node, "force_lower_limit", static_cast<RealType>(-max_force));
                    AddProperty(joint_node, "force_upper_limit", static_cast<RealType>(max_force));
                }
                const RealType initial = joint_type == JointType::Revolute
                                                 ? DEG_TO_RAD(target_position)
                                                 : target_position * meters_per_unit;
                AddProperty(joint_node, "joint_position", initial);
                AddProperty(joint_node, "initial_position", initial);
            } else if (actuator_iter != actuator_by_joint_path.end() && !has_affine_actuator) {
                AddProperty(joint_node, "drive_mode", JointDriveMode::Motor);
            }
            const int emitted_joint = state->AddNode(joint_node);
            emit_link(joint.child_path, emitted_joint, joint.frame1.inverse());
        }
        return link.node_index;
    };

    for (PhysicsLinkImportData& link : links) {
        if (child_link_paths.contains(link.path)) {
            continue;
        }
        Affine3 relative = ScaleTranslation(
                ToGobotTransform(link.world * model_world.GetInverse()), meters_per_unit);
        if (link.path == articulation_body.GetPath().GetString() &&
            !world_anchored_link_paths.contains(link.path)) {
            SceneState::NodeData floating;
            floating.type = "Joint3D";
            floating.name = "floating_base_joint";
            floating.parent = robot_index;
            AddRigidTransformProperties(floating, relative, link.path);
            AddProperty(floating, "joint_type", JointType::Floating);
            AddProperty(floating, "parent_link", std::string{});
            AddProperty(floating, "child_link", link.name);
            const int floating_index = state->AddNode(floating);
            emit_link(link.path, floating_index, Affine3::Identity());
        } else {
            emit_link(link.path, robot_index, relative);
        }
    }
    for (PhysicsLinkImportData& link : links) {
        if (link.node_index < 0) {
            Affine3 relative = ScaleTranslation(
                    ToGobotTransform(link.world * model_world.GetInverse()), meters_per_unit);
            emit_link(link.path, robot_index, relative);
        }
    }

    LOG_INFO("USD '{}' imported articulation '{}' with {} links, {} joints, and {} scene nodes.",
             path, robot_node.name, links.size(), joints.size(), state->GetNodeCount());
    GOB_UNUSED(global_path);
    return packed_scene;
}

#endif

} // namespace

bool ResourceFormatLoaderUSD::IsOpenUSDAvailable() {
#ifdef GOBOT_HAS_OPENUSD
    return true;
#else
    return false;
#endif
}

Ref<Resource> ResourceFormatLoaderUSD::Load(const std::string& path,
                                            const std::string& original_path,
                                            CacheMode cache_mode) {
    (void)original_path;
    (void)cache_mode;

#ifndef GOBOT_HAS_OPENUSD
    LOG_ERROR("OpenUSD support is disabled. Reconfigure with -DGOB_BUILD_OPENUSD=ON to load {}.", path);
    return {};
#else
    const std::string global_path = ProjectSettings::GetInstance()->GlobalizePath(path);
    pxr::UsdStageRefPtr stage = pxr::UsdStage::Open(global_path);
    if (!stage) {
        LOG_ERROR("Cannot open USD stage: {}.", path);
        return {};
    }

    if (IsPhysicsArticulationStage(stage)) {
        return ImportPhysicsArticulation(stage, original_path.empty() ? path : original_path, global_path);
    }

    Ref<PackedScene> packed_scene = MakeRef<PackedScene>();
    Ref<SceneState> state = packed_scene->GetState();

    SceneState::NodeData root;
    root.type = "Node3D";
    root.name = MakeSceneName(global_path);
    const RealType meters_per_unit = static_cast<RealType>(pxr::UsdGeomGetStageMetersPerUnit(stage));
    AddProperty(root, "scale", Vector3{meters_per_unit, meters_per_unit, meters_per_unit});
    if (pxr::UsdGeomGetStageUpAxis(stage) == pxr::UsdGeomTokens->y) {
        AddProperty(root, "rotation_degrees", Vector3{90.0f, 0.0f, 0.0f});
    }
    const int root_index = state->AddNode(root);

    std::unordered_map<std::string, int> prim_to_node;
    std::unordered_map<std::string, pxr::GfMatrix4d> xform_world;
    std::unordered_map<std::string, Ref<PBRMaterial3D>> material_cache;
    pxr::UsdGeomXformCache xform_cache(pxr::UsdTimeCode::Default());
    bool has_usd_physics = false;

    // UsdStage::Open composes layers and loads payloads by default. Build the
    // imported hierarchy from supported visual prims in that composed stage so
    // physics-only scopes, joints, and custom actuator metadata stay out of the
    // user-facing SceneTree.
    std::unordered_set<std::string> included_paths;
    for (const pxr::UsdPrim& prim :
         pxr::UsdPrimRange::Stage(stage, pxr::UsdTraverseInstanceProxies())) {
        if (!prim.IsActive() || prim.IsPseudoRoot()) {
            continue;
        }
        has_usd_physics = has_usd_physics || PrimContainsPhysicsSchema(prim);
        if (IsSupportedVisualMesh(prim)) {
            IncludePrimAndAncestors(prim, &included_paths);
        }
    }

    for (const pxr::UsdPrim& prim :
         pxr::UsdPrimRange::Stage(stage, pxr::UsdTraverseInstanceProxies())) {
        if (!prim.IsActive() || prim.IsPseudoRoot()) {
            continue;
        }

        const std::string prim_path = prim.GetPath().GetString();
        if (!included_paths.contains(prim_path)) {
            continue;
        }
        if (pxr::UsdShadeMaterial(prim) || pxr::UsdShadeShader(prim) ||
            pxr::UsdShadeNodeGraph(prim)) {
            continue;
        }

        SceneState::NodeData node;
        node.name = prim.GetName().GetString();
        const pxr::UsdGeomMesh usd_mesh(prim);
        const pxr::UsdGeomXformable xformable(prim);
        node.type = usd_mesh ? "MeshInstance3D" : (xformable ? "Node3D" : "Node");
        node.parent = root_index;

        pxr::UsdPrim parent = prim.GetParent();
        while (parent && !parent.IsPseudoRoot()) {
            if (const auto found = prim_to_node.find(parent.GetPath().GetString());
                found != prim_to_node.end()) {
                node.parent = found->second;
                break;
            }
            parent = parent.GetParent();
        }

        if (xformable) {
            const pxr::GfMatrix4d world = xform_cache.GetLocalToWorldTransform(prim);
            pxr::GfMatrix4d relative = world;
            pxr::UsdPrim transform_parent = prim.GetParent();
            while (transform_parent && !transform_parent.IsPseudoRoot()) {
                const auto found = xform_world.find(transform_parent.GetPath().GetString());
                if (found != xform_world.end()) {
                    relative = world * found->second.GetInverse();
                    break;
                }
                transform_parent = transform_parent.GetParent();
            }
            AddTransformProperties(node, relative, prim_path);
            xform_world.emplace(prim_path, world);

            if (const pxr::UsdGeomImageable imageable(prim);
                imageable && imageable.ComputeVisibility() == pxr::UsdGeomTokens->invisible) {
                AddProperty(node, "visible", false);
            }
        }

        if (usd_mesh) {
            const Ref<ArrayMesh> mesh = ImportMesh(usd_mesh, &material_cache);
            if (mesh.IsValid()) {
                const Ref<Material> material = DetachSingleSurfaceMaterial(mesh);
                AddProperty(node, "mesh", dynamic_pointer_cast<Mesh>(mesh));
                if (material.IsValid()) {
                    AddProperty(node, "material", material);
                }
            }
        }

        const int node_index = state->AddNode(node);
        prim_to_node.emplace(prim_path, node_index);
    }

    if (has_usd_physics) {
        LOG_WARN("USD stage '{}' contains UsdPhysics schemas. This importer currently preserves visual "
                 "scene data only; rigid bodies, colliders, joints, and drives were not imported.",
                 path);
    }

    return packed_scene;
#endif
}

void ResourceFormatLoaderUSD::GetRecognizedExtensionsForType(const std::string& type,
                                                             std::vector<std::string>* extensions) const {
    if (type.empty() || HandlesType(type)) {
        GetRecognizedExtensions(extensions);
    }
}

void ResourceFormatLoaderUSD::GetRecognizedExtensions(std::vector<std::string>* extensions) const {
    extensions->push_back("usd");
    extensions->push_back("usda");
    extensions->push_back("usdc");
}

bool ResourceFormatLoaderUSD::HandlesType(const std::string& type) const {
    return type.empty() || type == "PackedScene";
}

} // namespace gobot

GOBOT_REGISTRATION {

    Class_<gobot::ResourceFormatLoaderUSD>("ResourceFormatLoaderUSD")
            .constructor()(CtorAsRawPtr)
            .method("is_openusd_available", &gobot::ResourceFormatLoaderUSD::IsOpenUSDAvailable);

};
