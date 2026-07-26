/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/core/io/resource_format_usd.hpp"

#include "gobot/core/config/project_setting.hpp"
#include "gobot/core/math/geometry.hpp"
#include "gobot/core/registration.hpp"
#include "gobot/log.hpp"
#include "gobot/scene/resources/array_mesh.hpp"
#include "gobot/scene/resources/material.hpp"
#include "gobot/scene/resources/packed_scene.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#ifdef GOBOT_HAS_OPENUSD
#include <pxr/base/gf/matrix4d.h>
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
#include <pxr/usd/usdGeom/imageable.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/metrics.h>
#include <pxr/usd/usdGeom/primvar.h>
#include <pxr/usd/usdGeom/primvarsAPI.h>
#include <pxr/usd/usdGeom/tokens.h>
#include <pxr/usd/usdGeom/xformCache.h>
#include <pxr/usd/usdGeom/xformable.h>
#include <pxr/usd/usdShade/material.h>
#include <pxr/usd/usdShade/materialBindingAPI.h>
#include <pxr/usd/usdShade/nodeGraph.h>
#include <pxr/usd/usdShade/input.h>
#include <pxr/usd/usdShade/shader.h>
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

std::string MakeSceneName(const std::string& path) {
    const std::string stem = std::filesystem::path(path).stem().string();
    return stem.empty() ? "USDScene" : stem;
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

    for (const pxr::UsdPrim& prim :
         pxr::UsdPrimRange::Stage(stage, pxr::UsdTraverseInstanceProxies())) {
        if (!prim.IsActive() || prim.IsPseudoRoot()) {
            continue;
        }

        const std::string prim_type = prim.GetTypeName().GetString();
        has_usd_physics = has_usd_physics || prim_type.starts_with("Physics");
        if (!has_usd_physics) {
            for (const pxr::TfToken& schema : prim.GetAppliedSchemas()) {
                if (schema.GetString().starts_with("Physics")) {
                    has_usd_physics = true;
                    break;
                }
            }
        }
        if (pxr::UsdShadeMaterial(prim) || pxr::UsdShadeShader(prim) ||
            pxr::UsdShadeNodeGraph(prim)) {
            continue;
        }

        if (const pxr::UsdGeomImageable imageable(prim); imageable) {
            const pxr::TfToken purpose = imageable.ComputePurpose();
            if (purpose == pxr::UsdGeomTokens->proxy || purpose == pxr::UsdGeomTokens->guide) {
                continue;
            }
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
            AddTransformProperties(node, relative, prim.GetPath().GetString());
            xform_world.emplace(prim.GetPath().GetString(), world);

            if (const pxr::UsdGeomImageable imageable(prim);
                imageable && imageable.ComputeVisibility() == pxr::UsdGeomTokens->invisible) {
                AddProperty(node, "visible", false);
            }
        }

        if (usd_mesh) {
            const Ref<ArrayMesh> mesh = ImportMesh(usd_mesh, &material_cache);
            if (mesh.IsValid()) {
                AddProperty(node, "mesh", dynamic_pointer_cast<Mesh>(mesh));
            }
        }

        const int node_index = state->AddNode(node);
        prim_to_node.emplace(prim.GetPath().GetString(), node_index);
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
