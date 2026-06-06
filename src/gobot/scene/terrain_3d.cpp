/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/scene/terrain_3d.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

#include "gobot/core/registration.hpp"

namespace gobot {
namespace {

Affine3 BoxTransform(const TerrainBox& box) {
    Affine3 transform = Affine3::Identity();
    transform.translation() = box.center;
    transform.SetEulerAngle({
            DEG_TO_RAD(box.rotation_degrees.x()),
            DEG_TO_RAD(box.rotation_degrees.y()),
            DEG_TO_RAD(box.rotation_degrees.z())
    }, EulerOrder::SXYZ);
    return transform;
}

void AppendQuad(std::vector<Vector3>& vertices,
                std::vector<std::uint32_t>& indices,
                std::vector<Vector3>& normals,
                std::vector<Color>* colors,
                const Color& color,
                const Vector3& a,
                const Vector3& b,
                const Vector3& c,
                const Vector3& d) {
    const std::uint32_t base = static_cast<std::uint32_t>(vertices.size());
    Vector3 normal = (b - a).cross(c - a);
    const RealType length = normal.norm();
    if (length <= CMP_EPSILON) {
        normal = Vector3::UnitZ();
    } else {
        normal /= length;
    }
    vertices.push_back(a);
    vertices.push_back(b);
    vertices.push_back(c);
    vertices.push_back(d);
    normals.push_back(normal);
    normals.push_back(normal);
    normals.push_back(normal);
    normals.push_back(normal);
    if (colors != nullptr) {
        colors->push_back(color);
        colors->push_back(color);
        colors->push_back(color);
        colors->push_back(color);
    }
    indices.insert(indices.end(), {base, base + 1, base + 2, base, base + 2, base + 3});
}

void AppendBoxMesh(std::vector<Vector3>& vertices,
                   std::vector<std::uint32_t>& indices,
                   std::vector<Vector3>& normals,
                   std::vector<Color>* colors,
                   const TerrainBox& box) {
    const Vector3 half = box.size.cwiseMax(Vector3::Zero()) * 0.5;
    const Affine3 transform = BoxTransform(box);
    const std::array<Vector3, 8> corners = {
            transform * Vector3{-half.x(), -half.y(), -half.z()},
            transform * Vector3{ half.x(), -half.y(), -half.z()},
            transform * Vector3{ half.x(),  half.y(), -half.z()},
            transform * Vector3{-half.x(),  half.y(), -half.z()},
            transform * Vector3{-half.x(), -half.y(),  half.z()},
            transform * Vector3{ half.x(), -half.y(),  half.z()},
            transform * Vector3{ half.x(),  half.y(),  half.z()},
            transform * Vector3{-half.x(),  half.y(),  half.z()},
    };

    AppendQuad(vertices, indices, normals, colors, box.color, corners[4], corners[5], corners[6], corners[7]);
    AppendQuad(vertices, indices, normals, colors, box.color, corners[1], corners[0], corners[3], corners[2]);
    AppendQuad(vertices, indices, normals, colors, box.color, corners[0], corners[4], corners[7], corners[3]);
    AppendQuad(vertices, indices, normals, colors, box.color, corners[5], corners[1], corners[2], corners[6]);
    AppendQuad(vertices, indices, normals, colors, box.color, corners[3], corners[7], corners[6], corners[2]);
    AppendQuad(vertices, indices, normals, colors, box.color, corners[0], corners[1], corners[5], corners[4]);
}

Vector3 HeightFieldVertex(const TerrainHeightField& heightfield, int row, int col) {
    const RealType x = heightfield.cols <= 1
                               ? 0.0
                               : (-heightfield.size.x() * 0.5 +
                                  heightfield.size.x() * static_cast<RealType>(col) /
                                          static_cast<RealType>(heightfield.cols - 1));
    const RealType y = heightfield.rows <= 1
                               ? 0.0
                               : (-heightfield.size.y() * 0.5 +
                                  heightfield.size.y() * static_cast<RealType>(row) /
                                          static_cast<RealType>(heightfield.rows - 1));
    const std::size_t index = static_cast<std::size_t>(row * heightfield.cols + col);
    const RealType z = (index < heightfield.heights.size() ? heightfield.heights[index] : 0.0) +
                       heightfield.z_offset;
    return heightfield.center + Vector3{x, y, z};
}

Color HsvToRgb(RealType hue, RealType saturation, RealType value, RealType alpha = 1.0) {
    const RealType h = std::fmod(std::max<RealType>(hue, 0.0), 1.0) * 6.0;
    const int sector = static_cast<int>(std::floor(h));
    const RealType fraction = h - static_cast<RealType>(sector);
    const RealType p = value * (1.0 - saturation);
    const RealType q = value * (1.0 - saturation * fraction);
    const RealType t = value * (1.0 - saturation * (1.0 - fraction));

    RealType r = value;
    RealType g = t;
    RealType b = p;
    switch (sector % 6) {
        case 0: r = value; g = t; b = p; break;
        case 1: r = q; g = value; b = p; break;
        case 2: r = p; g = value; b = t; break;
        case 3: r = p; g = q; b = value; break;
        case 4: r = t; g = p; b = value; break;
        case 5: r = value; g = p; b = q; break;
    }
    return {
            static_cast<float>(std::clamp(r, static_cast<RealType>(0.0), static_cast<RealType>(1.0))),
            static_cast<float>(std::clamp(g, static_cast<RealType>(0.0), static_cast<RealType>(1.0))),
            static_cast<float>(std::clamp(b, static_cast<RealType>(0.0), static_cast<RealType>(1.0))),
            static_cast<float>(std::clamp(alpha, static_cast<RealType>(0.0), static_cast<RealType>(1.0))),
    };
}

Color PaletteHeightColor(RealType elevation) {
    const RealType e = std::clamp(elevation, static_cast<RealType>(0.0), static_cast<RealType>(1.0));
    return HsvToRgb(static_cast<RealType>(0.5) - e * static_cast<RealType>(0.45),
                   static_cast<RealType>(0.6) - e * static_cast<RealType>(0.2),
                   static_cast<RealType>(0.4) + e * static_cast<RealType>(0.3));
}

void AppendHeightFieldMesh(std::vector<Vector3>& vertices,
                           std::vector<std::uint32_t>& indices,
                           std::vector<Vector3>& normals,
                           std::vector<Color>* colors,
                           const TerrainHeightField& heightfield) {
    if (heightfield.rows < 2 || heightfield.cols < 2) {
        return;
    }

    const std::uint32_t base = static_cast<std::uint32_t>(vertices.size());
    vertices.reserve(vertices.size() + static_cast<std::size_t>(heightfield.rows * heightfield.cols));
    normals.resize(normals.size() + static_cast<std::size_t>(heightfield.rows * heightfield.cols), Vector3::Zero());
    for (int row = 0; row < heightfield.rows; ++row) {
        for (int col = 0; col < heightfield.cols; ++col) {
            vertices.push_back(HeightFieldVertex(heightfield, row, col));
            if (colors != nullptr) {
                const std::size_t index = static_cast<std::size_t>(row * heightfield.cols + col);
                const RealType elevation = index < heightfield.normalized_elevation.size()
                        ? heightfield.normalized_elevation[index]
                        : (index < heightfield.heights.size() ? heightfield.heights[index] : 0.0);
                colors->push_back(PaletteHeightColor(elevation));
            }
        }
    }

    for (int row = 0; row < heightfield.rows - 1; ++row) {
        for (int col = 0; col < heightfield.cols - 1; ++col) {
            const std::uint32_t i0 = base + static_cast<std::uint32_t>(row * heightfield.cols + col);
            const std::uint32_t i1 = i0 + 1;
            const std::uint32_t i2 = i0 + static_cast<std::uint32_t>(heightfield.cols);
            const std::uint32_t i3 = i2 + 1;
            indices.insert(indices.end(), {i0, i1, i3, i0, i3, i2});

            const Vector3& v0 = vertices[i0];
            const Vector3& v1 = vertices[i1];
            const Vector3& v2 = vertices[i2];
            const Vector3& v3 = vertices[i3];
            Vector3 normal_a = (v1 - v0).cross(v3 - v0);
            Vector3 normal_b = (v3 - v0).cross(v2 - v0);
            if (normal_a.norm() > CMP_EPSILON) {
                normal_a.normalize();
                normals[i0] += normal_a;
                normals[i1] += normal_a;
                normals[i3] += normal_a;
            }
            if (normal_b.norm() > CMP_EPSILON) {
                normal_b.normalize();
                normals[i0] += normal_b;
                normals[i3] += normal_b;
                normals[i2] += normal_b;
            }
        }
    }

    const std::size_t count = static_cast<std::size_t>(heightfield.rows * heightfield.cols);
    for (std::size_t offset = 0; offset < count; ++offset) {
        Vector3& normal = normals[static_cast<std::size_t>(base) + offset];
        const RealType length = normal.norm();
        if (length <= CMP_EPSILON) {
            normal = Vector3::UnitZ();
        } else {
            normal /= length;
        }
    }
}

Affine3 MeshPatchTransform(const TerrainMeshPatch& mesh_patch) {
    Affine3 transform = Affine3::Identity();
    transform.translation() = mesh_patch.center;
    transform.SetEulerAngle({
            DEG_TO_RAD(mesh_patch.rotation_degrees.x()),
            DEG_TO_RAD(mesh_patch.rotation_degrees.y()),
            DEG_TO_RAD(mesh_patch.rotation_degrees.z())
    }, EulerOrder::SXYZ);
    return transform;
}

void AppendMeshPatch(std::vector<Vector3>& vertices,
                     std::vector<std::uint32_t>& indices,
                     std::vector<Vector3>& normals,
                     std::vector<Color>* colors,
                     const TerrainMeshPatch& mesh_patch) {
    if (mesh_patch.vertices.empty() || mesh_patch.indices.empty()) {
        return;
    }

    const Affine3 transform = MeshPatchTransform(mesh_patch);
    const std::uint32_t base = static_cast<std::uint32_t>(vertices.size());
    vertices.reserve(vertices.size() + mesh_patch.vertices.size());
    normals.resize(normals.size() + mesh_patch.vertices.size(), Vector3::Zero());
    if (colors != nullptr) {
        colors->reserve(colors->size() + mesh_patch.vertices.size());
    }

    for (const Vector3& vertex : mesh_patch.vertices) {
        vertices.push_back(transform * vertex);
        if (colors != nullptr) {
            colors->push_back(mesh_patch.color);
        }
    }

    for (std::size_t index = 0; index + 2 < mesh_patch.indices.size(); index += 3) {
        const std::uint32_t i0 = base + mesh_patch.indices[index + 0];
        const std::uint32_t i1 = base + mesh_patch.indices[index + 1];
        const std::uint32_t i2 = base + mesh_patch.indices[index + 2];
        if (i0 >= vertices.size() || i1 >= vertices.size() || i2 >= vertices.size()) {
            continue;
        }
        indices.insert(indices.end(), {i0, i1, i2});
        Vector3 normal = (vertices[i1] - vertices[i0]).cross(vertices[i2] - vertices[i0]);
        if (normal.norm() > CMP_EPSILON) {
            normal.normalize();
            normals[i0] += normal;
            normals[i1] += normal;
            normals[i2] += normal;
        }
    }

    for (std::size_t index = static_cast<std::size_t>(base); index < normals.size(); ++index) {
        Vector3& normal = normals[index];
        const RealType length = normal.norm();
        if (length <= CMP_EPSILON) {
            normal = Vector3::UnitZ();
        } else {
            normal /= length;
        }
    }
}

Color LerpColor(const Color& from, const Color& to, RealType weight) {
    const RealType t = std::clamp(weight, static_cast<RealType>(0.0), static_cast<RealType>(1.0));
    return {
            static_cast<float>(from.red() + (to.red() - from.red()) * t),
            static_cast<float>(from.green() + (to.green() - from.green()) * t),
            static_cast<float>(from.blue() + (to.blue() - from.blue()) * t),
            static_cast<float>(from.alpha() + (to.alpha() - from.alpha()) * t),
    };
}

Color ScaleRgb(const Color& color, RealType scale) {
    const RealType s = std::clamp(scale, static_cast<RealType>(0.0), static_cast<RealType>(2.0));
    return {
            static_cast<float>(std::clamp(static_cast<RealType>(color.red()) * s,
                                          static_cast<RealType>(0.0),
                                          static_cast<RealType>(1.0))),
            static_cast<float>(std::clamp(static_cast<RealType>(color.green()) * s,
                                          static_cast<RealType>(0.0),
                                          static_cast<RealType>(1.0))),
            static_cast<float>(std::clamp(static_cast<RealType>(color.blue()) * s,
                                          static_cast<RealType>(0.0),
                                          static_cast<RealType>(1.0))),
            color.alpha(),
    };
}

void GetHeightRange(const std::vector<Vector3>& vertices,
                    RealType configured_min,
                    RealType configured_max,
                    RealType& out_min,
                    RealType& out_max) {
    if (std::abs(configured_max - configured_min) > CMP_EPSILON) {
        out_min = configured_min;
        out_max = configured_max;
        return;
    }

    out_min = std::numeric_limits<RealType>::max();
    out_max = std::numeric_limits<RealType>::lowest();
    for (const Vector3& vertex : vertices) {
        out_min = std::min(out_min, vertex.z());
        out_max = std::max(out_max, vertex.z());
    }
    if (vertices.empty()) {
        out_min = 0.0;
        out_max = 0.0;
    }
}

std::vector<Color> GenerateHeightRampColors(const std::vector<Vector3>& vertices,
                                            const Color& low_color,
                                            const Color& high_color,
                                            RealType range_min,
                                            RealType range_max) {
    std::vector<Color> colors;
    colors.reserve(vertices.size());
    const RealType range = range_max - range_min;
    for (const Vector3& vertex : vertices) {
        const RealType weight = std::abs(range) <= CMP_EPSILON
                ? static_cast<RealType>(0.0)
                : (vertex.z() - range_min) / range;
        const RealType clamped_weight = std::clamp(weight, static_cast<RealType>(0.0), static_cast<RealType>(1.0));
        Color base = clamped_weight < static_cast<RealType>(0.45)
                ? LerpColor(low_color,
                            Color{0.26f, 0.50f, 0.24f, 1.0f},
                            clamped_weight / static_cast<RealType>(0.45))
                : LerpColor(Color{0.26f, 0.50f, 0.24f, 1.0f},
                            high_color,
                            (clamped_weight - static_cast<RealType>(0.45)) / static_cast<RealType>(0.55));
        const RealType band = std::floor(clamped_weight * static_cast<RealType>(10.0));
        const RealType band_shade = (static_cast<int>(band) % 2 == 0)
                ? static_cast<RealType>(0.92)
                : static_cast<RealType>(1.06);
        const RealType contour = std::abs(std::fmod(clamped_weight * static_cast<RealType>(20.0),
                                                    static_cast<RealType>(1.0)) -
                                          static_cast<RealType>(0.5));
        const RealType contour_shade = contour > static_cast<RealType>(0.43)
                ? static_cast<RealType>(0.76)
                : static_cast<RealType>(1.0);
        colors.push_back(ScaleRgb(base, band_shade * contour_shade));
    }
    return colors;
}

} // namespace

void Terrain3D::ClearTerrain() {
    boxes_.clear();
    heightfields_.clear();
    mesh_patches_.clear();
    spawn_origins_.clear();
    MarkMeshDirty();
}

void Terrain3D::AddBox(const TerrainBox& box) {
    boxes_.push_back(box);
    MarkMeshDirty();
}

void Terrain3D::AddBox(const Vector3& center,
                       const Vector3& size,
                       const Vector3& rotation_degrees) {
    TerrainBox box;
    box.center = center;
    box.size = size;
    box.rotation_degrees = rotation_degrees;
    AddBox(box);
}

void Terrain3D::SetBoxes(const std::vector<TerrainBox>& boxes) {
    boxes_ = boxes;
    MarkMeshDirty();
}

const std::vector<TerrainBox>& Terrain3D::GetBoxes() const {
    return boxes_;
}

void Terrain3D::AddHeightField(const TerrainHeightField& heightfield) {
    heightfields_.push_back(heightfield);
    MarkMeshDirty();
}

void Terrain3D::SetHeightFields(const std::vector<TerrainHeightField>& heightfields) {
    heightfields_ = heightfields;
    MarkMeshDirty();
}

const std::vector<TerrainHeightField>& Terrain3D::GetHeightFields() const {
    return heightfields_;
}

void Terrain3D::AddMeshPatch(const TerrainMeshPatch& mesh_patch) {
    mesh_patches_.push_back(mesh_patch);
    MarkMeshDirty();
}

void Terrain3D::SetMeshPatches(const std::vector<TerrainMeshPatch>& mesh_patches) {
    mesh_patches_ = mesh_patches;
    MarkMeshDirty();
}

const std::vector<TerrainMeshPatch>& Terrain3D::GetMeshPatches() const {
    return mesh_patches_;
}

void Terrain3D::SetSpawnOrigins(const std::vector<Vector3>& spawn_origins) {
    spawn_origins_ = spawn_origins;
}

const std::vector<Vector3>& Terrain3D::GetSpawnOrigins() const {
    return spawn_origins_;
}

void Terrain3D::SetSurfaceColor(const Color& color) {
    surface_color_ = color;
}

Color Terrain3D::GetSurfaceColor() const {
    return surface_color_;
}

void Terrain3D::SetColorMode(TerrainColorMode color_mode) {
    color_mode_ = color_mode;
    MarkMeshDirty();
}

TerrainColorMode Terrain3D::GetColorMode() const {
    return color_mode_;
}

void Terrain3D::SetHeightLowColor(const Color& color) {
    height_low_color_ = color;
    MarkMeshDirty();
}

Color Terrain3D::GetHeightLowColor() const {
    return height_low_color_;
}

void Terrain3D::SetHeightHighColor(const Color& color) {
    height_high_color_ = color;
    MarkMeshDirty();
}

Color Terrain3D::GetHeightHighColor() const {
    return height_high_color_;
}

void Terrain3D::SetHeightRangeMin(RealType value) {
    height_range_min_ = value;
    MarkMeshDirty();
}

RealType Terrain3D::GetHeightRangeMin() const {
    return height_range_min_;
}

void Terrain3D::SetHeightRangeMax(RealType value) {
    height_range_max_ = value;
    MarkMeshDirty();
}

RealType Terrain3D::GetHeightRangeMax() const {
    return height_range_max_;
}

void Terrain3D::SetFriction(const Vector3& friction) {
    friction_ = friction;
}

const Vector3& Terrain3D::GetFriction() const {
    return friction_;
}

void Terrain3D::SetContactType(int contype) {
    contype_ = contype;
}

int Terrain3D::GetContactType() const {
    return contype_;
}

void Terrain3D::SetContactAffinity(int conaffinity) {
    conaffinity_ = conaffinity;
}

int Terrain3D::GetContactAffinity() const {
    return conaffinity_;
}

void Terrain3D::SetContactDimension(int condim) {
    condim_ = condim;
}

int Terrain3D::GetContactDimension() const {
    return condim_;
}

void Terrain3D::SetSolref(const Vector2& solref) {
    solref_ = solref;
}

const Vector2& Terrain3D::GetSolref() const {
    return solref_;
}

void Terrain3D::SetSolimp(const std::vector<RealType>& solimp) {
    solimp_ = solimp;
}

const std::vector<RealType>& Terrain3D::GetSolimp() const {
    return solimp_;
}

void Terrain3D::SetMargin(RealType margin) {
    margin_ = margin;
}

RealType Terrain3D::GetMargin() const {
    return margin_;
}

void Terrain3D::SetGap(RealType gap) {
    gap_ = gap;
}

RealType Terrain3D::GetGap() const {
    return gap_;
}

Ref<ArrayMesh> Terrain3D::GetRenderMesh() const {
    RebuildRenderMesh();
    return render_mesh_;
}

void Terrain3D::MarkMeshDirty() {
    render_mesh_dirty_ = true;
}

void Terrain3D::RebuildRenderMesh() const {
    if (!render_mesh_dirty_ && render_mesh_.IsValid()) {
        return;
    }

    if (!render_mesh_.IsValid()) {
        render_mesh_ = MakeRef<ArrayMesh>();
    }

    std::vector<Vector3> vertices;
    std::vector<std::uint32_t> indices;
    std::vector<Vector3> normals;
    std::vector<Color> colors;
    std::vector<Color>* per_vertex_colors =
            (color_mode_ == TerrainColorMode::Palette) ? &colors : nullptr;
    for (const TerrainBox& box : boxes_) {
        AppendBoxMesh(vertices, indices, normals, per_vertex_colors, box);
    }
    for (const TerrainHeightField& heightfield : heightfields_) {
        AppendHeightFieldMesh(vertices, indices, normals, per_vertex_colors, heightfield);
    }
    for (const TerrainMeshPatch& mesh_patch : mesh_patches_) {
        AppendMeshPatch(vertices, indices, normals, per_vertex_colors, mesh_patch);
    }

    if (color_mode_ == TerrainColorMode::HeightRamp) {
        RealType range_min = 0.0;
        RealType range_max = 0.0;
        GetHeightRange(vertices, height_range_min_, height_range_max_, range_min, range_max);
        colors = GenerateHeightRampColors(vertices, height_low_color_, height_high_color_, range_min, range_max);
    }

    render_mesh_->SetSurface(std::move(vertices), std::move(indices), std::move(normals), std::move(colors));
    render_mesh_dirty_ = false;
}

} // namespace gobot

GOBOT_REGISTRATION {

    USING_ENUM_BITWISE_OPERATORS;

    gobot::QuickEnumeration_<gobot::TerrainColorMode>("TerrainColorMode");

    Class_<gobot::TerrainBox>("TerrainBox")
            .constructor()(CtorAsRawPtr)
            .property("center", &gobot::TerrainBox::center)
            .property("size", &gobot::TerrainBox::size)
            .property("rotation_degrees", &gobot::TerrainBox::rotation_degrees)
            .property("color", &gobot::TerrainBox::color);

    Class_<gobot::TerrainHeightField>("TerrainHeightField")
            .constructor()(CtorAsRawPtr)
            .property("center", &gobot::TerrainHeightField::center)
            .property("size", &gobot::TerrainHeightField::size)
            .property("rows", &gobot::TerrainHeightField::rows)
            .property("cols", &gobot::TerrainHeightField::cols)
            .property("heights", &gobot::TerrainHeightField::heights)
            .property("normalized_elevation", &gobot::TerrainHeightField::normalized_elevation)
            .property("base_thickness", &gobot::TerrainHeightField::base_thickness)
            .property("z_offset", &gobot::TerrainHeightField::z_offset);

    Class_<gobot::TerrainMeshPatch>("TerrainMeshPatch")
            .constructor()(CtorAsRawPtr)
            .property("center", &gobot::TerrainMeshPatch::center)
            .property("rotation_degrees", &gobot::TerrainMeshPatch::rotation_degrees)
            .property("vertices", &gobot::TerrainMeshPatch::vertices)
            .property("indices", &gobot::TerrainMeshPatch::indices)
            .property("color", &gobot::TerrainMeshPatch::color);

    Class_<gobot::Terrain3D>("Terrain3D")
            .constructor()(CtorAsRawPtr)
            .property("boxes", &gobot::Terrain3D::GetBoxes, &gobot::Terrain3D::SetBoxes)
            .property("heightfields", &gobot::Terrain3D::GetHeightFields, &gobot::Terrain3D::SetHeightFields)
            .property("mesh_patches", &gobot::Terrain3D::GetMeshPatches, &gobot::Terrain3D::SetMeshPatches)
            .property("spawn_origins", &gobot::Terrain3D::GetSpawnOrigins, &gobot::Terrain3D::SetSpawnOrigins)
            .property("surface_color", &gobot::Terrain3D::GetSurfaceColor, &gobot::Terrain3D::SetSurfaceColor)
            .property("color_mode", &gobot::Terrain3D::GetColorMode, &gobot::Terrain3D::SetColorMode)
            .property("height_low_color", &gobot::Terrain3D::GetHeightLowColor, &gobot::Terrain3D::SetHeightLowColor)
            .property("height_high_color", &gobot::Terrain3D::GetHeightHighColor, &gobot::Terrain3D::SetHeightHighColor)
            .property("height_range_min", &gobot::Terrain3D::GetHeightRangeMin, &gobot::Terrain3D::SetHeightRangeMin)
            .property("height_range_max", &gobot::Terrain3D::GetHeightRangeMax, &gobot::Terrain3D::SetHeightRangeMax)
            .property("friction", &gobot::Terrain3D::GetFriction, &gobot::Terrain3D::SetFriction)
            .property("contype", &gobot::Terrain3D::GetContactType, &gobot::Terrain3D::SetContactType)
            .property("conaffinity", &gobot::Terrain3D::GetContactAffinity, &gobot::Terrain3D::SetContactAffinity)
            .property("condim", &gobot::Terrain3D::GetContactDimension, &gobot::Terrain3D::SetContactDimension)
            .property("solref", &gobot::Terrain3D::GetSolref, &gobot::Terrain3D::SetSolref)
            .property("solimp", &gobot::Terrain3D::GetSolimp, &gobot::Terrain3D::SetSolimp)
            .property("margin", &gobot::Terrain3D::GetMargin, &gobot::Terrain3D::SetMargin)
            .property("gap", &gobot::Terrain3D::GetGap, &gobot::Terrain3D::SetGap);

};
