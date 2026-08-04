/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/scene/tactile_sensor_3d.hpp"

#include <algorithm>
#include <cmath>

#include "gobot/core/registration.hpp"

namespace gobot {
namespace {

bool SetValidationError(std::string* error, std::string message) {
    if (error != nullptr) {
        *error = std::move(message);
    }
    return false;
}

} // namespace

void TactileSensorConfig::SetImageWidth(std::uint32_t image_width) {
    image_width_ = image_width;
    MarkChanged();
}

std::uint32_t TactileSensorConfig::GetImageWidth() const {
    return image_width_;
}

void TactileSensorConfig::SetImageHeight(std::uint32_t image_height) {
    image_height_ = image_height;
    MarkChanged();
}

std::uint32_t TactileSensorConfig::GetImageHeight() const {
    return image_height_;
}

void TactileSensorConfig::SetNearPlane(RealType near_plane) {
    near_plane_ = near_plane;
    MarkChanged();
}

RealType TactileSensorConfig::GetNearPlane() const {
    return near_plane_;
}

void TactileSensorConfig::SetFarPlane(RealType far_plane) {
    far_plane_ = far_plane;
    MarkChanged();
}

RealType TactileSensorConfig::GetFarPlane() const {
    return far_plane_;
}

void TactileSensorConfig::SetPixelSize(RealType pixel_size) {
    pixel_size_ = pixel_size;
    MarkChanged();
}

RealType TactileSensorConfig::GetPixelSize() const {
    return pixel_size_;
}

void TactileSensorConfig::SetDensity(RealType density) {
    density_ = density;
    MarkChanged();
}

RealType TactileSensorConfig::GetDensity() const {
    return density_;
}

void TactileSensorConfig::SetYoungModulus(RealType young_modulus) {
    young_modulus_ = young_modulus;
    MarkChanged();
}

RealType TactileSensorConfig::GetYoungModulus() const {
    return young_modulus_;
}

void TactileSensorConfig::SetPoissonRatio(RealType poisson_ratio) {
    poisson_ratio_ = poisson_ratio;
    MarkChanged();
}

RealType TactileSensorConfig::GetPoissonRatio() const {
    return poisson_ratio_;
}

void TactileSensorConfig::SetDamping(RealType damping) {
    damping_ = damping;
    MarkChanged();
}

RealType TactileSensorConfig::GetDamping() const {
    return damping_;
}

void TactileSensorConfig::SetFrictionCoefficient(RealType friction_coefficient) {
    friction_coefficient_ = friction_coefficient;
    MarkChanged();
}

RealType TactileSensorConfig::GetFrictionCoefficient() const {
    return friction_coefficient_;
}

void TactileSensorConfig::SetCoatVertexIndices(
        const std::vector<std::uint32_t>& coat_vertex_indices) {
    coat_vertex_indices_ = coat_vertex_indices;
    MarkChanged();
}

const std::vector<std::uint32_t>& TactileSensorConfig::GetCoatVertexIndices() const {
    return coat_vertex_indices_;
}

void TactileSensorConfig::SetStickVertexIndices(
        const std::vector<std::uint32_t>& stick_vertex_indices) {
    stick_vertex_indices_ = stick_vertex_indices;
    MarkChanged();
}

const std::vector<std::uint32_t>& TactileSensorConfig::GetStickVertexIndices() const {
    return stick_vertex_indices_;
}

void TactileSensorConfig::SetMarkerPositions(
        const std::vector<Vector2>& marker_positions) {
    marker_positions_ = marker_positions;
    MarkChanged();
}

const std::vector<Vector2>& TactileSensorConfig::GetMarkerPositions() const {
    return marker_positions_;
}

void TactileSensorConfig::SetMarkerTetrahedra(
        const std::vector<std::uint32_t>& marker_tetrahedra) {
    marker_tetrahedra_ = marker_tetrahedra;
    MarkChanged();
}

const std::vector<std::uint32_t>& TactileSensorConfig::GetMarkerTetrahedra() const {
    return marker_tetrahedra_;
}

void TactileSensorConfig::SetMarkerBarycentric(
        const std::vector<Vector4>& marker_barycentric) {
    marker_barycentric_ = marker_barycentric;
    MarkChanged();
}

const std::vector<Vector4>& TactileSensorConfig::GetMarkerBarycentric() const {
    return marker_barycentric_;
}

void TactileSensorConfig::SetRgbModel(const std::string& rgb_model) {
    rgb_model_ = rgb_model;
    MarkChanged();
}

const std::string& TactileSensorConfig::GetRgbModel() const {
    return rgb_model_;
}

bool TactileSensorConfig::Validate(
        const TetrahedralMesh& gel_mesh, std::string* error) const {
    if (image_width_ == 0 || image_height_ == 0) {
        return SetValidationError(error, "tactile image resolution must be positive");
    }
    if (!std::isfinite(near_plane_) || !std::isfinite(far_plane_) ||
        near_plane_ < 0.0 || far_plane_ <= near_plane_) {
        return SetValidationError(
                error, "tactile near/far planes must be finite and strictly ordered");
    }
    if (!std::isfinite(pixel_size_) || pixel_size_ <= 0.0) {
        return SetValidationError(error, "tactile pixel size must be finite and positive");
    }
    if (!std::isfinite(density_) || density_ <= 0.0 ||
        !std::isfinite(young_modulus_) || young_modulus_ <= 0.0 ||
        !std::isfinite(poisson_ratio_) || poisson_ratio_ <= -1.0 ||
        poisson_ratio_ >= 0.5 || !std::isfinite(damping_) || damping_ < 0.0 ||
        !std::isfinite(friction_coefficient_) || friction_coefficient_ < 0.0) {
        return SetValidationError(error, "tactile gel material parameters are invalid");
    }
    if (rgb_model_.empty()) {
        return SetValidationError(error, "tactile RGB model name must not be empty");
    }
    const auto validate_vertex_set = [&](const std::vector<std::uint32_t>& indices,
                                         const char* description) {
        std::vector<std::uint32_t> sorted = indices;
        std::sort(sorted.begin(), sorted.end());
        if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {
            return SetValidationError(
                    error, std::string("tactile ") + description + " contains duplicates");
        }
        if (std::any_of(sorted.begin(), sorted.end(), [&](std::uint32_t index) {
                return index >= gel_mesh.GetVertexCount();
            })) {
            return SetValidationError(
                    error, std::string("tactile ") + description + " references an invalid vertex");
        }
        return true;
    };
    if (!validate_vertex_set(coat_vertex_indices_, "coat vertex set") ||
        !validate_vertex_set(stick_vertex_indices_, "stick vertex set")) {
        return false;
    }
    if (marker_positions_.size() != marker_tetrahedra_.size() ||
        marker_positions_.size() != marker_barycentric_.size()) {
        return SetValidationError(
                error, "tactile marker position, tetrahedron, and barycentric tables must match");
    }
    for (std::size_t index = 0; index < marker_positions_.size(); ++index) {
        const Vector2& marker = marker_positions_[index];
        if (!marker.allFinite() || marker.x() < 0.0 || marker.y() < 0.0 ||
            marker.x() >= static_cast<RealType>(image_width_) ||
            marker.y() >= static_cast<RealType>(image_height_)) {
            return SetValidationError(
                    error, "tactile marker " + std::to_string(index) +
                                   " is outside the image in pixel coordinates");
        }
        if (marker_tetrahedra_[index] >= gel_mesh.GetTetrahedronCount()) {
            return SetValidationError(
                    error, "tactile marker references an invalid gel tetrahedron");
        }
        const Vector4& barycentric = marker_barycentric_[index];
        if (!barycentric.allFinite() ||
            (barycentric.array() < -CMP_EPSILON).any() ||
            std::abs(barycentric.sum() - 1.0) > 1.0e-5) {
            return SetValidationError(
                    error, "tactile marker barycentric weights must be finite, non-negative, and sum to one");
        }
    }
    if (error != nullptr) {
        error->clear();
    }
    return true;
}

void TactileSensor3D::SetConfig(const Ref<TactileSensorConfig>& config) {
    config_ = config;
}

const Ref<TactileSensorConfig>& TactileSensor3D::GetConfig() const {
    return config_;
}

void TactileSensor3D::SetGelMesh(const Ref<TetrahedralMesh>& gel_mesh) {
    gel_mesh_ = gel_mesh;
}

const Ref<TetrahedralMesh>& TactileSensor3D::GetGelMesh() const {
    return gel_mesh_;
}

void TactileSensor3D::SetCollisionLayer(std::uint32_t collision_layer) {
    collision_layer_ = collision_layer;
}

std::uint32_t TactileSensor3D::GetCollisionLayer() const {
    return collision_layer_;
}

void TactileSensor3D::SetCollisionMask(std::uint32_t collision_mask) {
    collision_mask_ = collision_mask;
}

std::uint32_t TactileSensor3D::GetCollisionMask() const {
    return collision_mask_;
}

} // namespace gobot

GOBOT_REGISTRATION {
    Class_<gobot::TactileSensorConfig>("TactileSensorConfig")
            .constructor()(CtorAsRawPtr)
            .property("image_width", &gobot::TactileSensorConfig::GetImageWidth,
                      &gobot::TactileSensorConfig::SetImageWidth)
            .property("image_height", &gobot::TactileSensorConfig::GetImageHeight,
                      &gobot::TactileSensorConfig::SetImageHeight)
            .property("near_plane", &gobot::TactileSensorConfig::GetNearPlane,
                      &gobot::TactileSensorConfig::SetNearPlane)
            .property("far_plane", &gobot::TactileSensorConfig::GetFarPlane,
                      &gobot::TactileSensorConfig::SetFarPlane)
            .property("pixel_size", &gobot::TactileSensorConfig::GetPixelSize,
                      &gobot::TactileSensorConfig::SetPixelSize)
            .property("density", &gobot::TactileSensorConfig::GetDensity,
                      &gobot::TactileSensorConfig::SetDensity)
            .property("young_modulus", &gobot::TactileSensorConfig::GetYoungModulus,
                      &gobot::TactileSensorConfig::SetYoungModulus)
            .property("poisson_ratio", &gobot::TactileSensorConfig::GetPoissonRatio,
                      &gobot::TactileSensorConfig::SetPoissonRatio)
            .property("damping", &gobot::TactileSensorConfig::GetDamping,
                      &gobot::TactileSensorConfig::SetDamping)
            .property("friction_coefficient",
                      &gobot::TactileSensorConfig::GetFrictionCoefficient,
                      &gobot::TactileSensorConfig::SetFrictionCoefficient)
            .property("coat_vertex_indices",
                      &gobot::TactileSensorConfig::GetCoatVertexIndices,
                      &gobot::TactileSensorConfig::SetCoatVertexIndices)
            .property("stick_vertex_indices",
                      &gobot::TactileSensorConfig::GetStickVertexIndices,
                      &gobot::TactileSensorConfig::SetStickVertexIndices)
            .property("marker_positions", &gobot::TactileSensorConfig::GetMarkerPositions,
                      &gobot::TactileSensorConfig::SetMarkerPositions)
            .property("marker_tetrahedra", &gobot::TactileSensorConfig::GetMarkerTetrahedra,
                      &gobot::TactileSensorConfig::SetMarkerTetrahedra)
            .property("marker_barycentric", &gobot::TactileSensorConfig::GetMarkerBarycentric,
                      &gobot::TactileSensorConfig::SetMarkerBarycentric)
            .property("rgb_model", &gobot::TactileSensorConfig::GetRgbModel,
                      &gobot::TactileSensorConfig::SetRgbModel);

    Class_<gobot::TactileSensor3D>("TactileSensor3D")
            .constructor()(CtorAsRawPtr)
            .property("config", &gobot::TactileSensor3D::GetConfig,
                      &gobot::TactileSensor3D::SetConfig)
            .property("gel_mesh", &gobot::TactileSensor3D::GetGelMesh,
                      &gobot::TactileSensor3D::SetGelMesh)
            .property("collision_layer", &gobot::TactileSensor3D::GetCollisionLayer,
                      &gobot::TactileSensor3D::SetCollisionLayer)
            .property("collision_mask", &gobot::TactileSensor3D::GetCollisionMask,
                      &gobot::TactileSensor3D::SetCollisionMask);

    gobot::Type::register_wrapper_converter_for_base_classes<
            gobot::Ref<gobot::TactileSensorConfig>, gobot::Ref<gobot::Resource>>();
};
