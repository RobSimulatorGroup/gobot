/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/scene/tactile_sensor_3d.hpp"

#include "gobot/core/registration.hpp"

namespace gobot {

void TactileSensorConfig::SetCameraIntrinsics(const Vector4& value) { parameters_.camera_intrinsics=value; MarkChanged(); }
const Vector4& TactileSensorConfig::GetCameraIntrinsics() const { return parameters_.camera_intrinsics; }
void TactileSensorConfig::SetCameraTransform(const Affine3& value) { parameters_.camera_transform=value; MarkChanged(); }
const Affine3& TactileSensorConfig::GetCameraTransform() const { return parameters_.camera_transform; }
void TactileSensorConfig::SetAttachmentStrength(const RealType& value) { parameters_.attachment_strength=value; MarkChanged(); }
const RealType& TactileSensorConfig::GetAttachmentStrength() const { return parameters_.attachment_strength; }
void TactileSensorConfig::SetOpticalCalibration(const Ref<TactileOpticalCalibration>& v) { optical_calibration_=v; MarkChanged(); }
const Ref<TactileOpticalCalibration>& TactileSensorConfig::GetOpticalCalibration() const { return optical_calibration_; }

void TactileSensorConfig::SetImageWidth(std::uint32_t image_width) {
    parameters_.image_width = image_width;
    MarkChanged();
}

std::uint32_t TactileSensorConfig::GetImageWidth() const {
    return parameters_.image_width;
}

void TactileSensorConfig::SetImageHeight(std::uint32_t image_height) {
    parameters_.image_height = image_height;
    MarkChanged();
}

std::uint32_t TactileSensorConfig::GetImageHeight() const {
    return parameters_.image_height;
}

void TactileSensorConfig::SetNearPlane(RealType near_plane) {
    parameters_.near_plane = near_plane;
    MarkChanged();
}

RealType TactileSensorConfig::GetNearPlane() const {
    return parameters_.near_plane;
}

void TactileSensorConfig::SetFarPlane(RealType far_plane) {
    parameters_.far_plane = far_plane;
    MarkChanged();
}

RealType TactileSensorConfig::GetFarPlane() const {
    return parameters_.far_plane;
}

void TactileSensorConfig::SetPixelSize(RealType pixel_size) {
    parameters_.pixel_size = pixel_size;
    MarkChanged();
}

RealType TactileSensorConfig::GetPixelSize() const {
    return parameters_.pixel_size;
}

void TactileSensorConfig::SetDensity(RealType density) {
    parameters_.density = density;
    MarkChanged();
}

RealType TactileSensorConfig::GetDensity() const {
    return parameters_.density;
}

void TactileSensorConfig::SetYoungModulus(RealType young_modulus) {
    parameters_.young_modulus = young_modulus;
    MarkChanged();
}

RealType TactileSensorConfig::GetYoungModulus() const {
    return parameters_.young_modulus;
}

void TactileSensorConfig::SetPoissonRatio(RealType poisson_ratio) {
    parameters_.poisson_ratio = poisson_ratio;
    MarkChanged();
}

RealType TactileSensorConfig::GetPoissonRatio() const {
    return parameters_.poisson_ratio;
}

void TactileSensorConfig::SetDamping(RealType damping) {
    parameters_.damping = damping;
    MarkChanged();
}

RealType TactileSensorConfig::GetDamping() const {
    return parameters_.damping;
}

void TactileSensorConfig::SetFrictionCoefficient(RealType friction_coefficient) {
    parameters_.friction_coefficient = friction_coefficient;
    MarkChanged();
}

RealType TactileSensorConfig::GetFrictionCoefficient() const {
    return parameters_.friction_coefficient;
}

void TactileSensorConfig::SetCoatVertexIndices(
        const std::vector<std::uint32_t>& coat_vertex_indices) {
    parameters_.coat_vertex_indices = coat_vertex_indices;
    MarkChanged();
}

const std::vector<std::uint32_t>& TactileSensorConfig::GetCoatVertexIndices() const {
    return parameters_.coat_vertex_indices;
}

void TactileSensorConfig::SetStickVertexIndices(
        const std::vector<std::uint32_t>& stick_vertex_indices) {
    parameters_.stick_vertex_indices = stick_vertex_indices;
    MarkChanged();
}

const std::vector<std::uint32_t>& TactileSensorConfig::GetStickVertexIndices() const {
    return parameters_.stick_vertex_indices;
}

void TactileSensorConfig::SetMarkerPositions(
        const std::vector<Vector2>& marker_positions) {
    parameters_.marker_positions = marker_positions;
    MarkChanged();
}

const std::vector<Vector2>& TactileSensorConfig::GetMarkerPositions() const {
    return parameters_.marker_positions;
}

void TactileSensorConfig::SetMarkerTetrahedra(
        const std::vector<std::uint32_t>& marker_tetrahedra) {
    parameters_.marker_tetrahedra = marker_tetrahedra;
    MarkChanged();
}

const std::vector<std::uint32_t>& TactileSensorConfig::GetMarkerTetrahedra() const {
    return parameters_.marker_tetrahedra;
}

void TactileSensorConfig::SetMarkerBarycentric(
        const std::vector<Vector4>& marker_barycentric) {
    parameters_.marker_barycentric = marker_barycentric;
    MarkChanged();
}

const std::vector<Vector4>& TactileSensorConfig::GetMarkerBarycentric() const {
    return parameters_.marker_barycentric;
}

void TactileSensorConfig::SetRgbModel(const std::string& rgb_model) {
    parameters_.rgb_model = rgb_model;
    MarkChanged();
}

const std::string& TactileSensorConfig::GetRgbModel() const {
    return parameters_.rgb_model;
}

bool TactileSensorConfig::Validate(const TetrahedralMesh& gel_mesh, std::string* error) const {
    return parameters_.Validate(gel_mesh.GetVertexCount(), gel_mesh.GetTetrahedronCount(), error);
}

void TactileSensor3D::SetRuntimeFrame(std::shared_ptr<const TactileSensorFrame> frame) {
    runtime_frame_ = std::move(frame);
}
const std::shared_ptr<const TactileSensorFrame>& TactileSensor3D::GetRuntimeFrame() const { return runtime_frame_; }
Ref<ArrayMesh> TactileSensor3D::GetRenderMesh() const {
    if (!gel_mesh_.IsValid()) return {};
    const auto& vertices = runtime_frame_ ? runtime_frame_->local_vertices : gel_mesh_->GetVertices();
    if (vertices.size() != gel_mesh_->GetVertexCount()) return {};
    const auto indices = gel_mesh_->GetResolvedSurfaceTriangles();
    if (!render_mesh_.IsValid()) render_mesh_ = MakeRef<ArrayMesh>();
    if (render_mesh_->GetVertices() != vertices || render_mesh_->GetIndices() != indices)
        render_mesh_->SetSurface(vertices, indices);
    return render_mesh_;
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
            .property("camera_intrinsics", &gobot::TactileSensorConfig::GetCameraIntrinsics, &gobot::TactileSensorConfig::SetCameraIntrinsics)
            .property("camera_transform", &gobot::TactileSensorConfig::GetCameraTransform, &gobot::TactileSensorConfig::SetCameraTransform)
            .property("attachment_strength", &gobot::TactileSensorConfig::GetAttachmentStrength, &gobot::TactileSensorConfig::SetAttachmentStrength)
            .property("optical_calibration", &gobot::TactileSensorConfig::GetOpticalCalibration, &gobot::TactileSensorConfig::SetOpticalCalibration)
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
