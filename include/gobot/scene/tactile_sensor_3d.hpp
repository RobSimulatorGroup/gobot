/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "gobot/core/io/resource.hpp"
#include "gobot/scene/resources/tetrahedral_mesh.hpp"
#include "gobot/scene/sensor_3d.hpp"

namespace gobot {

class GOBOT_EXPORT TactileSensorConfig : public Resource {
    GOBCLASS(TactileSensorConfig, Resource)

public:
    TactileSensorConfig() = default;

    void SetImageWidth(std::uint32_t image_width);
    std::uint32_t GetImageWidth() const;

    void SetImageHeight(std::uint32_t image_height);
    std::uint32_t GetImageHeight() const;

    void SetNearPlane(RealType near_plane);
    RealType GetNearPlane() const;

    void SetFarPlane(RealType far_plane);
    RealType GetFarPlane() const;

    void SetPixelSize(RealType pixel_size);
    RealType GetPixelSize() const;

    void SetDensity(RealType density);
    RealType GetDensity() const;

    void SetYoungModulus(RealType young_modulus);
    RealType GetYoungModulus() const;

    void SetPoissonRatio(RealType poisson_ratio);
    RealType GetPoissonRatio() const;

    void SetDamping(RealType damping);
    RealType GetDamping() const;

    void SetFrictionCoefficient(RealType friction_coefficient);
    RealType GetFrictionCoefficient() const;

    void SetCoatVertexIndices(const std::vector<std::uint32_t>& coat_vertex_indices);
    const std::vector<std::uint32_t>& GetCoatVertexIndices() const;

    void SetStickVertexIndices(const std::vector<std::uint32_t>& stick_vertex_indices);
    const std::vector<std::uint32_t>& GetStickVertexIndices() const;

    void SetMarkerPositions(const std::vector<Vector2>& marker_positions);
    const std::vector<Vector2>& GetMarkerPositions() const;

    void SetMarkerTetrahedra(const std::vector<std::uint32_t>& marker_tetrahedra);
    const std::vector<std::uint32_t>& GetMarkerTetrahedra() const;

    void SetMarkerBarycentric(const std::vector<Vector4>& marker_barycentric);
    const std::vector<Vector4>& GetMarkerBarycentric() const;

    void SetRgbModel(const std::string& rgb_model);
    const std::string& GetRgbModel() const;

    bool Validate(const TetrahedralMesh& gel_mesh, std::string* error = nullptr) const;

private:
    std::uint32_t image_width_{320};
    std::uint32_t image_height_{240};
    RealType near_plane_{0.0};
    RealType far_plane_{0.05};
    RealType pixel_size_{7.9375e-5};
    RealType density_{1000.0};
    RealType young_modulus_{500000.0};
    RealType poisson_ratio_{0.4};
    RealType damping_{0.0};
    RealType friction_coefficient_{1.0};
    std::vector<std::uint32_t> coat_vertex_indices_;
    std::vector<std::uint32_t> stick_vertex_indices_;
    std::vector<Vector2> marker_positions_;
    std::vector<std::uint32_t> marker_tetrahedra_;
    std::vector<Vector4> marker_barycentric_;
    std::string rgb_model_{"gobot_deterministic_v1"};
};

class GOBOT_EXPORT TactileSensor3D : public Sensor3D {
    GOBCLASS(TactileSensor3D, Sensor3D)

public:
    TactileSensor3D() = default;

    void SetConfig(const Ref<TactileSensorConfig>& config);
    const Ref<TactileSensorConfig>& GetConfig() const;

    void SetGelMesh(const Ref<TetrahedralMesh>& gel_mesh);
    const Ref<TetrahedralMesh>& GetGelMesh() const;

    void SetCollisionLayer(std::uint32_t collision_layer);
    std::uint32_t GetCollisionLayer() const;

    void SetCollisionMask(std::uint32_t collision_mask);
    std::uint32_t GetCollisionMask() const;

private:
    Ref<TactileSensorConfig> config_;
    Ref<TetrahedralMesh> gel_mesh_;
    std::uint32_t collision_layer_{1};
    std::uint32_t collision_mask_{0xffffffffU};
};

} // namespace gobot
