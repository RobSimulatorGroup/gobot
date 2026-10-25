#pragma once

#include <memory>
#include <span>
#include "gobot/core/tactile_sensor_parameters.hpp"
#include "gobot/core/tactile_optical_data.hpp"
#include "gobot/core/tactile_sensor_frame.hpp"

namespace gobot {

class GOBOT_EXPORT TactileSensorProcessor {
  public:
    TactileSensorProcessor(TactileSensorParameters parameters, std::vector<Vector3> rest_vertices,
                           std::vector<std::uint32_t> tetrahedra,
                           std::vector<std::uint32_t> surface_triangles, TactileOpticalData calibration = {});
    std::shared_ptr<const TactileSensorFrame>
    Process(std::span<const Vector3> world_vertices, std::span<const Vector3> world_contact_forces,
            const Affine3 &sensor_to_world, std::uint64_t tick, double time,
            std::span<const Vector3> contact_vertices = {},
            std::span<const std::uint32_t> contact_triangles = {}) const;
    const std::vector<float> &GetPixelAreas() const { return pixel_areas_; }

  private:
    struct Sample {
        std::uint32_t pixel, a, b, c;
        Vector3 weights;
    };
    Vector3 Project(const Vector3 &point) const;
    void Rasterize(std::span<const Vector3> vertices, std::vector<float> &depth,
                   std::vector<Vector2> *gradients = nullptr) const;
    void ContactOptics(std::span<const Vector3> local_vertices, std::span<const std::uint32_t> triangles,
                       TactileSensorFrame &frame, std::vector<Vector2> &gradients) const;
    std::vector<Vector3> SurfaceNormals(std::span<const Vector3> vertices) const;
    TactileSensorParameters parameters_;
    TactileOpticalData calibration_;
    std::vector<Vector3> rest_vertices_, reference_normals_;
    std::vector<std::uint32_t> tetrahedra_, surface_, coat_;
    std::vector<float> rest_depth_, pixel_areas_;
    std::vector<Sample> samples_;
    std::vector<double> vertex_weights_;
    std::vector<Vector2> rest_markers_;
};
} // namespace gobot
