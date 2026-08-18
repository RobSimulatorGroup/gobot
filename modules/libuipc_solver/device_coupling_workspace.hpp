#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace gobot::libuipc_solver {

struct DeviceDeformableContactRange {
  std::size_t output_offset{0};
  std::size_t vertex_count{0};
  std::int32_t global_vertex_offset{-1};
};

struct DeviceAffineContactRange {
  std::size_t output_offset{0};
  std::int32_t global_vertex_offset{-1};
  std::vector<double> local_vertices;
  double local_center_of_mass[3]{0.0, 0.0, 0.0};
};

struct DeviceAttachmentVertex {
  std::size_t deformable_output_offset{0};
  std::size_t affine_output_offset{0};
  double link_local_position[3]{0.0, 0.0, 0.0};
  double link_local_center_of_mass[3]{0.0, 0.0, 0.0};
  double vertex_mass{0.0};
  double strength_rate{0.0};
};

struct DeviceContactGradientView {
  const void *vertex_indices{nullptr};
  const void *gradients{nullptr};
  std::size_t count{0};
};

class DeviceCouplingWorkspace final {
public:
  DeviceCouplingWorkspace(
      std::uint32_t device_index, std::size_t deformable_vertex_count,
      std::size_t affine_body_count,
      std::vector<DeviceDeformableContactRange> deformable_ranges,
      std::vector<DeviceAffineContactRange> affine_ranges,
      std::vector<DeviceAttachmentVertex> attachment_vertices);
  ~DeviceCouplingWorkspace();

  DeviceCouplingWorkspace(const DeviceCouplingWorkspace &) = delete;
  DeviceCouplingWorkspace &operator=(const DeviceCouplingWorkspace &) = delete;

  void StageTargets(const double *row_major_targets,
                    const double *target_twists);

  void ExportReactions(std::span<const DeviceContactGradientView> gradients,
                       double inverse_time_step_squared,
                       double *deformable_contact_forces,
                       double *affine_contact_wrenches,
                       bool export_deformable_contact_forces,
                       bool export_affine_contact_wrenches);

  void *target_transforms() const;
  void *target_velocities() const;
  void *attachment_aim_positions() const;
  void *current_deformable_positions() const;
  void *current_affine_transforms() const;

  std::size_t deformable_vertex_count() const;
  std::size_t affine_body_count() const;
  std::size_t allocation_count() const;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace gobot::libuipc_solver
