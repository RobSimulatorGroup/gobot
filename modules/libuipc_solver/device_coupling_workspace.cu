#include "device_coupling_workspace.hpp"

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_reduce.cuh>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace gobot::libuipc_solver {
namespace {

struct Vec3 {
  double x;
  double y;
  double z;
};

struct Vec3Add {
  __host__ __device__ Vec3 operator()(const Vec3 &lhs, const Vec3 &rhs) const {
    return Vec3{lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z};
  }
};

struct AffineRangeData {
  std::size_t vertex_offset;
  std::size_t vertex_count;
  Vec3 center_of_mass;
};

struct AttachmentData {
  std::size_t deformable_output_offset;
  std::size_t affine_output_offset;
  Vec3 local_position;
  Vec3 local_center_of_mass;
  double vertex_mass;
  double strength_rate;
};

void RequireCuda(cudaError_t result, const char *operation) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(result));
  }
}

class DeviceAllocations final {
public:
  ~DeviceAllocations() {
    for (auto iterator = pointers_.rbegin(); iterator != pointers_.rend();
         ++iterator) {
      cudaFree(*iterator);
    }
  }

  DeviceAllocations() = default;
  DeviceAllocations(const DeviceAllocations &) = delete;
  DeviceAllocations &operator=(const DeviceAllocations &) = delete;

  template <typename T>
  T *Allocate(std::size_t count, const char *operation) {
    if (count == 0) {
      return nullptr;
    }
    T *result = nullptr;
    RequireCuda(
        cudaMalloc(reinterpret_cast<void **>(&result), count * sizeof(T)),
        operation);
    pointers_.push_back(result);
    return result;
  }

  void *AllocateBytes(std::size_t size, const char *operation) {
    if (size == 0) {
      return nullptr;
    }
    void *result = nullptr;
    RequireCuda(cudaMalloc(&result, size), operation);
    pointers_.push_back(result);
    return result;
  }

  template <typename T> void Release(T *&pointer) {
    if (pointer == nullptr) {
      return;
    }
    const auto found = std::ranges::find(pointers_, pointer);
    if (found != pointers_.end()) {
      cudaFree(pointer);
      pointers_.erase(found);
    }
    pointer = nullptr;
  }

private:
  std::vector<void *> pointers_;
};

template <typename T>
void Upload(T *destination, const std::vector<T> &source,
            const char *operation) {
  if (source.empty()) {
    return;
  }
  RequireCuda(cudaMemcpy(destination, source.data(), source.size() * sizeof(T),
                         cudaMemcpyHostToDevice),
              operation);
}

__device__ Vec3 TransformPoint(const double *transform, const Vec3 &point) {
  return Vec3{transform[0] * point.x + transform[4] * point.y +
                  transform[8] * point.z + transform[12],
              transform[1] * point.x + transform[5] * point.y +
                  transform[9] * point.z + transform[13],
              transform[2] * point.x + transform[6] * point.y +
                  transform[10] * point.z + transform[14]};
}

__device__ Vec3 Cross(const Vec3 &lhs, const Vec3 &rhs) {
  return Vec3{lhs.y * rhs.z - lhs.z * rhs.y, lhs.z * rhs.x - lhs.x * rhs.z,
              lhs.x * rhs.y - lhs.y * rhs.x};
}

__global__ void StageTargetsKernel(const double *row_major_targets,
                                   const double *target_twists,
                                   double *column_major_targets,
                                   double *column_major_velocities,
                                   std::size_t body_count) {
  const std::size_t body =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (body >= body_count) {
    return;
  }
  const double *source = row_major_targets + body * 16;
  double *target = column_major_targets + body * 16;
  double *velocity = column_major_velocities + body * 16;
  for (std::size_t row = 0; row < 4; ++row) {
    for (std::size_t column = 0; column < 4; ++column) {
      target[column * 4 + row] = source[row * 4 + column];
      velocity[column * 4 + row] = 0.0;
    }
  }

  const double wx = target_twists[body * 6 + 3];
  const double wy = target_twists[body * 6 + 4];
  const double wz = target_twists[body * 6 + 5];
  const double skew[9]{0.0, -wz, wy, wz, 0.0, -wx, -wy, wx, 0.0};
  for (std::size_t row = 0; row < 3; ++row) {
    for (std::size_t column = 0; column < 3; ++column) {
      double value = 0.0;
      for (std::size_t inner = 0; inner < 3; ++inner) {
        value += skew[row * 3 + inner] * source[inner * 4 + column];
      }
      velocity[column * 4 + row] = value;
    }
  }
  velocity[12] = target_twists[body * 6];
  velocity[13] = target_twists[body * 6 + 1];
  velocity[14] = target_twists[body * 6 + 2];
}

__global__ void StageAttachmentAimsKernel(const AttachmentData *attachments,
                                          std::size_t attachment_count,
                                          const double *target_transforms,
                                          Vec3 *attachment_aim_positions) {
  const std::size_t index =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (index >= attachment_count) {
    return;
  }
  const AttachmentData &attachment = attachments[index];
  const double *transform =
      target_transforms + attachment.affine_output_offset * 16;
  attachment_aim_positions[attachment.deformable_output_offset] =
      TransformPoint(transform, attachment.local_position);
}

__global__ void ScaleGradientsKernel(Vec3 *values, std::size_t count,
                                     double scale) {
  const std::size_t index =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (index >= count) {
    return;
  }
  values[index].x *= scale;
  values[index].y *= scale;
  values[index].z *= scale;
}

__global__ void ScatterUniqueGradientsKernel(
    const std::int32_t *keys, const Vec3 *values, const std::int32_t *run_count,
    std::size_t maximum_count, Vec3 *dense_values, std::size_t dense_count) {
  const std::size_t index =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (index >= maximum_count || index >= static_cast<std::size_t>(*run_count)) {
    return;
  }
  const std::int32_t key = keys[index];
  if (key >= 0 && static_cast<std::size_t>(key) < dense_count) {
    dense_values[key] = values[index];
  }
}

__global__ void ExportDeformableForcesKernel(const std::int32_t *global_indices,
                                             std::size_t count,
                                             const Vec3 *dense_values,
                                             Vec3 *output) {
  const std::size_t index =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (index >= count) {
    return;
  }
  output[index] = dense_values[global_indices[index]];
}

__global__ void ExportAffineWrenchesKernel(const AffineRangeData *ranges,
                                           std::size_t body_count,
                                           const std::int32_t *global_indices,
                                           const Vec3 *local_vertices,
                                           const Vec3 *dense_values,
                                           const double *current_transforms,
                                           double *output) {
  const std::size_t body =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (body >= body_count) {
    return;
  }
  const AffineRangeData &range = ranges[body];
  const double *transform = current_transforms + body * 16;
  const Vec3 center = TransformPoint(transform, range.center_of_mass);
  Vec3 force{0.0, 0.0, 0.0};
  Vec3 torque{0.0, 0.0, 0.0};
  for (std::size_t local = 0; local < range.vertex_count; ++local) {
    const std::size_t index = range.vertex_offset + local;
    const Vec3 value = dense_values[global_indices[index]];
    const Vec3 point = TransformPoint(transform, local_vertices[index]);
    const Vec3 arm{point.x - center.x, point.y - center.y, point.z - center.z};
    const Vec3 moment = Cross(arm, value);
    force.x += value.x;
    force.y += value.y;
    force.z += value.z;
    torque.x += moment.x;
    torque.y += moment.y;
    torque.z += moment.z;
  }
  output[body * 6] = force.x;
  output[body * 6 + 1] = force.y;
  output[body * 6 + 2] = force.z;
  output[body * 6 + 3] = torque.x;
  output[body * 6 + 4] = torque.y;
  output[body * 6 + 5] = torque.z;
}

__global__ void AccumulateAttachmentWrenchesKernel(
    const AttachmentData *attachments, std::size_t attachment_count,
    std::size_t body_count, const Vec3 *current_positions,
    const double *target_transforms, double inverse_time_step_squared,
    double *output) {
  const std::size_t body =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (body >= body_count) {
    return;
  }
  Vec3 force{0.0, 0.0, 0.0};
  Vec3 torque{0.0, 0.0, 0.0};
  const double *transform = target_transforms + body * 16;
  for (std::size_t index = 0; index < attachment_count; ++index) {
    const AttachmentData &attachment = attachments[index];
    if (attachment.affine_output_offset != body) {
      continue;
    }
    const Vec3 target = TransformPoint(transform, attachment.local_position);
    const Vec3 center =
        TransformPoint(transform, attachment.local_center_of_mass);
    const Vec3 position =
        current_positions[attachment.deformable_output_offset];
    const double scale = attachment.strength_rate * attachment.vertex_mass *
                         inverse_time_step_squared;
    const Vec3 value{scale * (position.x - target.x),
                     scale * (position.y - target.y),
                     scale * (position.z - target.z)};
    const Vec3 arm{target.x - center.x, target.y - center.y,
                   target.z - center.z};
    const Vec3 moment = Cross(arm, value);
    force.x += value.x;
    force.y += value.y;
    force.z += value.z;
    torque.x += moment.x;
    torque.y += moment.y;
    torque.z += moment.z;
  }
  output[body * 6] += force.x;
  output[body * 6 + 1] += force.y;
  output[body * 6 + 2] += force.z;
  output[body * 6 + 3] += torque.x;
  output[body * 6 + 4] += torque.y;
  output[body * 6 + 5] += torque.z;
}

constexpr std::size_t kThreads = 256;

std::size_t BlockCount(std::size_t count) {
  return (count + kThreads - 1) / kThreads;
}

} // namespace

class DeviceCouplingWorkspace::Impl final {
public:
  Impl(std::uint32_t device_index, std::size_t deformable_vertex_count,
       std::size_t affine_body_count,
       std::vector<DeviceDeformableContactRange> deformable_ranges,
       std::vector<DeviceAffineContactRange> affine_ranges,
       std::vector<DeviceAttachmentVertex> attachment_vertices)
      : device_index_(device_index),
        deformable_vertex_count_(deformable_vertex_count),
        affine_body_count_(affine_body_count) {
    RequireCuda(cudaSetDevice(static_cast<int>(device_index_)),
                "selecting CUDA device for coupling workspace");
    target_transforms_ = allocations_.Allocate<double>(
        affine_body_count_ * 16, "allocating affine targets");
    target_velocities_ = allocations_.Allocate<double>(
        affine_body_count_ * 16, "allocating affine velocities");
    attachment_aim_positions_ =
        allocations_.Allocate<Vec3>(deformable_vertex_count_,
                                    "allocating attachment aims");
    current_deformable_positions_ = allocations_.Allocate<Vec3>(
        deformable_vertex_count_, "allocating current FEM positions");
    current_affine_transforms_ = allocations_.Allocate<double>(
        affine_body_count_ * 16, "allocating current affine transforms");

    std::vector<std::int32_t> deformable_global_indices(
        deformable_vertex_count_, -1);
    std::size_t dense_count = 0;
    for (const DeviceDeformableContactRange &range : deformable_ranges) {
      if (range.output_offset + range.vertex_count > deformable_vertex_count_ ||
          range.global_vertex_offset < 0) {
        throw std::runtime_error(
            "invalid deformable contact range for CUDA coupling");
      }
      for (std::size_t local = 0; local < range.vertex_count; ++local) {
        const std::size_t output = range.output_offset + local;
        const std::size_t global =
            static_cast<std::size_t>(range.global_vertex_offset) + local;
        if (global > static_cast<std::size_t>(
                         std::numeric_limits<std::int32_t>::max())) {
          throw std::runtime_error(
              "contact vertex index exceeds CUDA coupling capacity");
        }
        deformable_global_indices[output] = static_cast<std::int32_t>(global);
        dense_count = std::max(dense_count, global + 1);
      }
    }
    if (!deformable_ranges.empty() &&
        std::ranges::any_of(deformable_global_indices,
                            [](std::int32_t value) { return value < 0; })) {
      throw std::runtime_error(
          "deformable contact ranges do not cover output storage");
    }
    deformable_global_indices_ = allocations_.Allocate<std::int32_t>(
        deformable_global_indices.size(),
        "allocating deformable contact indices");
    Upload(deformable_global_indices_, deformable_global_indices,
           "uploading deformable contact indices");

    std::vector<AffineRangeData> affine_range_data(affine_body_count_);
    std::vector<std::int32_t> affine_global_indices;
    std::vector<Vec3> affine_local_vertices;
    for (const DeviceAffineContactRange &range : affine_ranges) {
      if (range.output_offset >= affine_body_count_ ||
          range.global_vertex_offset < 0 ||
          range.local_vertices.size() % 3 != 0) {
        throw std::runtime_error(
            "invalid affine contact range for CUDA coupling");
      }
      const std::size_t count = range.local_vertices.size() / 3;
      AffineRangeData &output = affine_range_data[range.output_offset];
      output.vertex_offset = affine_global_indices.size();
      output.vertex_count = count;
      output.center_of_mass =
          Vec3{range.local_center_of_mass[0], range.local_center_of_mass[1],
               range.local_center_of_mass[2]};
      for (std::size_t local = 0; local < count; ++local) {
        const std::size_t global =
            static_cast<std::size_t>(range.global_vertex_offset) + local;
        if (global > static_cast<std::size_t>(
                         std::numeric_limits<std::int32_t>::max())) {
          throw std::runtime_error(
              "contact vertex index exceeds CUDA coupling capacity");
        }
        affine_global_indices.push_back(static_cast<std::int32_t>(global));
        affine_local_vertices.push_back(
            Vec3{range.local_vertices[local * 3],
                 range.local_vertices[local * 3 + 1],
                 range.local_vertices[local * 3 + 2]});
        dense_count = std::max(dense_count, global + 1);
      }
    }
    affine_ranges_ = allocations_.Allocate<AffineRangeData>(
        affine_range_data.size(), "allocating affine contact ranges");
    affine_global_indices_ = allocations_.Allocate<std::int32_t>(
        affine_global_indices.size(), "allocating affine contact indices");
    affine_local_vertices_ = allocations_.Allocate<Vec3>(
        affine_local_vertices.size(), "allocating affine local vertices");
    Upload(affine_ranges_, affine_range_data,
           "uploading affine contact ranges");
    Upload(affine_global_indices_, affine_global_indices,
           "uploading affine contact indices");
    Upload(affine_local_vertices_, affine_local_vertices,
           "uploading affine local vertices");

    std::vector<AttachmentData> attachments;
    attachments.reserve(attachment_vertices.size());
    for (const DeviceAttachmentVertex &value : attachment_vertices) {
      if (value.deformable_output_offset >= deformable_vertex_count_ ||
          value.affine_output_offset >= affine_body_count_) {
        throw std::runtime_error(
            "invalid attachment mapping for CUDA coupling");
      }
      attachments.push_back(AttachmentData{
          value.deformable_output_offset, value.affine_output_offset,
          Vec3{value.link_local_position[0], value.link_local_position[1],
               value.link_local_position[2]},
          Vec3{value.link_local_center_of_mass[0],
               value.link_local_center_of_mass[1],
               value.link_local_center_of_mass[2]},
          value.vertex_mass, value.strength_rate});
    }
    attachment_count_ = attachments.size();
    attachments_ = allocations_.Allocate<AttachmentData>(
        attachments.size(), "allocating attachment mappings");
    Upload(attachments_, attachments, "uploading attachment mappings");

    dense_force_count_ = dense_count;
    dense_forces_ = allocations_.Allocate<Vec3>(
        dense_force_count_, "allocating dense contact forces");
    const std::size_t initial_gradient_capacity = std::max<std::size_t>(
        4096, std::max(deformable_vertex_count_ + affine_global_indices.size(),
                       std::size_t{1}) *
                  16);
    EnsureGradientCapacity(initial_gradient_capacity);
  }

  ~Impl() {
    cudaSetDevice(static_cast<int>(device_index_));
  }

  void EnsureGradientCapacity(std::size_t required) {
    if (required <= gradient_capacity_) {
      return;
    }
    std::size_t capacity = std::max<std::size_t>(gradient_capacity_, 1);
    while (capacity < required) {
      if (capacity > std::numeric_limits<std::size_t>::max() / 2) {
        throw std::runtime_error("contact gradient capacity overflow");
      }
      capacity *= 2;
    }
    allocations_.Release(raw_keys_);
    allocations_.Release(raw_values_);
    allocations_.Release(sorted_keys_);
    allocations_.Release(sorted_values_);
    allocations_.Release(unique_keys_);
    allocations_.Release(unique_values_);
    raw_keys_ = allocations_.Allocate<std::int32_t>(
        capacity, "allocating contact gradient keys");
    raw_values_ = allocations_.Allocate<Vec3>(
        capacity, "allocating contact gradient values");
    sorted_keys_ = allocations_.Allocate<std::int32_t>(
        capacity, "allocating sorted contact keys");
    sorted_values_ = allocations_.Allocate<Vec3>(
        capacity, "allocating sorted contact values");
    unique_keys_ = allocations_.Allocate<std::int32_t>(
        capacity, "allocating unique contact keys");
    unique_values_ = allocations_.Allocate<Vec3>(
        capacity, "allocating unique contact values");
    if (unique_count_ == nullptr) {
      unique_count_ = allocations_.Allocate<std::int32_t>(
          1, "allocating unique contact count");
    }

    std::size_t sort_bytes = 0;
    std::size_t reduce_bytes = 0;
    RequireCuda(cub::DeviceRadixSort::SortPairs(nullptr, sort_bytes, raw_keys_,
                                                sorted_keys_, raw_values_,
                                                sorted_values_, capacity),
                "sizing contact radix-sort workspace");
    RequireCuda(cub::DeviceReduce::ReduceByKey(
                    nullptr, reduce_bytes, sorted_keys_, unique_keys_,
                    sorted_values_, unique_values_, unique_count_, Vec3Add{},
                    capacity),
                "sizing contact reduction workspace");
    const std::size_t temporary_bytes = std::max(sort_bytes, reduce_bytes);
    if (temporary_bytes > temporary_storage_bytes_) {
      allocations_.Release(temporary_storage_);
      temporary_storage_ = allocations_.AllocateBytes(
          temporary_bytes, "allocating contact reduction workspace");
      temporary_storage_bytes_ = temporary_bytes;
    }
    gradient_capacity_ = capacity;
    ++allocation_count_;
  }

  DeviceAllocations allocations_;
  std::uint32_t device_index_{0};
  std::size_t deformable_vertex_count_{0};
  std::size_t affine_body_count_{0};
  std::size_t attachment_count_{0};
  std::size_t dense_force_count_{0};
  std::size_t gradient_capacity_{0};
  std::size_t temporary_storage_bytes_{0};
  std::size_t allocation_count_{0};
  double *target_transforms_{nullptr};
  double *target_velocities_{nullptr};
  Vec3 *attachment_aim_positions_{nullptr};
  Vec3 *current_deformable_positions_{nullptr};
  double *current_affine_transforms_{nullptr};
  std::int32_t *deformable_global_indices_{nullptr};
  AffineRangeData *affine_ranges_{nullptr};
  std::int32_t *affine_global_indices_{nullptr};
  Vec3 *affine_local_vertices_{nullptr};
  AttachmentData *attachments_{nullptr};
  Vec3 *dense_forces_{nullptr};
  std::int32_t *raw_keys_{nullptr};
  Vec3 *raw_values_{nullptr};
  std::int32_t *sorted_keys_{nullptr};
  Vec3 *sorted_values_{nullptr};
  std::int32_t *unique_keys_{nullptr};
  Vec3 *unique_values_{nullptr};
  std::int32_t *unique_count_{nullptr};
  void *temporary_storage_{nullptr};
};

DeviceCouplingWorkspace::DeviceCouplingWorkspace(
    std::uint32_t device_index, std::size_t deformable_vertex_count,
    std::size_t affine_body_count,
    std::vector<DeviceDeformableContactRange> deformable_ranges,
    std::vector<DeviceAffineContactRange> affine_ranges,
    std::vector<DeviceAttachmentVertex> attachment_vertices)
    : impl_(std::make_unique<Impl>(
          device_index, deformable_vertex_count, affine_body_count,
          std::move(deformable_ranges), std::move(affine_ranges),
          std::move(attachment_vertices))) {}

DeviceCouplingWorkspace::~DeviceCouplingWorkspace() = default;

void DeviceCouplingWorkspace::StageTargets(const double *row_major_targets,
                                           const double *target_twists) {
  if (impl_->affine_body_count_ == 0) {
    return;
  }
  StageTargetsKernel<<<BlockCount(impl_->affine_body_count_), kThreads>>>(
      row_major_targets, target_twists, impl_->target_transforms_,
      impl_->target_velocities_, impl_->affine_body_count_);
  RequireCuda(cudaGetLastError(), "staging affine targets and twists");
  if (impl_->attachment_count_ != 0) {
    StageAttachmentAimsKernel<<<BlockCount(impl_->attachment_count_),
                                kThreads>>>(
        impl_->attachments_, impl_->attachment_count_,
        impl_->target_transforms_, impl_->attachment_aim_positions_);
    RequireCuda(cudaGetLastError(), "staging attachment aims");
  }
}

void DeviceCouplingWorkspace::ExportReactions(
    std::span<const DeviceContactGradientView> gradients,
    double inverse_time_step_squared, double *deformable_contact_forces,
    double *affine_contact_wrenches, bool export_deformable_contact_forces,
    bool export_affine_contact_wrenches) {
  std::size_t total_count = 0;
  for (const DeviceContactGradientView &view : gradients) {
    if (view.count > std::numeric_limits<std::size_t>::max() - total_count) {
      throw std::runtime_error("contact gradient count overflow");
    }
    total_count += view.count;
  }
  impl_->EnsureGradientCapacity(std::max<std::size_t>(total_count, 1));
  if (impl_->dense_force_count_ != 0) {
    RequireCuda(cudaMemsetAsync(impl_->dense_forces_, 0,
                                impl_->dense_force_count_ * sizeof(Vec3)),
                "clearing dense contact forces");
  }
  if (export_affine_contact_wrenches && impl_->affine_body_count_ != 0) {
    RequireCuda(cudaMemsetAsync(affine_contact_wrenches, 0,
                                impl_->affine_body_count_ * 6 * sizeof(double)),
                "clearing affine contact wrenches");
  }

  std::size_t offset = 0;
  for (const DeviceContactGradientView &view : gradients) {
    if (view.count == 0) {
      continue;
    }
    RequireCuda(cudaMemcpyAsync(impl_->raw_keys_ + offset, view.vertex_indices,
                                view.count * sizeof(std::int32_t),
                                cudaMemcpyDeviceToDevice),
                "gathering contact gradient indices");
    RequireCuda(cudaMemcpyAsync(impl_->raw_values_ + offset, view.gradients,
                                view.count * sizeof(Vec3),
                                cudaMemcpyDeviceToDevice),
                "gathering contact gradient values");
    offset += view.count;
  }

  if (total_count != 0) {
    ScaleGradientsKernel<<<BlockCount(total_count), kThreads>>>(
        impl_->raw_values_, total_count, -inverse_time_step_squared);
    RequireCuda(cudaGetLastError(), "scaling contact gradients");

    std::size_t temporary_bytes = impl_->temporary_storage_bytes_;
    RequireCuda(cub::DeviceRadixSort::SortPairs(
                    impl_->temporary_storage_, temporary_bytes,
                    impl_->raw_keys_, impl_->sorted_keys_, impl_->raw_values_,
                    impl_->sorted_values_, total_count),
                "sorting contact gradients");
    temporary_bytes = impl_->temporary_storage_bytes_;
    RequireCuda(cub::DeviceReduce::ReduceByKey(
                    impl_->temporary_storage_, temporary_bytes,
                    impl_->sorted_keys_, impl_->unique_keys_,
                    impl_->sorted_values_, impl_->unique_values_,
                    impl_->unique_count_, Vec3Add{}, total_count),
                "reducing contact gradients");
    ScatterUniqueGradientsKernel<<<BlockCount(total_count), kThreads>>>(
        impl_->unique_keys_, impl_->unique_values_, impl_->unique_count_,
        total_count, impl_->dense_forces_, impl_->dense_force_count_);
    RequireCuda(cudaGetLastError(), "scattering contact gradients");
  }

  if (export_deformable_contact_forces &&
      impl_->deformable_vertex_count_ != 0) {
    ExportDeformableForcesKernel<<<BlockCount(impl_->deformable_vertex_count_),
                                   kThreads>>>(
        impl_->deformable_global_indices_, impl_->deformable_vertex_count_,
        impl_->dense_forces_,
        reinterpret_cast<Vec3 *>(deformable_contact_forces));
    RequireCuda(cudaGetLastError(), "exporting deformable contact forces");
  }
  if (export_affine_contact_wrenches && impl_->affine_body_count_ != 0) {
    ExportAffineWrenchesKernel<<<BlockCount(impl_->affine_body_count_),
                                 kThreads>>>(
        impl_->affine_ranges_, impl_->affine_body_count_,
        impl_->affine_global_indices_, impl_->affine_local_vertices_,
        impl_->dense_forces_, impl_->current_affine_transforms_,
        affine_contact_wrenches);
    RequireCuda(cudaGetLastError(), "exporting affine contact wrenches");
  }
  if (export_affine_contact_wrenches && impl_->attachment_count_ != 0) {
    AccumulateAttachmentWrenchesKernel<<<BlockCount(impl_->affine_body_count_),
                                         kThreads>>>(
        impl_->attachments_, impl_->attachment_count_,
        impl_->affine_body_count_, impl_->current_deformable_positions_,
        impl_->target_transforms_, inverse_time_step_squared,
        affine_contact_wrenches);
    RequireCuda(cudaGetLastError(), "exporting attachment contact wrenches");
  }
}

void *DeviceCouplingWorkspace::target_transforms() const {
  return impl_->target_transforms_;
}

void *DeviceCouplingWorkspace::target_velocities() const {
  return impl_->target_velocities_;
}

void *DeviceCouplingWorkspace::attachment_aim_positions() const {
  return impl_->attachment_aim_positions_;
}

void *DeviceCouplingWorkspace::current_deformable_positions() const {
  return impl_->current_deformable_positions_;
}

void *DeviceCouplingWorkspace::current_affine_transforms() const {
  return impl_->current_affine_transforms_;
}

std::size_t DeviceCouplingWorkspace::deformable_vertex_count() const {
  return impl_->deformable_vertex_count_;
}

std::size_t DeviceCouplingWorkspace::affine_body_count() const {
  return impl_->affine_body_count_;
}

std::size_t DeviceCouplingWorkspace::allocation_count() const {
  return impl_->allocation_count_;
}

} // namespace gobot::libuipc_solver
