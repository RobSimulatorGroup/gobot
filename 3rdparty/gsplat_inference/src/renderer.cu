/*
 * SPDX-FileCopyrightText: Copyright 2025 Nerfstudio Team
 * SPDX-FileCopyrightText: Copyright 2026 NVIDIA CORPORATION & AFFILIATES
 * SPDX-License-Identifier: Apache-2.0
 *
 * Modified by the Gobot project for raw-CUDA, inference-only use. This file
 * implements a gsplat-derived projection/binning/sorting/compositing path
 * without its ATen ownership layer or training dependencies.
 */

#include "gsplat_inference/renderer.h"

#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <new>
#include <string>

namespace gsplat_inference {
namespace {

constexpr int kTileSize = 16;
constexpr float kMinimumAlpha = 1.0f / 255.0f;
constexpr float kMaximumAlpha = 0.99f;

struct ProjectedGaussian {
    float2 mean;
    float depth;
    float3 conic;
    float3 color;
    float opacity;
    int radius;
    int valid;
};

void SetError(char* destination, std::size_t size, const std::string& message) {
    if (destination == nullptr || size == 0) return;
    const std::size_t count = std::min(size - 1u, message.size());
    std::memcpy(destination, message.data(), count);
    destination[count] = '\0';
}

std::string CudaError(cudaError_t result, const char* operation) {
    return std::string(operation) + " failed: " + cudaGetErrorName(result) +
           " (" + cudaGetErrorString(result) + ")";
}

template <typename T>
bool ResizeDevice(T** pointer, std::size_t* capacity, std::size_t count, std::string* error) {
    if (count <= *capacity) return true;
    if (*pointer != nullptr) cudaFree(*pointer);
    *pointer = nullptr;
    *capacity = 0;
    const cudaError_t result = cudaMalloc(reinterpret_cast<void**>(pointer), count * sizeof(T));
    if (result != cudaSuccess) {
        *error = CudaError(result, "cudaMalloc");
        return false;
    }
    *capacity = count;
    return true;
}

template <typename T>
void FreeDevice(T*& pointer) {
    if (pointer != nullptr) cudaFree(pointer);
    pointer = nullptr;
}

__device__ float MatrixAt(const float* matrix, int row, int column) {
    return matrix[column * 4 + row];
}

__device__ float3 TransformPoint(const float* matrix, const float3& point) {
    return make_float3(
            MatrixAt(matrix, 0, 0) * point.x + MatrixAt(matrix, 0, 1) * point.y +
                    MatrixAt(matrix, 0, 2) * point.z + MatrixAt(matrix, 0, 3),
            MatrixAt(matrix, 1, 0) * point.x + MatrixAt(matrix, 1, 1) * point.y +
                    MatrixAt(matrix, 1, 2) * point.z + MatrixAt(matrix, 1, 3),
            MatrixAt(matrix, 2, 0) * point.x + MatrixAt(matrix, 2, 1) * point.y +
                    MatrixAt(matrix, 2, 2) * point.z + MatrixAt(matrix, 2, 3));
}

__device__ float3 Normalize(const float3& value) {
    const float length = sqrtf(value.x * value.x + value.y * value.y + value.z * value.z);
    const float inverse = length > 1e-12f ? 1.0f / length : 0.0f;
    return make_float3(value.x * inverse, value.y * inverse, value.z * inverse);
}

__device__ void QuaternionMatrix(const float4& q, float rotation[9]) {
    const float w = q.x;
    const float x = q.y;
    const float y = q.z;
    const float z = q.w;
    rotation[0] = 1.0f - 2.0f * (y * y + z * z);
    rotation[1] = 2.0f * (x * y - z * w);
    rotation[2] = 2.0f * (x * z + y * w);
    rotation[3] = 2.0f * (x * y + z * w);
    rotation[4] = 1.0f - 2.0f * (x * x + z * z);
    rotation[5] = 2.0f * (y * z - x * w);
    rotation[6] = 2.0f * (x * z - y * w);
    rotation[7] = 2.0f * (y * z + x * w);
    rotation[8] = 1.0f - 2.0f * (x * x + y * y);
}

__device__ float3 EvaluateSh(const float* coefficients, int degree, const float3& direction) {
    constexpr float c0 = 0.28209479177387814f;
    constexpr float c1 = 0.4886025119029199f;
    constexpr float c2[5] = {
            1.0925484305920792f, -1.0925484305920792f, 0.31539156525252005f,
            -1.0925484305920792f, 0.5462742152960396f};
    constexpr float c3[7] = {
            -0.5900435899266435f, 2.890611442640554f, -0.4570457994644658f,
            0.3731763325901154f, -0.4570457994644658f, 1.445305721320277f,
            -0.5900435899266435f};
    float3 result = make_float3(
            c0 * coefficients[0], c0 * coefficients[1], c0 * coefficients[2]);
    const auto add = [&](int coefficient, float basis) {
        result.x += basis * coefficients[coefficient * 3];
        result.y += basis * coefficients[coefficient * 3 + 1];
        result.z += basis * coefficients[coefficient * 3 + 2];
    };
    if (degree >= 1) {
        add(1, -c1 * direction.y);
        add(2, c1 * direction.z);
        add(3, -c1 * direction.x);
    }
    if (degree >= 2) {
        const float xx = direction.x * direction.x;
        const float yy = direction.y * direction.y;
        const float zz = direction.z * direction.z;
        add(4, c2[0] * direction.x * direction.y);
        add(5, c2[1] * direction.y * direction.z);
        add(6, c2[2] * (2.0f * zz - xx - yy));
        add(7, c2[3] * direction.x * direction.z);
        add(8, c2[4] * (xx - yy));
    }
    if (degree >= 3) {
        const float xx = direction.x * direction.x;
        const float yy = direction.y * direction.y;
        const float zz = direction.z * direction.z;
        add(9, c3[0] * direction.y * (3.0f * xx - yy));
        add(10, c3[1] * direction.x * direction.y * direction.z);
        add(11, c3[2] * direction.y * (4.0f * zz - xx - yy));
        add(12, c3[3] * direction.z * (2.0f * zz - 3.0f * xx - 3.0f * yy));
        add(13, c3[4] * direction.x * (4.0f * zz - xx - yy));
        add(14, c3[5] * direction.z * (xx - yy));
        add(15, c3[6] * direction.x * (xx - 3.0f * yy));
    }
    result.x = fmaxf(result.x + 0.5f, 0.0f);
    result.y = fmaxf(result.y + 0.5f, 0.0f);
    result.z = fmaxf(result.z + 0.5f, 0.0f);
    return result;
}

__global__ void ProjectKernel(const float* means,
                              const float* rotations,
                              const float* scales,
                              const float* opacities,
                              const float* sh,
                              std::size_t count,
                              int sh_degree,
                              const float* view,
                              const float* projection,
                              const float* model,
                              float3 camera_position,
                              float near_plane,
                              float far_plane,
                              int width,
                              int height,
                              bool top_left_origin,
                              int tile_columns,
                              int tile_rows,
                              ProjectedGaussian* projected,
                              std::uint32_t* intersection_counts) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    ProjectedGaussian output{};
    const float3 local_mean = make_float3(
            means[index * 3], means[index * 3 + 1], means[index * 3 + 2]);
    const float3 world_mean = TransformPoint(model, local_mean);
    const float3 view_mean = TransformPoint(view, world_mean);
    const float depth = -view_mean.z;
    if (!(depth > near_plane && depth < far_plane)) {
        projected[index] = output;
        intersection_counts[index] = 0;
        return;
    }

    const float fx = 0.5f * static_cast<float>(width) * MatrixAt(projection, 0, 0);
    const float fy = 0.5f * static_cast<float>(height) * MatrixAt(projection, 1, 1);
    const float sign_y = top_left_origin ? -1.0f : 1.0f;
    const float camera_x = view_mean.x;
    const float camera_y = sign_y * view_mean.y;
    output.mean = make_float2(
            fx * camera_x / depth + 0.5f * static_cast<float>(width),
            fy * camera_y / depth + 0.5f * static_cast<float>(height));
    output.depth = depth;

    float quaternion_rotation[9];
    QuaternionMatrix(make_float4(
            rotations[index * 4], rotations[index * 4 + 1],
            rotations[index * 4 + 2], rotations[index * 4 + 3]), quaternion_rotation);
    float a[9]{};
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            float value = 0.0f;
            for (int k = 0; k < 3; ++k) {
                value += MatrixAt(model, row, k) * quaternion_rotation[k * 3 + column];
            }
            a[row * 3 + column] = value * scales[index * 3 + column];
        }
    }
    float covariance_world[9]{};
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            for (int k = 0; k < 3; ++k) {
                covariance_world[row * 3 + column] += a[row * 3 + k] * a[column * 3 + k];
            }
        }
    }
    float camera_rotation[9];
    for (int column = 0; column < 3; ++column) {
        camera_rotation[column] = MatrixAt(view, 0, column);
        camera_rotation[3 + column] = sign_y * MatrixAt(view, 1, column);
        camera_rotation[6 + column] = -MatrixAt(view, 2, column);
    }
    float covariance_camera[9]{};
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    covariance_camera[row * 3 + column] +=
                            camera_rotation[row * 3 + i] * covariance_world[i * 3 + j] *
                            camera_rotation[column * 3 + j];
                }
            }
        }
    }
    const float inverse_depth = 1.0f / depth;
    const float j00 = fx * inverse_depth;
    const float j02 = -fx * camera_x * inverse_depth * inverse_depth;
    const float j11 = fy * inverse_depth;
    const float j12 = -fy * camera_y * inverse_depth * inverse_depth;
    const float covariance_xx =
            j00 * j00 * covariance_camera[0] + 2.0f * j00 * j02 * covariance_camera[2] +
            j02 * j02 * covariance_camera[8] + 0.3f;
    const float covariance_xy =
            j00 * j11 * covariance_camera[1] + j00 * j12 * covariance_camera[2] +
            j02 * j11 * covariance_camera[7] + j02 * j12 * covariance_camera[8];
    const float covariance_yy =
            j11 * j11 * covariance_camera[4] + 2.0f * j11 * j12 * covariance_camera[5] +
            j12 * j12 * covariance_camera[8] + 0.3f;
    const float determinant = covariance_xx * covariance_yy - covariance_xy * covariance_xy;
    if (!(determinant > 1e-12f)) {
        projected[index] = output;
        intersection_counts[index] = 0;
        return;
    }
    output.conic = make_float3(
            covariance_yy / determinant, -covariance_xy / determinant,
            covariance_xx / determinant);
    const float midpoint = 0.5f * (covariance_xx + covariance_yy);
    const float eigen_radius = sqrtf(fmaxf(
            0.0f, midpoint + sqrtf(fmaxf(0.0f, midpoint * midpoint - determinant))));
    output.radius = static_cast<int>(ceilf(3.0f * eigen_radius));
    if (output.radius <= 0) {
        projected[index] = output;
        intersection_counts[index] = 0;
        return;
    }

    float3 world_direction = make_float3(
            world_mean.x - camera_position.x,
            world_mean.y - camera_position.y,
            world_mean.z - camera_position.z);
    const float3 local_direction = Normalize(make_float3(
            MatrixAt(model, 0, 0) * world_direction.x + MatrixAt(model, 1, 0) * world_direction.y +
                    MatrixAt(model, 2, 0) * world_direction.z,
            MatrixAt(model, 0, 1) * world_direction.x + MatrixAt(model, 1, 1) * world_direction.y +
                    MatrixAt(model, 2, 1) * world_direction.z,
            MatrixAt(model, 0, 2) * world_direction.x + MatrixAt(model, 1, 2) * world_direction.y +
                    MatrixAt(model, 2, 2) * world_direction.z));
    const int coefficient_count = (sh_degree + 1) * (sh_degree + 1);
    output.color = EvaluateSh(sh + index * coefficient_count * 3, sh_degree, local_direction);
    output.opacity = fminf(fmaxf(opacities[index], 0.0f), 1.0f);

    const int min_x = max(0, static_cast<int>(floorf(output.mean.x - output.radius)) / kTileSize);
    const int min_y = max(0, static_cast<int>(floorf(output.mean.y - output.radius)) / kTileSize);
    const int max_x = min(tile_columns, static_cast<int>(ceilf(output.mean.x + output.radius)) / kTileSize + 1);
    const int max_y = min(tile_rows, static_cast<int>(ceilf(output.mean.y + output.radius)) / kTileSize + 1);
    if (min_x >= max_x || min_y >= max_y) {
        projected[index] = output;
        intersection_counts[index] = 0;
        return;
    }
    output.valid = 1;
    projected[index] = output;
    intersection_counts[index] = static_cast<std::uint32_t>((max_x - min_x) * (max_y - min_y));
}

__global__ void FillIntersectionsKernel(const ProjectedGaussian* projected,
                                        const std::uint32_t* offsets,
                                        std::size_t count,
                                        int tile_columns,
                                        int tile_rows,
                                        std::uint64_t* keys,
                                        std::uint32_t* gaussian_ids) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count || projected[index].valid == 0) return;
    const ProjectedGaussian gaussian = projected[index];
    const int min_x = max(0, static_cast<int>(floorf(gaussian.mean.x - gaussian.radius)) / kTileSize);
    const int min_y = max(0, static_cast<int>(floorf(gaussian.mean.y - gaussian.radius)) / kTileSize);
    const int max_x = min(tile_columns, static_cast<int>(ceilf(gaussian.mean.x + gaussian.radius)) / kTileSize + 1);
    const int max_y = min(tile_rows, static_cast<int>(ceilf(gaussian.mean.y + gaussian.radius)) / kTileSize + 1);
    std::uint32_t output = offsets[index];
    for (int tile_y = min_y; tile_y < max_y; ++tile_y) {
        for (int tile_x = min_x; tile_x < max_x; ++tile_x) {
            const std::uint32_t tile = static_cast<std::uint32_t>(tile_y * tile_columns + tile_x);
            keys[output] = (static_cast<std::uint64_t>(tile) << 32u) |
                           static_cast<std::uint32_t>(__float_as_uint(gaussian.depth));
            gaussian_ids[output] = static_cast<std::uint32_t>(index);
            ++output;
        }
    }
}

__global__ void BuildRangesKernel(const std::uint64_t* keys,
                                  std::uint32_t intersection_count,
                                  uint2* ranges) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= intersection_count) return;
    const std::uint32_t tile = static_cast<std::uint32_t>(keys[index] >> 32u);
    const std::uint32_t previous = index > 0
                                           ? static_cast<std::uint32_t>(keys[index - 1] >> 32u)
                                           : 0xffffffffu;
    const std::uint32_t next = index + 1u < intersection_count
                                       ? static_cast<std::uint32_t>(keys[index + 1] >> 32u)
                                       : 0xffffffffu;
    if (tile != previous) ranges[tile].x = index;
    if (tile != next) ranges[tile].y = index + 1u;
}

__global__ void RasterKernel(const ProjectedGaussian* projected,
                             const std::uint32_t* sorted_ids,
                             const uint2* ranges,
                             int tile_columns,
                             int width,
                             int height,
                             float3 clear_color,
                             const std::uint32_t* geometry_coverage,
                             std::uint32_t* output) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const std::size_t pixel = static_cast<std::size_t>(y) * width + x;
    if (geometry_coverage != nullptr && geometry_coverage[pixel] == 1u) return;
    const std::uint32_t tile = static_cast<std::uint32_t>((y / kTileSize) * tile_columns + x / kTileSize);
    const uint2 range = ranges[tile];
    float3 color = make_float3(0.0f, 0.0f, 0.0f);
    float transmittance = 1.0f;
    for (std::uint32_t intersection = range.x; intersection < range.y; ++intersection) {
        const ProjectedGaussian gaussian = projected[sorted_ids[intersection]];
        const float dx = static_cast<float>(x) + 0.5f - gaussian.mean.x;
        const float dy = static_cast<float>(y) + 0.5f - gaussian.mean.y;
        const float exponent = -0.5f *
                               (gaussian.conic.x * dx * dx +
                                2.0f * gaussian.conic.y * dx * dy +
                                gaussian.conic.z * dy * dy);
        if (exponent > 0.0f || exponent < -12.0f) continue;
        const float alpha = fminf(kMaximumAlpha, gaussian.opacity * expf(exponent));
        if (alpha < kMinimumAlpha) continue;
        const float weight = alpha * transmittance;
        color.x += weight * gaussian.color.x;
        color.y += weight * gaussian.color.y;
        color.z += weight * gaussian.color.z;
        transmittance *= 1.0f - alpha;
        if (transmittance < 1e-4f) break;
    }
    color.x = fminf(fmaxf(color.x + transmittance * clear_color.x, 0.0f), 1.0f);
    color.y = fminf(fmaxf(color.y + transmittance * clear_color.y, 0.0f), 1.0f);
    color.z = fminf(fmaxf(color.z + transmittance * clear_color.z, 0.0f), 1.0f);
    const std::uint32_t red = static_cast<std::uint32_t>(color.x * 255.0f + 0.5f);
    const std::uint32_t green = static_cast<std::uint32_t>(color.y * 255.0f + 0.5f);
    const std::uint32_t blue = static_cast<std::uint32_t>(color.z * 255.0f + 0.5f);
    output[pixel] = red | (green << 8u) | (blue << 16u) | (255u << 24u);
}

} // namespace

struct Renderer {
    float* means = nullptr;
    float* rotations = nullptr;
    float* scales = nullptr;
    float* opacities = nullptr;
    float* sh = nullptr;
    std::size_t scene_capacity = 0;
    std::size_t sh_capacity = 0;
    std::size_t count = 0;
    int sh_degree = 0;

    ProjectedGaussian* projected = nullptr;
    std::size_t projected_capacity = 0;
    std::uint32_t* counts = nullptr;
    std::size_t counts_capacity = 0;
    std::uint32_t* offsets = nullptr;
    std::size_t offsets_capacity = 0;
    std::uint64_t* keys = nullptr;
    std::uint64_t* sorted_keys = nullptr;
    std::uint32_t* ids = nullptr;
    std::uint32_t* sorted_ids = nullptr;
    std::size_t intersection_capacity = 0;
    uint2* ranges = nullptr;
    std::size_t range_capacity = 0;
    void* scan_storage = nullptr;
    std::size_t scan_storage_capacity = 0;
    void* sort_storage = nullptr;
    std::size_t sort_storage_capacity = 0;
    float* camera_matrices = nullptr;
};

Renderer* Create() {
    return new (std::nothrow) Renderer{};
}

void Release(Renderer* renderer) {
    if (renderer == nullptr) return;
    FreeDevice(renderer->means);
    FreeDevice(renderer->rotations);
    FreeDevice(renderer->scales);
    FreeDevice(renderer->opacities);
    FreeDevice(renderer->sh);
    FreeDevice(renderer->projected);
    FreeDevice(renderer->counts);
    FreeDevice(renderer->offsets);
    FreeDevice(renderer->keys);
    FreeDevice(renderer->sorted_keys);
    FreeDevice(renderer->ids);
    FreeDevice(renderer->sorted_ids);
    FreeDevice(renderer->ranges);
    if (renderer->scan_storage != nullptr) cudaFree(renderer->scan_storage);
    if (renderer->sort_storage != nullptr) cudaFree(renderer->sort_storage);
    FreeDevice(renderer->camera_matrices);
    *renderer = {};
}

void Destroy(Renderer* renderer) {
    if (renderer == nullptr) return;
    Release(renderer);
    delete renderer;
}

bool Upload(Renderer* renderer,
            const SceneData& scene,
            void* cuda_stream,
            char* error,
            std::size_t error_size) {
    if (renderer == nullptr || scene.means == nullptr || scene.rotations_wxyz == nullptr ||
        scene.scales == nullptr || scene.opacities == nullptr || scene.sh_coefficients == nullptr ||
        scene.count == 0 || scene.sh_degree < 0 || scene.sh_degree > 3) {
        SetError(error, error_size, "Invalid gsplat scene upload.");
        return false;
    }
    std::string message;
    const std::size_t coefficient_count = static_cast<std::size_t>((scene.sh_degree + 1) * (scene.sh_degree + 1));
    if (!ResizeDevice(&renderer->means, &renderer->scene_capacity, scene.count * 3u, &message)) {
        SetError(error, error_size, message); return false;
    }
    // The arrays have different element counts, so use independent exact reallocations.
    if (renderer->rotations != nullptr) cudaFree(renderer->rotations);
    if (renderer->scales != nullptr) cudaFree(renderer->scales);
    if (renderer->opacities != nullptr) cudaFree(renderer->opacities);
    renderer->rotations = nullptr; renderer->scales = nullptr; renderer->opacities = nullptr;
    cudaError_t result = cudaMalloc(reinterpret_cast<void**>(&renderer->rotations), scene.count * 4u * sizeof(float));
    if (result == cudaSuccess) result = cudaMalloc(reinterpret_cast<void**>(&renderer->scales), scene.count * 3u * sizeof(float));
    if (result == cudaSuccess) result = cudaMalloc(reinterpret_cast<void**>(&renderer->opacities), scene.count * sizeof(float));
    if (result != cudaSuccess) {
        SetError(error, error_size, CudaError(result, "cudaMalloc scene arrays"));
        return false;
    }
    if (!ResizeDevice(&renderer->sh, &renderer->sh_capacity,
                      scene.count * coefficient_count * 3u, &message) ||
        !ResizeDevice(&renderer->projected, &renderer->projected_capacity, scene.count, &message) ||
        !ResizeDevice(&renderer->counts, &renderer->counts_capacity, scene.count + 1u, &message) ||
        !ResizeDevice(&renderer->offsets, &renderer->offsets_capacity, scene.count + 1u, &message)) {
        SetError(error, error_size, message); return false;
    }
    auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
    result = cudaMemcpyAsync(renderer->means, scene.means, scene.count * 3u * sizeof(float), cudaMemcpyHostToDevice, stream);
    if (result == cudaSuccess) result = cudaMemcpyAsync(renderer->rotations, scene.rotations_wxyz, scene.count * 4u * sizeof(float), cudaMemcpyHostToDevice, stream);
    if (result == cudaSuccess) result = cudaMemcpyAsync(renderer->scales, scene.scales, scene.count * 3u * sizeof(float), cudaMemcpyHostToDevice, stream);
    if (result == cudaSuccess) result = cudaMemcpyAsync(renderer->opacities, scene.opacities, scene.count * sizeof(float), cudaMemcpyHostToDevice, stream);
    if (result == cudaSuccess) result = cudaMemcpyAsync(renderer->sh, scene.sh_coefficients,
                                                         scene.count * coefficient_count * 3u * sizeof(float),
                                                         cudaMemcpyHostToDevice, stream);
    if (result != cudaSuccess) {
        SetError(error, error_size, CudaError(result, "cudaMemcpyAsync scene upload"));
        return false;
    }
    renderer->count = scene.count;
    renderer->sh_degree = scene.sh_degree;
    return true;
}

bool Render(Renderer* renderer,
            const CameraData& camera,
            const RenderTarget& target,
            void* cuda_stream,
            char* error,
            std::size_t error_size) {
    if (renderer == nullptr || renderer->count == 0 || camera.view_column_major == nullptr ||
        camera.projection_column_major == nullptr || camera.model_column_major == nullptr ||
        camera.width <= 0 || camera.height <= 0 || target.rgba8 == nullptr) {
        SetError(error, error_size, "Invalid gsplat render request.");
        return false;
    }
    auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
    cudaError_t result = cudaSuccess;
    if (renderer->camera_matrices == nullptr) {
        result = cudaMalloc(reinterpret_cast<void**>(&renderer->camera_matrices), 48u * sizeof(float));
    }
    if (result != cudaSuccess) {
        SetError(error, error_size, CudaError(result, "cudaMalloc camera matrices"));
        return false;
    }
    float* view = renderer->camera_matrices;
    float* projection = renderer->camera_matrices + 16u;
    float* model = renderer->camera_matrices + 32u;
    result = cudaMemcpyAsync(view, camera.view_column_major, 16u * sizeof(float), cudaMemcpyHostToDevice, stream);
    if (result == cudaSuccess) result = cudaMemcpyAsync(projection, camera.projection_column_major, 16u * sizeof(float), cudaMemcpyHostToDevice, stream);
    if (result == cudaSuccess) result = cudaMemcpyAsync(model, camera.model_column_major, 16u * sizeof(float), cudaMemcpyHostToDevice, stream);
    if (result == cudaSuccess) result = cudaMemsetAsync(renderer->counts, 0,
                                                         (renderer->count + 1u) * sizeof(std::uint32_t), stream);
    if (result != cudaSuccess) {
        SetError(error, error_size, CudaError(result, "gsplat camera upload"));
        return false;
    }

    const int tile_columns = (camera.width + kTileSize - 1) / kTileSize;
    const int tile_rows = (camera.height + kTileSize - 1) / kTileSize;
    const std::size_t tile_count = static_cast<std::size_t>(tile_columns) * tile_rows;
    const int threads = 256;
    const int blocks = static_cast<int>((renderer->count + threads - 1u) / threads);
    ProjectKernel<<<blocks, threads, 0, stream>>>(
            renderer->means, renderer->rotations, renderer->scales, renderer->opacities,
            renderer->sh, renderer->count, renderer->sh_degree, view, projection, model,
            make_float3(camera.camera_position[0], camera.camera_position[1], camera.camera_position[2]),
            camera.near_plane, camera.far_plane, camera.width, camera.height,
            camera.top_left_origin, tile_columns, tile_rows, renderer->projected, renderer->counts);
    result = cudaGetLastError();
    if (result != cudaSuccess) {
        SetError(error, error_size, CudaError(result, "gsplat projection kernel"));
        return false;
    }

    std::size_t scan_bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, scan_bytes, renderer->counts, renderer->offsets,
                                  renderer->count + 1u, stream);
    std::string message;
    if (scan_bytes > renderer->scan_storage_capacity) {
        if (renderer->scan_storage != nullptr) cudaFree(renderer->scan_storage);
        result = cudaMalloc(&renderer->scan_storage, scan_bytes);
        if (result != cudaSuccess) {
            SetError(error, error_size, CudaError(result, "cudaMalloc scan storage"));
            return false;
        }
        renderer->scan_storage_capacity = scan_bytes;
    }
    result = cub::DeviceScan::ExclusiveSum(renderer->scan_storage, scan_bytes,
                                           renderer->counts, renderer->offsets,
                                           renderer->count + 1u, stream);
    std::uint32_t intersection_count = 0;
    if (result == cudaSuccess) result = cudaMemcpyAsync(
            &intersection_count, renderer->offsets + renderer->count,
            sizeof(intersection_count), cudaMemcpyDeviceToHost, stream);
    if (result == cudaSuccess) result = cudaStreamSynchronize(stream);
    if (result != cudaSuccess) {
        SetError(error, error_size, CudaError(result, "gsplat intersection count"));
        return false;
    }

    if (!ResizeDevice(&renderer->ranges, &renderer->range_capacity, tile_count, &message)) {
        SetError(error, error_size, message); return false;
    }
    result = cudaMemsetAsync(renderer->ranges, 0, tile_count * sizeof(uint2), stream);
    if (intersection_count > 0u) {
        if (intersection_count > renderer->intersection_capacity) {
            FreeDevice(renderer->keys); FreeDevice(renderer->sorted_keys);
            FreeDevice(renderer->ids); FreeDevice(renderer->sorted_ids);
            result = cudaMalloc(reinterpret_cast<void**>(&renderer->keys), intersection_count * sizeof(std::uint64_t));
            if (result == cudaSuccess) result = cudaMalloc(reinterpret_cast<void**>(&renderer->sorted_keys), intersection_count * sizeof(std::uint64_t));
            if (result == cudaSuccess) result = cudaMalloc(reinterpret_cast<void**>(&renderer->ids), intersection_count * sizeof(std::uint32_t));
            if (result == cudaSuccess) result = cudaMalloc(reinterpret_cast<void**>(&renderer->sorted_ids), intersection_count * sizeof(std::uint32_t));
            if (result != cudaSuccess) {
                SetError(error, error_size, CudaError(result, "cudaMalloc intersections"));
                return false;
            }
            renderer->intersection_capacity = intersection_count;
        }
        FillIntersectionsKernel<<<blocks, threads, 0, stream>>>(
                renderer->projected, renderer->offsets, renderer->count,
                tile_columns, tile_rows, renderer->keys, renderer->ids);
        std::size_t sort_bytes = 0;
        cub::DeviceRadixSort::SortPairs(nullptr, sort_bytes,
                                        renderer->keys, renderer->sorted_keys,
                                        renderer->ids, renderer->sorted_ids,
                                        intersection_count, 0, 64, stream);
        if (sort_bytes > renderer->sort_storage_capacity) {
            if (renderer->sort_storage != nullptr) cudaFree(renderer->sort_storage);
            result = cudaMalloc(&renderer->sort_storage, sort_bytes);
            if (result != cudaSuccess) {
                SetError(error, error_size, CudaError(result, "cudaMalloc sort storage"));
                return false;
            }
            renderer->sort_storage_capacity = sort_bytes;
        }
        result = cub::DeviceRadixSort::SortPairs(renderer->sort_storage, sort_bytes,
                                                 renderer->keys, renderer->sorted_keys,
                                                 renderer->ids, renderer->sorted_ids,
                                                 intersection_count, 0, 64, stream);
        if (result == cudaSuccess) {
            const int range_blocks = static_cast<int>((intersection_count + threads - 1u) / threads);
            BuildRangesKernel<<<range_blocks, threads, 0, stream>>>(
                    renderer->sorted_keys, intersection_count, renderer->ranges);
            result = cudaGetLastError();
        }
    }
    if (result == cudaSuccess) {
        const dim3 block(kTileSize, kTileSize);
        const dim3 grid(tile_columns, tile_rows);
        RasterKernel<<<grid, block, 0, stream>>>(
                renderer->projected, renderer->sorted_ids, renderer->ranges,
                tile_columns, camera.width, camera.height,
                make_float3(camera.clear_color[0], camera.clear_color[1], camera.clear_color[2]),
                target.geometry_coverage, target.rgba8);
        result = cudaGetLastError();
    }
    if (result != cudaSuccess) {
        SetError(error, error_size, CudaError(result, "gsplat render"));
        return false;
    }
    return true;
}

} // namespace gsplat_inference
