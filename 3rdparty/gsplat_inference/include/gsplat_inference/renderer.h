/*
 * SPDX-FileCopyrightText: Copyright 2025 Nerfstudio Team
 * SPDX-FileCopyrightText: Copyright 2026 NVIDIA CORPORATION & AFFILIATES
 * SPDX-License-Identifier: Apache-2.0
 *
 * Modified by the Gobot project for raw-CUDA, inference-only use.
 */

#pragma once

#include <cstddef>
#include <cstdint>

namespace gsplat_inference {

struct SceneData {
    const float* means = nullptr;            // [N, 3]
    const float* rotations_wxyz = nullptr;   // [N, 4]
    const float* scales = nullptr;           // [N, 3], activated and positive
    const float* opacities = nullptr;         // [N], activated to [0, 1]
    const float* sh_coefficients = nullptr;   // [N, K, 3]
    std::size_t count = 0;
    int sh_degree = 0;
};

struct CameraData {
    const float* view_column_major = nullptr;
    const float* projection_column_major = nullptr;
    const float* model_column_major = nullptr;
    float camera_position[3]{};
    float clear_color[3]{};
    float near_plane = 0.05f;
    float far_plane = 4000.0f;
    int width = 0;
    int height = 0;
    bool top_left_origin = false;
};

struct RenderTarget {
    std::uint32_t* rgba8 = nullptr;
    // Optional values: 0=background, 1=normal RGB geometry, 2=proxy geometry.
    // Pixels marked 1 are preserved; background and proxy pixels receive the splat image.
    const std::uint32_t* geometry_coverage = nullptr;
};

struct Renderer;

Renderer* Create();
void Destroy(Renderer* renderer);

bool Upload(Renderer* renderer,
            const SceneData& scene,
            void* cuda_stream,
            char* error,
            std::size_t error_size);

bool Render(Renderer* renderer,
            const CameraData& camera,
            const RenderTarget& target,
            void* cuda_stream,
            char* error,
            std::size_t error_size);

void Release(Renderer* renderer);

} // namespace gsplat_inference
