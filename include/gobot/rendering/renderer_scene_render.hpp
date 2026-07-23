/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * This file is created by Qiqi Wu, 23-6-23
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "gobot/core/rid.hpp"
#include "gobot/rendering/scene_render_items.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>

namespace gobot {

enum class SceneRendererMode {
    Raster,
    RayTracingAuto,
    RealtimeRayTracing,
    ProgressivePathTracing
};

enum class RasterAntiAliasingMode {
    Disabled,
    Fxaa
};

enum class RasterShadowQuality {
    Disabled,
    Low,
    Medium,
    High
};

struct RasterRendererSettings {
    bool frustum_culling = true;
    RasterAntiAliasingMode anti_aliasing = RasterAntiAliasingMode::Fxaa;
    RasterShadowQuality shadow_quality = RasterShadowQuality::Medium;
    RealType shadow_distance = 50.0;
};

struct SceneRendererSettings {
    SceneRendererMode mode = SceneRendererMode::Raster;
    int target_fps = 30;
    int samples_per_frame = 1;
    int max_accumulated_samples = 4096;
    int max_bounces = 4;
    bool denoise = true;
    bool adaptive_quality = true;
    RasterRendererSettings raster;
};

struct SceneRendererCapabilities {
    bool ray_tracing_available = false;
    bool realtime = false;
    bool progressive = false;
    bool denoise = false;
    bool direct_presentation_interop = false;
    bool cpu_render_products = true;
    bool cuda_render_products = false;
    bool raster_frustum_culling = false;
    bool raster_directional_shadows = false;
    bool raster_fxaa = false;
    std::string backend_name = "OpenGL";
    std::string status;
};

struct SceneRendererStats {
    SceneRendererMode active_mode = SceneRendererMode::Raster;
    std::uint64_t accumulated_samples = 0;
    double scene_update_ms = 0.0;
    double acceleration_build_ms = 0.0;
    double render_ms = 0.0;
    double denoise_ms = 0.0;
    double presentation_ms = 0.0;
    double shadow_ms = 0.0;
    double post_process_ms = 0.0;
    std::uint64_t visible_items = 0;
    std::uint64_t culled_items = 0;
    std::uint64_t draw_calls = 0;
    std::uint64_t shadow_draw_calls = 0;
    std::string status;
};

struct RendererRenderProductBuffer {
    void* device_pointer = nullptr;
    std::size_t allocation_size = 0;
    std::size_t pixel_stride_bytes = 0;
};

struct RendererRenderProductFrame {
    std::array<RendererRenderProductBuffer, 5> buffers{};
    int device_id = 0;
    std::shared_ptr<void> owner;
    std::function<bool(std::uint32_t, void*, std::size_t)> copy_to_host;
};

class RendererSceneRender {
public:
    virtual ~RendererSceneRender() = default;

    virtual void RenderScene(const RID& render_target,
                             const RenderSceneSnapshot& scene,
                             const RenderViewSnapshot& view) = 0;

    virtual void SetSettings(const SceneRendererSettings& settings) { settings_ = settings; }

    [[nodiscard]] virtual SceneRendererSettings GetSettings() const { return settings_; }

    [[nodiscard]] virtual SceneRendererCapabilities GetCapabilities() const { return {}; }

    [[nodiscard]] virtual SceneRendererStats GetStats() const { return {}; }

    virtual bool CaptureCudaRenderProduct(const RenderSceneSnapshot&,
                                          const RenderViewSnapshot&,
                                          int,
                                          int,
                                          std::uint32_t,
                                          std::uint32_t,
                                          RendererRenderProductFrame*,
                                          std::string* error) {
        if (error != nullptr) {
            *error = "CUDA render products are unavailable through the active renderer";
        }
        return false;
    }

protected:
    SceneRendererSettings settings_;
};

}
