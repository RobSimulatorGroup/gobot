/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * This file is created by Qiqi Wu, 23-6-23
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "gobot/core/rid.hpp"
#include "gobot/rendering/scene_render_items.hpp"

#include <cstdint>
#include <string>

namespace gobot {

enum class SceneRendererMode {
    Raster,
    RayTracingAuto,
    RealtimeRayTracing,
    ProgressivePathTracing
};

struct SceneRendererSettings {
    SceneRendererMode mode = SceneRendererMode::Raster;
    int target_fps = 30;
    int samples_per_frame = 1;
    int max_accumulated_samples = 4096;
    int max_bounces = 4;
    bool denoise = true;
    bool adaptive_quality = true;
};

struct SceneRendererCapabilities {
    bool ray_tracing_available = false;
    bool realtime = false;
    bool progressive = false;
    bool denoise = false;
    bool direct_presentation_interop = false;
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
    std::string status;
};

class RendererSceneRender {
public:
    virtual ~RendererSceneRender() = default;

    virtual void RenderScene(const RID& render_target, const SceneRenderSnapshot& snapshot) = 0;

    virtual void SetSettings(const SceneRendererSettings& settings) { settings_ = settings; }

    [[nodiscard]] virtual SceneRendererSettings GetSettings() const { return settings_; }

    [[nodiscard]] virtual SceneRendererCapabilities GetCapabilities() const { return {}; }

    [[nodiscard]] virtual SceneRendererStats GetStats() const { return {}; }

protected:
    SceneRendererSettings settings_;
};

}
