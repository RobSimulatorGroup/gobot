/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * This file is created by Qiqi Wu, 23-3-23
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/rendering/renderer_compositor.hpp"
#include "gobot/error_macros.hpp"
#include "gobot/rendering/texture_storage.hpp"
#include "gobot/rendering/frame_buffer_cache.hpp"
#include "gobot/rendering/scene_renderer.hpp"

namespace gobot {

Rasterizer* Rasterizer::s_singleton = nullptr;

Rasterizer *(*Rasterizer::CreateFunc)() = nullptr;

Rasterizer::Rasterizer() {
    s_singleton = this;
}

Rasterizer* Rasterizer::Create() {
    ERR_FAIL_COND_V_MSG(CreateFunc == nullptr, nullptr, "No rasterizer backend is registered.");
    return CreateFunc();
}

Rasterizer::~Rasterizer() {
    s_singleton = nullptr;
}

Rasterizer* Rasterizer::GetInstance() {
    ERR_FAIL_COND_V_MSG(s_singleton == nullptr, nullptr, "Must call this after initialize Rasterizer");
    return s_singleton;
}



}
