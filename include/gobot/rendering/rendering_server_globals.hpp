/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * This file is created by Qiqi Wu, 23-3-23
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "gobot/rendering/renderer_compositor.hpp"
#include "gobot/rendering/texture_storage.hpp"
#include "gobot/rendering/renderer_viewport.hpp"
#include "gobot/rendering/render_utilities.hpp"
#include "gobot/rendering/material_storage.hpp"
#include "gobot/rendering/mesh_storage.hpp"
#include "gobot/rendering/renderer_debug_draw.hpp"
#include "gobot/rendering/renderer_scene_render.hpp"

namespace gobot {

#define RSG RenderingServerGlobals

class RenderingServerGlobals {
public:
    static bool threaded;

    static RendererUtilities *utilities;

    static Rasterizer *rasterizer;

    static RendererTextureStorage* texture_storage;

    static RendererViewport* viewport;

    static MaterialStorage* material_storage;

    static MeshStorage* mesh_storage;

    static RendererSceneRender* scene;

    static RendererDebugDraw* debug_draw;

};


}
