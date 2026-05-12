/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * This file is created by Qiqi Wu, 23-3-23
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace gobot {

class FrameBufferCache;
class RendererSceneRender;
class RendererTextureStorage;
class MaterialStorage;
class MeshStorage;
class RendererUtilities;
class RendererDebugDraw;

class Rasterizer {
protected:
    static Rasterizer*(*CreateFunc)();

public:
    Rasterizer();

    virtual ~Rasterizer();

    static Rasterizer* Create();

    static Rasterizer* GetInstance();

    virtual RendererSceneRender* GetScene() = 0;

    virtual RendererTextureStorage* GetTextureStorage() = 0;

    virtual MaterialStorage* GetMaterialStorage() = 0;

    virtual MeshStorage* GetMeshStorage() = 0;

    virtual RendererDebugDraw* GetDebugDraw() = 0;

    virtual RendererUtilities* GetUtilities() = 0;

    virtual void Initialize() = 0;

    virtual void BeginFrame(double frame_step) = 0;

    virtual void EndFrame(bool p_swap_buffers) = 0;

    virtual void Finalize() = 0;

private:

    static Rasterizer* s_singleton;

};


}
