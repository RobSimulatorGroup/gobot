/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-6-23
*/

#pragma once

#include "gobot/rendering/renderer_compositor.hpp"
#include "gobot/drivers/opengl/texture_storage.hpp"
#include "gobot/drivers/opengl/rasterizer_gles3.hpp"
#include "gobot/drivers/opengl/rasterizer_scene_gles3.hpp"

namespace gobot::opengl {

class RasterizerGLES3 : public RendererCompositor {
public:
    RasterizerGLES3();

    ~RasterizerGLES3() override;

    RendererSceneRender* GetScene() override {
        return scene_;
    }

    RendererTextureStorage* GetTextureStorage() override {
        return texture_storage_;
    }

    void Initialize() override;

    void BeginFrame(double frame_step) override;

    void EndFrame(bool p_swap_buffers) override;

    void Finalize() override;

public:
    static RendererCompositor* CreateCurrent() {
        return new RasterizerGLES3();
    }

    static void MakeCurrent() {
        CreateFunc = CreateCurrent;
    }

    static RasterizerGLES3* GetInstance() { return s_singleton; }

private:
    static RasterizerGLES3* s_singleton;

    TextureStorage* texture_storage_ = nullptr;
    RasterizerSceneGLES3* scene_ = nullptr;
};

}
