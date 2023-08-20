/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-6-23
*/

#pragma once

#include "gobot/rendering/renderer_compositor.hpp"
#include "gobot/drivers/opengl/texture_storage.hpp"
#include "gobot/drivers/opengl/rasterizer_gl.hpp"
#include "gobot/drivers/opengl/rasterizer_scene_gl.hpp"
#include "gobot/drivers/opengl/material_storage_gl.hpp"
#include "gobot/drivers/opengl/mesh_storage_gl.hpp"

namespace gobot::opengl {

class GLRasterizer : public Rasterizer {
public:
    GLRasterizer();

    ~GLRasterizer() override;

    RendererSceneRender* GetScene() override {
        return scene_;
    }

    RendererTextureStorage* GetTextureStorage() override {
        return texture_storage_;
    }

    MaterialStorage* GetMaterialStorage() override {
        return material_storage_;
    }

    MeshStorage* GetMeshStorage() override {
        return mesh_storage_;
    }

    RendererUtilities* GetUtilities() override {
        return utilities_;
    }

    void Initialize() override;

    void BeginFrame(double frame_step) override;

    void EndFrame(bool p_swap_buffers) override;

    void Finalize() override;

public:
    static Rasterizer* CreateCurrent() {
        return new GLRasterizer();
    }

    static void MakeCurrent() {
        CreateFunc = CreateCurrent;
    }

    static GLRasterizer* GetInstance() { return s_singleton; }

private:
    static GLRasterizer* s_singleton;

    TextureStorage* texture_storage_ = nullptr;
    GLRasterizerScene* scene_ = nullptr;
    GLMaterialStorage* material_storage_ = nullptr;
    GLMeshStorage* mesh_storage_ = nullptr;
    RendererUtilities*  utilities_ = nullptr;
};

}
