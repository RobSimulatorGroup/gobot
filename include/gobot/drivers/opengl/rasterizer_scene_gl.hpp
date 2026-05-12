/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * This file is created by Qiqi Wu, 23-6-23
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "glad/glad.h"
#include "gobot/rendering/renderer_scene_render.hpp"
#include "gobot/rendering/scene_render_items.hpp"

namespace gobot::opengl {

struct GLMeshData;
class GLMeshStorage;

class GLRasterizerScene : public RendererSceneRender {
public:
    explicit GLRasterizerScene(GLMeshStorage* mesh_storage);

    ~GLRasterizerScene() override;

    void RenderScene(const RID& render_target, const Node* scene_root, const Camera3D* camera) override;

private:
    GLMeshStorage* mesh_storage_{nullptr};
    GLuint default_program_ = 0;

    void EnsureDefaultProgram();

    void UploadMesh(GLMeshData* mesh);

    void DrawVisualItem(const VisualMeshRenderItem& item);
};

}
