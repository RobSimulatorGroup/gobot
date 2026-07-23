/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/drivers/opengl/rasterizer_gl.hpp"

#include "gobot/drivers/opengl/material_storage_gl.hpp"
#include "gobot/drivers/opengl/render_utilities_gl.hpp"

namespace gobot::opengl {

GLRasterizer* GLRasterizer::s_singleton = nullptr;

GLRasterizer::GLRasterizer() {
    s_singleton = this;
    texture_storage_ = new TextureStorage();
    material_storage_ = new GLMaterialStorage();
    mesh_storage_ = new GLMeshStorage();
    scene_ = new GLRasterizerScene();
    debug_draw_ = new GLRendererDebugDraw();
    utilities_ = new GLRendererUtilities();
}

GLRasterizer::~GLRasterizer() {
    s_singleton = nullptr;
    delete scene_;
    delete texture_storage_;
    delete material_storage_;
    delete mesh_storage_;
    delete debug_draw_;
    delete utilities_;
}

void GLRasterizer::Initialize() {}
void GLRasterizer::BeginFrame(double) {}
void GLRasterizer::EndFrame(bool) {}
void GLRasterizer::Finalize() {}

} // namespace gobot::opengl
