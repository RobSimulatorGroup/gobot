/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-6-23
*/

#include "gobot/drivers/opengl/rasterizer_gl.hpp"
#include "gobot/drivers/opengl/render_utilities_gl.hpp"
#include "gobot/drivers/opengl/material_storage_gl.hpp"

namespace gobot::opengl {

GLRasterizer *GLRasterizer::s_singleton = nullptr;

GLRasterizer::GLRasterizer() {
    s_singleton = this;

    texture_storage_ = new opengl::TextureStorage();
    material_storage_ = new opengl::GLMaterialStorage();
    mesh_storage_ =  new opengl::GLMeshStorage();
    utilities_ = new opengl::GLRendererUtilities();
}

GLRasterizer::~GLRasterizer() {
    s_singleton = nullptr;
    delete texture_storage_;
    delete material_storage_;
    delete mesh_storage_;
    delete utilities_;
}

void GLRasterizer::Initialize() {

}

void GLRasterizer::BeginFrame(double frame_step) {

}

void GLRasterizer::EndFrame(bool p_swap_buffers) {

}

void GLRasterizer::Finalize() {

}


}
