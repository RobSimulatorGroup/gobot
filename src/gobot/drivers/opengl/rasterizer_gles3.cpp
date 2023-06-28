/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-6-23
*/

#include "gobot/drivers/opengl/rasterizer_gles3.hpp"

namespace gobot::opengl {

RasterizerGLES3 *RasterizerGLES3::s_singleton = nullptr;

RasterizerGLES3::RasterizerGLES3() {
    s_singleton = this;

    texture_storage_ = new opengl::TextureStorage();
    shader_storage_ = new opengl::GLShaderStorage;
}

RasterizerGLES3::~RasterizerGLES3() {
    s_singleton = nullptr;
    delete texture_storage_;
}

void RasterizerGLES3::Initialize() {

}

void RasterizerGLES3::BeginFrame(double frame_step) {

}

void RasterizerGLES3::EndFrame(bool p_swap_buffers) {

}

void RasterizerGLES3::Finalize() {

}


}
