/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-6-28
*/

#include "gobot/drivers/opengl/render_utilities_gl.hpp"
#include "gobot/drivers/opengl/shader_storage.hpp"
#include "gobot/drivers/opengl/texture_storage.hpp"

namespace gobot::opengl {

RendererUtilities* GLRendererUtilities::s_singleton = nullptr;

GLRendererUtilities::GLRendererUtilities() {
    s_singleton = this;
}

GLRendererUtilities::~GLRendererUtilities() {
    s_singleton = nullptr;
}

bool GLRendererUtilities::Free(gobot::RID p_rid) {
    if (opengl::TextureStorage::GetInstance()->OwnsRenderTarget(p_rid)) {
        opengl::TextureStorage::GetInstance()->RenderTargetFree(p_rid);
        return true;
    } else if (opengl::TextureStorage::GetInstance()->OwnsTexture(p_rid)) {
        opengl::TextureStorage::GetInstance()->TextureFree(p_rid);
        return true;
    } else if (opengl::GLShaderStorage::GetInstance()->OwnsShader(p_rid)) {
        opengl::GLShaderStorage::GetInstance()->ShaderFree(p_rid);
        return true;
    } else if (opengl::GLShaderProgramStorage::GetInstance()->OwnsShaderProgram(p_rid)) {
        opengl::GLShaderProgramStorage::GetInstance()->ShaderProgramFree(p_rid);
        return true;
    } else {
        return false;
    }
}



}