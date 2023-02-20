/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-20
*/

#include "gobot/drivers/opengl/gl_swap_chain.hpp"
#include "gobot/drivers/opengl/gl_command_buffer.hpp"
#include "gobot/drivers/opengl/gl.hpp"
#include "gobot/log.hpp"

namespace gobot {

static std::string GetStringForType(GLenum type)
{
    switch(type) {
        case GL_DEBUG_TYPE_ERROR:
            return "Error";
        case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR:
            return "Deprecated behavior";
        case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:
            return "Undefined behavior";
        case GL_DEBUG_TYPE_PORTABILITY:
            return "Portability issue";
        case GL_DEBUG_TYPE_PERFORMANCE:
            return "Performance issue";
        case GL_DEBUG_TYPE_MARKER:
            return "Stream annotation";
        case GL_DEBUG_TYPE_OTHER:
            return "Other";
        default:
            return "";
    }
}

static bool PrintMessage(GLenum type)
{
    switch(type) {
        case GL_DEBUG_TYPE_ERROR:
        case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR:
        case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:
        case GL_DEBUG_TYPE_PORTABILITY:
            return true;
        case GL_DEBUG_TYPE_PERFORMANCE:
        case GL_DEBUG_TYPE_MARKER:
        case GL_DEBUG_TYPE_OTHER:
        default:
            return false;
    }
}

static std::string GetStringForSource(GLenum source)
{
    switch(source) {
    case GL_DEBUG_SOURCE_API:
        return "API";
    case GL_DEBUG_SOURCE_WINDOW_SYSTEM:
        return "Window System";
    case GL_DEBUG_SOURCE_SHADER_COMPILER:
        return "Shader compiler";
    case GL_DEBUG_SOURCE_THIRD_PARTY:
        return "Third party";
    case GL_DEBUG_SOURCE_APPLICATION:
        return "Application";
    case GL_DEBUG_SOURCE_OTHER:
        return "Other";
    default:
        return "";
    }
}

static std::string GetStringForSeverity(GLenum severity)
{
    switch(severity) {
        case GL_DEBUG_SEVERITY_HIGH:
            return "High";
        case GL_DEBUG_SEVERITY_MEDIUM:
            return "Medium";
        case GL_DEBUG_SEVERITY_LOW:
            return "Low";
        case GL_DEBUG_SEVERITY_NOTIFICATION:
            return "Notification";
        case GL_DEBUG_SOURCE_API:
            return "Source API";
        default:
            return ("");
    }
}

void APIENTRY openglCallbackFunction(GLenum source,
                                     GLenum type,
                                     GLuint id,
                                     GLenum severity,
                                     GLsizei length,
                                     const GLchar* message,
                                     const void* userParam)
{
    if(!PrintMessage(type))
        return;

    LOG_INFO("[OpenGL] Message: {0}", message);
    LOG_INFO("[OpenGL] Type: {0}", GetStringForType(type));
    LOG_INFO("[OpenGL] Source: {0}", GetStringForSource(source));
    LOG_INFO("[OpenGL] ID: {0}", id);
    LOG_INFO("[OpenGL] Severity: {0}", GetStringForSeverity(source));
}

GLSwapChain::GLSwapChain(uint32_t width, uint32_t height)
{
    m_Width  = width;
    m_Height = height;
    //            FramebufferDesc info {};
    //            info.width = width;
    //            info.height = height;
    //            info.attachments = nullptr;
}

GLSwapChain::~GLSwapChain()
{
    for(auto& buffer : swapChainBuffers)
        delete buffer;
}

bool GLSwapChain::Init(bool vsync)
{

//#if GOBOT_DEBUG
//    #ifdef GL_DEBUD_CALLBACK
//#ifndef LUMOS_PLATFORM_MACOS
//            LUMOS_LOG_INFO(OPENGLLOG "Registering OpenGL debug callback");
//
//            glEnable(GL_DEBUG_OUTPUT);
//            glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
//            glDebugMessageCallback(Lumos::openglCallbackFunction, nullptr);
//            GLuint unusedIds = 0;
//            glDebugMessageControl(GL_DONT_CARE,
//                                  GL_DONT_CARE,
//                                  GL_DONT_CARE,
//                                  0,
//                                  &unusedIds,
//                                  true);
//#else
//            LUMOS_LOG_INFO(OPENGLLOG "glDebugMessageCallback not available");
//#endif
//#endif
//#endif

    MainCommandBuffer = std::make_shared<GLCommandBuffer>();
    return true;
}

Texture* GLSwapChain::GetCurrentImage()
{
    return nullptr; // swapChainBuffers[0];
}

uint32_t GLSwapChain::GetCurrentBufferIndex() const
{
    return 0;
}

size_t GLSwapChain::GetSwapChainBufferCount() const
{
    return 1;
}

void GLSwapChain::MakeDefault()
{
    CreateFunc = CreateFuncGL;
}

SwapChain* GLSwapChain::CreateFuncGL(uint32_t width, uint32_t height)
{
    return new GLSwapChain(width, height);
}

CommandBuffer* GLSwapChain::GetCurrentCommandBuffer()
{
    return MainCommandBuffer.get();
}

}