/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-7
*/

#include "gobot/platform/glfw/glfw_window.hpp"
#include "gobot/core/events/key_event.hpp"
#include "gobot/core/events/mouse_event.hpp"
#include "gobot/platform/glfw/glfw_keycodes.hpp"
#include "gobot/graphics/RHI/graphics_context.hpp"
#include "gobot/log.hpp"
#include "gobot/error_macros.hpp"

namespace gobot {

static int s_num_glfw_windows   = 0;

static void GLFWErrorCallback(int error, const char* description)
{
    LOG_ERROR("GLFW Error - {0} : {1}", error, description);
}

GLFWWindow::GLFWWindow(const WindowDesc& properties) {
    init_  = false;
    v_sync_ = properties.vsync;
    LOG_INFO("VSync : {}", v_sync_ ? "True" : "False");
}

GLFWWindow::~GLFWWindow() {

}

bool GLFWWindow::Init(const WindowDesc& properties) {
    LOG_INFO("Creating window - Title : {0}, Width : {1}, Height : {2}", properties.title, properties.width, properties.height);

    window_data_.title = properties.title;
    window_data_.width = properties.width;
    window_data_.height = properties.height;

    if (s_num_glfw_windows == 0) {
        int success = glfwInit();
        CRASH_COND_MSG(success, "Could not initialize GLFW!");
        glfwSetErrorCallback(GLFWErrorCallback);
    }
    s_num_glfw_windows++;

    GLFWmonitor* monitor = glfwGetPrimaryMonitor();
    float xscale, yscale;
    glfwGetMonitorContentScale(monitor, &xscale, &yscale);
    window_data_.dpi_scale = xscale;

    {
#ifdef GOBOT_DEBUG
        if (window_data_.render_api == RenderAPI::OpenGL)
            glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GLFW_TRUE);
#endif

        native_handle_ = glfwCreateWindow(static_cast<int>(properties.width),
                                          static_cast<int>(properties.height),
                                          window_data_.title.toStdString().c_str(), nullptr, nullptr);
        ++s_num_glfw_windows;
    }

    if(window_data_.render_api == RenderAPI::OpenGL) {
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    }

    SetBorderlessWindow(properties.borderless);

    const GLFWvidmode* mode = glfwGetVideoMode(monitor);
    uint32_t screen_width  = 0;
    uint32_t screen_height = 0;
    if(properties.full_screen) {
        screen_width  = mode->width;
        screen_height = mode->height;
    } else {
        screen_width  = properties.width;
        screen_height = properties.height;
    }

    native_handle_ = glfwCreateWindow(screen_width,
                                      screen_height,
                                      properties.title.toStdString().c_str(),
                                      nullptr,
                                      nullptr);

    int w, h;
    glfwGetFramebufferSize(native_handle_, &w, &h);

    window_data_.title = properties.title;
    window_data_.exit = false;
    window_data_.width  = w;
    window_data_.height = h;

    if(window_data_.render_api == RenderAPI::OpenGL)
    {
        glfwMakeContextCurrent(native_handle_);

        if(!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress)))
        {
            LOG_ERROR("Failed to initialise OpenGL context");
        }
    }

    return true;
}

void GLFWWindow::SetBorderlessWindow(bool borderless)
{
    if(borderless) {
        glfwWindowHint(GLFW_DECORATED, false);
    } else {
        glfwWindowHint(GLFW_DECORATED, true);
    }
}

void GLFWWindow::OnUpdate() {

    {
        // TODO(wqq): profile
        glfwPollEvents();
    }


}

}