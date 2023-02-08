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
    v_sync_ = properties.vsync_;
    LOG_INFO("VSync : {}", v_sync_ ? "True" : "False");
}

GLFWWindow::~GLFWWindow() {

}

bool GLFWWindow::Init(const WindowDesc& properties) {
    LOG_INFO("Creating window - Title : {0}, Width : {1}, Height : {2}", properties.title_, properties.width_, properties.height_);

    window_data_.title = properties.title_;
    window_data_.width = properties.width_;
    window_data_.height = properties.height_;

    if (s_num_glfw_windows == 0) {
        int success = glfwInit();
        CRASH_COND_MSG(success, "Could not initialize GLFW!");
        glfwSetErrorCallback(GLFWErrorCallback);
    }

    {
#ifdef ARC_DEBUG
        if (Renderer::GetAPI() == RendererAPI::API::OpenGL)
					glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GLFW_TRUE);
#endif

#ifdef ARC_PLATFORM_WINDOWS
        glfwWindowHint(GLFW_TITLEBAR, GLFW_FALSE);
#endif

        native_window_handle_ = glfwCreateWindow(static_cast<int>(properties.width_),
                                    static_cast<int>(properties.height_),
                                    window_data_.title.toStdString().c_str(), nullptr, nullptr);
        ++s_num_glfw_windows;
    }

}

void GLFWWindow::OnUpdate() {

    {
        // TODO(wqq): profile
        glfwPollEvents();
    }


}

}