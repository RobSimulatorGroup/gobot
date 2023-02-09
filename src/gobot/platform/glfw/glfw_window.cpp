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
    initialised_  = false;
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

    native_handle_ = glfwCreateWindow(static_cast<int>(screen_width),
                                      static_cast<int>(screen_height),
                                      properties.title.toStdString().c_str(),
                                      nullptr,
                                      nullptr);

    int w, h;
    glfwGetFramebufferSize(native_handle_, &w, &h);

    window_data_.title = properties.title;
    window_data_.exit = false;
    window_data_.width  = w;
    window_data_.height = h;

    if(window_data_.render_api == RenderAPI::OpenGL) {
        glfwMakeContextCurrent(native_handle_);

        if(!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress)))
        {
            LOG_ERROR("Failed to initialise OpenGL context");
        }
    }

    glfwSetWindowUserPointer(native_handle_, &window_data_);

    if(glfwRawMouseMotionSupported())
        glfwSetInputMode(native_handle_, GLFW_RAW_MOUSE_MOTION, GLFW_TRUE);


    return true;
}

void GLFWWindow::Shutdown() const
{
    glfwDestroyWindow(native_handle_);
    --s_num_glfw_windows;

    if(s_num_glfw_windows == 0)
        glfwTerminate();
}

void GLFWWindow::SetWindowTitle(const String& title) {
    glfwSetWindowTitle(native_handle_, title.toStdString().c_str());
}

void GLFWWindow::ToggleVSync() {
    if(v_sync_) {
        SetVSync(false);
    } else {
        SetVSync(true);
    }

    LOG_INFO("VSync : {0}", v_sync_ ? "True" : "False");
}

void GLFWWindow::SetVSync(bool v_sync) {
    v_sync_ = v_sync;
    if(window_data_.render_api == RenderAPI::OpenGL)
        glfwSwapInterval(v_sync ? 1 : 0);

    LOG_INFO("VSync : {0}", v_sync ? "True" : "False");
}


void GLFWWindow::SetBorderlessWindow(bool borderless)
{
    if(borderless) {
        glfwWindowHint(GLFW_DECORATED, false);
    } else {
        glfwWindowHint(GLFW_DECORATED, true);
    }
}

bool GLFWWindow::IsMaximized() {
    return glfwGetWindowAttrib(native_handle_, GLFW_MAXIMIZED);
}

void GLFWWindow::Maximize() {
    glfwMaximizeWindow(native_handle_);
}

void GLFWWindow::Minimize() {
    glfwIconifyWindow(native_handle_);
}

void GLFWWindow::Restore()  {
    glfwRestoreWindow(native_handle_);
}

void GLFWWindow::OnUpdate() {
    // Check if any events have been activated (key pressed, mouse moved etc.) and call corresponding response functions
    glfwPollEvents();

    if(window_data_.render_api == RenderAPI::OpenGL)
    {
        // Swap the screen buffers
        glfwSwapBuffers(native_handle_);
    }


}

}