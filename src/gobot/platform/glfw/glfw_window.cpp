/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-7
*/

#include "gobot/platform/glfw/glfw_window.hpp"

#include "gobot/core/events/application_event.hpp"
#include "gobot/core/events/key_event.hpp"
#include "gobot/core/events/mouse_event.hpp"
#include "gobot/platform/glfw/glfw_keycodes.hpp"
#include "gobot/graphics/RHI/graphics_context.hpp"
#include "gobot/core/os/input.hpp"
#include "gobot/log.hpp"
#include "gobot/error_macros.hpp"

#include <imgui.h>

namespace gobot {

static int s_num_glfw_windows = 0;

static GLFWcursor* g_mouse_cursors[ImGuiMouseCursor_COUNT] = { nullptr };

static void GLFWErrorCallback(int error, const char* description)
{
    LOG_ERROR("GLFW Error - {0} : {1}", error, description);
}

GLFWWindow::GLFWWindow(const WindowDesc& properties) {
    initialised_  = false;
    v_sync_ = properties.vsync;
    LOG_INFO("VSync : {}", v_sync_ ? "True" : "False");

    Init(properties);
}

GLFWWindow::~GLFWWindow() {
    for(auto& g_mouse_cursor : g_mouse_cursors) {
        glfwDestroyCursor(g_mouse_cursor);
        g_mouse_cursor = nullptr;
    }

    glfwDestroyWindow(native_handle_);
    --s_num_glfw_windows;

    if(s_num_glfw_windows < 1)
        glfwTerminate();

}

bool GLFWWindow::Init(const WindowDesc& properties) {
    LOG_INFO("Creating window - Title : {0}, Width : {1}, Height : {2}", properties.title, properties.width, properties.height);

    window_data_.title = properties.title;
    window_data_.width = properties.width;
    window_data_.height = properties.height;

    if (s_num_glfw_windows == 0) {
        int success = glfwInit();
        CRASH_COND_MSG(!success, "Could not initialize GLFW!");
        glfwSetErrorCallback(GLFWErrorCallback);
    }
    s_num_glfw_windows++;

    GLFWmonitor* monitor = glfwGetPrimaryMonitor();
    float x_scale, y_scale;
    glfwGetMonitorContentScale(monitor, &x_scale, &y_scale);
    window_data_.dpi_scale = x_scale;

    if(window_data_.render_api == RenderAPI::OpenGL) {
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef GOBOT_DEBUG
        glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GLFW_TRUE);
#endif
    }

    if(properties.borderless) {
        glfwWindowHint(GLFW_DECORATED, false);
    } else {
        glfwWindowHint(GLFW_DECORATED, true);
    }

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
    window_data_.width  = w;
    window_data_.height = h;
    window_data_.holder_ = this;

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


    glfwSetInputMode(native_handle_, GLFW_STICKY_KEYS, true);

    // Set GLFW callbacks
    glfwSetWindowSizeCallback(native_handle_, [](GLFWwindow* window, int width, int height) {
        WindowData& data = *static_cast<WindowData*>((glfwGetWindowUserPointer(window)));

        int w, h;
        glfwGetFramebufferSize(window, &w, &h);

        data.dpi_scale = (float)w / (float)width;

        data.width =  static_cast<std::uint32_t>(width * data.dpi_scale);
        data.height = static_cast<std::uint32_t>(height * data.dpi_scale);

        WindowResizeEvent event(data.width, data.height, data.dpi_scale);
        data.event_callback(event);
    });
    glfwSetWindowCloseCallback(native_handle_, [](GLFWwindow* window) {
        WindowData& data = *static_cast<WindowData*>((glfwGetWindowUserPointer(window)));
        WindowCloseEvent event;
        data.event_callback(event);
    });
    glfwSetWindowFocusCallback(native_handle_, [](GLFWwindow* window, int focused) {
        WindowData& data = *static_cast<WindowData*>((glfwGetWindowUserPointer(window)));
        data.holder_->SetWindowFocus(focused);
    });
    glfwSetWindowIconifyCallback(native_handle_, [](GLFWwindow* window, int32_t state) {
        WindowData& data = *static_cast<WindowData*>((glfwGetWindowUserPointer(window)));
        switch(state) {
            case GL_TRUE:
                data.holder_->SetWindowFocus(false);
                break;
            case GL_FALSE:
                data.holder_->SetWindowFocus(true);
                break;
            default:
                LOG_INFO("Unsupported window iconify state from callback");
        }
    });
    glfwSetKeyCallback(native_handle_, [](GLFWwindow* window, int key, int scancode, int action, int mods)
    {
        WindowData& data = *static_cast<WindowData*>((glfwGetWindowUserPointer(window)));
        switch(action) {
            case GLFW_PRESS: {
                KeyPressedEvent event(GLFWToGobotKeyboardKey(key), 0);
                data.event_callback(event);
                break;
            }
            case GLFW_RELEASE: {
                KeyReleasedEvent event(GLFWToGobotKeyboardKey(key));
                data.event_callback(event);
                break;
            }
            case GLFW_REPEAT:
            {
                KeyPressedEvent event(GLFWToGobotKeyboardKey(key), 1);
                data.event_callback(event);
                break;
            }
        }
    });
    glfwSetMouseButtonCallback(native_handle_, [](GLFWwindow* window, int button, int action, int mods)
    {
        WindowData& data = *static_cast<WindowData*>((glfwGetWindowUserPointer(window)));
        switch(action) {
            case GLFW_PRESS: {
                MouseButtonPressedEvent event(GLFWToGobotMouseKey(button));
                data.event_callback(event);
                break;
            }
            case GLFW_RELEASE: {
                MouseButtonReleasedEvent event(GLFWToGobotMouseKey(button));
                data.event_callback(event);
                break;
            }
        }
    });
    glfwSetScrollCallback(native_handle_, [](GLFWwindow* window, double xOffset, double yOffset) {
        WindowData& data = *static_cast<WindowData*>((glfwGetWindowUserPointer(window)));
        MouseScrolledEvent event((float)xOffset, (float)yOffset);
        data.event_callback(event);
    });
    glfwSetCursorPosCallback(native_handle_, [](GLFWwindow* window, double xPos, double yPos) {
        WindowData& data = *static_cast<WindowData*>((glfwGetWindowUserPointer(window)));
        MouseMovedEvent event((float)xPos /* * data.DPIScale*/, (float)yPos /* * data.DPIScale*/);
        data.event_callback(event);
    });
    glfwSetCursorEnterCallback(native_handle_, [](GLFWwindow* window, int enter) {
        WindowData& data = *static_cast<WindowData*>((glfwGetWindowUserPointer(window)));
        MouseEnterEvent event(enter > 0);
        data.event_callback(event);
    });
    glfwSetCharCallback(native_handle_, [](GLFWwindow* window, unsigned int keycode) {
        WindowData& data = *static_cast<WindowData*>((glfwGetWindowUserPointer(window)));
        KeyTypedEvent event(GLFWToGobotKeyboardKey(keycode));
        data.event_callback(event);
    });
    glfwSetDropCallback(native_handle_, [](GLFWwindow* window, int numDropped, const char** filenames) {
        WindowData& data = *static_cast<WindowData*>((glfwGetWindowUserPointer(window)));
        String file_path = filenames[0];
        WindowFileEvent event(file_path);
        data.event_callback(event);
    });

    g_mouse_cursors[ImGuiMouseCursor_Arrow]      = glfwCreateStandardCursor(GLFW_ARROW_CURSOR);
    g_mouse_cursors[ImGuiMouseCursor_TextInput]  = glfwCreateStandardCursor(GLFW_IBEAM_CURSOR);
    g_mouse_cursors[ImGuiMouseCursor_ResizeAll]  = glfwCreateStandardCursor(GLFW_ARROW_CURSOR);
    g_mouse_cursors[ImGuiMouseCursor_ResizeNS]   = glfwCreateStandardCursor(GLFW_VRESIZE_CURSOR);
    g_mouse_cursors[ImGuiMouseCursor_ResizeEW]   = glfwCreateStandardCursor(GLFW_HRESIZE_CURSOR);
    g_mouse_cursors[ImGuiMouseCursor_ResizeNESW] = glfwCreateStandardCursor(GLFW_ARROW_CURSOR);
    g_mouse_cursors[ImGuiMouseCursor_ResizeNWSE] = glfwCreateStandardCursor(GLFW_ARROW_CURSOR);
    g_mouse_cursors[ImGuiMouseCursor_Hand]       = glfwCreateStandardCursor(GLFW_HAND_CURSOR);

    LOG_INFO("Initialised GLFW version : {0}", glfwGetVersionString());
    return true;
}

void GLFWWindow::SetIcon(const std::string& file_path, const std::string& small_icon_file_path)
{
    // TODO(wqq): Consider godot's Image
    //  glfwSetWindowIcon(native_handle_, int(images.size()), images.data());
}

void GLFWWindow::UpdateCursorImGui()
{
    ImGuiIO& io = ImGui::GetIO();
    ImGuiMouseCursor imgui_cursor = ImGui::GetMouseCursor();

    if((io.ConfigFlags & ImGuiConfigFlags_NoMouseCursorChange) ||
        glfwGetInputMode(native_handle_, GLFW_CURSOR) == GLFW_CURSOR_DISABLED)
        return;

    if(imgui_cursor == ImGuiMouseCursor_None || io.MouseDrawCursor)
    {
        // Hide OS mouse cursor if imgui is drawing it or if it wants no cursor

        // TODO: This was disabled as it would override control of hiding the cursor
        //       Need to find a solution to support both
        // glfwSetInputMode(m_Handle, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);
    }
    else
    {
        glfwSetCursor(native_handle_,
                      g_mouse_cursors[imgui_cursor] ? g_mouse_cursors[imgui_cursor] :
                                                      g_mouse_cursors[ImGuiMouseCursor_Arrow]);
        // glfwSetInputMode(m_Handle, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
    }
}

void GLFWWindow::HideMouse(bool hide)
{
    if(hide) {
        glfwSetInputMode(native_handle_, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);
    } else {
        glfwSetInputMode(native_handle_, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
    }
}


void GLFWWindow::SetMousePosition(const Eigen::Vector2f& pos)
{
    Input::GetInstance()->StoreMousePosition(pos);
    glfwSetCursorPos(native_handle_, pos.x(), pos.y());
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