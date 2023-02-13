/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-11
*/


#include "gobot/drivers/sdl/sdl_window.hpp"
#include "gobot/log.hpp"
#include "gobot/error_macros.hpp"

#include <imgui.h>
#include <SDL.h>
#include <SDL_image.h>

namespace gobot {

static const int s_default_width = 1280;
static const int s_default_height = 720;
static const char* s_default_window_title = "Gobot";

SDLWindow::SDLWindow()
    : render_api_(RenderAPI::OpenGL)
{
    if (SDL_WasInit(SDL_INIT_VIDEO) != SDL_INIT_VIDEO) {
        CRASH_COND_MSG(SDL_Init(SDL_INIT_VIDEO) < 0, "Could not initialize SDL2!");
    }

    native_window_ = SDL_CreateWindow(s_default_window_title,
                                      SDL_WINDOWPOS_CENTERED,
                                      SDL_WINDOWPOS_CENTERED,
                                      s_default_width,
                                      s_default_height,
                                      SDL_WINDOW_SHOWN);


    CRASH_COND_MSG(native_window_ == nullptr, fmt::format("Error creating window: {}", SDL_GetError()));
}

SDLWindow::~SDLWindow()
{
    SDL_DestroyWindow(native_window_);
}

//void SDLWindow::Init() {
//
//    Uint32 flags = SDL_WasInit(0);
//    if ((flags | SDL_INIT_VIDEO) && (flags | SDL_INIT_EVENTS) ) {
//        CRASH_COND_MSG(SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS) < 0, "Could not initialize GLFW!");
//    }
//
//    native_window_ = SDL_CreateWindow(s_default_window_title,
//                                      SDL_WINDOWPOS_UNDEFINED,
//                                      SDL_WINDOWPOS_UNDEFINED,
//                                      s_default_width,
//                                      s_default_height,
//                                      SDL_WINDOW_SHOWN);
//
//    CRASH_COND_MSG(native_window_ == nullptr, fmt::format("Error creating window: {}", SDL_GetError()));
//
//    if(glfwRawMouseMotionSupported())
//        glfwSetInputMode(native_handle_, GLFW_RAW_MOUSE_MOTION, GLFW_TRUE);
//
//
//    glfwSetInputMode(native_handle_, GLFW_STICKY_KEYS, true);
//
//    // Set GLFW callbacks
//    glfwSetWindowSizeCallback(native_handle_, [](GLFWwindow* window, int width, int height) {
//        WindowData& data = *static_cast<WindowData*>((glfwGetWindowUserPointer(window)));
//
//        int w, h;
//        glfwGetFramebufferSize(window, &w, &h);
//
//        data.dpi_scale = (float)w / (float)width;
//
//        data.width =  static_cast<std::uint32_t>(width * data.dpi_scale);
//        data.height = static_cast<std::uint32_t>(height * data.dpi_scale);
//
//        WindowResizeEvent event(data.width, data.height, data.dpi_scale);
//        data.event_callback(event);
//    });
//    glfwSetWindowCloseCallback(native_handle_, [](GLFWwindow* window) {
//        WindowData& data = *static_cast<WindowData*>((glfwGetWindowUserPointer(window)));
//        WindowCloseEvent event;
//        data.event_callback(event);
//    });
//    glfwSetWindowFocusCallback(native_handle_, [](GLFWwindow* window, int focused) {
//        WindowData& data = *static_cast<WindowData*>((glfwGetWindowUserPointer(window)));
//        data.holder_->SetWindowFocus(focused);
//    });
//    glfwSetWindowIconifyCallback(native_handle_, [](GLFWwindow* window, int32_t state) {
//        WindowData& data = *static_cast<WindowData*>((glfwGetWindowUserPointer(window)));
//        switch(state) {
//            case GL_TRUE:
//                data.holder_->SetWindowFocus(false);
//                break;
//            case GL_FALSE:
//                data.holder_->SetWindowFocus(true);
//                break;
//            default:
//                LOG_INFO("Unsupported window iconify state from callback");
//        }
//    });
//    glfwSetKeyCallback(native_handle_, [](GLFWwindow* window, int key, int scancode, int action, int mods)
//    {
//        WindowData& data = *static_cast<WindowData*>((glfwGetWindowUserPointer(window)));
//        switch(action) {
//            case GLFW_PRESS: {
//                KeyPressedEvent event(GLFWToGobotKeyboardKey(key), 0);
//                data.event_callback(event);
//                break;
//            }
//            case GLFW_RELEASE: {
//                KeyReleasedEvent event(GLFWToGobotKeyboardKey(key));
//                data.event_callback(event);
//                break;
//            }
//            case GLFW_REPEAT:
//            {
//                KeyPressedEvent event(GLFWToGobotKeyboardKey(key), 1);
//                data.event_callback(event);
//                break;
//            }
//        }
//    });
//    glfwSetMouseButtonCallback(native_handle_, [](GLFWwindow* window, int button, int action, int mods)
//    {
//        WindowData& data = *static_cast<WindowData*>((glfwGetWindowUserPointer(window)));
//        switch(action) {
//            case GLFW_PRESS: {
//                MouseButtonPressedEvent event(GLFWToGobotMouseKey(button));
//                data.event_callback(event);
//                break;
//            }
//            case GLFW_RELEASE: {
//                MouseButtonReleasedEvent event(GLFWToGobotMouseKey(button));
//                data.event_callback(event);
//                break;
//            }
//        }
//    });
//    glfwSetScrollCallback(native_handle_, [](GLFWwindow* window, double xOffset, double yOffset) {
//        WindowData& data = *static_cast<WindowData*>((glfwGetWindowUserPointer(window)));
//        MouseScrolledEvent event((float)xOffset, (float)yOffset);
//        data.event_callback(event);
//    });
//    glfwSetCursorPosCallback(native_handle_, [](GLFWwindow* window, double xPos, double yPos) {
//        WindowData& data = *static_cast<WindowData*>((glfwGetWindowUserPointer(window)));
//        MouseMovedEvent event((float)xPos /* * data.DPIScale*/, (float)yPos /* * data.DPIScale*/);
//        data.event_callback(event);
//    });
//    glfwSetCursorEnterCallback(native_handle_, [](GLFWwindow* window, int enter) {
//        WindowData& data = *static_cast<WindowData*>((glfwGetWindowUserPointer(window)));
//        MouseEnterEvent event(enter > 0);
//        data.event_callback(event);
//    });
//    glfwSetCharCallback(native_handle_, [](GLFWwindow* window, unsigned int keycode) {
//        WindowData& data = *static_cast<WindowData*>((glfwGetWindowUserPointer(window)));
//        KeyTypedEvent event(GLFWToGobotKeyboardKey(keycode));
//        data.event_callback(event);
//    });
//    glfwSetDropCallback(native_handle_, [](GLFWwindow* window, int numDropped, const char** filenames) {
//        WindowData& data = *static_cast<WindowData*>((glfwGetWindowUserPointer(window)));
//        String file_path = filenames[0];
//        WindowFileEvent event(file_path);
//        data.event_callback(event);
//    });
//
//    g_mouse_cursors[ImGuiMouseCursor_Arrow]      = glfwCreateStandardCursor(GLFW_ARROW_CURSOR);
//    g_mouse_cursors[ImGuiMouseCursor_TextInput]  = glfwCreateStandardCursor(GLFW_IBEAM_CURSOR);
//    g_mouse_cursors[ImGuiMouseCursor_ResizeAll]  = glfwCreateStandardCursor(GLFW_ARROW_CURSOR);
//    g_mouse_cursors[ImGuiMouseCursor_ResizeNS]   = glfwCreateStandardCursor(GLFW_VRESIZE_CURSOR);
//    g_mouse_cursors[ImGuiMouseCursor_ResizeEW]   = glfwCreateStandardCursor(GLFW_HRESIZE_CURSOR);
//    g_mouse_cursors[ImGuiMouseCursor_ResizeNESW] = glfwCreateStandardCursor(GLFW_ARROW_CURSOR);
//    g_mouse_cursors[ImGuiMouseCursor_ResizeNWSE] = glfwCreateStandardCursor(GLFW_ARROW_CURSOR);
//    g_mouse_cursors[ImGuiMouseCursor_Hand]       = glfwCreateStandardCursor(GLFW_HAND_CURSOR);
//
//    LOG_INFO("Initialised GLFW version : {0}", glfwGetVersionString());
//}

std::uint32_t SDLWindow::GetWidth() const
{
    int width;
    SDL_GetWindowSize(native_window_, &width, nullptr);
    return width;
}

std::uint32_t SDLWindow::GetHeight() const
{
    int height;
    SDL_GetWindowSize(native_window_, nullptr, &height);
    return height;
}

Eigen::Vector2i SDLWindow::GetWindowSize() const
{
    int width, height;
    SDL_GetWindowSize(native_window_, &width, &height);
    return {width, height};
}

bool SDLWindow::SetWindowFullscreen()
{
    auto ret = SDL_SetWindowFullscreen(native_window_, SDL_WINDOW_FULLSCREEN_DESKTOP);
    ERR_FAIL_COND_V_MSG(ret != 0, false, fmt::format("Error creating window: {}", SDL_GetError()));
    return true;
}

String SDLWindow::GetTitle() const {
    return SDL_GetWindowTitle(native_window_);
}

void SDLWindow::SetWindowTitle(const String& title)
{
    SDL_SetWindowTitle(native_window_, title.toStdString().c_str());
}

WindowInterface::WindowHandle SDLWindow::GetNativeWindowHandle() const {
    return native_window_;
}

bool SDLWindow::IsMaximized()
{
    auto flag = SDL_GetWindowFlags(native_window_);
    return flag & SDL_WINDOW_MAXIMIZED;
}

bool SDLWindow::IsMinimized()
{
    auto flag = SDL_GetWindowFlags(native_window_);
    return flag & SDL_WINDOW_MINIMIZED;
}

bool SDLWindow::IsFullscreen()
{
    auto flag = SDL_GetWindowFlags(native_window_);
    return flag & SDL_WINDOW_FULLSCREEN_DESKTOP;
}

void SDLWindow::SetWindowBordered(bool bordered)
{
    SDL_SetWindowBordered(native_window_, bordered ? SDL_TRUE : SDL_FALSE);
}

bool SDLWindow::IsWindowBordered()
{
    auto flag = SDL_GetWindowFlags(native_window_);
    return flag & SDL_WINDOW_BORDERLESS;
}

void SDLWindow::Maximize()
{
    SDL_MaximizeWindow(native_window_);
}

void SDLWindow::Minimize()
{
    SDL_MinimizeWindow(native_window_);
}

void SDLWindow::Restore()
{
    SDL_RestoreWindow(native_window_);
}

void SDLWindow::RaiseWindow()
{
    SDL_RaiseWindow(native_window_);
}

void SDLWindow::SetIcon(const Ref<Image>& image)
{
    if (image && image->IsSDLImage()) {
        SDL_SetWindowIcon(native_window_, image->GetSDLImage());
    }
    LOG_ERROR("Input image is not sdl image");
}


void SDLWindow::UpdateCursorImGui()
{
    ImGuiIO& io = ImGui::GetIO();
    ImGuiMouseCursor imgui_cursor = ImGui::GetMouseCursor();

//    if((io.ConfigFlags & ImGuiConfigFlags_NoMouseCursorChange) ||
//       glfwGetInputMode(native_handle_, GLFW_CURSOR) == GLFW_CURSOR_DISABLED)
//        return;
//
//    if(imgui_cursor == ImGuiMouseCursor_None || io.MouseDrawCursor)
//    {
//        // Hide OS mouse cursor if imgui is drawing it or if it wants no cursor
//
//        // TODO: This was disabled as it would override control of hiding the cursor
//        //       Need to find a solution to support both
//        // glfwSetInputMode(m_Handle, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);
//    }
//    else
//    {
//        glfwSetCursor(native_handle_,
//                      g_mouse_cursors[imgui_cursor] ? g_mouse_cursors[imgui_cursor] :
//                      g_mouse_cursors[ImGuiMouseCursor_Arrow]);
//        // glfwSetInputMode(m_Handle, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
//    }
}

void SDLWindow::OnUpdate() {
    // Check if any events have been activated (key pressed, mouse moved etc.) and call corresponding response functions
//    glfwPollEvents();
//
//    if(window_data_.render_api == RenderAPI::OpenGL)
//    {
//        // Swap the screen buffers
//        glfwSwapBuffers(native_handle_);
//    }
}

}
