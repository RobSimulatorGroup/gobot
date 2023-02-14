/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-11
*/


#include "gobot/drivers/sdl/sdl_window.hpp"
#include "gobot/core/events/application_event.hpp"
#include "gobot/core/events/mouse_event.hpp"
#include "gobot/core/events/key_event.hpp"
#include "gobot/log.hpp"
#include "gobot/error_macros.hpp"

#include <imgui.h>
#include <imgui_impl_sdl.h>
#include <SDL.h>

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

    windows_id_ = SDL_GetWindowID(native_window_);
}

SDLWindow::~SDLWindow()
{
    SDL_DestroyWindow(native_window_);
}


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

void SDLWindow::SetTitle(const String& title)
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
    } else {
        LOG_ERROR("Input image is not sdl image");
    }
}

std::uint32_t SDLWindow::GetWindowID() const {
    return windows_id_;
}

void SDLWindow::ProcessEvents() {
    // Check if any events have been activated (key pressed, mouse moved etc.) and call corresponding response functions
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
        // this should only run if there's imgui on
        ImGui_ImplSDL2_ProcessEvent(&event);
        switch (event.type) {
            case SDL_WINDOWEVENT: {
                if (event.window.windowID == windows_id_) {
                    switch (event.window.event) {
                        case SDL_WINDOWEVENT_RESIZED: {
                            WindowResizeEvent resize_event(event.window.data1, event.window.data2);
                            event_callback(resize_event);
                            break;
                        }
                        case SDL_WINDOWEVENT_CLOSE: {
                            WindowCloseEvent close_event;
                            event_callback(close_event);
                            break;
                        }
                        case SDL_WINDOWEVENT_ENTER:
                        case SDL_WINDOWEVENT_TAKE_FOCUS:
                        case SDL_WINDOWEVENT_FOCUS_GAINED:
                        case SDL_WINDOWEVENT_MAXIMIZED: {
                            WindowFocusEvent focus_event;
                            event_callback(focus_event);
                            break;
                        }
                        case SDL_WINDOWEVENT_MINIMIZED:
                        case SDL_WINDOWEVENT_LEAVE:
                        case SDL_WINDOWEVENT_FOCUS_LOST: {
                            WindowLostFocusEvent lose_focus_event;
                            event_callback(lose_focus_event);
                            break;
                        }
                        case SDL_WINDOWEVENT_MOVED: {
                            WindowMovedEvent moved_event(event.window.data1, event.window.data2);
                            event_callback(moved_event);
                            break;
                        }
                        default:
                            break;
                    }
                }
                break;
            }
            case SDL_DROPFILE: {
                if (event.drop.windowID == windows_id_) {
                    char* dropped_file_dir = event.drop.file;
                    WindowDropFileEvent drop_file_event(dropped_file_dir);
                    event_callback(drop_file_event);
                    SDL_free(dropped_file_dir);    // Free dropped_file_dir memory
                }
                break;
            }
            case SDL_KEYDOWN: {
                if (event.key.windowID == windows_id_) {
                    KeyPressedEvent key_press_event((KeyCode)event.key.keysym.scancode, event.key.repeat);
                    event_callback(key_press_event);
                }
                break;
            }
            case SDL_KEYUP: {
                if (event.key.windowID == windows_id_) {
                    KeyReleasedEvent key_released_event((KeyCode)event.key.keysym.scancode);
                    event_callback(key_released_event);
                }
                break;
            }
            case SDL_MOUSEBUTTONUP: {
                if (event.button.windowID == windows_id_) {
                    MouseButtonPressedEvent mouse_button_pressed_event((MouseButton)event.button.button,
                                                                 event.button.x,
                                                                 event.button.y,
                                                                 (MouseButtonClickMode)event.button.clicks);
                    event_callback(mouse_button_pressed_event);
                }
                break;
            }
            case SDL_MOUSEBUTTONDOWN:
            {
                if (event.button.windowID == windows_id_) {
                    MouseButtonReleasedEvent mouse_released_event((MouseButton)event.button.button,
                                                                 event.button.x,
                                                                 event.button.y,
                                                                 (MouseButtonClickMode)event.button.clicks);
                    event_callback(mouse_released_event);
                }
                break;
            }
            case SDL_MOUSEWHEEL:
            {
                MouseScrolledEvent mouseScrolledEvent(static_cast<float>(event.wheel.x), static_cast<float>(event.wheel.y));
                event_callback(mouseScrolledEvent);
                break;
            }
            case SDL_MOUSEMOTION:
            {
                MouseMovedEvent mouseMovedEvent(static_cast<float>(event.motion.x), static_cast<float>(event.motion.y));
                event_callback(mouseMovedEvent);
                break;
            }
        }
    }
}


}
