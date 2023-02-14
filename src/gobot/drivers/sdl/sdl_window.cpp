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
                        case SDL_WINDOWEVENT_ENTER: {
                            MouseEnterEvent enter_event;
                            event_callback(enter_event);
                            break;
                        }
                        case SDL_WINDOWEVENT_FOCUS_GAINED: {
                            KeyboardFocusEvent focus_event;
                            event_callback(focus_event);
                            break;
                        }
                        case SDL_WINDOWEVENT_FOCUS_LOST: {
                            KeyboardLoseFocusEvent lose_focus_event;
                            event_callback(lose_focus_event);
                            break;
                        }
                        case SDL_WINDOWEVENT_TAKE_FOCUS: {
                            WindowTakeFocusEvent take_focus_event;
                            event_callback(take_focus_event);
                            break;
                        }
                        case SDL_WINDOWEVENT_MAXIMIZED: {
                            WindowMaximizedEvent maximized_event;
                            event_callback(maximized_event);
                            break;
                        }
                        case SDL_WINDOWEVENT_MINIMIZED: {
                            WindowMinimizedEvent minimized_event;
                            event_callback(minimized_event);
                            break;
                        }
                        case SDL_WINDOWEVENT_LEAVE: {
                            MouseLeaveEvent leave_event;
                            event_callback(leave_event);
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
                    KeyPressedEvent key_press_event((KeyCode)event.key.keysym.scancode, event.key.keysym.mod, event.key.repeat);
                    event_callback(key_press_event);
                }
                break;
            }
            case SDL_KEYUP: {
                if (event.key.windowID == windows_id_) {
                    KeyReleasedEvent key_released_event((KeyCode)event.key.keysym.scancode, event.key.keysym.mod);
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
            case SDL_MOUSEWHEEL: {
                if (event.wheel.windowID == windows_id_) {
                    MouseScrolledEvent mouse_scrolled_event(event.wheel.preciseX, event.wheel.preciseY);
                    event_callback(mouse_scrolled_event);
                }
                break;
            }
            case SDL_MOUSEMOTION: {
                if (event.motion.windowID == windows_id_) {
                    MouseMovedEvent mouse_moved_event(event.motion.x,
                                                      event.motion.y,
                                                      event.motion.xrel,
                                                      event.motion.yrel,
                                                      event.motion.state);
                    event_callback(mouse_moved_event);
                }
                break;
            }
        }
    }
}


}
