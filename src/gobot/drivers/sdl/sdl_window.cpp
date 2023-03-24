/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-11
*/


#include "gobot/drivers/sdl/sdl_window.hpp"
#include "gobot/core/events/window_event.hpp"
#include "gobot/core/events/mouse_event.hpp"
#include "gobot/core/events/key_event.hpp"
#include "gobot/rendering/render_server.hpp"
#include "gobot/platfom.hpp"
#include "gobot/log.hpp"
#include "gobot/error_macros.hpp"

#include <imgui_impl_sdl2.h>
#include <SDL.h>
#include <SDL_syswm.h>

#include <bgfx/platform.h>

#ifndef ENTRY_CONFIG_USE_WAYLAND
#	define ENTRY_CONFIG_USE_WAYLAND 0
#endif // ENTRY_CONFIG_USE_WAYLAND


namespace gobot {

// <X11/X.h>
#ifdef None
#undef None
#endif

static const int s_default_width = 1280;
static const int s_default_height = 720;
static const char* s_default_window_title = "Gobot";

static RenderDebugFlags s_render_debug_flag = RenderDebugFlags::None;
static RenderResetFlags s_render_reset_flag = RenderResetFlags::None;

SDLWindow::SDLWindow()
{
    if (SDL_WasInit(SDL_INIT_VIDEO) != SDL_INIT_VIDEO) {
        CRASH_COND_MSG(SDL_Init(SDL_INIT_VIDEO) < 0, "Could not initialize SDL2!");
    }

    sdl2_window_ = SDL_CreateWindow(s_default_window_title,
                                    SDL_WINDOWPOS_CENTERED,
                                    SDL_WINDOWPOS_CENTERED,
                                    s_default_width,
                                    s_default_height,
                                      SDL_WINDOW_SHOWN | SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE);

    CRASH_COND_MSG(sdl2_window_ == nullptr, fmt::format("Error creating window: {}", SDL_GetError()));

    windows_id_ = SDL_GetWindowID(sdl2_window_);


    bgfx::renderFrame();
}

SDLWindow::~SDLWindow()
{
    SDL_DestroyWindow(sdl2_window_);
}

std::uint32_t SDLWindow::GetWidth() const
{
    int width;
    SDL_GetWindowSize(sdl2_window_, &width, nullptr);
    return width;
}

std::uint32_t SDLWindow::GetHeight() const
{
    int height;
    SDL_GetWindowSize(sdl2_window_, nullptr, &height);
    return height;
}

Eigen::Vector2i SDLWindow::GetWindowSize() const
{
    int width, height;
    SDL_GetWindowSize(sdl2_window_, &width, &height);
    return {width, height};
}

bool SDLWindow::SetWindowFullscreen()
{
    auto ret = SDL_SetWindowFullscreen(sdl2_window_, SDL_WINDOW_FULLSCREEN_DESKTOP);
    ERR_FAIL_COND_V_MSG(ret != 0, false, fmt::format("Error creating window: {}", SDL_GetError()));
    return true;
}

String SDLWindow::GetTitle() const {
    return SDL_GetWindowTitle(sdl2_window_);
}

void SDLWindow::SetTitle(const String& title)
{
    SDL_SetWindowTitle(sdl2_window_, title.toStdString().c_str());
}

WindowInterface::NativeWindowHandle SDLWindow::GetNativeWindowHandle() const {
    SDL_SysWMinfo wmi;
    SDL_VERSION(&wmi.version);
    if (!SDL_GetWindowWMInfo(sdl2_window_, &wmi)) {
        return nullptr;
    }

#if GOB_PLATFORM_LINUX || GOB_PLATFORM_BSD
		return (void*)wmi.info.x11.window;
#elif GOB_PLATFORM_OSX || GOB_PLATFORM_IOS
        return wmi.info.cocoa.window;
#elif GOB_PLATFORM_WINDOWS
        return wmi.info.win.window;
#elif GOB_PLATFORM_ANDROID
        return wmi.info.android.window;
#endif
}

void* SDLWindow::GetNativeDisplayHandle() const
{
    SDL_SysWMinfo wmi;
    SDL_VERSION(&wmi.version);
    if (!SDL_GetWindowWMInfo(sdl2_window_, &wmi)) {
        return nullptr;
    }
#if GOB_PLATFORM_LINUX || GOB_PLATFORM_BSD
    return wmi.info.x11.display;
#else
    return nullptr;
#endif
}

bool SDLWindow::IsMaximized()
{
    auto flag = SDL_GetWindowFlags(sdl2_window_);
    return flag & SDL_WINDOW_MAXIMIZED;
}

bool SDLWindow::IsMinimized()
{
    auto flag = SDL_GetWindowFlags(sdl2_window_);
    return flag & SDL_WINDOW_MINIMIZED;
}

bool SDLWindow::IsFullscreen()
{
    auto flag = SDL_GetWindowFlags(sdl2_window_);
    return flag & SDL_WINDOW_FULLSCREEN_DESKTOP;
}

void SDLWindow::SetWindowBordered(bool bordered)
{
    SDL_SetWindowBordered(sdl2_window_, bordered ? SDL_TRUE : SDL_FALSE);
}

bool SDLWindow::IsWindowBordered()
{
    auto flag = SDL_GetWindowFlags(sdl2_window_);
    return flag & SDL_WINDOW_BORDERLESS;
}

void SDLWindow::Maximize()
{
    SDL_MaximizeWindow(sdl2_window_);
}

void SDLWindow::Minimize()
{
    SDL_MinimizeWindow(sdl2_window_);
}

void SDLWindow::Restore()
{
    SDL_RestoreWindow(sdl2_window_);
}

void SDLWindow::RaiseWindow()
{
    SDL_RaiseWindow(sdl2_window_);
}

void SDLWindow::SetIcon(const Ref<Image>& image)
{
    if (image && image->IsSDLImage()) {
        SDL_SetWindowIcon(sdl2_window_, image->GetSDLImage());
    } else {
        LOG_ERROR("Input image is not sdl image");
    }
}

void SDLWindow::ShowWindow()
{
    SDL_ShowWindow(sdl2_window_);
}

void  SDLWindow::HideWindow()
{
    SDL_HideWindow(sdl2_window_);
}

bool SDLWindow::IsWindowHide()
{
    auto flag = SDL_GetWindowFlags(sdl2_window_);
    return flag & SDL_WINDOW_HIDDEN;
}

std::uint32_t SDLWindow::GetWindowID() const {
    return windows_id_;
}

void SDLWindow::ProcessEvents() {
    if (RenderServer::HasInit()) {
        auto flags = GET_RS()->GetResetFlags();
        render_need_reset_ = s_render_reset_flag != flags;
        s_render_reset_flag = flags;
    }

    // Check if any events have been activated (key pressed, mouse moved etc.) and call corresponding response functions
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
        if (ImGui::GetCurrentContext()) {
            // this should only run if there's imgui on
            ImGui_ImplSDL2_ProcessEvent(&event);
        }

        switch (event.type) {
            case SDL_WINDOWEVENT: {
                if (event.window.windowID == windows_id_) {
                    switch (event.window.event) {
                        case SDL_WINDOWEVENT_RESIZED: {
                            WindowResizeEvent resize_event(event.window.data1, event.window.data2);
                            RunEventCallback(resize_event);
                            render_need_reset_ = true;
                            break;
                        }
                        case SDL_WINDOWEVENT_CLOSE: {
                            WindowCloseEvent close_event;
                            RunEventCallback(close_event);
                            break;
                        }
                        case SDL_WINDOWEVENT_ENTER: {
                            MouseEnterEvent enter_event;
                            RunEventCallback(enter_event);
                            break;
                        }
                        case SDL_WINDOWEVENT_FOCUS_GAINED: {
                            KeyboardFocusEvent focus_event;
                            RunEventCallback(focus_event);
                            break;
                        }
                        case SDL_WINDOWEVENT_FOCUS_LOST: {
                            KeyboardLoseFocusEvent lose_focus_event;
                            RunEventCallback(lose_focus_event);
                            break;
                        }
                        case SDL_WINDOWEVENT_TAKE_FOCUS: {
                            WindowTakeFocusEvent take_focus_event;
                            RunEventCallback(take_focus_event);
                            break;
                        }
                        case SDL_WINDOWEVENT_MAXIMIZED: {
                            WindowMaximizedEvent maximized_event;
                            RunEventCallback(maximized_event);
                            break;
                        }
                        case SDL_WINDOWEVENT_MINIMIZED: {
                            WindowMinimizedEvent minimized_event;
                            RunEventCallback(minimized_event);
                            break;
                        }
                        case SDL_WINDOWEVENT_LEAVE: {
                            MouseLeaveEvent leave_event;
                            RunEventCallback(leave_event);
                            break;
                        }
                        case SDL_WINDOWEVENT_MOVED: {
                            WindowMovedEvent moved_event(event.window.data1, event.window.data2);
                            RunEventCallback(moved_event);
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
                    RunEventCallback(drop_file_event);
                    SDL_free(dropped_file_dir);    // Free dropped_file_dir memory
                }
                break;
            }
            case SDL_KEYDOWN: {
                if (event.key.windowID == windows_id_) {
                    KeyPressedEvent key_press_event((KeyCode)event.key.keysym.scancode, (KeyModifiers)event.key.keysym.mod, event.key.repeat);
                    RunEventCallback(key_press_event);
                }
                break;
            }
            case SDL_KEYUP: {
                if (event.key.windowID == windows_id_) {
                    KeyReleasedEvent key_released_event((KeyCode)event.key.keysym.scancode, (KeyModifiers)event.key.keysym.mod);
                    RunEventCallback(key_released_event);
                }
                break;
            }
            case SDL_MOUSEBUTTONDOWN: {
                if (event.button.windowID == windows_id_) {
                    MouseButtonPressedEvent mouse_button_pressed_event((MouseButton)event.button.button,
                                                                 event.button.x,
                                                                 event.button.y,
                                                                 (MouseButtonClickMode)event.button.clicks);
                    RunEventCallback(mouse_button_pressed_event);
                }
                break;
            }
            case SDL_MOUSEBUTTONUP:
            {
                if (event.button.windowID == windows_id_) {
                    MouseButtonReleasedEvent mouse_released_event((MouseButton)event.button.button,
                                                                 event.button.x,
                                                                 event.button.y,
                                                                 (MouseButtonClickMode)event.button.clicks);
                    RunEventCallback(mouse_released_event);
                }
                break;
            }
            case SDL_MOUSEWHEEL: {
                if (event.wheel.windowID == windows_id_) {
                    MouseScrolledEvent mouse_scrolled_event(event.wheel.preciseX, event.wheel.preciseY);
                    RunEventCallback(mouse_scrolled_event);
                }
                break;
            }
            case SDL_MOUSEMOTION: {
                if (event.motion.windowID == windows_id_) {
                    MouseMovedEvent mouse_moved_event(event.motion.x,
                                                      event.motion.y,
                                                      event.motion.xrel,
                                                      event.motion.yrel,
                                                      (MouseButtonMask) event.motion.state);
                    RunEventCallback(mouse_moved_event);
                }
                break;
            }
        }
    }

    if (RenderServer::HasInit()) {
        if (render_need_reset_) {
            GET_RS()->Reset(GetWidth(), GetHeight(), s_render_reset_flag);
        }
    }

}

void SDLWindow::RunEventCallback(Event& event)
{
    if (event_callback_) {
        event_callback_(event);
    }
}


}
