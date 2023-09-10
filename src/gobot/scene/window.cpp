/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-1-14
*/


#include "gobot/scene/window.hpp"
#include "gobot/drivers/sdl/sdl_window.hpp"
#include "gobot/core/os/input.hpp"
#include "gobot/log.hpp"
#include "gobot/error_macros.hpp"

namespace gobot {

Window::Window(bool p_init_sdl_window) {
    if (p_init_sdl_window) {
        window_ = std::make_unique<SDLWindow>();
        window_->Maximize();
    }
}


Window::~Window() {

}

void Window::SetVisible(bool visible)
{
    if (visible)
        window_->ShowWindow();
    else
        window_->HideWindow();
}

bool Window::IsVisible() const
{
    return !window_->IsWindowHide();
}

void Window::PullEvent()
{
    window_->ProcessEvents();
}

void Window::SwapBuffers() {
    window_->SwapBuffers();
}


}