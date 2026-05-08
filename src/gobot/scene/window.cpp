/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-1-14
*/


#include "gobot/scene/window.hpp"
#include "gobot/core/os/input.hpp"
#include "gobot/log.hpp"
#include "gobot/error_macros.hpp"

namespace gobot {

namespace {

Window::WindowFactory& MutableWindowFactory() {
    static Window::WindowFactory factory;
    return factory;
}

}

Window::Window(bool p_init_sdl_window) {
    if (p_init_sdl_window) {
        if (auto& factory = MutableWindowFactory()) {
            window_ = factory();
            window_->Maximize();
        }
    }
}


Window::~Window() {

}

void Window::SetWindowFactory(WindowFactory factory) {
    MutableWindowFactory() = std::move(factory);
}

void Window::SetVisible(bool visible)
{
    ERR_FAIL_COND(window_ == nullptr);
    if (visible)
        window_->ShowWindow();
    else
        window_->HideWindow();
}

bool Window::IsVisible() const
{
    ERR_FAIL_COND_V(window_ == nullptr, false);
    return !window_->IsWindowHide();
}

void Window::PullEvent()
{
    ERR_FAIL_COND(window_ == nullptr);
    window_->ProcessEvents();
}

void Window::SwapBuffers() {
    ERR_FAIL_COND(window_ == nullptr);
    window_->SwapBuffers();
}


}
