/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-1-14
*/


#include "gobot/scene/window.hpp"
#include "gobot/drivers/sdl/sdl_window.hpp"
#include "gobot/core/os/input.hpp"
#include "gobot/render/graphics_context.hpp"
#include "gobot/log.hpp"
#include "gobot/error_macros.hpp"

namespace gobot {

Window::Window() {
    switch (window_driver_) {
        case WindowDriver::SDL:
        case WindowDriver::Win32:
        case WindowDriver::IOSWindow:
            window_interface_ = std::make_unique<SDLWindow>();
    }

    window_interface_->Maximize();

    RegisterWindowCallbacks();
}


Window::~Window() {

}

void Window::SetVisible(bool visible)
{
    if (visible)
        window_interface_->ShowWindow();
    else
        window_interface_->HideWindow();
}

bool Window::IsVisible() const
{
    return !window_interface_->IsWindowHide();
}

void Window::OnEvent(Event& e)
{
    if (e.GetCategoryFlags() == EventCategory::EventCategoryWindow) {
        if (e.GetEventType() == EventType::WindowClose) {
            Q_EMIT windowCloseRequested();
        }
        else if (e.GetEventType() == EventType::WindowResize) {
            Q_EMIT windowResizeRequested(dynamic_cast<WindowResizeEvent&>(e));
        }
        else if (e.GetEventType() == EventType::WindowMaximized) {
            Q_EMIT windowMaximizedRequested();
        }
        else if (e.GetEventType() == EventType::WindowMinimized) {
            Q_EMIT windowMinimizedRequested();
        }
        else if (e.GetEventType() == EventType::WindowMoved) {
            Q_EMIT windowMovedRequested();
        }
        else if (e.GetEventType() == EventType::WindowTakeFocus) {
            Q_EMIT windowTakeFocusRequested();
        }
        else if (e.GetEventType() == EventType::WindowDropFile) {
            Q_EMIT windowDropFileRequested();
        }
    }
}

void Window::PullEvent()
{
    window_interface_->PollEvents();
}

void Window::RegisterWindowCallbacks()
{
    window_interface_->SetEventCallback(BIND_EVENT_FN(Window::OnEvent));
}



}