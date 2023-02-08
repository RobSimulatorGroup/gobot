/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-8
*/

#include "gobot/core/os/input.hpp"


namespace gobot {

Input *Input::s_singleton = nullptr;

Input::Input()
    :  mouse_mode_(MouseMode::Visible)
{
}

void Input::Reset()
{
    memset(key_pressed_, 0, static_cast<std::uint32_t>(KeyCode::MaxKey));
    memset(key_held_, 0, static_cast<std::uint32_t>(KeyCode::MaxKey));

    memset(mouse_clicked_, 0, static_cast<std::uint32_t>(MouseKeyCode::MaxButton));
    memset(mouse_held_, 0, static_cast<std::uint32_t>(MouseKeyCode::MaxButton));

    mouse_on_screen_ = true;
    scroll_offset_  = 0.0f;
}

void Input::ResetPressed()
{
    memset(key_pressed_, 0, static_cast<std::uint32_t>(KeyCode::MaxKey));
    memset(mouse_clicked_, 0, static_cast<std::uint32_t>(MouseKeyCode::MaxButton));
    scroll_offset_ = 0;
}

void Input::OnEvent(Event& e) {
    EventDispatcher dispatcher(e);
    dispatcher.Dispatch<KeyPressedEvent>(BIND_EVENT_FN(Input::OnKeyPressed));
    dispatcher.Dispatch<KeyReleasedEvent>(BIND_EVENT_FN(Input::OnKeyReleased));
    dispatcher.Dispatch<MouseButtonPressedEvent>(BIND_EVENT_FN(Input::OnMousePressed));
    dispatcher.Dispatch<MouseButtonReleasedEvent>(BIND_EVENT_FN(Input::OnMouseReleased));
    dispatcher.Dispatch<MouseScrolledEvent>(BIND_EVENT_FN(Input::OnMouseScrolled));
    dispatcher.Dispatch<MouseMovedEvent>(BIND_EVENT_FN(Input::OnMouseMoved));
    dispatcher.Dispatch<MouseEnterEvent>(BIND_EVENT_FN(Input::OnMouseEnter));
}

bool Input::OnKeyPressed(KeyPressedEvent& e)
{
    SetKeyPressed(e.GetKeyCode(), e.GetRepeatCount() < 1);
    SetKeyHeld(e.GetKeyCode(), true);
    return false;
}

bool Input::OnKeyReleased(KeyReleasedEvent& e)
{
    SetKeyPressed(e.GetKeyCode(), false);
    SetKeyHeld(e.GetKeyCode(), false);
    return false;
}

bool Input::OnMousePressed(MouseButtonPressedEvent& e)
{
    SetMouseClicked(e.GetMouseButton(), true);
    SetMouseHeld(e.GetMouseButton(), true);
    return false;
}

bool Input::OnMouseReleased(MouseButtonReleasedEvent& e)
{
    SetMouseClicked(e.GetMouseButton(), false);
    SetMouseHeld(e.GetMouseButton(), false);
    return false;
}

bool Input::OnMouseScrolled(MouseScrolledEvent& e)
{
    SetScrollOffset(e.GetYOffset());
    return false;
}

bool Input::OnMouseMoved(MouseMovedEvent& e)
{
    StoreMousePosition(e.GetX(), e.GetY());
    return false;
}

bool Input::OnMouseEnter(MouseEnterEvent& e)
{
    SetMouseOnScreen(e.GetEntered());
    return false;
}


Input* Input::GetInstance() {
    return s_singleton;
}

void Input::SetMouseMode(MouseMode p_mode) {
    mouse_mode_ = p_mode;
}

MouseMode Input::GetMouseMode() const {
    return mouse_mode_;
}



}