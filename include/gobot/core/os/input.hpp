/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-7
*/

#pragma once

#include "gobot/core/object.hpp"
#include "gobot/core/os/keycodes.hpp"
#include "gobot/core/events/event.hpp"
#include "gobot/core/events/key_event.hpp"
#include "gobot/core/events/mouse_event.hpp"

#include "Eigen/Dense"

#include <mutex>

namespace gobot {

enum class MouseMode
{
    Visible,
    Hidden,
    Captured
};


class Input : public Object {
    GOBCLASS(Input, Object);

public:
    using KeyCodeUInt = std::underlying_type_t<KeyCode>;
    using MouseKeyCodeUInt = std::underlying_type_t<MouseKeyCode>;

    Input();

    void Reset();

    void ResetPressed();

    void OnEvent(Event& e);

    FORCE_INLINE void SetKeyPressed(KeyCode key, bool pressed) { key_pressed_[KeyCodeUInt(key)] = pressed; }

    FORCE_INLINE void SetKeyHeld(KeyCode key, bool held) { key_held_[KeyCodeUInt(key)] = held; }

    void SetMouseClicked(MouseKeyCode key, bool clicked) { mouse_clicked_[MouseKeyCodeUInt(key)] = clicked; }

    void SetMouseHeld(MouseKeyCode key, bool held) { mouse_held_[MouseKeyCodeUInt(key)] = held; }

    void SetScrollOffset(float offset) { scroll_offset_ = offset; }

    float GetScrollOffset() const { return scroll_offset_; }

    void StoreMousePosition(float xpos, float ypos) { mouse_position_ = Eigen::Vector2f(xpos, ypos); }

    const Eigen::Vector2f& GetMousePosition() const { return mouse_position_; }

    void SetMouseOnScreen(bool on_screen) { mouse_on_screen_ = on_screen; }

    bool GetMouseOnScreen() const { return mouse_on_screen_; }

    static Input* GetInstance();

    void SetMouseMode(MouseMode mode);

    MouseMode GetMouseMode() const;

    bool GetKeyPressed(KeyCode key_code) const { return key_pressed_[KeyCodeUInt(key_code)]; }

private:
    bool OnKeyPressed(KeyPressedEvent& e);

    bool OnKeyReleased(KeyReleasedEvent& e);

    bool OnMousePressed(MouseButtonPressedEvent& e);

    bool OnMouseReleased(MouseButtonReleasedEvent& e);

    bool OnMouseScrolled(MouseScrolledEvent& e);

    bool OnMouseMoved(MouseMovedEvent& e);

    bool OnMouseEnter(MouseEnterEvent& e);

private:
    static Input* s_singleton;

    mutable std::mutex mutex_;

    bool key_pressed_[static_cast<KeyCodeUInt>(KeyCode::MaxKey)];
    bool key_held_[static_cast<KeyCodeUInt>(KeyCode::MaxKey)];

    bool mouse_held_[static_cast<MouseKeyCodeUInt>(MouseKeyCode::MaxButton)];
    bool mouse_clicked_[static_cast<MouseKeyCodeUInt>(MouseKeyCode::MaxButton)];

    float scroll_offset_ = 0.0f;

    bool mouse_on_screen_;
    MouseMode mouse_mode_;

    Eigen::Vector2f mouse_position_;
};

}