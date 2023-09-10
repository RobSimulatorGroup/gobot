/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-7
*/

#pragma once

#include "gobot/core/object.hpp"
#include "gobot/core/os/keycodes.hpp"
#include "gobot/core/os/mouse_button.hpp"
#include "gobot/core/events/event.hpp"
#include "gobot/core/events/key_event.hpp"
#include "gobot/core/events/mouse_event.hpp"
#include "gobot/core/math/matrix.hpp"
#include <gobot_export.h>

#include "Eigen/Dense"

#include <mutex>

namespace gobot {

enum class MouseMode
{
    Visible,
    Hidden,
    Captured
};

enum class MouseClickedState {
    None = 0,
    SingleClicked,
    DoubleClicked
};

class GOBOT_EXPORT Input : public Object {
    GOBCLASS(Input, Object);

public:
    using KeyCodeUInt = std::underlying_type_t<KeyCode>;
    using MouseButtonUInt = std::underlying_type_t<MouseButton>;

    Input();

    void Reset();

    void ResetPressed();

    static Input* GetInstance();

    FORCE_INLINE const Eigen::Vector2i& GetMousePosition() const { return mouse_position_; }

    FORCE_INLINE MouseClickedState GetMouseClickedState(MouseButton mouse_button) const {
        return mouse_clicked_[MouseButtonUInt(mouse_button)];
    }

    void SetMouseMode(MouseMode mode);

    MouseMode GetMouseMode() const;

    FORCE_INLINE bool GetKeyPressed(KeyCode key_code) const { return key_pressed_[KeyCodeUInt(key_code)]; }

    FORCE_INLINE float GetScrollOffset() const { return scroll_offset_; }

    void SetScrollOffset(float offset);

private:
    FORCE_INLINE void SetKeyPressed(KeyCode key, bool pressed);

    FORCE_INLINE void SetKeyHeld(KeyCode key, bool held);

    void SetMouseClicked(MouseButton key, MouseClickedState clicked);

private:
    bool OnKeyPressed(const KeyPressedEvent& e);

    bool OnKeyReleased(const KeyReleasedEvent& e);

    bool OnMousePressed(const MouseButtonPressedEvent& e);

    bool OnMouseReleased(const MouseButtonReleasedEvent& e);

    bool OnMouseScrolled(const MouseScrolledEvent& e);

    bool OnMouseMoved(const MouseMovedEvent& e);

    bool OnMouseEnter(const MouseEnterEvent& e);

    bool OnMouseLeave(const MouseLeaveEvent& e);

private:

    static Input* s_singleton;

    mutable std::mutex mutex_;

    bool key_pressed_[static_cast<KeyCodeUInt>(KeyCode::KeyCodeMaxNum)];
    bool key_held_[static_cast<KeyCodeUInt>(KeyCode::KeyCodeMaxNum)];

    MouseClickedState mouse_clicked_[static_cast<MouseButtonUInt>(MouseButton::ButtonMaxNum)];

    float scroll_offset_{0.0f};

    bool mouse_on_screen_{true};
    MouseMode mouse_mode_{MouseMode::Visible};

    Vector2i mouse_position_{0.0, 0.0};
};

}