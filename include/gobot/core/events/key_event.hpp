/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-7
*/

#pragma once

#include "gobot/core/os/keycodes.hpp"
#include "gobot/core/events/event.hpp"
#include "gobot/core/macros.hpp"

namespace gobot {

class GOBOT_EXPORT KeyEvent : public Event {
    GOBCLASS(KeyEvent, Event)
public:

    explicit KeyEvent(KeyCode key_code);

    [[nodiscard]] FORCE_INLINE KeyCode GetKeyCode() const { return key_code_; }

    EVENT_CLASS_CATEGORY(EventCategoryKeyboard | EventCategoryInput)

protected:
    KeyCode key_code_;
};

class GOBOT_EXPORT KeyPressedEvent : public KeyEvent {
    GOBCLASS(KeyPressedEvent, KeyEvent)
public:
    KeyPressedEvent(KeyCode key_code, uint16_t repeat_count);

    [[nodiscard]] FORCE_INLINE uint16_t GetRepeatCount() const { return repeat_count_; }

    [[nodiscard]] String ToString() const override;

    EVENT_CLASS_TYPE(KeyPressed)

private:
    uint16_t repeat_count_;
};

class GOBOT_EXPORT KeyReleasedEvent : public KeyEvent {
    GOBCLASS(KeyReleasedEvent, KeyEvent)
public:
    explicit KeyReleasedEvent(KeyCode key_code);

    [[nodiscard]] String ToString() const override;

    EVENT_CLASS_TYPE(KeyReleased)
};

class GOBOT_EXPORT KeyTypedEvent : public KeyEvent {
    GOBCLASS(KeyTypedEvent, KeyEvent)
public:
    explicit KeyTypedEvent(KeyCode key_code);

    [[nodiscard]] String ToString() const override;

    EVENT_CLASS_TYPE(KeyTyped)
};

}
