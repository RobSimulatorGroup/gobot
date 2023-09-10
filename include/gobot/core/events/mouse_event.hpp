/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-7
*/

#pragma once

#include "gobot/core/os/keycodes.hpp"
#include "gobot/core/os/mouse_button.hpp"
#include "gobot/core/events/event.hpp"

namespace gobot {

class GOBOT_EXPORT MouseMovedEvent : public Event {
    GOBCLASS(MouseMovedEvent, Event)
public:
    explicit MouseMovedEvent(std::int32_t x,
                             std::int32_t y,
                             std::int32_t rel_x,
                             std::int32_t rel_y,
                             MouseButtonMask state);

    [[nodiscard]] FORCE_INLINE std::int32_t GetX() const { return x_; }

    [[nodiscard]] FORCE_INLINE std::int32_t GetY() const { return y_; }

    [[nodiscard]] FORCE_INLINE std::int32_t GetRelX() const { return rel_x_; }

    [[nodiscard]] FORCE_INLINE std::int32_t GetRelY() const { return rel_y_; }

    [[nodiscard]] FORCE_INLINE MouseButtonMask GetButtonMask() const { return state_; }

    [[nodiscard]] std::string ToString() const override;

    EVENT_CLASS_TYPE(MouseMoved)
    EVENT_CLASS_CATEGORY(EventCategoryMouse | EventCategoryInput)

private:
    std::int32_t x_{0};
    std::int32_t y_{0};
    std::int32_t rel_x_{0};
    std::int32_t rel_y_{0};
    MouseButtonMask state_;
};

class GOBOT_EXPORT MouseScrolledEvent : public Event {
    GOBCLASS(MouseScrolledEvent, Event)
public:
    explicit MouseScrolledEvent(float x_offset, float y_offset);

    [[nodiscard]] FORCE_INLINE float GetXOffset() const { return x_offset_; }

    [[nodiscard]] FORCE_INLINE float GetYOffset() const { return y_offset_; }

    [[nodiscard]] std::string ToString() const override;

    EVENT_CLASS_TYPE(MouseScrolled)
    EVENT_CLASS_CATEGORY(EventCategoryMouse | EventCategoryInput)

private:
    float x_offset_;
    float y_offset_;
};

class GOBOT_EXPORT MouseButtonEvent : public Event {
    GOBCLASS(MouseButtonEvent, Event)
public:
    explicit MouseButtonEvent(MouseButton button,
                              std::int32_t x_coordinate,
                              std::int32_t y_coordinate,
                              MouseButtonClickMode click_mode);

    [[nodiscard]] FORCE_INLINE MouseButton GetMouseButton() const { return button_; }

    [[nodiscard]] FORCE_INLINE std::int32_t GetXCoordinate() const { return x_coordinate_; }

    [[nodiscard]] FORCE_INLINE std::int32_t GetYCoordinate() const { return y_coordinate_; }

    [[nodiscard]] FORCE_INLINE MouseButtonClickMode GetMouseButtonClickMode() const { return click_mode_; }

    EVENT_CLASS_CATEGORY(EventCategoryMouse | EventCategoryInput | EventCategoryMouseButton)

protected:
    MouseButton button_;
    std::int32_t x_coordinate_;
    std::int32_t y_coordinate_;
    MouseButtonClickMode click_mode_;
 };

class GOBOT_EXPORT MouseButtonPressedEvent : public MouseButtonEvent {
    GOBCLASS(MouseButtonPressedEvent, MouseButtonEvent)
public:
    explicit MouseButtonPressedEvent(MouseButton button,
                                     std::int32_t x_coordinate,
                                     std::int32_t y_coordinate,
                                     MouseButtonClickMode click_mode);

    [[nodiscard]] std::string ToString() const override;

    EVENT_CLASS_TYPE(MouseButtonPressed)
};

class GOBOT_EXPORT MouseButtonReleasedEvent : public MouseButtonEvent {
    GOBCLASS(MouseButtonReleasedEvent, MouseButtonEvent)
public:
    explicit MouseButtonReleasedEvent(MouseButton button,
                                      std::int32_t x_coordinate,
                                      std::int32_t y_coordinate,
                                      MouseButtonClickMode click_mode);

    [[nodiscard]] std::string ToString() const override;

    EVENT_CLASS_TYPE(MouseButtonReleased)
};


class GOBOT_EXPORT MouseEnterEvent : public Event {
    GOBCLASS(MouseEnterEvent, Event)
public:
    MouseEnterEvent() = default;

    EVENT_CLASS_TYPE(MouseEnter)
    EVENT_CLASS_CATEGORY(EventCategoryMouse | EventCategoryInput)
};

class GOBOT_EXPORT MouseLeaveEvent : public Event {
    GOBCLASS(MouseLeaveEvent, Event)
public:
    MouseLeaveEvent() = default;

    EVENT_CLASS_TYPE(MouseLeave)
    EVENT_CLASS_CATEGORY(EventCategoryMouse | EventCategoryInput)
};

}
