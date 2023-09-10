/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-7
*/

#pragma once

#include "gobot/core/events/event.hpp"

namespace gobot {

class GOBOT_EXPORT WindowResizeEvent : public Event {
    GOBCLASS(WindowResizeEvent, Event)
public:
    WindowResizeEvent(std::uint32_t width, std::uint32_t height);

    [[nodiscard]] FORCE_INLINE unsigned int GetWidth() const { return width_; }

    [[nodiscard]] FORCE_INLINE unsigned int GetHeight() const { return height_; }

    [[nodiscard]] std::string ToString() const override;

    EVENT_CLASS_TYPE(WindowResize)
    EVENT_CLASS_CATEGORY(EventCategoryWindow)

private:
    std::uint32_t width_;
    std::uint32_t height_;
};

class GOBOT_EXPORT WindowCloseEvent : public Event {
    GOBCLASS(WindowCloseEvent, Event)
public:
    WindowCloseEvent() = default;

    EVENT_CLASS_TYPE(WindowClose)
    EVENT_CLASS_CATEGORY(EventCategoryWindow)
};

class WindowMaximizedEvent : public Event {
    GOBCLASS(WindowMaximizedEvent, Event)
public:
    WindowMaximizedEvent() = default;

    EVENT_CLASS_TYPE(WindowMaximized)
    EVENT_CLASS_CATEGORY(EventCategoryWindow)
};

class WindowMinimizedEvent : public Event {
    GOBCLASS(WindowMinimizedEvent, Event)
public:
    WindowMinimizedEvent() = default;

    EVENT_CLASS_TYPE(WindowMinimized)
    EVENT_CLASS_CATEGORY(EventCategoryWindow)
};

class WindowMovedEvent : public Event {
    GOBCLASS(WindowMovedEvent, Event)
public:
    WindowMovedEvent(std::uint32_t x, std::uint32_t y);

    [[nodiscard]] std::string ToString() const override;

    EVENT_CLASS_TYPE(WindowMoved)
    EVENT_CLASS_CATEGORY(EventCategoryWindow)
private:
    std::uint32_t x_;
    std::uint32_t y_;
};

class WindowTakeFocusEvent : public Event {
    GOBCLASS(WindowTakeFocusEvent, Event)
public:
    WindowTakeFocusEvent() = default;

    EVENT_CLASS_TYPE(WindowTakeFocus)
    EVENT_CLASS_CATEGORY(EventCategoryWindow)
};

class GOBOT_EXPORT WindowDropFileEvent : public Event {
    GOBCLASS(WindowDropFileEvent, Event)
public:
    explicit WindowDropFileEvent(std::string file_path);

    [[nodiscard]] FORCE_INLINE const std::string& GetFilePath() const { return file_path_; }

    [[nodiscard]] std::string ToString() const override;

    EVENT_CLASS_TYPE(WindowDropFile)
    EVENT_CLASS_CATEGORY(EventCategoryWindow)
private:
    std::string file_path_;
};

}