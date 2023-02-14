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

    [[nodiscard]] String ToString() const override;

    EVENT_CLASS_TYPE(WindowResize)
    EVENT_CLASS_CATEGORY(EventCategoryApplication)

private:
    std::uint32_t width_;
    std::uint32_t height_;
};

class GOBOT_EXPORT WindowCloseEvent : public Event {
    GOBCLASS(WindowCloseEvent, Event)
public:
    WindowCloseEvent() = default;

    EVENT_CLASS_TYPE(WindowClose)
    EVENT_CLASS_CATEGORY(EventCategoryApplication)
};

class WindowFocusEvent : public Event {
    GOBCLASS(WindowFocusEvent, Event)
public:
    WindowFocusEvent() = default;
    EVENT_CLASS_TYPE(WindowFocus)
    EVENT_CLASS_CATEGORY(EventCategoryApplication)
};

class WindowLostFocusEvent : public Event {
    GOBCLASS(WindowLostFocusEvent, Event)
public:
    WindowLostFocusEvent() = default;
    EVENT_CLASS_TYPE(WindowLostFocus)
    EVENT_CLASS_CATEGORY(EventCategoryApplication)
};

class WindowMovedEvent : public Event {
    GOBCLASS(WindowMovedEvent, Event)
public:
    WindowMovedEvent(std::uint32_t x, std::uint32_t y);

    [[nodiscard]] String ToString() const override;

    EVENT_CLASS_TYPE(WindowMoved)
    EVENT_CLASS_CATEGORY(EventCategoryApplication)
private:
    std::uint32_t x_;
    std::uint32_t y_;
};


class GOBOT_EXPORT WindowDropFileEvent : public Event {
    GOBCLASS(WindowDropFileEvent, Event)
public:
    explicit WindowDropFileEvent(String file_path);

    [[nodiscard]] FORCE_INLINE const String& GetFilePath() const { return file_path_; }

    [[nodiscard]] String ToString() const override;

    EVENT_CLASS_TYPE(WindowDropFile)
    EVENT_CLASS_CATEGORY(EventCategoryApplication)
private:
    String file_path_;
};

class GOBOT_EXPORT AppTickEvent : public Event {
    GOBCLASS(AppTickEvent, Event)
public:
    AppTickEvent() = default;

    EVENT_CLASS_TYPE(AppTick)
    EVENT_CLASS_CATEGORY(EventCategoryApplication)
};

class GOBOT_EXPORT AppUpdateEvent : public Event {
    GOBCLASS(AppUpdateEvent, Event)
public:
    AppUpdateEvent() = default;

    EVENT_CLASS_TYPE(AppUpdate)
    EVENT_CLASS_CATEGORY(EventCategoryApplication)
};

class GOBOT_EXPORT AppRenderEvent : public Event {
    GOBCLASS(AppRenderEvent, Event)
public:
    AppRenderEvent() = default;

    EVENT_CLASS_TYPE(AppRender)
    EVENT_CLASS_CATEGORY(EventCategoryApplication)
};


}