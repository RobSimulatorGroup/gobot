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
    WindowResizeEvent(unsigned int width, unsigned int height);

    [[nodiscard]] FORCE_INLINE unsigned int GetWidth() const { return width_; }

    [[nodiscard]] FORCE_INLINE unsigned int GetHeight() const { return height_; }

    [[nodiscard]] String ToString() const override;

    EVENT_CLASS_TYPE(WindowResize)
    EVENT_CLASS_CATEGORY(EventCategoryApplication)

private:
    unsigned int width_;
    unsigned int height_;
};

class GOBOT_EXPORT WindowCloseEvent : public Event {
    GOBCLASS(WindowCloseEvent, Event)
public:
    WindowCloseEvent() = default;

    EVENT_CLASS_TYPE(WindowClose)
    EVENT_CLASS_CATEGORY(EventCategoryApplication)
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