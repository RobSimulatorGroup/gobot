/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-7
*/

#pragma once

#include "gobot/core/types.hpp"
#include "gobot/core/object.hpp"

namespace gobot {

enum class EventType
{
    None = 0,
    WindowClose,
    WindowResize,
    WindowMaximized,
    WindowMinimized,
    WindowMoved,
    WindowTakeFocus,
    WindowDropFile,
    KeyPressed,
    KeyReleased,
    KeyboardFocus,
    KeyboardLoseFocus,
    MouseButtonPressed,
    MouseButtonReleased,
    MouseMoved,
    MouseScrolled,
    MouseEnter,
    MouseLeave
};

enum EventCategory
{
    None                     = 0,
    EventCategoryWindow      = 1 << 0,
    EventCategoryInput       = 1 << 1,
    EventCategoryKeyboard    = 1 << 2,
    EventCategoryMouse       = 1 << 3,
    EventCategoryMouseButton = 1 << 4
};


#define EVENT_CLASS_TYPE(type)                                                 \
static EventType GetStaticType() { return EventType::type; }                   \
virtual EventType GetEventType() const override { return GetStaticType(); }    \
virtual const char* GetName() const override { return #type; }


#define EVENT_CLASS_CATEGORY(category)                                         \
virtual int GetCategoryFlags() const override { return category; }


#define BIND_EVENT_FN(fn) [this](auto&&... args) -> decltype(auto) {           \
    return this->fn(std::forward<decltype(args)>(args)...);                    \
}


class GOBOT_EXPORT Event : public Object {
    GOBCLASS(Event, Object)
public:
    using Subscriber = std::function<void(const Event&)>;

    [[nodiscard]] virtual EventType GetEventType() const = 0;

    [[nodiscard]] virtual const char* GetName() const = 0;

    [[nodiscard]] virtual int GetCategoryFlags() const = 0;

    [[nodiscard]] virtual std::string ToString() const { return GetName(); }

    [[nodiscard]] FORCE_INLINE bool IsInCategory(EventCategory category) const
    {
        return GetCategoryFlags() & category;
    }

    [[nodiscard]] FORCE_INLINE bool Handled() const { return handled_; }

    friend std::ostream& operator<<(std::ostream& os, const Event& e)
    {
        return os << e.ToString();
    }

    static void Shutdown();

    static void Subscribe(const EventType event_type, Subscriber&& function);

    static void Fire(const Event& event);

protected:
    bool handled_ = false;
};


}
