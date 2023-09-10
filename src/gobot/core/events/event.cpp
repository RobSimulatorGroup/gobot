/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-9-10.
*/


#include "gobot/core/events/event.hpp"

namespace gobot {

namespace
{
    std::array<std::vector<Event::Subscriber>, 32> s_event_subscribers;
}

void Event::Shutdown()
{
    for (std::vector<Event::Subscriber>& subscribers : s_event_subscribers) {
        subscribers.clear();
    }
}

void Event::Subscribe(const EventType event_type, Subscriber&& function)
{
    s_event_subscribers[static_cast<uint32_t>(event_type)].push_back(std::forward<Subscriber>(function));
}

void Event::Fire(const Event& event)
{
    for (const auto& subscriber : s_event_subscribers[static_cast<uint32_t>(event.GetEventType())]) {
        subscriber(event);
    }
}


}
