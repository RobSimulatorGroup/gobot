/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * This file is created by Qiqi Wu, 23-9-10.
 * SPDX-License-Identifier: Apache-2.0
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

void Event::Subscribe(const EventType& event_type, Subscriber&& function)
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
