/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * This file is created by Qiqi Wu, 23-9-10.
 * SPDX-License-Identifier: Apache-2.0
 */


#include "gobot/core/events/event.hpp"

#include <algorithm>
#include <utility>

namespace gobot {

struct Event::Connection::Slot {
    Subscriber callback;
    bool connected{true};
};

std::array<std::vector<std::weak_ptr<Event::Connection::Slot>>, 32> Event::subscribers_;
std::vector<Event::Connection> Event::permanent_connections_;

Event::Connection::Connection(std::shared_ptr<Slot> slot) : slot_(std::move(slot)) {}
Event::Connection::~Connection() { Disconnect(); }
Event::Connection::Connection(Connection&& other) noexcept = default;
Event::Connection& Event::Connection::operator=(Connection&& other) noexcept {
    if (this != &other) {
        Disconnect();
        slot_ = std::move(other.slot_);
    }
    return *this;
}

void Event::Connection::Disconnect() {
    if (slot_) {
        slot_->connected = false;
        slot_.reset();
    }
}

void Event::Shutdown()
{
    for (auto& subscribers : subscribers_) {
        for (auto& weak : subscribers) {
            if (auto slot = weak.lock()) {
                slot->connected = false;
            }
        }
        subscribers.clear();
    }
    permanent_connections_.clear();
}

void Event::Subscribe(const EventType& event_type, Subscriber&& function)
{
    permanent_connections_.push_back(SubscribeScoped(event_type, std::move(function)));
}

Event::Connection Event::SubscribeScoped(EventType event_type, Subscriber function) {
    const auto index = static_cast<std::size_t>(event_type);
    if (index >= subscribers_.size() || !function) {
        return {};
    }
    auto& subscribers = subscribers_[index];
    std::erase_if(subscribers, [](const auto& weak) { return weak.expired(); });
    auto slot = std::make_shared<Connection::Slot>();
    slot->callback = std::move(function);
    subscribers.push_back(slot);
    return Connection(std::move(slot));
}

void Event::Fire(const Event& event)
{
    const auto index = static_cast<std::size_t>(event.GetEventType());
    if (index >= subscribers_.size()) {
        return;
    }
    auto& subscribers = subscribers_[index];
    std::erase_if(subscribers, [](const auto& weak) { return weak.expired(); });
    // Snapshot the list: callbacks may subscribe, disconnect, or destroy an owner.
    const auto dispatch = subscribers;
    for (const auto& weak : dispatch) {
        if (auto slot = weak.lock(); slot && slot->connected) {
            slot->callback(event);
        }
    }
}


}
