#include <gtest/gtest.h>

#include "gobot/core/events/event.hpp"
#include "gobot/core/events/window_event.hpp"
#include "gobot/scene/scene_tree.hpp"

using namespace gobot;

TEST(EventConnection, DestructionUnsubscribes) {
    int calls = 0;
    const WindowCloseEvent event;
    {
        auto connection = Event::SubscribeScoped(EventType::WindowClose, [&](const Event&) { ++calls; });
        Event::Fire(event);
        EXPECT_EQ(calls, 1);
    }
    Event::Fire(event);
    EXPECT_EQ(calls, 1);
}

TEST(EventConnection, DisconnectDuringDispatchSkipsDestroyedOwner) {
    int calls = 0;
    Event::Connection second;
    auto first = Event::SubscribeScoped(EventType::WindowClose, [&](const Event&) { second.Disconnect(); });
    second = Event::SubscribeScoped(EventType::WindowClose, [&](const Event&) { ++calls; });
    Event::Fire(WindowCloseEvent{});
    EXPECT_EQ(calls, 0);
}

TEST(EventConnection, SubscribeDuringDispatchStartsNextEvent) {
    int calls = 0;
    Event::Connection second;
    auto first = Event::SubscribeScoped(EventType::WindowClose, [&](const Event&) {
        second = Event::SubscribeScoped(EventType::WindowClose, [&](const Event&) { ++calls; });
    });
    Event::Fire(WindowCloseEvent{});
    EXPECT_EQ(calls, 0);
    first.Disconnect();
    Event::Fire(WindowCloseEvent{});
    EXPECT_EQ(calls, 1);
}

TEST(EventConnection, MoveAssignmentDisconnectsPreviousSubscription) {
    int old_calls = 0;
    int new_calls = 0;
    auto first = Event::SubscribeScoped(EventType::WindowClose, [&](const Event&) { ++old_calls; });
    auto second = Event::SubscribeScoped(EventType::WindowClose, [&](const Event&) { ++new_calls; });
    first = std::move(second);
    Event::Fire(WindowCloseEvent{});
    EXPECT_EQ(old_calls, 0);
    EXPECT_EQ(new_calls, 1);
}

TEST(EventConnection, HeadlessTreesDoNotReceiveWindowEvents) {
    for (int i = 0; i < 100; ++i) {
        auto* tree = Object::New<SceneTree>(false);
        Event::Fire(WindowCloseEvent{});
        EXPECT_FALSE(tree->IsQuitRequested());
        Object::Delete(tree);
    }
    Event::Fire(WindowCloseEvent{});
}
