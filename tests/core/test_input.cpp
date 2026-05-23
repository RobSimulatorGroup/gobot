#include <gtest/gtest.h>

#include <gobot/core/events/event.hpp>
#include <gobot/core/events/key_event.hpp>
#include <gobot/core/os/input.hpp>

namespace {

gobot::Input* GetTestInput() {
    static gobot::Input* input = gobot::Object::New<gobot::Input>();
    input->Reset();
    return input;
}

void FireKeyPressed(gobot::KeyCode key) {
    const gobot::KeyPressedEvent event(key, gobot::KeyModifiers::None, 0);
    gobot::Event::Fire(event);
}

void FireKeyReleased(gobot::KeyCode key) {
    const gobot::KeyReleasedEvent event(key, gobot::KeyModifiers::None);
    gobot::Event::Fire(event);
}

} // namespace

TEST(TestInput, KeyQueriesRequireControlFocus) {
    auto* input = GetTestInput();

    FireKeyPressed(gobot::KeyCode::W);
    EXPECT_FALSE(input->IsKeyHeldByName("W"));

    input->SetControlFocus(true);
    EXPECT_TRUE(input->IsKeyHeldByName("W"));
    EXPECT_TRUE(input->IsKeyHeldByName("w"));

    FireKeyReleased(gobot::KeyCode::W);
    EXPECT_FALSE(input->IsKeyHeldByName("W"));
}

TEST(TestInput, ControlFocusClearsJustPressedState) {
    auto* input = GetTestInput();

    input->SetControlFocus(true);
    FireKeyPressed(gobot::KeyCode::R);
    EXPECT_TRUE(input->IsKeyPressedByName("R"));

    input->SetControlFocus(false);
    input->SetControlFocus(true);
    EXPECT_FALSE(input->IsKeyPressedByName("R"));

    FireKeyReleased(gobot::KeyCode::R);
}

TEST(TestInput, EscapeReleasesControlFocus) {
    auto* input = GetTestInput();

    input->SetControlFocus(true);
    FireKeyPressed(gobot::KeyCode::Escape);
    EXPECT_FALSE(input->HasControlFocus());

    FireKeyReleased(gobot::KeyCode::Escape);
}
