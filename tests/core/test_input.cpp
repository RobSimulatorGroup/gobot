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

TEST(TestInput, KeyQueriesIncludeHighValueModifierScancodes) {
    auto* input = GetTestInput();
    input->SetControlFocus(true);

    gobot::KeyCode parsed = gobot::KeyCode::Unknown;
    ASSERT_TRUE(gobot::Input::TryParseKeyName("LeftShift", parsed));
    EXPECT_EQ(parsed, gobot::KeyCode::LeftShift);
    ASSERT_TRUE(gobot::Input::TryParseKeyName("right_shift", parsed));
    EXPECT_EQ(parsed, gobot::KeyCode::RightShift);

    FireKeyPressed(gobot::KeyCode::LeftShift);
    EXPECT_TRUE(input->IsKeyHeldByName("LeftShift"));
    FireKeyReleased(gobot::KeyCode::LeftShift);
    EXPECT_FALSE(input->IsKeyHeldByName("LeftShift"));
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
