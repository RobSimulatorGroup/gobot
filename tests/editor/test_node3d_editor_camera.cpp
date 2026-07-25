/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <memory>

#include <gobot/core/events/event.hpp>
#include <gobot/core/events/key_event.hpp>
#include <gobot/core/events/mouse_event.hpp>
#include <gobot/core/os/input.hpp>
#include <gobot/editor/node3d_editor.hpp>
#include <gobot/rendering/render_server.hpp>

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

void FireMouseMoved(int x, int y) {
    const gobot::MouseMovedEvent event(x, y, 0, 0, gobot::MouseButtonMask{});
    gobot::Event::Fire(event);
}

void FireMousePressed(gobot::MouseButton button) {
    const gobot::MouseButtonPressedEvent event(
            button, 0, 0, gobot::MouseButtonClickMode::Single);
    gobot::Event::Fire(event);
}

void FireMouseReleased(gobot::MouseButton button) {
    const gobot::MouseButtonReleasedEvent event(
            button, 0, 0, gobot::MouseButtonClickMode::Single);
    gobot::Event::Fire(event);
}

void FireMouseScrolled(float y_offset) {
    const gobot::MouseScrolledEvent event(0.0f, y_offset);
    gobot::Event::Fire(event);
}

class TestNode3DEditorCamera : public testing::Test {
protected:
    void SetUp() override {
        input_ = GetTestInput();
        FireMouseMoved(0, 0);
        render_server_ = std::make_unique<gobot::RenderServer>();
        editor_ = gobot::Object::New<gobot::Node3DEditor>();
        editor_->ApplySceneViewState({
                .eye = gobot::Vector3::Zero(),
                .at = gobot::Vector3::UnitX(),
                .up = gobot::Vector3::UnitZ(),
                .fov_y = 50.0,
        });
    }

    void TearDown() override {
        gobot::Object::Delete(editor_);
        render_server_.reset();
    }

    gobot::Input* input_{nullptr};
    gobot::Node3DEditor* editor_{nullptr};
    std::unique_ptr<gobot::RenderServer> render_server_;
};

TEST_F(TestNode3DEditorCamera, RightMouseLooksWithoutChangingPosition) {
    FireMousePressed(gobot::MouseButton::Right);
    editor_->UpdateCamera(1.0 / 60.0);

    FireMouseMoved(100, 0);
    editor_->UpdateCamera(1.0 / 60.0);

    const gobot::EditorSceneViewState state = editor_->GetSceneViewState();
    EXPECT_TRUE(state.eye.isApprox(gobot::Vector3::Zero(), CMP_EPSILON));
    EXPECT_FALSE((state.at - state.eye).normalized().isApprox(gobot::Vector3::UnitX(), CMP_EPSILON));
    EXPECT_NEAR((state.at - state.eye).norm(), 1.0, CMP_EPSILON);

    FireMouseReleased(gobot::MouseButton::Right);
}

TEST_F(TestNode3DEditorCamera, RightMouseAndWasdMovesThroughScene) {
    FireMousePressed(gobot::MouseButton::Right);
    FireKeyPressed(gobot::KeyCode::W);
    editor_->UpdateCamera(0.1);

    const gobot::EditorSceneViewState state = editor_->GetSceneViewState();
    EXPECT_NEAR(state.eye.x(), 0.3, CMP_EPSILON);
    EXPECT_NEAR(state.at.x(), 1.3, CMP_EPSILON);
    EXPECT_NEAR((state.at - state.eye).norm(), 1.0, CMP_EPSILON);

    FireKeyReleased(gobot::KeyCode::W);
    FireMouseReleased(gobot::MouseButton::Right);
}

TEST_F(TestNode3DEditorCamera, MouseWheelDolliesInsteadOfChangingFocalLength) {
    FireMouseScrolled(1.0f);
    editor_->UpdateCamera(1.0 / 60.0);

    const gobot::EditorSceneViewState state = editor_->GetSceneViewState();
    EXPECT_NEAR(state.eye.x(), 0.2, CMP_EPSILON);
    EXPECT_NEAR(state.at.x(), 1.2, CMP_EPSILON);
    EXPECT_NEAR((state.at - state.eye).norm(), 1.0, CMP_EPSILON);
}

TEST_F(TestNode3DEditorCamera, AltLeftMouseOrbitsAroundFocus) {
    FireKeyPressed(gobot::KeyCode::LeftAlt);
    FireMousePressed(gobot::MouseButton::Left);
    editor_->UpdateCamera(1.0 / 60.0);

    FireMouseMoved(100, 0);
    editor_->UpdateCamera(1.0 / 60.0);

    const gobot::EditorSceneViewState state = editor_->GetSceneViewState();
    EXPECT_TRUE(state.at.isApprox(gobot::Vector3::UnitX(), CMP_EPSILON));
    EXPECT_FALSE(state.eye.isApprox(gobot::Vector3::Zero(), CMP_EPSILON));
    EXPECT_NEAR((state.at - state.eye).norm(), 1.0, CMP_EPSILON);

    FireMouseReleased(gobot::MouseButton::Left);
    FireKeyReleased(gobot::KeyCode::LeftAlt);
}

TEST_F(TestNode3DEditorCamera, SceneViewStateRestoresFieldOfView) {
    EXPECT_DOUBLE_EQ(editor_->GetCamera3D()->GetFovy(), 50.0);
    EXPECT_DOUBLE_EQ(editor_->GetSceneViewState().fov_y, 50.0);
}

} // namespace
