/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * This file is created by Qiqi Wu, 23-3-15
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "gobot/scene/resources/texture.hpp"
#include "gobot/scene/imgui_window.hpp"
#include "imgui.h"

#include <memory>

namespace gobot {

class SceneTree;
class EditorViewportRenderer;
class Joint3D;
class Node;

class GOBOT_EXPORT SceneView3DPanel : public ImGuiWindow {
    GOBCLASS(SceneView3DPanel, ImGuiWindow)
public:
    SceneView3DPanel();

    ~SceneView3DPanel() override;

    bool Begin() override;

    void OnImGuiContent() override;

    void ToolBar();
    void ToolBar(const ImVec2& screen_position);

    void Resize(uint32_t width, uint32_t height);

    SceneTree* current_scene_ = nullptr;
    std::uint32_t width_{0};
    std::uint32_t height_{0};

    RID view_port_{};

private:
    void ProcessViewportInput(Node* scene_root,
                              const ImVec2& viewport_position,
                              const ImVec2& viewport_size,
                              bool mouse_inside_rect);

    std::unique_ptr<EditorViewportRenderer> viewport_renderer_;
    Node* hovered_node_{nullptr};
    Joint3D* motion_target_joint_{nullptr};
    Joint3D* pressed_joint_{nullptr};
    Joint3D* dragged_joint_{nullptr};
    ImVec2 drag_last_mouse_{0.0f, 0.0f};
    ImVec2 drag_joint_screen_center_{0.0f, 0.0f};
    ImVec2 drag_joint_screen_axis_{1.0f, 0.0f};
    float drag_last_angle_{0.0f};
    float drag_joint_rotation_sign_{1.0f};
    bool drag_joint_screen_center_valid_{false};
    bool drag_joint_screen_axis_valid_{false};
    bool drag_last_angle_valid_{false};
};


}
