/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-3-15
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

    void OnImGuiContent() override;

    void ToolBar();

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
    float drag_start_joint_position_{0.0f};
    ImVec2 drag_start_mouse_{0.0f, 0.0f};
};


}
