/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-3-15
*/

#pragma once

#include "gobot/scene/resources/texture.hpp"
#include "gobot/rendering/scene_renderer.hpp"
#include "gobot/scene/imgui_window.hpp"

namespace gobot {

class SceneTree;
class SceneRenderer;

class GOBOT_EXPORT SceneViewPanel : public ImGuiWindow {
    GOBCLASS(SceneViewPanel, ImGuiWindow)
public:
    SceneViewPanel();

    ~SceneViewPanel() = default;

    void OnImGuiContent() override;

    void ToolBar();

    void Resize(uint32_t width, uint32_t height);

    SceneTree* current_scene_ = nullptr;
    std::uint32_t width_{0};
    std::uint32_t height_{0};

    RID view_rid_{};

};


}