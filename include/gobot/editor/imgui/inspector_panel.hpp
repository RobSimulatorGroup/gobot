/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-3-20
*/

#pragma once

#include "gobot/scene/imgui_window.hpp"

class ImGuiTextFilter;

namespace gobot {

class EditorInspector;
class Node;

class GOBOT_EXPORT InspectorPanel : public ImGuiWindow {
    GOBCLASS(InspectorPanel, ImGuiWindow)
public:
    InspectorPanel();

    ~InspectorPanel() override;

    void OnImGuiContent() override;

private:
    void RebuildInspector(Node* selected);

    Node* inspected_node_{nullptr};
    std::uint64_t inspected_scene_change_version_{0};

    ImGuiTextFilter* filter_;

    EditorInspector* editor_inspector_{nullptr};
};


}
