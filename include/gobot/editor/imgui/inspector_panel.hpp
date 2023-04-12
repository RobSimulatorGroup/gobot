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

class TestPropertyNode;
class EditorInspector;

class GOBOT_EXPORT InspectorPanel : public ImGuiWindow {
    GOBCLASS(InspectorPanel, ImGuiWindow)
public:
    InspectorPanel();

    ~InspectorPanel() override;

    void OnImGuiContent() override;

private:
    TestPropertyNode* test_node_{nullptr};

    ImGuiTextFilter* filter_;

    // edit history of inspector
    std::vector<EditorInspector*> editor_inspectors_{};
    int current_inspector_index_{-1};
};


}