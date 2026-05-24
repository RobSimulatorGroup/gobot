/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * This file is created by Qiqi Wu, 23-3-20
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "gobot/core/object_id.hpp"
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
    ObjectID inspected_node_id_{};

    ImGuiTextFilter* filter_;

    EditorInspector* editor_inspector_{nullptr};
};


}
