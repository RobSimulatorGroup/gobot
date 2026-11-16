/* SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include "imgui_internal.h"

namespace gobot {

// Editor-only docking policy, shared by first startup and Reset Layout.
inline void BuildDefaultEditorDockLayout(ImGuiID dockspace_id, ImVec2 size) {
    ImGui::DockBuilderRemoveNode(dockspace_id);
    ImGui::DockBuilderAddNode(dockspace_id);
    ImGui::DockBuilderSetNodeSize(dockspace_id, size);

    ImGuiID main = dockspace_id;
    ImGuiID left = ImGui::DockBuilderSplitNode(main, ImGuiDir_Left, 0.14f, nullptr, &main);
    ImGuiID right = ImGui::DockBuilderSplitNode(main, ImGuiDir_Right, 0.35f, nullptr, &main);
    ImGuiID resources = ImGui::DockBuilderSplitNode(left, ImGuiDir_Down, 0.40f, nullptr, &left);
    ImGuiID simulation = ImGui::DockBuilderSplitNode(right, ImGuiDir_Down, 0.56f, nullptr, &right);
    ImGuiID python = ImGui::DockBuilderSplitNode(main, ImGuiDir_Down, 0.35f, nullptr, &main);
    ImGuiID console = ImGui::DockBuilderSplitNode(python, ImGuiDir_Down, 0.43f, nullptr, &python);

    ImGui::DockBuilderDockWindow("###scene_view3d", main);
    ImGui::DockBuilderDockWindow("###python", python);
    ImGui::DockBuilderDockWindow("###inspector", right);
    ImGui::DockBuilderDockWindow("###physics", simulation);
    ImGui::DockBuilderDockWindow("###tactile", simulation);
    ImGui::DockBuilderDockWindow("###console", console);
    ImGui::DockBuilderDockWindow("###resources", resources);
    ImGui::DockBuilderDockWindow("###scene_editor", left);

    const auto set_tab_order = [](const char* name, short order) {
        if (ImGuiWindow* window = ImGui::FindWindowByName(name)) window->DockOrder = order;
        if (ImGuiWindowSettings* settings = ImGui::FindWindowSettingsByID(ImHashStr(name))) {
            settings->DockOrder = order;
        }
    };
    set_tab_order("###physics", 0);
    set_tab_order("###tactile", 1);
    ImGui::DockBuilderGetNode(simulation)->SelectedTabId = ImHashStr("###physics");
    ImGui::DockBuilderFinish(dockspace_id);
}

} // namespace gobot
