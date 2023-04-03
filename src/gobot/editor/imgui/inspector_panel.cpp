/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-3-20
*/

#include "gobot/editor/imgui/inspector_panel.hpp"
#include "gobot/editor/property_inspector/editor_inspector.hpp"
#include "imgui_extension/icon_fonts/icons_material_design_icons.h"
#include "gobot/scene/node_3d.hpp"

#include "imgui.h"

namespace gobot {

InspectorPanel::InspectorPanel() {
    SetName(ICON_MDI_INFORMATION " Inspector###inspector");
    node_3d_ = Node3D::New<Node3D>();
    node_3d_->SetName("node3d");
    Variant variant(node_3d_);
    editor_inspectors_.emplace_back(EditorInspector::New<EditorInspector>(variant));
    current_inspector_index_ = 0;
    AddChild(editor_inspectors_.at(current_inspector_index_));
}

void InspectorPanel::OnImGuiContent() {
    ImGui::Begin(GetName().toStdString().c_str());

    if (ImGui::Button(ICON_MDI_FILE_PLUS)) {
        // TODO(wqq): new
    }

    ImGui::SameLine();
    if (ImGui::Button(ICON_MDI_ARROW_UP_BOX)) {
        // TODO(wqq): load
    }

    ImGui::SameLine();
    if (ImGui::Button(ICON_MDI_CONTENT_SAVE)) {
        // TODO(wqq): save
    }

    auto button_size = ImGui::GetItemRectSize();

    ImGui::SameLine(ImGui::GetWindowWidth() - 3 * button_size.x - 20);
    if (ImGui::ArrowButton("##left", ImGuiDir_Left)) {

    }
    ImGui::SameLine(0.0f, ImGui::GetStyle().ItemInnerSpacing.x);

    if (ImGui::ArrowButton("##right", ImGuiDir_Right)) {

    }

    ImGui::SameLine(ImGui::GetWindowWidth() - 30);
    if (ImGui::Button(ICON_MDI_HISTORY)) {
        // TODO(wqq): select current_inspector_index_
    }



//    editor_inspectors_.at(current_inspector_index_)->OnImGui();

    ImGui::End();
}

}