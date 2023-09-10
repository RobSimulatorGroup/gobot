/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-3-20
*/

#include "gobot/editor/imgui/inspector_panel.hpp"
#include "gobot/editor/property_inspector/editor_inspector.hpp"
#include "gobot/editor/imgui/type_icons.hpp"
#include "imgui_extension/icon_fonts/icons_material_design_icons.h"
#include "gobot/scene/test_property_node.hpp"

#include "imgui.h"
#include "imgui_stdlib.h"

namespace gobot {

InspectorPanel::InspectorPanel() {
    SetName(ICON_MDI_INFORMATION " Inspector###inspector");
    test_node_ = Object::New<TestPropertyNode>();
    test_node_->SetName("test");
    Variant variant(test_node_);
    editor_inspectors_.emplace_back(EditorInspector::New<EditorInspector>(variant));
    current_inspector_index_ = 0;
    AddChild(editor_inspectors_.at(current_inspector_index_));

    filter_ = new ImGuiTextFilter();
}

InspectorPanel::~InspectorPanel() {
    delete filter_;
}

void InspectorPanel::OnImGuiContent() {
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

    auto& cache = editor_inspectors_.at(current_inspector_index_)->GetVariantCache();
    auto* property_name = editor_inspectors_.at(current_inspector_index_)->GetNameProperty();

    ImGui::TextUnformatted(GetTypeIcon(cache.type));


    if (property_name) {
        ImGui::SameLine();
        auto str = property_name->GetValue().to_string();
        ImGui::SetNextItemWidth(ImGui::GetWindowWidth() - 60);
        if (ImGui::InputText(fmt::format("##{}", property_name->GetPropertyName()).c_str(), &str)) {
            property_name->SetValue(str);
        }

        ImGui::SameLine(ImGui::GetWindowWidth() - 30);
        if (ImGui::Button(ICON_MDI_COGS)) {
            ImGui::OpenPopup("Inspector setting");
        }
        if (ImGui::BeginPopup("Inspector setting"))
        {
            if (ImGui::Button("Expand All")) {
                // TODO(wqq)
            }
            if (ImGui::Button("Collapse All")) {
                // TODO(wqq)
            }
            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();
            if (ImGui::Button("Copy Properties")) {
                // TODO(wqq)
            }
            if (ImGui::Button("Paste Properties")) {
                // TODO(wqq)
            }

            ImGui::EndPopup();
        }
    }


    filter_->Draw("###PropertyFilter", ImGui::GetWindowWidth() - 10);

    if(!filter_->IsActive()) {
        ImGui::SameLine();
        ImGui::SetCursorPosX(ImGui::GetFontSize() * 0.5f);
        ImGui::TextUnformatted("Filter Properties");
        ImGui::SameLine();
        ImGui::SetCursorPosX(ImGui::GetWindowWidth() - 30);
        ImGui::TextUnformatted(ICON_MDI_MAGNIFY);
    }

}

}