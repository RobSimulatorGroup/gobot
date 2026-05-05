/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-3-20
*/

#include "gobot/editor/imgui/inspector_panel.hpp"
#include "gobot/editor/editor.hpp"
#include "gobot/editor/property_inspector/editor_inspector.hpp"
#include "gobot/editor/imgui/type_icons.hpp"
#include "imgui_extension/icon_fonts/icons_material_design_icons.h"

#include "imgui.h"
#include "imgui_stdlib.h"

#include <algorithm>

namespace gobot {

InspectorPanel::InspectorPanel() {
    SetName("InspectorPanel");
    SetImGuiWindow(ICON_MDI_INFORMATION " Inspector", "inspector");

    filter_ = new ImGuiTextFilter();
}

InspectorPanel::~InspectorPanel() {
    delete filter_;
}

void InspectorPanel::RebuildInspector(Node* selected) {
    const std::uint64_t scene_change_version = Editor::GetInstance()->GetSceneChangeVersion();
    if (selected == inspected_node_ &&
        scene_change_version == inspected_scene_change_version_) {
        return;
    }

    if (editor_inspector_) {
        Object::Delete(editor_inspector_);
        editor_inspector_ = nullptr;
    }

    inspected_node_ = selected;
    inspected_scene_change_version_ = scene_change_version;
    if (!inspected_node_) {
        return;
    }

    Variant variant(inspected_node_);
    editor_inspector_ = Object::New<EditorInspector>(variant);
    AddChild(editor_inspector_);
}

void InspectorPanel::OnImGuiContent() {
    RebuildInspector(Editor::GetInstance()->GetSelected());

    const float frame_height = ImGui::GetFrameHeight();
    const float item_spacing = ImGui::GetStyle().ItemSpacing.x;
    const float content_right = ImGui::GetCursorScreenPos().x + ImGui::GetContentRegionAvail().x;
    const float icon_button_width = frame_height;

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

    ImGui::SameLine(std::max(ImGui::GetCursorPosX() + item_spacing,
                             ImGui::GetContentRegionMax().x - 3.0f * button_size.x - item_spacing * 2.0f));
    if (ImGui::ArrowButton("##left", ImGuiDir_Left)) {

    }
    ImGui::SameLine(0.0f, ImGui::GetStyle().ItemInnerSpacing.x);

    if (ImGui::ArrowButton("##right", ImGuiDir_Right)) {

    }

    ImGui::SameLine(std::max(ImGui::GetCursorPosX() + item_spacing,
                             ImGui::GetContentRegionMax().x - icon_button_width));
    if (ImGui::Button(ICON_MDI_HISTORY)) {
        // TODO(wqq): select current_inspector_index_
    }

    if (!editor_inspector_) {
        ImGui::TextUnformatted("No node selected");
        return;
    }

    auto& cache = editor_inspector_->GetVariantCache();
    auto* property_name = editor_inspector_->GetNameProperty();

    ImGui::TextUnformatted(GetTypeIcon(cache.type));


    if (property_name) {
        ImGui::SameLine();
        auto str = property_name->GetValue().to_string();
        const float settings_button_width = frame_height;
        const float name_input_width = std::max(48.0f,
                                                content_right - ImGui::GetCursorScreenPos().x -
                                                settings_button_width - item_spacing);
        ImGui::SetNextItemWidth(name_input_width);
        if (ImGui::InputText(fmt::format("##{}", property_name->GetPropertyName()).c_str(), &str)) {
            property_name->SetValue(str);
        }

        ImGui::SameLine();
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


    filter_->Draw("###PropertyFilter", ImGui::GetContentRegionAvail().x);

    if(!filter_->IsActive()) {
        ImGui::SameLine();
        ImGui::SetCursorPosX(ImGui::GetFontSize() * 0.5f);
        ImGui::TextUnformatted("Filter Properties");
        ImGui::SameLine();
        ImGui::SetCursorPosX(ImGui::GetContentRegionMax().x - icon_button_width);
        ImGui::TextUnformatted(ICON_MDI_MAGNIFY);
    }

}

}
