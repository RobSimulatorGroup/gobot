/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * This file is created by Qiqi Wu, 23-3-20
 * SPDX-License-Identifier: Apache-2.0
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
namespace {

void DrawHoverTooltip(const char* text) {
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal | ImGuiHoveredFlags_NoSharedDelay)) {
        ImGui::SetTooltip("%s", text);
    }
}

} // namespace

InspectorPanel::InspectorPanel() {
    SetName("InspectorPanel");
    SetImGuiWindow(ICON_MDI_INFORMATION " Inspector", "inspector");

    filter_ = new ImGuiTextFilter();
}

InspectorPanel::~InspectorPanel() {
    delete filter_;
}

void InspectorPanel::RebuildInspector(Node* selected) {
    ObjectID selected_id{};
    if (selected != nullptr) {
        selected_id = selected->GetInstanceId();
    }

    if (selected == inspected_node_ &&
        selected_id == inspected_node_id_ &&
        (selected == nullptr || editor_inspector_->GetVariantCache().type == selected->GetType())) {
        return;
    }

    if (editor_inspector_) {
        Object::Delete(editor_inspector_);
        editor_inspector_ = nullptr;
    }

    inspected_node_ = selected;
    inspected_node_id_ = selected_id;
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
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("New resource");
    }

    ImGui::SameLine();
    if (ImGui::Button(ICON_MDI_ARROW_UP_BOX)) {
        // TODO(wqq): load
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Load resource");
    }

    ImGui::SameLine();
    if (ImGui::Button(ICON_MDI_CONTENT_SAVE)) {
        // TODO(wqq): save
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Save resource");
    }

    auto button_size = ImGui::GetItemRectSize();

    ImGui::SameLine(std::max(ImGui::GetCursorPosX() + item_spacing,
                             ImGui::GetContentRegionMax().x - 3.0f * button_size.x - item_spacing * 2.0f));
    if (ImGui::ArrowButton("##left", ImGuiDir_Left)) {

    }
    DrawHoverTooltip("Go to the previous inspected object");
    ImGui::SameLine(0.0f, ImGui::GetStyle().ItemInnerSpacing.x);

    if (ImGui::ArrowButton("##right", ImGuiDir_Right)) {

    }
    DrawHoverTooltip("Go to the next inspected object");

    ImGui::SameLine(std::max(ImGui::GetCursorPosX() + item_spacing,
                             ImGui::GetContentRegionMax().x - icon_button_width));
    if (ImGui::Button(ICON_MDI_HISTORY)) {
        // TODO(wqq): select current_inspector_index_
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Inspector history");
    }

    if (!editor_inspector_) {
        ImGui::TextUnformatted("No node selected");
        return;
    }

    auto& cache = editor_inspector_->GetVariantCache();
    auto* property_name = editor_inspector_->GetNameProperty();
    auto* inspected_node = Object::PointerCastTo<Node>(cache.object);
    const bool runtime_node = Editor::GetInstance()->IsRuntimeNode(inspected_node);

    const float icon_size = ImGui::GetTextLineHeight();
    DrawEditorIcon(GetTypeEditorIcon(cache.type), {icon_size, icon_size});
    if (ImGui::IsItemHovered()) {
        ImGui::BeginTooltip();
        ImGui::Text("Selected type: %s", cache.type.get_name().data());
        ImGui::TextUnformatted("Properties are grouped by inheritance below.");
        ImGui::EndTooltip();
    }


    if (property_name) {
        ImGui::SameLine();
        auto str = property_name->GetValue().to_string();
        const float settings_button_width = frame_height;
        const float name_input_width = std::max(48.0f,
                                                content_right - ImGui::GetCursorScreenPos().x -
                                                settings_button_width - item_spacing);
        ImGui::SetNextItemWidth(name_input_width);
        if (runtime_node) {
            ImGui::BeginDisabled();
        }
        const bool name_changed = ImGui::InputText(fmt::format("##{}", property_name->GetPropertyName()).c_str(), &str);
        if (runtime_node) {
            ImGui::EndDisabled();
        }
        if (!runtime_node && name_changed) {
            property_name->SetValue(str);
        }
        DrawHoverTooltip(runtime_node
                         ? "Runtime node name. Play Mode Inspector is read-only."
                         : "Selected node name. Editing this renames the node in SceneTree.");

        ImGui::SameLine();
        if (ImGui::Button(ICON_MDI_COGS)) {
            ImGui::OpenPopup("Inspector setting");
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Inspector options");
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

    if (runtime_node) {
        ImGui::TextDisabled("Runtime state");
        DrawHoverTooltip("Showing the active Play Mode clone. Properties are synchronized from the simulation world once per editor frame.");
    }

    filter_->Draw("###PropertyFilter", ImGui::GetContentRegionAvail().x);
    DrawHoverTooltip("Filter visible properties by name");

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
