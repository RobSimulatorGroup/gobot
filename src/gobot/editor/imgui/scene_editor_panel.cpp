/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-3-20
*/

#include "gobot/editor/imgui/scene_editor_panel.hpp"
#include "gobot/core/os/input.hpp"
#include "gobot/core/os/main_loop.hpp"
#include "gobot/editor/imgui/imgui_utilities.hpp"
#include "gobot/editor/editor.hpp"
#include "gobot/core/os/os.hpp"
#include "gobot/scene/window.hpp"
#include "imgui_extension/icon_fonts/icons_material_design_icons.h"
#include "imgui_stdlib.h"
#include "imgui.h"
#include "imgui_internal.h"

namespace gobot {

SceneEditorPanel::SceneEditorPanel()
{
    SetName(ICON_MDI_FILE_TREE " SceneTree###scene_editor");

    filter_ = new ImGuiTextFilter();
}


SceneEditorPanel::~SceneEditorPanel()
{
    delete filter_;
}

void SceneEditorPanel::DrawNode(Node* node)
{
    bool show = true;

    auto* editor = Editor::GetInstance();

    if(!node)
        return;

    std::string name = node->GetName();

    if(filter_->IsActive()) {
        if(!filter_->PassFilter(name.c_str())) {
            show = false;
        }
    }

    if (show) {
        ImGui::PushID(node);

        ImGuiTreeNodeFlags node_flags = ((editor->GetSelected() == node) ? ImGuiTreeNodeFlags_Selected : 0);
        node_flags |=  ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_OpenOnArrow |
                       ImGuiTreeNodeFlags_FramePadding | ImGuiTreeNodeFlags_AllowItemOverlap | ImGuiTreeNodeFlags_SpanAvailWidth;

        if(node->GetChildCount() == 0) {
            node_flags |= ImGuiTreeNodeFlags_Leaf;
        }

        bool double_clicked = false;
        if(node == double_clicked_) {
            double_clicked = true;
        }

        if(double_clicked)
            ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, { 1.0f, 2.0f });


        ImGui::PushStyleColor(ImGuiCol_Text, ImGuiUtilities::GetIconColor());

        bool node_open = ImGui::TreeNodeEx(node, node_flags, "%s", ICON_MDI_CUBE_OUTLINE);
        {
            // Allow clicking of icon and text. Need twice as they are separated
            if(ImGui::IsItemClicked())
                editor->SetSelected(node);
            else if(double_clicked_ == node && ImGui::IsMouseClicked(ImGuiMouseButton_Left) &&
                         !ImGui::IsItemHovered(ImGuiHoveredFlags_None))
                double_clicked_ = nullptr;
        }
        ImGui::PopStyleColor();
        ImGui::SameLine();
        if(!double_clicked)
            ImGui::TextUnformatted(name.c_str());

        if(double_clicked) {
            auto value = name;
            ImGui::PushItemWidth(-1);
            if(ImGui::InputText("##Name", &value)) {
                node->SetName(value);
            }
            ImGui::PopStyleVar();
        }

        bool delete_node = false;
        if(ImGui::BeginPopupContextItem(name.c_str())) {

            ImGui::EndPopup();
        }

        if(ImGui::IsItemClicked() && !delete_node)
            editor->SetSelected(node);
        else if(double_clicked_ == node && ImGui::IsMouseClicked(ImGuiMouseButton_Left) && !ImGui::IsItemHovered(ImGuiHoveredFlags_None))
            double_clicked_ = nullptr;

        if(ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left) && ImGui::IsItemHovered(ImGuiHoveredFlags_None)) {
            double_clicked_ = node;
        }

        if(delete_node) {
            node->GetParent()->RemoveChild(node);
            if(node_open)
                ImGui::TreePop();

            ImGui::PopID();
            return;
        }

        if(!node_open) {
            ImGui::PopID();
            return;
        }

        const ImColor TreeLineColor = ImColor(128, 128, 128, 128);
        const float SmallOffsetX    = 6.0f * ImGui::GetWindowDpiScale();
        ImDrawList* drawList        = ImGui::GetWindowDrawList();

        ImVec2 verticalLineStart = ImGui::GetCursorScreenPos();
        verticalLineStart.x += SmallOffsetX; // to nicely line up with the arrow symbol
        ImVec2 verticalLineEnd = verticalLineStart;

        auto child_count = node->GetChildCount();
        for (int i = 0; i < child_count; i++) {
            auto* child = node->GetChild(i);
            float HorizontalTreeLineSize = 16.0f * ImGui::GetWindowDpiScale(); // chosen arbitrarily
            auto currentPos              = ImGui::GetCursorScreenPos();
            ImGui::Indent(10.0f);

            DrawNode(child);
            ImGui::Unindent(10.0f);

            const ImRect childRect = ImRect(currentPos, currentPos + ImVec2(0.0f, ImGui::GetFontSize() + ImGui::GetStyle().FramePadding.y));

            const float midpoint = (childRect.Min.y + childRect.Max.y) * 0.5f;
            drawList->AddLine(ImVec2(verticalLineStart.x, midpoint), ImVec2(verticalLineStart.x + HorizontalTreeLineSize, midpoint), TreeLineColor);
            verticalLineEnd.y = midpoint;
        }

        drawList->AddLine(verticalLineStart, verticalLineEnd, TreeLineColor);

        ImGui::TreePop();
        ImGui::PopID();
    }
}

void SceneEditorPanel::OnImGuiContent()
{
    auto flags        = ImGuiWindowFlags_NoCollapse;
    current_ = nullptr;

    select_up_ = Input::GetInstance()->GetKeyPressed(KeyCode::Up);
    select_down_ = Input::GetInstance()->GetKeyPressed(KeyCode::Down);

    auto* scene_tree = Object::PointerCastTo<SceneTree>(OS::GetInstance()->GetMainLoop());

    if(!scene_tree) {
        return;
    }

    ImGui::PushStyleColor(ImGuiCol_MenuBarBg, ImGui::GetStyleColorVec4(ImGuiCol_TabActive));

    if(ImGui::Button(ICON_MDI_PLUS))
    {
        // Add Entity Menu
        ImGui::OpenPopup("Add New Node");
    }

    bool open = true;
    if(ImGui::BeginPopupModal("Add New Node", &open))
    {
        ImGui::Text("Hello dsjfhds fhjs hfj dshfj hds");
        if (ImGui::Button("Close"))
            ImGui::CloseCurrentPopup();
        ImGui::EndPopup();
    }

    ImGui::SameLine();
    ImGui::TextUnformatted(ICON_MDI_MAGNIFY);
    ImGui::SameLine();

    {
        ImGuiUtilities::ScopedFont boldFont(ImGui::GetIO().Fonts->Fonts[1]);
        ImGuiUtilities::ScopedStyle frameBorder(ImGuiStyleVar_FrameBorderSize, 0.0f);
        ImGuiUtilities::ScopedColor frameColor(ImGuiCol_FrameBg, IM_COL32(0, 0, 0, 0));
        filter_->Draw("##HierarchyFilter", ImGui::GetContentRegionAvail().x - ImGui::GetStyle().IndentSpacing);
        ImGuiUtilities::DrawItemActivityOutline(2.0f, false);
    }

    if(!filter_->IsActive()) {
        ImGui::SameLine();
        ImGuiUtilities::ScopedFont boldFont(ImGui::GetIO().Fonts->Fonts[1]);
        ImGui::SetCursorPosX(ImGui::GetFontSize() * 4.0f);
        ImGuiUtilities::ScopedStyle padding(ImGuiStyleVar_FramePadding, ImVec2(0.0f, ImGui::GetStyle().FramePadding.y));
        ImGui::TextUnformatted("Search...");
    }

    ImGui::PopStyleColor();
    ImGui::Unindent();

    ImGui::Separator();

    ImGui::BeginChild("Nodes");

    {
        ImGui::Indent();
        DrawNode(static_cast<Node*>(scene_tree->GetRoot()));
    }

    ImGui::EndChild();
}


}
