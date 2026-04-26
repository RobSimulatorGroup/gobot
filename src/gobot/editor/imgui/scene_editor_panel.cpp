/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-3-20
*/

#include "gobot/editor/imgui/scene_editor_panel.hpp"

#include <algorithm>
#include <cctype>
#include <unordered_set>

#include "gobot/core/os/input.hpp"
#include "gobot/core/os/main_loop.hpp"
#include "gobot/core/types.hpp"
#include "gobot/editor/imgui/imgui_utilities.hpp"
#include "gobot/editor/editor.hpp"
#include "gobot/scene/mesh_instance_3d.hpp"
#include "gobot/scene/node_3d.hpp"
#include "gobot/scene/resources/primitive_mesh.hpp"
#include "imgui_extension/icon_fonts/icons_material_design_icons.h"
#include "imgui_stdlib.h"
#include "imgui.h"
#include "imgui_internal.h"

namespace gobot {

namespace {

bool ContainsCaseInsensitive(const std::string& text, const std::string& query) {
    if (query.empty()) {
        return true;
    }

    auto to_lower = [](std::string value) {
        std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        return value;
    };

    return to_lower(text).find(to_lower(query)) != std::string::npos;
}

bool IsEditorOnlyNodeType(const std::string& type_name) {
    std::string name = type_name;
    const std::string namespace_prefix = "gobot::";
    if (name.starts_with(namespace_prefix)) {
        name = name.substr(namespace_prefix.size());
    }

    return name == "Editor" ||
           name == "EditedScene" ||
           name == "Node3DEditor" ||
           name == "ImGuiNode" ||
           name == "ImGuiWindow" ||
           name == "ImGuiCustomNode" ||
           name == "TestPropertyNode";
}

} // namespace

SceneEditorPanel::SceneEditorPanel()
{
    SetName("SceneTreePanel");
    SetImGuiWindow(ICON_MDI_FILE_TREE " SceneTree", "scene_editor");

    filter_ = new ImGuiTextFilter();
}


SceneEditorPanel::~SceneEditorPanel()
{
    delete filter_;
}

std::vector<SceneEditorPanel::AddNodeEntry> SceneEditorPanel::BuildAddNodeEntries() {
    std::vector<AddNodeEntry> entries;
    std::unordered_set<std::string> seen_labels;
    Type node_type = Type::get<Node>();

    for (const Type& type : Type::get_types()) {
        if (!type.is_valid()) {
            continue;
        }

        const std::string type_name = type.get_name().data();
        if (type_name.empty() ||
            type_name.find('*') != std::string::npos ||
            type.is_pointer() ||
            type.is_wrapper() ||
            IsEditorOnlyNodeType(type_name)) {
            continue;
        }

        if (type != node_type && !type.is_derived_from(node_type)) {
            continue;
        }

        if (!type.get_constructor().is_valid() || seen_labels.contains(type_name)) {
            continue;
        }

        seen_labels.insert(type_name);
        entries.push_back({
            type_name,
            [type_name]() -> Node* {
                Type current_type = Type::get_by_name(type_name);
                if (!current_type.is_valid()) {
                    return nullptr;
                }

                Variant new_obj = current_type.create();
                bool success = false;
                auto* node = new_obj.convert<Node*>(&success);
                return success ? node : nullptr;
            }
        });
    }

    entries.push_back({
        "Box Mesh",
        []() -> Node* {
            auto* node = Object::New<MeshInstance3D>();
            node->SetName("Box");
            node->SetMesh(MakeRef<BoxMesh>());
            return node;
        }
    });

    std::sort(entries.begin(), entries.end(), [](const AddNodeEntry& lhs, const AddNodeEntry& rhs) {
        return lhs.label < rhs.label;
    });

    return entries;
}

Node* SceneEditorPanel::GetAddChildTarget(Node* scene_root) const {
    auto* selected = Editor::GetInstance()->GetSelected();
    if (selected == nullptr || scene_root == nullptr) {
        return scene_root;
    }

    if (selected == scene_root || scene_root->IsAncestorOf(selected)) {
        return selected;
    }

    return scene_root;
}

void SceneEditorPanel::RequestOpenAddChildDialog(Node* parent) {
    add_child_parent_ = parent;
    open_add_child_dialog_ = true;
    add_node_search_.clear();
    selected_add_node_index_ = -1;
}

bool SceneEditorPanel::CreateSelectedAddNode() {
    static const std::vector<AddNodeEntry> entries = BuildAddNodeEntries();
    if (add_child_parent_ == nullptr ||
        selected_add_node_index_ < 0 ||
        static_cast<std::size_t>(selected_add_node_index_) >= entries.size()) {
        return false;
    }

    Node* node = entries[static_cast<std::size_t>(selected_add_node_index_)].create();
    if (node == nullptr) {
        return false;
    }

    add_child_parent_->AddChild(node, true);
    Editor::GetInstance()->SetSelected(node);
    return true;
}

void SceneEditorPanel::DrawAddChildDialog() {
    if (open_add_child_dialog_) {
        ImGui::OpenPopup("Create New Node");
        open_add_child_dialog_ = false;
    }

    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->GetCenter(), ImGuiCond_Appearing, {0.5f, 0.5f});
    ImGui::SetNextWindowSize({820.0f, 620.0f}, ImGuiCond_Appearing);

    if (!ImGui::BeginPopupModal("Create New Node", nullptr, ImGuiWindowFlags_NoCollapse)) {
        return;
    }

    if (add_child_parent_ == nullptr) {
        ImGui::TextUnformatted("No parent selected");
        if (ImGui::Button("Cancel", {120.0f, 0.0f})) {
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
        return;
    }

    const float footer_height = ImGui::GetFrameHeightWithSpacing() + ImGui::GetStyle().ItemSpacing.y;
    ImGui::BeginChild("CreateNodeBody", {0.0f, -footer_height}, false);

    ImGui::BeginChild("CreateNodeSideBar", {170.0f, 0.0f}, true);
    ImGui::TextUnformatted("Favorites");
    ImGui::Separator();
    ImGui::TextDisabled("Node3D");
    ImGui::TextDisabled("Box Mesh");
    ImGui::Spacing();
    ImGui::TextUnformatted("Recent");
    ImGui::Separator();
    ImGui::TextDisabled("Node3D");
    ImGui::EndChild();

    ImGui::SameLine();

    ImGui::BeginChild("CreateNodeMain", {0.0f, 0.0f}, false);
    ImGui::TextUnformatted("Search");
    ImGui::SetNextItemWidth(-1.0f);
    if (ImGui::IsWindowAppearing()) {
        ImGui::SetKeyboardFocusHere();
    }
    ImGui::InputTextWithHint("##AddNodeSearch", "Search node type...", &add_node_search_);
    ImGui::TextUnformatted("Matches");
    ImGui::Separator();

    static const std::vector<AddNodeEntry> entries = BuildAddNodeEntries();
    ImGui::BeginChild("CreateNodeMatches", {0.0f, -110.0f}, true);
    bool had_visible_entry = false;
    for (std::size_t i = 0; i < entries.size(); ++i) {
        const auto& entry = entries[i];
        if (!ContainsCaseInsensitive(entry.label, add_node_search_)) {
            continue;
        }

        had_visible_entry = true;
        const bool selected = selected_add_node_index_ == static_cast<int>(i);
        std::string label = entry.label;
        if (entry.label == "Box Mesh") {
            label = ICON_MDI_CUBE_OUTLINE "  " + label;
        } else if (entry.label == "Node3D") {
            label = ICON_MDI_VECTOR_POINT "  " + label;
        } else {
            label = ICON_MDI_CUBE_OUTLINE "  " + label;
        }

        if (ImGui::Selectable(label.c_str(), selected)) {
            selected_add_node_index_ = static_cast<int>(i);
            if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left) && CreateSelectedAddNode()) {
                ImGui::CloseCurrentPopup();
            }
        }
    }

    if (!had_visible_entry) {
        ImGui::TextDisabled("No node types found");
    }
    ImGui::EndChild();

    ImGui::TextUnformatted("Description");
    ImGui::Separator();
    if (selected_add_node_index_ >= 0 && static_cast<std::size_t>(selected_add_node_index_) < entries.size()) {
        const auto& selected_entry = entries[static_cast<std::size_t>(selected_add_node_index_)];
        if (selected_entry.label == "Box Mesh") {
            ImGui::TextWrapped("Creates a MeshInstance3D with a default BoxMesh resource.");
        } else {
            ImGui::TextWrapped("Creates a %s node as a child of %s.",
                               selected_entry.label.c_str(),
                               add_child_parent_->GetName().c_str());
        }
    } else {
        ImGui::TextDisabled("Select a node type to see details.");
    }

    ImGui::EndChild();
    ImGui::EndChild();

    ImGui::Separator();
    const float button_width = 120.0f;
    if (ImGui::Button("Cancel", {button_width, 0.0f})) {
        ImGui::CloseCurrentPopup();
    }
    ImGui::SameLine(ImGui::GetWindowWidth() - button_width - ImGui::GetStyle().WindowPadding.x);
    const bool can_create = selected_add_node_index_ >= 0;
    if (!can_create) {
        ImGui::BeginDisabled();
    }
    if (ImGui::Button("Create", {button_width, 0.0f})) {
        if (CreateSelectedAddNode()) {
            ImGui::CloseCurrentPopup();
        }
    }
    if (!can_create) {
        ImGui::EndDisabled();
    }

    ImGui::EndPopup();
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
        if(ImGui::BeginPopupContextItem("NodeContext")) {
            editor->SetSelected(node);
            if (ImGui::MenuItem("Add Child")) {
                RequestOpenAddChildDialog(node);
            }
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

    auto* scene_root = Editor::GetInstance()->GetEditedSceneRoot();
    if(!scene_root) {
        return;
    }

    ImGui::PushStyleColor(ImGuiCol_MenuBarBg, ImGui::GetStyleColorVec4(ImGuiCol_TabActive));

    if(ImGui::Button(ICON_MDI_PLUS)) {
        RequestOpenAddChildDialog(GetAddChildTarget(scene_root));
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
        DrawNode(scene_root);
    }

    ImGui::EndChild();

    DrawAddChildDialog();
}


}
