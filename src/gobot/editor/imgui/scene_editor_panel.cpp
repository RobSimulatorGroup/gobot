/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-3-20
*/

#include "gobot/editor/imgui/scene_editor_panel.hpp"

#include <algorithm>
#include <cctype>
#include <functional>
#include <unordered_map>
#include <vector>

#include "gobot/core/os/input.hpp"
#include "gobot/core/os/main_loop.hpp"
#include "gobot/editor/imgui/imgui_utilities.hpp"
#include "gobot/editor/editor.hpp"
#include "gobot/scene/node_3d.hpp"
#include "gobot/scene/node_creation_registry.hpp"
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

struct AddNodeTreeItem {
    const NodeCreationEntry* entry = nullptr;
    std::vector<std::size_t> children;
};

struct AddNodeTree {
    std::vector<AddNodeTreeItem> items;
    std::vector<std::size_t> roots;
};

AddNodeTree BuildAddNodeTree(const std::vector<NodeCreationEntry>& entries) {
    AddNodeTree tree;
    std::unordered_map<std::string, std::size_t> index_by_id;

    tree.items.reserve(entries.size());
    for (const auto& entry : entries) {
        index_by_id[entry.id] = tree.items.size();
        tree.items.push_back({&entry, {}});
    }

    for (std::size_t i = 0; i < tree.items.size(); ++i) {
        const auto* entry = tree.items[i].entry;
        const auto parent_iter = index_by_id.find(entry->parent_id);
        if (entry->parent_id.empty() || parent_iter == index_by_id.end()) {
            tree.roots.push_back(i);
        } else {
            tree.items[parent_iter->second].children.push_back(i);
        }
    }

    auto sort_indices = [&tree](std::vector<std::size_t>& indices) {
        std::sort(indices.begin(), indices.end(), [&tree](std::size_t lhs, std::size_t rhs) {
            return tree.items[lhs].entry->display_name < tree.items[rhs].entry->display_name;
        });
    };

    sort_indices(tree.roots);
    for (auto& item : tree.items) {
        sort_indices(item.children);
    }

    return tree;
}

bool AddNodeTreeItemMatches(const AddNodeTree& tree,
                            std::size_t item_index,
                            const std::string& search) {
    const auto* entry = tree.items[item_index].entry;
    if (ContainsCaseInsensitive(entry->display_name, search) ||
        ContainsCaseInsensitive(entry->id, search)) {
        return true;
    }

    for (const auto child_index : tree.items[item_index].children) {
        if (AddNodeTreeItemMatches(tree, child_index, search)) {
            return true;
        }
    }

    return false;
}

const char* GetAddNodeIcon(const NodeCreationEntry& entry) {
    if (entry.id == "Node") {
        return ICON_MDI_CIRCLE_OUTLINE;
    }
    if (entry.id == "Node3D") {
        return ICON_MDI_VECTOR_POINT;
    }
    return ICON_MDI_CUBE_OUTLINE;
}

bool AcceptSceneResourceDrop() {
    if (!ImGui::BeginDragDropTarget()) {
        return false;
    }

    bool accepted = false;
    const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("GobotSceneResource");
    if (payload != nullptr && payload->Data != nullptr && payload->DataSize > 0) {
        const std::string scene_path(static_cast<const char*>(payload->Data));
        accepted = Editor::GetInstance()->AddSceneToEditedScene(scene_path);
        if (accepted) {
            LOG_INFO("Added scene from Resources drop: {}", scene_path);
        } else {
            LOG_ERROR("Failed to add scene from Resources drop: {}", scene_path);
        }
    }

    ImGui::EndDragDropTarget();
    return accepted;
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
    selected_add_node_id_.clear();
}

bool SceneEditorPanel::CreateSelectedAddNode() {
    if (add_child_parent_ == nullptr ||
        selected_add_node_id_.empty()) {
        return false;
    }

    Node* node = NodeCreationRegistry::CreateNode(selected_add_node_id_);
    if (node == nullptr) {
        return false;
    }

    add_child_parent_->AddChild(node, true);
    Editor::GetInstance()->SetSelected(node);
    return true;
}

bool SceneEditorPanel::CanDeleteNode(Node* node) const {
    auto* editor = Editor::GetInstance();
    return node != nullptr &&
           node != editor->GetEditedSceneRoot() &&
           node->GetParent() != nullptr;
}

bool SceneEditorPanel::DeleteNode(Node* node) {
    if (!CanDeleteNode(node)) {
        return false;
    }

    auto* editor = Editor::GetInstance();
    Node* parent = node->GetParent();
    editor->SetSelected(parent);
    if (double_clicked_ == node) {
        double_clicked_ = nullptr;
    }

    Node::Delete(node);
    editor->MarkSceneDirty();
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
    ImGui::TextDisabled("MeshInstance3D");
    ImGui::TextDisabled("Box Mesh");
    ImGui::TextDisabled("Cylinder Mesh");
    ImGui::TextDisabled("Sphere Mesh");
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

    ImGui::BeginChild("CreateNodeMatches", {0.0f, -110.0f}, true);
    const AddNodeTree node_tree = BuildAddNodeTree(NodeCreationRegistry::GetNodeTypes());
    bool had_visible_entry = false;
    bool create_requested = false;

    std::function<void(std::size_t)> draw_tree_item = [&](std::size_t item_index) {
        if (!AddNodeTreeItemMatches(node_tree, item_index, add_node_search_)) {
            return;
        }

        had_visible_entry = true;

        const auto& item = node_tree.items[item_index];
        const auto* entry = item.entry;
        std::vector<std::size_t> visible_children;
        visible_children.reserve(item.children.size());
        for (const auto child_index : item.children) {
            if (AddNodeTreeItemMatches(node_tree, child_index, add_node_search_)) {
                visible_children.push_back(child_index);
            }
        }

        ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_OpenOnArrow |
                                   ImGuiTreeNodeFlags_SpanAvailWidth |
                                   ImGuiTreeNodeFlags_FramePadding;
        if (selected_add_node_id_ == entry->id) {
            flags |= ImGuiTreeNodeFlags_Selected;
        }
        if (visible_children.empty()) {
            flags |= ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
        }
        if (!add_node_search_.empty()) {
            ImGui::SetNextItemOpen(true, ImGuiCond_Always);
        }

        const bool open = ImGui::TreeNodeEx(entry->id.c_str(),
                                           flags,
                                           "%s  %s",
                                           GetAddNodeIcon(*entry),
                                           entry->display_name.c_str());
        if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen()) {
            selected_add_node_id_ = entry->id;
        }
        if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
            selected_add_node_id_ = entry->id;
            create_requested = CreateSelectedAddNode();
        }

        if (open && !visible_children.empty()) {
            for (const auto child_index : visible_children) {
                draw_tree_item(child_index);
                if (create_requested) {
                    break;
                }
            }
            ImGui::TreePop();
        }
    };

    for (const auto root_index : node_tree.roots) {
        draw_tree_item(root_index);
        if (create_requested) {
            break;
        }
    }

    if (create_requested) {
        ImGui::CloseCurrentPopup();
    }

    if (!had_visible_entry) {
        ImGui::TextDisabled("No node types found");
    }
    ImGui::EndChild();

    ImGui::TextUnformatted("Description");
    ImGui::Separator();
    const NodeCreationEntry* selected_entry = nullptr;
    if (!selected_add_node_id_.empty()) {
        selected_entry = NodeCreationRegistry::FindNodeType(selected_add_node_id_);
    }
    if (selected_entry != nullptr) {
        ImGui::TextWrapped("%s", selected_entry->description.c_str());
    } else {
        ImGui::TextDisabled("Select a node type to see details.");
    }

    if (selected_entry != nullptr && ImGui::IsKeyPressed(ImGuiKey_Enter)) {
        if (CreateSelectedAddNode()) {
            ImGui::CloseCurrentPopup();
        }
    }

    ImGui::EndChild();
    ImGui::EndChild();

    ImGui::Separator();
    const float button_width = 120.0f;
    if (ImGui::Button("Cancel", {button_width, 0.0f})) {
        ImGui::CloseCurrentPopup();
    }
    ImGui::SameLine(ImGui::GetWindowWidth() - button_width - ImGui::GetStyle().WindowPadding.x);
    const bool can_create = selected_entry != nullptr;
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

bool SceneEditorPanel::DrawNode(Node* node)
{
    bool show = true;

    auto* editor = Editor::GetInstance();

    if(!node)
        return false;

    std::string name = node->GetName();

    if(filter_->IsActive()) {
        if(!filter_->PassFilter(name.c_str())) {
            show = false;
        }
    }

    if (show) {
        const bool can_delete = CanDeleteNode(node);

        ImGui::PushID(node);

        ImGuiTreeNodeFlags node_flags = ((editor->GetSelected() == node) ? ImGuiTreeNodeFlags_Selected : 0);
        node_flags |=  ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_OpenOnArrow |
                       ImGuiTreeNodeFlags_FramePadding | ImGuiTreeNodeFlags_AllowOverlap | ImGuiTreeNodeFlags_SpanAvailWidth;

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
        const bool dropped_scene_on_row = AcceptSceneResourceDrop();
        ImGui::SameLine();
        if(!double_clicked)
            ImGui::TextUnformatted(name.c_str());
        if (!dropped_scene_on_row) {
            AcceptSceneResourceDrop();
        }

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
            if (!can_delete) {
                ImGui::BeginDisabled();
            }
            if (ImGui::MenuItem(ICON_MDI_DELETE " Delete Node")) {
                delete_node = can_delete;
            }
            if (!can_delete) {
                ImGui::EndDisabled();
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
            DeleteNode(node);
            if(node_open)
                ImGui::TreePop();

            ImGui::PopID();
            return true;
        }

        if(!node_open) {
            ImGui::PopID();
            return false;
        }

        const ImColor TreeLineColor = ImColor(128, 128, 128, 128);
        const float SmallOffsetX    = 6.0f * ImGui::GetWindowDpiScale();
        ImDrawList* drawList        = ImGui::GetWindowDrawList();

        ImVec2 verticalLineStart = ImGui::GetCursorScreenPos();
        verticalLineStart.x += SmallOffsetX; // to nicely line up with the arrow symbol
        ImVec2 verticalLineEnd = verticalLineStart;

        for (int i = 0; i < static_cast<int>(node->GetChildCount()); i++) {
            auto* child = node->GetChild(i);
            float HorizontalTreeLineSize = 16.0f * ImGui::GetWindowDpiScale(); // chosen arbitrarily
            auto currentPos              = ImGui::GetCursorScreenPos();
            ImGui::Indent(10.0f);

            const bool child_deleted = DrawNode(child);
            ImGui::Unindent(10.0f);
            if (child_deleted) {
                i--;
            }

            const ImRect childRect = ImRect(currentPos, currentPos + ImVec2(0.0f, ImGui::GetFontSize() + ImGui::GetStyle().FramePadding.y));

            const float midpoint = (childRect.Min.y + childRect.Max.y) * 0.5f;
            drawList->AddLine(ImVec2(verticalLineStart.x, midpoint), ImVec2(verticalLineStart.x + HorizontalTreeLineSize, midpoint), TreeLineColor);
            verticalLineEnd.y = midpoint;
        }

        drawList->AddLine(verticalLineStart, verticalLineEnd, TreeLineColor);

        ImGui::TreePop();
        ImGui::PopID();
    }

    return false;
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
    Node* selected = Editor::GetInstance()->GetSelected();
    const bool can_delete_selected = CanDeleteNode(selected);
    if (!can_delete_selected) {
        ImGui::BeginDisabled();
    }
    if (ImGui::Button(ICON_MDI_DELETE)) {
        DeleteNode(selected);
    }
    if (!can_delete_selected) {
        ImGui::EndDisabled();
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

    AcceptSceneResourceDrop();

    ImGui::EndChild();

    DrawAddChildDialog();
}


}
