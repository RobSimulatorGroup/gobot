/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * This file is created by Qiqi Wu, 23-3-20
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/editor/imgui/scene_editor_panel.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <functional>
#include <fstream>
#include <unordered_map>
#include <vector>

#include "gobot/core/io/python_script.hpp"
#include "gobot/core/io/resource_loader.hpp"
#include "gobot/core/os/input.hpp"
#include "gobot/core/os/main_loop.hpp"
#include "gobot/core/config/project_setting.hpp"
#include "gobot/core/string_utils.hpp"
#include "gobot/editor/imgui/imgui_utilities.hpp"
#include "gobot/editor/editor.hpp"
#include "gobot/main/engine_context.hpp"
#include "gobot/scene/node_3d.hpp"
#include "gobot/scene/node_creation_registry.hpp"
#include "gobot/scene/scene_command.hpp"
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

bool IsPythonScriptPath(const std::filesystem::path& path) {
    std::string extension = path.extension().string();
    std::ranges::transform(extension, extension.begin(), [](unsigned char character) {
        return static_cast<char>(std::tolower(character));
    });
    return extension == ".py";
}

std::string EnsurePythonScriptExtension(std::string path) {
    if (path.empty() || IsPythonScriptPath(path)) {
        return path;
    }
    return path + ".py";
}

bool IsProjectLocalScriptPath(const std::string& path) {
    return path.starts_with("res://") && IsPythonScriptPath(path);
}

std::vector<std::string> FindProjectPythonScripts() {
    std::vector<std::string> scripts;
    const auto* settings = ProjectSettings::GetInstance();
    if (settings->GetProjectPath().empty()) {
        return scripts;
    }

    const std::filesystem::path project_path = settings->GetProjectPath();
    std::error_code error;
    if (!std::filesystem::exists(project_path, error)) {
        return scripts;
    }

    for (std::filesystem::recursive_directory_iterator it(project_path,
                                                         std::filesystem::directory_options::skip_permission_denied,
                                                         error);
         it != std::filesystem::recursive_directory_iterator();
         it.increment(error)) {
        if (error) {
            error.clear();
            continue;
        }
        if (!it->is_regular_file(error) || !IsPythonScriptPath(it->path())) {
            continue;
        }
        scripts.push_back(settings->LocalizePath(it->path().string()));
    }

    std::ranges::sort(scripts);
    return scripts;
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

std::string PythonScriptTemplate() {
    return "import gobot\n\n"
           "\n"
           "class Script(gobot.NodeScript):\n"
           "    def _ready(self):\n"
           "        pass\n"
           "\n"
           "    def _process(self, delta: float):\n"
           "        pass\n"
           "\n"
           "    def _physics_process(self, delta: float):\n"
           "        pass\n";
}

std::string SanitizeScriptBaseName(std::string value) {
    if (value.empty()) {
        value = "node";
    }
    std::ranges::transform(value, value.begin(), [](unsigned char character) {
        if (std::isalnum(character)) {
            return static_cast<char>(std::tolower(character));
        }
        return '_';
    });
    while (value.find("__") != std::string::npos) {
        value = ReplaceAll(value, "__", "_");
    }
    while (!value.empty() && value.front() == '_') {
        value.erase(value.begin());
    }
    while (!value.empty() && value.back() == '_') {
        value.pop_back();
    }
    return value.empty() ? "node" : value;
}

std::string UniqueScriptPathForNode(const Node& node) {
    const std::string base_name = SanitizeScriptBaseName(node.GetName()) + "_script";
    std::string candidate = "res://scripts/" + base_name + ".py";
    for (int index = 1; ResourceLoader::Exists(candidate, "PythonScript"); ++index) {
        candidate = "res://scripts/" + base_name + "_" + std::to_string(index) + ".py";
    }
    return candidate;
}

bool CreateScriptFileIfNeeded(const std::string& local_path, bool write_template) {
    const std::string global_path = ProjectSettings::GetInstance()->GlobalizePath(local_path);
    if (std::filesystem::exists(global_path)) {
        return true;
    }

    std::error_code error;
    std::filesystem::create_directories(std::filesystem::path(global_path).parent_path(), error);
    if (error) {
        LOG_ERROR("Failed to create Python script directory '{}': {}",
                  std::filesystem::path(global_path).parent_path().string(),
                  error.message());
        return false;
    }

    std::ofstream output(global_path, std::ios::out | std::ios::trunc);
    if (!output.is_open()) {
        LOG_ERROR("Failed to create Python script '{}'.", local_path);
        return false;
    }
    if (write_template) {
        output << PythonScriptTemplate();
    }
    return true;
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
        if (IsSceneInstanceNode(selected)) {
            return scene_root;
        }
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

    auto* editor = Editor::GetInstance();
    auto* context = editor->GetEngineContext();
    if (context == nullptr ||
        !context->ExecuteSceneCommand(std::make_unique<AddChildNodeCommand>(
                add_child_parent_->GetInstanceId(),
                node->GetInstanceId(),
                true))) {
        Object::Delete(node);
        return false;
    }
    Editor::GetInstance()->SetSelected(node);
    return true;
}

void SceneEditorPanel::RequestOpenAttachScriptDialog(Node* node) {
    if (node == nullptr) {
        return;
    }

    attach_script_node_ = node;
    open_attach_script_dialog_ = true;
    attach_script_create_new_ = true;
    attach_script_template_enabled_ = true;
    attach_script_path_ = UniqueScriptPathForNode(*node);
    attach_script_search_.clear();
    attach_selected_script_path_.clear();
    attach_script_candidates_ = FindProjectPythonScripts();
}

bool SceneEditorPanel::AttachSelectedScript() {
    if (attach_script_node_ == nullptr) {
        return false;
    }

    std::string local_path;
    bool created = false;
    if (attach_script_create_new_) {
        local_path = EnsurePythonScriptExtension(attach_script_path_);
        if (!IsProjectLocalScriptPath(local_path)) {
            LOG_ERROR("Python node scripts must use a res:// path ending in .py: {}", local_path);
            return false;
        }
        const std::string global_path = ProjectSettings::GetInstance()->GlobalizePath(local_path);
        created = !std::filesystem::exists(global_path);
        if (!CreateScriptFileIfNeeded(local_path, attach_script_template_enabled_)) {
            return false;
        }
    } else {
        local_path = attach_selected_script_path_;
        if (local_path.empty()) {
            LOG_ERROR("Select a Python script to attach.");
            return false;
        }
    }

    if (!AttachScript(attach_script_node_, local_path)) {
        return false;
    }

    if (created) {
        LOG_INFO("Created Python script: {}", local_path);
    }
    Editor::GetInstance()->RefreshResourcePanel();
    attach_script_node_ = nullptr;
    attach_script_candidates_.clear();
    return true;
}

bool SceneEditorPanel::IsSceneInstanceNode(Node* node) const {
    return node != nullptr && node->GetSceneInstance().IsValid();
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

    auto* context = editor->GetEngineContext();
    return context != nullptr &&
           context->ExecuteSceneCommand(std::make_unique<RemoveChildNodeCommand>(
                   parent->GetInstanceId(),
                   node->GetInstanceId(),
                   true));
}

bool SceneEditorPanel::AttachScript(Node* node) {
    RequestOpenAttachScriptDialog(node);
    return node != nullptr;
}

bool SceneEditorPanel::AttachScript(Node* node, const std::string& script_path) {
    if (node == nullptr || script_path.empty()) {
        return false;
    }

    auto* editor = Editor::GetInstance();
    auto* context = editor->GetEngineContext();
    if (context == nullptr) {
        return false;
    }

    Ref<Resource> resource = ResourceLoader::Load(script_path,
                                                  "PythonScript",
                                                  ResourceFormatLoader::CacheMode::Replace);
    Ref<PythonScript> script = dynamic_pointer_cast<PythonScript>(resource);
    if (!script.IsValid()) {
        LOG_ERROR("Failed to load Python script '{}'.", script_path);
        return false;
    }

    if (!context->ExecuteSceneCommand(std::make_unique<SetNodePropertyCommand>(
                node->GetInstanceId(),
                "script",
                Variant(script)))) {
        return false;
    }

    editor->SetSelected(node);
    RequestOpenScript(script->GetPath().empty() ? script_path : script->GetPath());
    return true;
}

bool SceneEditorPanel::AcceptResourceDropOnNode(Node* node) {
    if (node == nullptr || !ImGui::BeginDragDropTarget()) {
        return false;
    }

    bool accepted = false;
    const ImGuiPayload* python_payload = ImGui::AcceptDragDropPayload("GobotPythonScriptResource");
    if (python_payload != nullptr && python_payload->Data != nullptr && python_payload->DataSize > 0) {
        const std::string script_path(static_cast<const char*>(python_payload->Data));
        accepted = AttachScript(node, script_path);
        if (accepted) {
            LOG_INFO("Attached Python script from Resources drop: {} -> {}", script_path, node->GetName());
        } else {
            LOG_ERROR("Failed to attach Python script from Resources drop: {} -> {}", script_path, node->GetName());
        }
    }

    if (!accepted) {
        const ImGuiPayload* scene_payload = ImGui::AcceptDragDropPayload("GobotSceneResource");
        if (scene_payload != nullptr && scene_payload->Data != nullptr && scene_payload->DataSize > 0) {
            const std::string scene_path(static_cast<const char*>(scene_payload->Data));
            accepted = Editor::GetInstance()->AddSceneToEditedScene(scene_path);
            if (accepted) {
                LOG_INFO("Added scene from Resources drop: {}", scene_path);
            } else {
                LOG_ERROR("Failed to add scene from Resources drop: {}", scene_path);
            }
        }
    }

    ImGui::EndDragDropTarget();
    return accepted;
}

bool SceneEditorPanel::DetachScript(Node* node) {
    if (node == nullptr || !node->GetScript().IsValid()) {
        return false;
    }

    auto* context = Editor::GetInstance()->GetEngineContext();
    return context != nullptr &&
           context->ExecuteSceneCommand(std::make_unique<SetNodePropertyCommand>(
                   node->GetInstanceId(),
                   "script",
                   Variant(Ref<PythonScript>())));
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

void SceneEditorPanel::DrawAttachScriptDialog() {
    if (open_attach_script_dialog_) {
        ImGui::OpenPopup("Attach Node Script");
        open_attach_script_dialog_ = false;
    }

    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->GetCenter(), ImGuiCond_Appearing, {0.5f, 0.5f});
    ImGui::SetNextWindowSize({720.0f, 520.0f}, ImGuiCond_Appearing);

    if (!ImGui::BeginPopupModal("Attach Node Script", nullptr, ImGuiWindowFlags_NoCollapse)) {
        return;
    }

    if (attach_script_node_ == nullptr) {
        ImGui::TextUnformatted("No node selected.");
        if (ImGui::Button("Close", {120.0f, 0.0f})) {
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
        return;
    }

    ImGui::Text("Node:");
    ImGui::SameLine(150.0f);
    ImGui::TextUnformatted(attach_script_node_->GetName().c_str());

    ImGui::Text("Script:");
    ImGui::SameLine(150.0f);
    const float mode_button_width = 180.0f;
    if (ImGui::Selectable("New Script",
                          attach_script_create_new_,
                          ImGuiSelectableFlags_DontClosePopups,
                          {mode_button_width, ImGui::GetFrameHeight()})) {
        attach_script_create_new_ = true;
    }
    ImGui::SameLine();
    if (ImGui::Selectable("Existing Script",
                          !attach_script_create_new_,
                          ImGuiSelectableFlags_DontClosePopups,
                          {mode_button_width, ImGui::GetFrameHeight()})) {
        attach_script_create_new_ = false;
    }

    ImGui::Separator();

    const float footer_height = ImGui::GetFrameHeightWithSpacing() + ImGui::GetStyle().ItemSpacing.y;
    ImGui::BeginChild("AttachScriptBody", {0.0f, -footer_height}, false);

    if (attach_script_create_new_) {
        ImGui::TextUnformatted("Path:");
        ImGui::SameLine(150.0f);
        ImGui::SetNextItemWidth(-1.0f);
        if (ImGui::IsWindowAppearing()) {
            ImGui::SetKeyboardFocusHere();
        }
        ImGui::InputText("##AttachScriptPath", &attach_script_path_);

        ImGui::TextUnformatted("Template:");
        ImGui::SameLine(150.0f);
        ImGui::Checkbox("Use default NodeScript template", &attach_script_template_enabled_);

        const std::string normalized_path = EnsurePythonScriptExtension(attach_script_path_);
        const bool valid_path = IsProjectLocalScriptPath(normalized_path);
        const std::string global_path = ProjectSettings::GetInstance()->GlobalizePath(normalized_path);
        const bool file_exists = valid_path && std::filesystem::exists(global_path);

        ImGui::Spacing();
        ImGui::BeginChild("AttachScriptStatus", {0.0f, 140.0f}, true);
        if (!valid_path) {
            ImGui::TextColored({1.0f, 0.35f, 0.35f, 1.0f},
                               "%s Path must be a res:// .py file.",
                               ICON_MDI_ALERT_CIRCLE);
        } else if (file_exists) {
            ImGui::TextColored({0.45f, 1.0f, 0.45f, 1.0f},
                               "%s File exists; it will be reused.",
                               ICON_MDI_CHECK_CIRCLE);
            ImGui::TextColored({0.45f, 1.0f, 0.45f, 1.0f},
                               "%s Existing script will be loaded and attached.",
                               ICON_MDI_CHECK_CIRCLE);
        } else {
            ImGui::TextColored({0.45f, 1.0f, 0.45f, 1.0f},
                               "%s New script file will be created.",
                               ICON_MDI_CHECK_CIRCLE);
            ImGui::TextColored({0.45f, 1.0f, 0.45f, 1.0f},
                               "%s It will be attached to the selected node.",
                               ICON_MDI_CHECK_CIRCLE);
        }
        ImGui::EndChild();
    } else {
        ImGui::TextUnformatted("Filter:");
        ImGui::SameLine(150.0f);
        ImGui::SetNextItemWidth(-1.0f);
        if (ImGui::IsWindowAppearing()) {
            ImGui::SetKeyboardFocusHere();
        }
        ImGui::InputTextWithHint("##AttachScriptSearch", "Search Python scripts...", &attach_script_search_);

        if (ImGui::Button(ICON_MDI_REFRESH " Refresh")) {
            attach_script_candidates_ = FindProjectPythonScripts();
        }

        ImGui::BeginChild("AttachScriptList", {0.0f, 0.0f}, true);
        bool showed_any = false;
        for (const std::string& script_path : attach_script_candidates_) {
            if (!ContainsCaseInsensitive(script_path, attach_script_search_)) {
                continue;
            }
            showed_any = true;
            const bool selected = attach_selected_script_path_ == script_path;
            const std::string item_label = std::string(ICON_MDI_LANGUAGE_PYTHON "  ") + script_path;
            if (ImGui::Selectable(item_label.c_str(), selected)) {
                attach_selected_script_path_ = script_path;
            }
            if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
                attach_selected_script_path_ = script_path;
                if (AttachSelectedScript()) {
                    ImGui::CloseCurrentPopup();
                }
            }
        }
        if (!showed_any) {
            ImGui::TextDisabled("No Python scripts found in the current project.");
        }
        ImGui::EndChild();
    }

    ImGui::EndChild();

    ImGui::Separator();
    const float button_width = 120.0f;
    const char* action_label = attach_script_create_new_
                               ? (std::filesystem::exists(ProjectSettings::GetInstance()->GlobalizePath(
                                          EnsurePythonScriptExtension(attach_script_path_)))
                                  ? "Load"
                                  : "Create")
                               : "Attach";
    const bool can_attach = attach_script_create_new_
                            ? IsProjectLocalScriptPath(EnsurePythonScriptExtension(attach_script_path_))
                            : !attach_selected_script_path_.empty();

    if (!can_attach) {
        ImGui::BeginDisabled();
    }
    if (ImGui::Button(action_label, {button_width, 0.0f}) ||
        (can_attach && ImGui::IsKeyPressed(ImGuiKey_Enter))) {
        if (AttachSelectedScript()) {
            ImGui::CloseCurrentPopup();
        }
    }
    if (!can_attach) {
        ImGui::EndDisabled();
    }
    ImGui::SameLine();
    if (ImGui::Button("Cancel", {button_width, 0.0f})) {
        attach_script_node_ = nullptr;
        attach_script_candidates_.clear();
        ImGui::CloseCurrentPopup();
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
        const Ref<PackedScene> scene_instance = node->GetSceneInstance();
        const Ref<PythonScript> python_script = node->GetScript();
        const bool has_script = python_script.IsValid();
        const bool is_scene_instance = IsSceneInstanceNode(node);
        const bool can_open_scene_instance = is_scene_instance && !scene_instance->GetPath().empty();
        const bool can_open_script = has_script && !python_script->GetPath().empty();
        bool delete_node = false;
        bool script_icon_clicked = false;

        ImGui::PushID(node);

        ImGuiTreeNodeFlags node_flags = ((editor->GetSelected() == node) ? ImGuiTreeNodeFlags_Selected : 0);
        node_flags |=  ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_OpenOnArrow |
                       ImGuiTreeNodeFlags_FramePadding | ImGuiTreeNodeFlags_AllowOverlap | ImGuiTreeNodeFlags_SpanAvailWidth;

        const bool suppress_tree_push = node->GetChildCount() == 0 || is_scene_instance;
        if(suppress_tree_push) {
            node_flags |= ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
        }

        bool double_clicked = false;
        if(node == double_clicked_) {
            double_clicked = true;
        }

        if(double_clicked)
            ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, { 1.0f, 2.0f });


        ImGui::PushStyleColor(ImGuiCol_Text, ImGuiUtilities::GetIconColor());

        bool node_open = ImGui::TreeNodeEx(node,
                                           node_flags,
                                           "%s",
                                           is_scene_instance ? ICON_MDI_LINK_BOX_OUTLINE : ICON_MDI_CUBE_OUTLINE);
        ImRect node_row_rect(ImGui::GetItemRectMin(), ImGui::GetItemRectMax());
        node_row_rect.Max.x = ImGui::GetWindowPos().x + ImGui::GetWindowContentRegionMax().x;
        {
            // Allow clicking of icon and text. Need twice as they are separated
            if(ImGui::IsItemClicked())
                editor->SetSelected(node);
            else if(double_clicked_ == node && ImGui::IsMouseClicked(ImGuiMouseButton_Left) &&
                         !ImGui::IsItemHovered(ImGuiHoveredFlags_None))
                double_clicked_ = nullptr;
        }
        ImGui::PopStyleColor();
        const bool dropped_resource_on_row = AcceptResourceDropOnNode(node);
        ImGui::SameLine();
        if(!double_clicked)
            ImGui::TextUnformatted(name.c_str());
        node_row_rect.Add(ImRect(ImGui::GetItemRectMin(), ImGui::GetItemRectMax()));
        if (!dropped_resource_on_row) {
            AcceptResourceDropOnNode(node);
        }

        const float icon_button_width = ImGui::GetFrameHeight();
        const float button_spacing = ImGui::GetStyle().ItemInnerSpacing.x;
        const int scene_instance_button_count = is_scene_instance ? (can_delete ? 2 : 1) : 0;
        const float scene_instance_buttons_width =
                icon_button_width * static_cast<float>(scene_instance_button_count) +
                button_spacing * static_cast<float>(std::max(0, scene_instance_button_count - 1));
        const float script_icon_right_padding = is_scene_instance && scene_instance_button_count > 0
                                                ? scene_instance_buttons_width + button_spacing
                                                : 0.0f;
        const float row_end_x = ImGui::GetWindowContentRegionMax().x;

        if (has_script) {
            const float script_icon_x = row_end_x - script_icon_right_padding - icon_button_width;
            if (ImGui::GetCursorPosX() + icon_button_width < script_icon_x) {
                ImGui::SameLine(script_icon_x);
            } else {
                ImGui::SameLine();
            }
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImGui::GetStyleColorVec4(ImGuiCol_HeaderHovered));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImGui::GetStyleColorVec4(ImGuiCol_HeaderActive));
            if (ImGui::SmallButton(ICON_MDI_LANGUAGE_PYTHON "##OpenNodeScript")) {
                if (can_open_script) {
                    script_icon_clicked = true;
                    double_clicked_ = nullptr;
                    Editor::GetInstance()->OpenPythonScriptFromPath(python_script->GetPath());
                }
            }
            ImGui::PopStyleColor(3);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("%s",
                                  can_open_script ? python_script->GetPath().c_str()
                                                  : "Python script has no resource path");
            }
        }

        if (is_scene_instance) {
            if (ImGui::GetCursorPosX() + scene_instance_buttons_width < row_end_x - scene_instance_buttons_width) {
                ImGui::SameLine(row_end_x - scene_instance_buttons_width);
            } else {
                ImGui::SameLine();
            }

            if (!can_open_scene_instance) {
                ImGui::BeginDisabled();
            }
            if (ImGui::SmallButton(ICON_MDI_OPEN_IN_NEW "##OpenSceneInstance")) {
                RequestOpenSceneInstance(scene_instance->GetPath());
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Open scene instance");
            }
            if (!can_open_scene_instance) {
                ImGui::EndDisabled();
            }

            if (can_delete) {
                ImGui::SameLine();
                if (ImGui::SmallButton(ICON_MDI_DELETE "##DeleteSceneInstance")) {
                    delete_node = true;
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Remove scene instance from the current scene");
                }
            }
        }

        const bool node_row_hovered =
                ImGui::IsMouseHoveringRect(node_row_rect.Min, node_row_rect.Max, false) &&
                ImGui::IsWindowHovered(ImGuiHoveredFlags_AllowWhenBlockedByPopup);
        if (node_row_hovered && ImGui::IsMouseReleased(ImGuiMouseButton_Right)) {
            editor->SetSelected(node);
            ImGui::OpenPopup("NodeContext");
        }

        if(double_clicked) {
            auto value = name;
            ImGui::PushItemWidth(-1);
            if(ImGui::InputText("##Name", &value)) {
                if (auto* context = editor->GetEngineContext()) {
                    context->ExecuteSceneCommand(std::make_unique<RenameNodeCommand>(node->GetInstanceId(), value));
                }
            }
            ImGui::PopStyleVar();
        }

        if(ImGui::BeginPopup("NodeContext")) {
            editor->SetSelected(node);
            if (is_scene_instance) {
                ImGui::BeginDisabled();
            }
            if (ImGui::MenuItem("Add Child")) {
                RequestOpenAddChildDialog(node);
            }
            if (is_scene_instance) {
                ImGui::EndDisabled();
            }
            if (!can_open_scene_instance) {
                ImGui::BeginDisabled();
            }
            if (ImGui::MenuItem(ICON_MDI_OPEN_IN_NEW " Open Scene Instance")) {
                RequestOpenSceneInstance(scene_instance->GetPath());
            }
            if (!can_open_scene_instance) {
                ImGui::EndDisabled();
            }
            if (!can_open_script) {
                ImGui::BeginDisabled();
            }
            if (ImGui::MenuItem(ICON_MDI_LANGUAGE_PYTHON " Open Script")) {
                RequestOpenScript(python_script->GetPath());
            }
            if (!can_open_script) {
                ImGui::EndDisabled();
            }
            if (has_script) {
                ImGui::BeginDisabled();
            }
            if (ImGui::MenuItem(ICON_MDI_LANGUAGE_PYTHON " Attach Script")) {
                AttachScript(node);
            }
            if (has_script) {
                ImGui::EndDisabled();
            }
            if (!has_script) {
                ImGui::BeginDisabled();
            }
            if (ImGui::MenuItem(ICON_MDI_CLOSE " Detach Script")) {
                DetachScript(node);
            }
            if (!has_script) {
                ImGui::EndDisabled();
            }
            if (!can_delete) {
                ImGui::BeginDisabled();
            }
            const char* delete_label = is_scene_instance
                                       ? ICON_MDI_DELETE " Remove Scene Instance"
                                       : ICON_MDI_DELETE " Delete Node";
            if (ImGui::MenuItem(delete_label)) {
                delete_node = can_delete;
            }
            if (!can_delete) {
                ImGui::EndDisabled();
            }
            ImGui::EndPopup();
        }

        if(!script_icon_clicked && node_row_hovered && ImGui::IsMouseReleased(ImGuiMouseButton_Left) && !delete_node)
            editor->SetSelected(node);
        else if(double_clicked_ == node && ImGui::IsMouseClicked(ImGuiMouseButton_Left) && !node_row_hovered)
            double_clicked_ = nullptr;

        if(!script_icon_clicked && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left) && node_row_hovered) {
            double_clicked_ = node;
            editor->FocusSceneViewerPanel();
        }

        if(delete_node) {
            DeleteNode(node);
            if(node_open && !suppress_tree_push)
                ImGui::TreePop();

            ImGui::PopID();
            return true;
        }

        if(!node_open || suppress_tree_push) {
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

void SceneEditorPanel::RequestOpenSceneInstance(const std::string& path) {
    pending_open_scene_instance_path_ = path;
}

void SceneEditorPanel::RequestOpenScript(const std::string& path) {
    pending_open_script_path_ = path;
}

void SceneEditorPanel::FlushPendingSceneInstanceOpen() {
    if (pending_open_scene_instance_path_.empty()) {
        return;
    }

    std::string path = std::move(pending_open_scene_instance_path_);
    pending_open_scene_instance_path_.clear();
    double_clicked_ = nullptr;
    add_child_parent_ = nullptr;
    Editor::GetInstance()->RequestOpenSceneFromPath(path);
}

void SceneEditorPanel::FlushPendingScriptOpen() {
    if (pending_open_script_path_.empty()) {
        return;
    }

    std::string path = std::move(pending_open_script_path_);
    pending_open_script_path_.clear();
    double_clicked_ = nullptr;
    add_child_parent_ = nullptr;
    Editor::GetInstance()->OpenPythonScriptFromPath(path);
}

void SceneEditorPanel::OnImGuiContent()
{
    FlushPendingScriptOpen();
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
    DrawAttachScriptDialog();

    FlushPendingSceneInstanceOpen();
}


}
