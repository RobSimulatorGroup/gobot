/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-28
*/

#include "gobot/editor/editor.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <string>

#include "gobot/core/config/project_setting.hpp"
#include "gobot/core/io/resource_loader.hpp"
#include "gobot/core/os/input.hpp"
#include "gobot/editor/edited_scene.hpp"
#include "gobot/editor/node3d_editor.hpp"
#include "gobot/core/registration.hpp"
#include "gobot/log.hpp"
#include "gobot/editor/imgui/console_sink.hpp"
#include "gobot/editor/imgui/imgui_manager.hpp"
#include "gobot/editor/imgui/scene_view_3d_panel.hpp"
#include "gobot/editor/imgui/scene_editor_panel.hpp"
#include "gobot/editor/imgui/inspector_panel.hpp"
#include "gobot/editor/imgui/physics_panel.hpp"
#include "gobot/editor/imgui/python_panel.hpp"
#include "gobot/editor/imgui/resource_panel.hpp"
#include "gobot/editor/property_inspector/editor_inspector.hpp"
#include "gobot/main/engine_context.hpp"
#include "gobot/main/main.hpp"
#include "gobot/physics/physics_server.hpp"
#include "gobot/python/python_app_context.hpp"
#include "gobot/scene/collision_shape_3d.hpp"
#include "gobot/scene/imgui_window.hpp"
#include "gobot/scene/mesh_instance_3d.hpp"
#include "gobot/scene/node_3d.hpp"
#include "gobot/scene/resources/box_shape_3d.hpp"
#include "gobot/scene/resources/packed_scene.hpp"
#include "gobot/scene/resources/primitive_mesh.hpp"
#include "gobot/simulation/simulation_server.hpp"
#include "gobot/core/config/engine.hpp"
#include "gobot/core/config/project_setting.hpp"
#include "gobot/editor/imgui/imgui_utilities.hpp"
#include "gobot/scene/scene_command.hpp"
#include "imgui.h"
#include "imgui_internal.h"
#include "imgui_extension/file_browser/ImFileBrowser.h"

namespace gobot {
namespace {

std::string ToLower(std::string value) {
    std::ranges::transform(value, value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

std::string NativeScenePathForImport(const std::filesystem::path& selected_path) {
    std::filesystem::path native_filename = selected_path.filename();
    native_filename.replace_extension(".jscn");

    ProjectSettings* settings = ProjectSettings::GetInstance();
    if (settings != nullptr && !settings->GetProjectPath().empty()) {
        return "res://" + native_filename.string();
    }

    std::filesystem::path native_path = selected_path;
    native_path.replace_extension(".jscn");
    return settings != nullptr ? settings->LocalizePath(native_path.string()) : native_path.string();
}

std::string FileStemOrFilename(const std::string& path) {
    const std::filesystem::path fs_path(path);
    std::string name = fs_path.stem().string();
    if (name.empty()) {
        name = fs_path.filename().string();
    }
    return name.empty() ? "[unsaved]" : name;
}

} // namespace

Editor* Editor::s_singleton = nullptr;

Editor::Editor() {
    s_singleton = this;
    SetName("Editor");

    imgui_manager_ = Object::New<ImGuiManager>();

    edited_scene_ = Object::New<EditedScene>();
    AddChild(edited_scene_, true);
    selected_ = edited_scene_->GetRoot();
    engine_context_ = new EngineContext(ProjectSettings::GetInstance(),
                                        PhysicsServer::GetInstance(),
                                        SimulationServer::GetInstance());
    engine_context_->SetSceneChangedCallback([this]() {
        ++scene_change_version_;
        scene_dirty_ = engine_context_ != nullptr && engine_context_->IsSceneDirty();
    });
    engine_context_->SetLoadSceneCallback([this](const std::string& path) {
        return OpenSceneFromPath(path);
    });
    BindEngineContextToEditedScene();
    python::SetActiveAppContext(engine_context_);

    node3d_editor_ = Object::New<Node3DEditor>();
    node3d_editor_->SetName("Node3DEditor");
    AddChild(node3d_editor_);

    spdlog::sink_ptr sink = std::make_shared<ImGuiConsoleSinkMultiThreaded>();
    Logger::GetInstance().AddSink(sink);

    EditorInspector::AddInspectorPlugin(MakeRef<EditorInspectorDefaultPlugin>());

    AddChild(Object::New<ConsolePanel>());
    python_panel_ = Object::New<PythonPanel>();
    AddChild(python_panel_);
    AddChild(Object::New<SceneEditorPanel>());
    AddChild(Object::New<InspectorPanel>());
    AddChild(Object::New<PhysicsPanel>());
    resource_panel_ = Object::New<ResourcePanel>();
    AddChild(resource_panel_);


    file_browser_ = new ImGui::FileBrowser();
    file_browser_->SetTitle("File Browser");
}

Editor::~Editor() {
    if (python::GetActiveAppContextOrNull() == engine_context_) {
        python::SetActiveAppContext(nullptr);
    }
    delete engine_context_;
    engine_context_ = nullptr;
    s_singleton = nullptr;

    delete file_browser_;
    delete imgui_manager_;
}

Editor* Editor::GetInstance() {
    ERR_FAIL_COND_V_MSG(s_singleton == nullptr, nullptr, "Must call this after initialize Editor");
    return s_singleton;
}

Node3D* Editor::GetEditedSceneRoot() const {
    return edited_scene_ ? edited_scene_->GetRoot() : nullptr;
}

void Editor::BindEngineContextToEditedScene() {
    if (engine_context_ == nullptr) {
        return;
    }

    engine_context_->SetSceneRoot(GetEditedSceneRoot(),
                                  false,
                                  current_scene_path_);
}

void Editor::SetSelected(Node* selected) {
    Node* scene_root = GetEditedSceneRoot();
    if (selected != nullptr && scene_root != nullptr) {
        Node* instance_root = nullptr;
        for (Node* node = selected; node != nullptr && node != scene_root; node = node->GetParent()) {
            if (node->GetSceneInstance().IsValid()) {
                instance_root = node;
            }
        }

        if (instance_root != nullptr) {
            selected = instance_root;
        }
    }

    selected_ = selected;
}

bool Editor::SaveEditedScene(const std::string& path) const {
    if (edited_scene_ == nullptr || !edited_scene_->SaveToPath(path)) {
        return false;
    }

    if (resource_panel_ != nullptr) {
        resource_panel_->Refresh();
    }
    return true;
}

bool Editor::SaveCurrentScene() {
    if (!HasCurrentScenePath()) {
        OpenSceneFileDialog(SceneFileDialogMode::SaveAs);
        return false;
    }

    if (SaveEditedScene(current_scene_path_)) {
        ClearSceneDirty();
        LOG_INFO("Saved scene: {}", current_scene_path_);
        return true;
    }

    LOG_ERROR("Failed to save scene: {}", current_scene_path_);
    return false;
}

bool Editor::LoadEditedScene(const std::string& path) {
    const std::string extension = ToLower(std::filesystem::path(path).extension().string());
    const bool default_robot_motion_mode = extension == ".urdf" || extension == ".xml";
    if (edited_scene_ == nullptr || !edited_scene_->LoadFromPath(path, default_robot_motion_mode)) {
        return false;
    }

    selected_ = edited_scene_->GetRoot();
    BindEngineContextToEditedScene();
    return true;
}

bool Editor::OpenSceneFromPath(const std::string& path) {
    if (!LoadEditedScene(path)) {
        LOG_ERROR("Failed to open scene: {}", path);
        return false;
    }

    current_scene_path_ = path;
    BindEngineContextToEditedScene();
    ClearSceneDirty();
    LOG_INFO("Opened scene: {}", current_scene_path_);
    return true;
}

void Editor::RequestOpenSceneFromPath(const std::string& path) {
    RequestSceneSwitch([this, path]() {
        OpenSceneFromPath(path);
    });
}

bool Editor::NewEditedScene() {
    if (edited_scene_ == nullptr || !edited_scene_->NewScene()) {
        LOG_ERROR("Failed to create a new scene.");
        return false;
    }

    selected_ = edited_scene_->GetRoot();
    current_scene_path_.clear();
    BindEngineContextToEditedScene();
    ClearSceneDirty();
    LOG_INFO("Created a new scene.");
    return true;
}

void Editor::RequestNewEditedScene() {
    RequestSceneSwitch([this]() {
        NewEditedScene();
    });
}

void Editor::RequestImportSceneFromPath(const std::string& path) {
    RequestSceneSwitch([this, path]() {
        if (LoadEditedScene(path)) {
            current_scene_path_ = NativeScenePathForImport(std::filesystem::path(path));
            BindEngineContextToEditedScene();
            if (engine_context_ != nullptr) {
                engine_context_->MarkSceneDirtyBaseline();
            }
            LOG_INFO("Imported scene: {}. Native save target: {}", path, current_scene_path_);
        } else {
            LOG_ERROR("Failed to import scene: {}", path);
        }
    });
}

bool Editor::AddSceneToEditedScene(const std::string& path) {
    Node3D* root = GetEditedSceneRoot();
    if (root == nullptr || engine_context_ == nullptr) {
        LOG_ERROR("Cannot add scene '{}': no edited scene.", path);
        return false;
    }

    Ref<Resource> resource = ResourceLoader::Load(path, "PackedScene", ResourceFormatLoader::CacheMode::Reuse);
    Ref<PackedScene> packed_scene = dynamic_pointer_cast<PackedScene>(resource);
    if (!packed_scene.IsValid()) {
        LOG_ERROR("Cannot add scene '{}': ResourceLoader did not return a PackedScene.", path);
        return false;
    }
    if (packed_scene->GetPath().empty()) {
        packed_scene->SetPath(path, false);
    }

    auto command = std::make_unique<AddPackedSceneChildCommand>(
            root->GetInstanceId(),
            packed_scene,
            true,
            true);
    AddPackedSceneChildCommand* command_ptr = command.get();
    if (!engine_context_->ExecuteSceneCommand(std::move(command))) {
        return false;
    }

    auto* added = Object::PointerCastTo<Node3D>(ObjectDB::GetInstance(command_ptr->GetChildId()));
    selected_ = added;
    return true;
}

bool Editor::AddGroundToEditedScene() {
    Node3D* root = GetEditedSceneRoot();
    if (root == nullptr) {
        LOG_ERROR("Cannot add ground: edited scene root is null.");
        return false;
    }

    constexpr RealType kGroundSize = 8.0;
    constexpr RealType kGroundThickness = 0.05;
    const Vector3 ground_size{kGroundSize, kGroundSize, kGroundThickness};
    const Vector3 ground_position{0.0, 0.0, -kGroundThickness * 0.5};

    auto* visual = Object::New<MeshInstance3D>();
    visual->SetName("ground_visual");
    auto ground_mesh = MakeRef<BoxMesh>();
    ground_mesh->SetSize(ground_size);
    visual->SetMesh(ground_mesh);
    visual->SetSurfaceColor({0.28f, 0.30f, 0.32f, 1.0f});
    visual->SetPosition(ground_position);

    auto* collision = Object::New<CollisionShape3D>();
    collision->SetName("ground_collision");
    auto ground_shape = MakeRef<BoxShape3D>();
    ground_shape->SetSize(ground_size);
    collision->SetShape(ground_shape);
    collision->SetPosition(ground_position);

    if (engine_context_ != nullptr) {
        if (!engine_context_->BeginSceneTransaction("Add Ground")) {
            Object::Delete(visual);
            Object::Delete(collision);
            return false;
        }
        if (!engine_context_->ExecuteSceneCommand(std::make_unique<AddChildNodeCommand>(
                root->GetInstanceId(),
                visual->GetInstanceId(),
                true))) {
            engine_context_->CancelSceneTransaction();
            Object::Delete(visual);
            Object::Delete(collision);
            return false;
        }
        if (!engine_context_->ExecuteSceneCommand(std::make_unique<AddChildNodeCommand>(
                root->GetInstanceId(),
                collision->GetInstanceId(),
                true))) {
            engine_context_->CancelSceneTransaction();
            Object::Delete(collision);
            return false;
        }
        if (!engine_context_->CommitSceneTransaction()) {
            Object::Delete(collision);
            return false;
        }
    } else {
        Object::Delete(visual);
        Object::Delete(collision);
        return false;
    }

    selected_ = visual;
    LOG_INFO("Added ground to scene root '{}'.", root->GetName());
    return true;
}

void Editor::RefreshResourcePanel() {
    if (resource_panel_ != nullptr) {
        resource_panel_->Refresh();
    }
}

bool Editor::OpenPythonScriptFromPath(const std::string& path) {
    if (python_panel_ == nullptr) {
        return false;
    }
    python_panel_->SetOpen(true);
    return python_panel_->OpenScript(path);
}

void Editor::MarkSceneDirty() {
    LOG_ERROR("Editor::MarkSceneDirty() is not a mutation API. Use SceneCommand through EngineContext.");
}

void Editor::ClearSceneDirty() {
    if (engine_context_ != nullptr) {
        engine_context_->MarkSceneClean();
        scene_dirty_ = engine_context_->IsSceneDirty();
        return;
    }
    ++scene_change_version_;
    scene_dirty_ = false;
}

bool Editor::IsSceneDirty() const {
    return engine_context_ != nullptr ? engine_context_->IsSceneDirty() : scene_dirty_;
}

std::uint64_t Editor::GetSceneChangeVersion() const {
    return engine_context_ != nullptr ? engine_context_->GetSceneCommandVersion() : scene_change_version_;
}

std::string Editor::GetSceneDisplayName() const {
    return HasCurrentScenePath() ? FileStemOrFilename(current_scene_path_) : "[unsaved]";
}

std::string Editor::GetSceneViewTitle() const {
    std::string title = GetSceneDisplayName();
    if (IsSceneDirty()) {
        title += "(*)";
    }
    return title;
}

void Editor::NotificationCallBack(NotificationType notification) {
    switch (notification) {
        case NotificationType::Process: {
            OnImGui();
        }
    }
}


bool Editor::Begin() {
    imgui_manager_->BeginFrame();

    HandleGlobalShortcuts();
    DrawMenuBar();
    BeginDockSpace();

    return true;
}

void Editor::End() {
    EndDockSpace();

    imgui_manager_->EndFrame();
}

void Editor::OnImGuiContent() {
    file_browser_->Display();
    HandleSceneFileDialogSelection();
    DrawUnsavedSceneDialog();
}

void Editor::DrawMenuBar() {
    if (ImGui::BeginMainMenuBar()) {
        if (ImGui::BeginMenu("File")) {
            if (ImGui::MenuItem("New Scene")) {
                RequestNewEditedScene();
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Save Scene", "Ctrl+S")) {
                SaveCurrentScene();
            }
            if (ImGui::MenuItem("Save Scene As...")) {
                OpenSceneFileDialog(SceneFileDialogMode::SaveAs);
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Load Scene...")) {
                OpenSceneFileDialog(SceneFileDialogMode::Load);
            }
            if (ImGui::MenuItem("Add Scene...")) {
                OpenSceneFileDialog(SceneFileDialogMode::AddScene);
            }
            if (ImGui::MenuItem("Import URDF...")) {
                OpenSceneFileDialog(SceneFileDialogMode::Import);
            }
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Edit")) {
            const bool can_undo = engine_context_ != nullptr && engine_context_->CanUndoSceneCommand();
            const bool can_redo = engine_context_ != nullptr && engine_context_->CanRedoSceneCommand();
            if (ImGui::MenuItem("Undo", "CTRL+Z", false, can_undo)) {
                UndoSceneCommand();
            }
            if (ImGui::MenuItem("Redo", "CTRL+Y", false, can_redo)) {
                RedoSceneCommand();
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Cut", "CTRL+X")) {}
            if (ImGui::MenuItem("Copy", "CTRL+C")) {}
            if (ImGui::MenuItem("Paste", "CTRL+V")) {}
            ImGui::Separator();
            if (ImGui::MenuItem("Add Ground")) {
                AddGroundToEditedScene();
            }
            ImGui::EndMenu();
        }
        DrawViewMenu();
        ImGui::EndMainMenuBar();
    }
}

void Editor::DrawViewMenu() {
    if (!ImGui::BeginMenu("View")) {
        return;
    }

    DrawPanelViewMenuItems(this);
    ImGui::EndMenu();
}

void Editor::DrawPanelViewMenuItems(Node* node) {
    if (node == nullptr) {
        return;
    }

    if (auto* window = Object::PointerCastTo<ImGuiWindow>(node)) {
        bool open = window->IsOpened();
        if (ImGui::MenuItem(window->GetImGuiWindowTitle().c_str(), nullptr, &open)) {
            window->SetOpen(open);
        }
    }

    for (std::size_t i = 0; i < node->GetChildCount(); ++i) {
        DrawPanelViewMenuItems(node->GetChild(static_cast<int>(i)));
    }
}

void Editor::HandleGlobalShortcuts() {
    if (scene_file_dialog_mode_ != SceneFileDialogMode::None ||
        ImGui::IsPopupOpen(nullptr, ImGuiPopupFlags_AnyPopupId)) {
        return;
    }

    const bool ctrl_down = Input::GetInstance()->GetKeyHeld(KeyCode::LeftCtrl) ||
                           Input::GetInstance()->GetKeyHeld(KeyCode::RightCtrl) ||
                           ImGui::GetIO().KeyCtrl;
    const bool s_down = Input::GetInstance()->GetKeyHeld(KeyCode::S) ||
                        ImGui::IsKeyDown(ImGuiKey_S);
    const bool z_down = Input::GetInstance()->GetKeyHeld(KeyCode::Z) ||
                        ImGui::IsKeyDown(ImGuiKey_Z);
    const bool y_down = Input::GetInstance()->GetKeyHeld(KeyCode::Y) ||
                        ImGui::IsKeyDown(ImGuiKey_Y);

    static bool undo_shortcut_down = false;
    static bool redo_shortcut_down = false;

    const bool save_shortcut_down = ctrl_down && s_down;
    if (save_shortcut_down && !save_shortcut_down_) {
        SaveCurrentScene();
    }
    save_shortcut_down_ = save_shortcut_down;

    const bool undo_down = ctrl_down && z_down;
    if (undo_down && !undo_shortcut_down) {
        UndoSceneCommand();
    }
    undo_shortcut_down = undo_down;

    const bool redo_down = ctrl_down && y_down;
    if (redo_down && !redo_shortcut_down) {
        RedoSceneCommand();
    }
    redo_shortcut_down = redo_down;
}

void Editor::UndoSceneCommand() {
    if (engine_context_ == nullptr || !engine_context_->UndoSceneCommand()) {
        return;
    }
    selected_ = GetEditedSceneRoot();
}

void Editor::RedoSceneCommand() {
    if (engine_context_ == nullptr || !engine_context_->RedoSceneCommand()) {
        return;
    }
    selected_ = GetEditedSceneRoot();
}

void Editor::DrawUnsavedSceneDialog() {
    if (request_unsaved_scene_dialog_) {
        ImGui::OpenPopup("Unsaved Scene");
        request_unsaved_scene_dialog_ = false;
    }

    if (!ImGui::BeginPopupModal("Unsaved Scene", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        return;
    }

    ImGui::TextUnformatted("The current scene has unsaved changes.");
    ImGui::TextUnformatted("Save before switching scenes?");
    ImGui::Separator();

    if (ImGui::Button("Save")) {
        if (!HasCurrentScenePath()) {
            OpenSceneFileDialog(SceneFileDialogMode::SaveAs);
            ImGui::CloseCurrentPopup();
        } else if (SaveCurrentScene()) {
            ContinuePendingSceneSwitch();
            ImGui::CloseCurrentPopup();
        }
    }
    ImGui::SameLine();
    if (ImGui::Button("Don't Save")) {
        ClearSceneDirty();
        ContinuePendingSceneSwitch();
        ImGui::CloseCurrentPopup();
    }
    ImGui::SameLine();
    if (ImGui::Button("Cancel")) {
        pending_scene_switch_action_ = nullptr;
        ImGui::CloseCurrentPopup();
    }

    ImGui::EndPopup();
}

void Editor::RequestSceneSwitch(std::function<void()> action) {
    if (!action) {
        return;
    }

    if (IsSceneDirty()) {
        pending_scene_switch_action_ = std::move(action);
        request_unsaved_scene_dialog_ = true;
        return;
    }

    action();
}

void Editor::ContinuePendingSceneSwitch() {
    if (!pending_scene_switch_action_) {
        return;
    }

    auto action = std::move(pending_scene_switch_action_);
    pending_scene_switch_action_ = nullptr;
    action();
}

void Editor::OpenSceneFileDialog(SceneFileDialogMode mode) {
    if (file_browser_ == nullptr) {
        return;
    }

    scene_file_dialog_mode_ = mode;
    if (mode == SceneFileDialogMode::SaveAs) {
        file_browser_->SetFlags(ImGuiFileBrowserFlags_EnterNewFilename);
    } else {
        file_browser_->SetFlags(0);
    }

    if (mode == SceneFileDialogMode::SaveAs) {
        file_browser_->SetTitle("Save Scene");
        file_browser_->SetOkText("Save");
        file_browser_->SetFileFilters({".jscn"});
    } else if (mode == SceneFileDialogMode::AddScene) {
        file_browser_->SetTitle("Add Scene");
        file_browser_->SetOkText("Add");
        file_browser_->SetFileFilters({".jscn"});
    } else if (mode == SceneFileDialogMode::Import) {
        file_browser_->SetTitle("Import URDF");
        file_browser_->SetOkText("Import");
        file_browser_->SetFileFilters({".urdf", ".xml"});
    } else {
        file_browser_->SetTitle("Load Scene");
        file_browser_->SetOkText("Load");
        file_browser_->SetFileFilters({".jscn", ".urdf", ".xml"});
    }

    std::string browser_path = ProjectSettings::GetInstance()->GetProjectPath();
    if (!browser_path.empty()) {
        file_browser_->SetPwd(browser_path);
    }

    file_browser_->Open();
}

void Editor::HandleSceneFileDialogSelection() {
    if (scene_file_dialog_mode_ == SceneFileDialogMode::None || file_browser_ == nullptr) {
        return;
    }

    if (!file_browser_->IsOpened() && !file_browser_->HasSelected()) {
        scene_file_dialog_mode_ = SceneFileDialogMode::None;
        ResetFileDialogDefaults();
        return;
    }

    if (!file_browser_->HasSelected()) {
        return;
    }

    std::filesystem::path selected_path = file_browser_->GetSelected();
    if (scene_file_dialog_mode_ == SceneFileDialogMode::SaveAs && selected_path.extension() != ".jscn") {
        selected_path += ".jscn";
    }

    const std::string scene_path = ProjectSettings::GetInstance()->LocalizePath(selected_path.string());
    if (scene_file_dialog_mode_ == SceneFileDialogMode::SaveAs) {
        if (SaveEditedScene(scene_path)) {
            current_scene_path_ = scene_path;
            ClearSceneDirty();
            LOG_INFO("Saved scene: {}", current_scene_path_);
            ContinuePendingSceneSwitch();
        } else {
            LOG_ERROR("Failed to save scene: {}", scene_path);
        }
    } else if (scene_file_dialog_mode_ == SceneFileDialogMode::Load) {
        if (IsNativeScenePath(scene_path)) {
            RequestOpenSceneFromPath(scene_path);
        } else {
            RequestImportSceneFromPath(scene_path);
        }
    } else if (scene_file_dialog_mode_ == SceneFileDialogMode::AddScene) {
        if (AddSceneToEditedScene(scene_path)) {
            LOG_INFO("Added scene: {}", scene_path);
        } else {
            LOG_ERROR("Failed to add scene: {}", scene_path);
        }
    } else if (scene_file_dialog_mode_ == SceneFileDialogMode::Import) {
        RequestImportSceneFromPath(scene_path);
    }

    file_browser_->ClearSelected();
    scene_file_dialog_mode_ = SceneFileDialogMode::None;
    ResetFileDialogDefaults();
}

void Editor::ResetFileDialogDefaults() {
    if (file_browser_ == nullptr) {
        return;
    }

    file_browser_->SetFlags(0);
    file_browser_->SetTitle("File Browser");
    file_browser_->SetOkText("OK");
    file_browser_->ClearFilters();
}

bool Editor::IsNativeScenePath(const std::string& path) {
    return ToLower(std::filesystem::path(path).extension().string()) == ".jscn";
}

void Editor::BeginDockSpace() {
    static bool p_open                    = true;
    static bool opt_fullscreen_persistant = true;
    static ImGuiDockNodeFlags opt_flags   = ImGuiDockNodeFlags_NoWindowMenuButton | ImGuiDockNodeFlags_NoCloseButton;
    bool opt_fullscreen                   = opt_fullscreen_persistant;

    ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoDocking;
    if(opt_fullscreen) {
        ImGuiViewport* viewport = ImGui::GetMainViewport();
        auto pos     = viewport->Pos;
        auto size    = viewport->Size;
        bool menu_bar = true;
        if(menu_bar) {
            const float infoBarSize = ImGui::GetFrameHeight();
            pos.y += infoBarSize;
            size.y -= infoBarSize;
        }

        ImGui::SetNextWindowPos(pos);
        ImGui::SetNextWindowSize(size);
        ImGui::SetNextWindowViewport(viewport->ID);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
        window_flags |= ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove;
        window_flags |= ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;
    }

    // When using ImGuiDockNodeFlags_PassthruDockspace, DockSpace() will render our background and handle the
    // pass-thru hole, so we ask Begin() to not render a background.
    if(opt_flags & ImGuiDockNodeFlags_DockSpace)
        window_flags |= ImGuiWindowFlags_NoBackground;

    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    ImGui::Begin("GobotDockspace###gobot_dockspace_v2", &p_open, window_flags);
    ImGui::PopStyleVar();

    if(opt_fullscreen)
        ImGui::PopStyleVar(2);

    ImGuiID DockspaceID = ImGui::GetID("gobot_dockspace_v2");

    if(!ImGui::DockBuilderGetNode(DockspaceID)) {
        ImGui::DockBuilderRemoveNode(DockspaceID); // Clear out existing layout
        ImGui::DockBuilderAddNode(DockspaceID);    // Add empty node
        ImGui::DockBuilderSetNodeSize(DockspaceID, ImGui::GetIO().DisplaySize);

        ImGuiID dock_main_id = DockspaceID;
        ImGuiID DockLeft     = ImGui::DockBuilderSplitNode(dock_main_id, ImGuiDir_Left, 0.2f, nullptr, &dock_main_id);
        ImGuiID DockRight    = ImGui::DockBuilderSplitNode(dock_main_id, ImGuiDir_Right, 0.20f, nullptr, &dock_main_id);

        ImGuiID DockingBottomLeftChild  = ImGui::DockBuilderSplitNode(DockLeft, ImGuiDir_Down, 0.4f, nullptr, &DockLeft);

        ImGuiID DockMiddle       = ImGui::DockBuilderSplitNode(dock_main_id, ImGuiDir_Right, 0.8f, nullptr, &dock_main_id);
        ImGuiID DockBottomMiddle = ImGui::DockBuilderSplitNode(DockMiddle, ImGuiDir_Down, 0.3f, nullptr, &DockMiddle);

        ImGui::DockBuilderDockWindow("###scene_view3d", DockMiddle);
        ImGui::DockBuilderDockWindow("###inspector", DockRight);
        ImGui::DockBuilderDockWindow("###physics", DockRight);
        ImGui::DockBuilderDockWindow("###console", DockBottomMiddle);
        ImGui::DockBuilderDockWindow("###resources", DockingBottomLeftChild);
        ImGui::DockBuilderDockWindow("###scene_editor", DockLeft);

        ImGui::DockBuilderFinish(DockspaceID);
    }

    // Submit the DockSpace
    ImGuiIO& io = ImGui::GetIO();
    if(io.ConfigFlags & ImGuiConfigFlags_DockingEnable) {
        ImGui::DockSpace(DockspaceID, ImVec2(0.0f, 0.0f), opt_flags);
    }
}

void Editor::EndDockSpace() {
    ImGui::End();
}

}

GOBOT_REGISTRATION {
    Class_<Editor>("Editor")
        .constructor()(CtorAsRawPtr);

};
