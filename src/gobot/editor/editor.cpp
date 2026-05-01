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
#include "gobot/editor/imgui/resource_panel.hpp"
#include "gobot/editor/property_inspector/editor_inspector.hpp"
#include "gobot/main/main.hpp"
#include "gobot/scene/node_3d.hpp"
#include "gobot/core/config/engine.hpp"
#include "gobot/core/config/project_setting.hpp"
#include "gobot/editor/imgui/imgui_utilities.hpp"
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
    std::filesystem::path native_path = selected_path;
    native_path.replace_extension(".jscn");
    return ProjectSettings::GetInstance()->LocalizePath(native_path.string());
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

    node3d_editor_ = Object::New<Node3DEditor>();
    node3d_editor_->SetName("Node3DEditor");
    AddChild(node3d_editor_);

    spdlog::sink_ptr sink = std::make_shared<ImGuiConsoleSinkMultiThreaded>();
    Logger::GetInstance().AddSink(sink);

    EditorInspector::AddInspectorPlugin(MakeRef<EditorInspectorDefaultPlugin>());

    // for test
    ProjectSettings::GetInstance()->SetProjectPath(".");

    AddChild(Object::New<ConsolePanel>());
    AddChild(Object::New<SceneEditorPanel>());
    AddChild(Object::New<InspectorPanel>());
    AddChild(Object::New<PhysicsPanel>());
    AddChild(Object::New<ResourcePanel>());


    file_browser_ = new ImGui::FileBrowser();
    file_browser_->SetTitle("File Browser");
}

Editor::~Editor() {
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

bool Editor::SaveEditedScene(const std::string& path) const {
    return edited_scene_ != nullptr && edited_scene_->SaveToPath(path);
}

bool Editor::LoadEditedScene(const std::string& path) {
    const std::string extension = ToLower(std::filesystem::path(path).extension().string());
    const bool default_robot_motion_mode = extension == ".urdf" || extension == ".xml";
    if (edited_scene_ == nullptr || !edited_scene_->LoadFromPath(path, default_robot_motion_mode)) {
        return false;
    }

    selected_ = edited_scene_->GetRoot();
    return true;
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

    DrawMenuBar();
    BeginDockSpace();

    return true;
}

void Editor::End() {
    EndDockSpace();

    imgui_manager_->EndFrame();
}

void Editor::OnImGuiContent() {
    // your drawing here
    ImGui::ShowDemoWindow();

    file_browser_->Display();
    HandleSceneFileDialogSelection();
}

void Editor::DrawMenuBar() {
    if (ImGui::BeginMainMenuBar()) {
        if (ImGui::BeginMenu("File")) {
            if (ImGui::MenuItem("Save Scene", "Ctrl+S")) {
                if (SaveEditedScene(current_scene_path_)) {
                    LOG_INFO("Saved scene: {}", current_scene_path_);
                } else {
                    LOG_ERROR("Failed to save scene: {}", current_scene_path_);
                }
            }
            if (ImGui::MenuItem("Save Scene As...")) {
                OpenSceneFileDialog(SceneFileDialogMode::SaveAs);
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Load Scene...")) {
                OpenSceneFileDialog(SceneFileDialogMode::Load);
            }
            if (ImGui::MenuItem("Import URDF...")) {
                OpenSceneFileDialog(SceneFileDialogMode::Import);
            }
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Edit")) {
            if (ImGui::MenuItem("Undo", "CTRL+Z")) {}
            if (ImGui::MenuItem("Redo", "CTRL+Y", false, false)) {}  // Disabled item
            ImGui::Separator();
            if (ImGui::MenuItem("Cut", "CTRL+X")) {}
            if (ImGui::MenuItem("Copy", "CTRL+C")) {}
            if (ImGui::MenuItem("Paste", "CTRL+V")) {}
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
    }
}

void Editor::OpenSceneFileDialog(SceneFileDialogMode mode) {
    if (file_browser_ == nullptr) {
        return;
    }

    scene_file_dialog_mode_ = mode;
    file_browser_->SetFlags(mode == SceneFileDialogMode::SaveAs
                            ? ImGuiFileBrowserFlags_EnterNewFilename
                            : 0);
    if (mode == SceneFileDialogMode::SaveAs) {
        file_browser_->SetTitle("Save Scene");
        file_browser_->SetOkText("Save");
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

    const std::string& project_path = ProjectSettings::GetInstance()->GetProjectPath();
    if (!project_path.empty()) {
        file_browser_->SetPwd(project_path);
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
            LOG_INFO("Saved scene: {}", current_scene_path_);
        } else {
            LOG_ERROR("Failed to save scene: {}", scene_path);
        }
    } else if (scene_file_dialog_mode_ == SceneFileDialogMode::Load) {
        if (LoadEditedScene(scene_path)) {
            if (IsNativeScenePath(scene_path)) {
                current_scene_path_ = scene_path;
            } else {
                current_scene_path_ = NativeScenePathForImport(selected_path);
            }
            LOG_INFO("Loaded scene: {}", current_scene_path_);
        } else {
            LOG_ERROR("Failed to load scene: {}", scene_path);
        }
    } else if (scene_file_dialog_mode_ == SceneFileDialogMode::Import) {
        if (LoadEditedScene(scene_path)) {
            current_scene_path_ = NativeScenePathForImport(selected_path);
            LOG_INFO("Imported scene: {}. Native save target: {}", scene_path, current_scene_path_);
        } else {
            LOG_ERROR("Failed to import scene: {}", scene_path);
        }
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
    ImGui::Begin("MyDockspace", &p_open, window_flags);
    ImGui::PopStyleVar();

    if(opt_fullscreen)
        ImGui::PopStyleVar(2);

    ImGuiID DockspaceID = ImGui::GetID("MyDockspace");

    if(!ImGui::DockBuilderGetNode(DockspaceID)) {
        ImGui::DockBuilderRemoveNode(DockspaceID); // Clear out existing layout
        ImGui::DockBuilderAddNode(DockspaceID);    // Add empty node
        ImGui::DockBuilderSetNodeSize(DockspaceID, ImGui::GetIO().DisplaySize * ImGui::GetIO().DisplayFramebufferScale);

        ImGuiID dock_main_id = DockspaceID;
        ImGuiID DockLeft     = ImGui::DockBuilderSplitNode(dock_main_id, ImGuiDir_Left, 0.2f, nullptr, &dock_main_id);
        ImGuiID DockRight    = ImGui::DockBuilderSplitNode(dock_main_id, ImGuiDir_Right, 0.20f, nullptr, &dock_main_id);

        ImGuiID DockingBottomLeftChild  = ImGui::DockBuilderSplitNode(DockLeft, ImGuiDir_Down, 0.4f, nullptr, &DockLeft);

        ImGuiID DockMiddle       = ImGui::DockBuilderSplitNode(dock_main_id, ImGuiDir_Right, 0.8f, nullptr, &dock_main_id);
        ImGuiID DockBottomMiddle = ImGui::DockBuilderSplitNode(DockMiddle, ImGuiDir_Down, 0.3f, nullptr, &DockMiddle);

        ImGui::DockBuilderDockWindow("###scene_view3d", DockMiddle);
        ImGui::DockBuilderDockWindow("###inspector", DockRight);
        ImGui::DockBuilderDockWindow("###console", DockBottomMiddle);
        ImGui::DockBuilderDockWindow("###physics", DockBottomMiddle);
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
