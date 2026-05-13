/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * This file is created by Qiqi Wu, 23-2-28
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <functional>
#include <memory>
#include <string>

#include "gobot/scene/node.hpp"
#include "gobot/scene/imgui_node.hpp"

namespace ImGui {
class FileBrowser;
}

namespace gobot {

class Node3DEditor;
class ImGuiManager;
class Node3D;
class EditedScene;
class EngineContext;
class ResourcePanel;
class PythonPanel;
class ScenePlaySession;

class GOBOT_EXPORT Editor : public ImGuiNode {
    GOBCLASS(Editor, Node)
public:
    Editor();

    ~Editor() override;

    void NotificationCallBack(NotificationType notification);

    static Editor* GetInstance();

    static Editor* GetInstanceOrNull() { return s_singleton; }

    FORCE_INLINE Node3DEditor* GetNode3dEditor() { return node3d_editor_; }

    FORCE_INLINE ImGui::FileBrowser* GetFileBrowser() { return file_browser_; }

    bool Begin() override;

    void End() override;

    void OnImGuiContent() override;

    FORCE_INLINE Node* GetSelected() const { return selected_; }

    void SetSelected(Node* selected);

    FORCE_INLINE EditedScene* GetEditedScene() const { return edited_scene_; }

    Node3D* GetEditedSceneRoot() const;

    Node* GetActiveSceneRoot() const;

    EngineContext* GetEngineContext() const { return engine_context_; }

    bool SaveEditedScene(const std::string& path) const;

    bool SaveCurrentScene();

    bool LoadEditedScene(const std::string& path);

    bool OpenSceneFromPath(const std::string& path);

    void RequestOpenSceneFromPath(const std::string& path);

    bool NewEditedScene();

    void RequestNewEditedScene();

    void RequestImportSceneFromPath(const std::string& path);

    bool AddSceneToEditedScene(const std::string& path);

    bool AddGroundToEditedScene();

    void RefreshResourcePanel();

    bool OpenPythonScriptFromPath(const std::string& path);

    void FocusSceneViewerPanel();

    void FocusPythonPanel();

    void MarkSceneDirty();

    void ClearSceneDirty();

    [[nodiscard]] bool IsSceneDirty() const;

    [[nodiscard]] std::uint64_t GetSceneChangeVersion() const;

    [[nodiscard]] bool HasCurrentScenePath() const { return !current_scene_path_.empty(); }

    [[nodiscard]] const std::string& GetCurrentScenePath() const { return current_scene_path_; }

    [[nodiscard]] std::string GetSceneDisplayName() const;

    [[nodiscard]] std::string GetSceneViewTitle() const;

    bool StartScenePlaySession();

    bool PlayScene();

    bool PauseScene();

    bool StopScene();

    bool StepScene();

    void StopScenePlaySession();

    bool ResetScenePlaySession();

    [[nodiscard]] bool IsScenePlaySessionRunning() const;

    [[nodiscard]] std::size_t GetActiveSceneScriptCount() const;

    [[nodiscard]] std::string GetScenePlaySessionLastError() const;

private:
    enum class SceneFileDialogMode {
        None,
        Load,
        Import,
        AddScene,
        SaveAs,
    };

    void DrawMenuBar();

    void DrawPlayToolbar();

    void DrawViewMenu();

    void DrawPanelViewMenuItems(Node* node);

    void HandleGlobalShortcuts();

    void UndoSceneCommand();

    void RedoSceneCommand();

    void DrawUnsavedSceneDialog();

    void HandleQuitRequest();

    void NotifyScenePlaySessionProcess();

    void NotifyScenePlaySessionPhysicsProcess();

    void RequestSceneSwitch(std::function<void()> action);

    void ContinuePendingSceneSwitch();

    void OpenSceneFileDialog(SceneFileDialogMode mode);

    void HandleSceneFileDialogSelection();

    void ResetFileDialogDefaults();

    void BindEngineContextToEditedScene();

    void UpdatePythonPanelFromSceneRootScript();

    static bool IsNativeScenePath(const std::string& path);

    void BeginDockSpace();

    void EndDockSpace();

private:
    static Editor* s_singleton;

    Node3DEditor* node3d_editor_{nullptr};
    ImGuiManager* imgui_manager_{nullptr};
    ImGui::FileBrowser* file_browser_{nullptr};
    EditedScene* edited_scene_{nullptr};
    EngineContext* engine_context_{nullptr};
    ResourcePanel* resource_panel_{nullptr};
    PythonPanel* python_panel_{nullptr};
    std::unique_ptr<ScenePlaySession> scene_play_session_;

    SceneFileDialogMode scene_file_dialog_mode_{SceneFileDialogMode::None};
    std::string current_scene_path_;
    bool scene_dirty_{false};
    std::uint64_t scene_change_version_{0};
    bool save_shortcut_down_{false};
    bool request_unsaved_scene_dialog_{false};
    bool pending_quit_request_{false};
    bool request_scene_viewer_focus_{true};
    bool request_python_panel_focus_{false};
    std::function<void()> pending_scene_switch_action_;

    Node* selected_{nullptr};

};


}
