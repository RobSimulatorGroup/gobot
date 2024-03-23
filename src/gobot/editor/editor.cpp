/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-28
*/

#include "gobot/editor/editor.hpp"
#include "gobot/core/config/project_setting.hpp"
#include "gobot/editor/node3d_editor.hpp"
#include "gobot/core/registration.hpp"
#include "gobot/log.hpp"
#include "gobot/editor/imgui/console_sink.hpp"
#include "gobot/editor/imgui/imgui_manager.hpp"
#include "gobot/editor/imgui/scene_view_3d_panel.hpp"
#include "gobot/editor/imgui/scene_editor_panel.hpp"
#include "gobot/editor/imgui/inspector_panel.hpp"
#include "gobot/editor/imgui/resource_panel.hpp"
#include "gobot/editor/property_inspector/editor_inspector.hpp"
#include "gobot/main/main.hpp"
#include "gobot/core/config/engine.hpp"
#include "gobot/core/config/project_setting.hpp"
#include "gobot/editor/imgui/imgui_utilities.hpp"
#include "imgui.h"
#include "imgui_internal.h"
#include "imgui_extension/file_browser/ImFileBrowser.h"

namespace gobot {

Editor* Editor::s_singleton = nullptr;

Editor::Editor() {
    s_singleton = this;

    imgui_manager_ = Object::New<ImGuiManager>();

    node3d_editor_ = Object::New<Node3DEditor>();
    AddChild(node3d_editor_);

    spdlog::sink_ptr sink = std::make_shared<ImGuiConsoleSinkMultiThreaded>();
    Logger::GetInstance().AddSink(sink);

    EditorInspector::AddInspectorPlugin(MakeRef<EditorInspectorDefaultPlugin>());

    // for test
    ProjectSettings::GetInstance()->SetProjectPath(".");

    AddChild(Object::New<ConsolePanel>());
    AddChild(Object::New<SceneEditorPanel>());
    AddChild(Object::New<InspectorPanel>());
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
}

void Editor::DrawMenuBar() {
    if (ImGui::BeginMainMenuBar()) {
        if (ImGui::BeginMenu("File")) {
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