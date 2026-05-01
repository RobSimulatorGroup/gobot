/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
*/

#include "gobot/editor/imgui/physics_panel.hpp"

#include <algorithm>
#include <string>
#include <vector>

#include "gobot/core/registration.hpp"
#include "gobot/editor/editor.hpp"
#include "gobot/physics/physics_server.hpp"
#include "gobot/scene/node_3d.hpp"
#include "gobot/simulation/simulation_server.hpp"
#include "imgui.h"
#include "imgui_extension/icon_fonts/icons_material_design_icons.h"

namespace gobot {
namespace {

const char* BackendTypeLabel(PhysicsBackendType backend_type) {
    switch (backend_type) {
        case PhysicsBackendType::Null:
            return "Null";
        case PhysicsBackendType::MuJoCoCpu:
            return "MuJoCo CPU";
        case PhysicsBackendType::PhysXCpu:
            return "PhysX CPU";
        case PhysicsBackendType::PhysXGpu:
            return "PhysX GPU";
        case PhysicsBackendType::NewtonGpu:
            return "Newton GPU";
        case PhysicsBackendType::RigidIpcCpu:
            return "Rigid IPC CPU";
    }
    return "Unknown";
}

std::vector<PhysicsBackendInfo> GetBackendInfos() {
    if (PhysicsServer::HasInstance()) {
        return PhysicsServer::GetInstance()->GetBackendInfos();
    }
    return PhysicsServer::GetBackendInfosForAllBackends();
}

PhysicsBackendInfo GetBackendInfo(PhysicsBackendType backend_type) {
    if (PhysicsServer::HasInstance()) {
        return PhysicsServer::GetInstance()->GetBackendInfo(backend_type);
    }
    return PhysicsServer::GetBackendInfoForBackend(backend_type);
}

void DrawStatusText(bool ok, const char* available_text, const char* unavailable_text) {
    const ImVec4 color = ok ? ImVec4(0.35f, 0.85f, 0.35f, 1.0f)
                            : ImVec4(0.95f, 0.35f, 0.35f, 1.0f);
    ImGui::TextColored(color, "%s", ok ? available_text : unavailable_text);
}

} // namespace

PhysicsPanel::PhysicsPanel() {
    SetName("PhysicsPanel");
    SetImGuiWindow(ICON_MDI_COGS " Physics", "physics");
}

void PhysicsPanel::OnImGuiContent() {
    SimulationServer* simulation = SimulationServer::HasInstance()
                                           ? SimulationServer::GetInstance()
                                           : nullptr;
    if (simulation == nullptr) {
        ImGui::TextColored(ImVec4(0.95f, 0.35f, 0.35f, 1.0f),
                           "SimulationServer is not initialized.");
        return;
    }

    if (!simulation->HasWorld() && simulation->GetBackendType() != selected_backend_) {
        simulation->SetBackendType(selected_backend_);
    }

    const std::vector<PhysicsBackendInfo> backend_infos = GetBackendInfos();
    const PhysicsBackendInfo selected_info = GetBackendInfo(selected_backend_);

    if (ImGui::BeginCombo("Backend", BackendTypeLabel(selected_backend_))) {
        for (const PhysicsBackendInfo& info : backend_infos) {
            const bool selected = info.type == selected_backend_;
            std::string label = info.name;
            if (!info.available) {
                label += " (Unavailable)";
            }

            if (ImGui::Selectable(label.c_str(), selected)) {
                selected_backend_ = info.type;
                simulation->SetBackendType(selected_backend_);
                simulation->ClearWorld();
            }

            if (selected) {
                ImGui::SetItemDefaultFocus();
            }
        }
        ImGui::EndCombo();
    }

    ImGui::SameLine();
    DrawStatusText(selected_info.available, "Available", "Unavailable");

    ImGui::TextWrapped("%s", selected_info.status.c_str());
    ImGui::Separator();

    const bool has_world = simulation->HasWorld();
    DrawStatusText(has_world, "World built", "No world");

    ImGui::Text("Time: %.6f", static_cast<double>(simulation->GetSimulationTime()));
    ImGui::Text("Frame: %llu", static_cast<unsigned long long>(simulation->GetFrameCount()));
    ImGui::Text("Fixed dt: %.6f", static_cast<double>(simulation->GetFixedTimeStep()));

    bool paused = simulation->IsPaused();
    if (ImGui::Checkbox("Paused", &paused)) {
        simulation->SetPaused(paused);
    }

    Node* scene_root = Editor::GetInstance() ? Editor::GetInstance()->GetEditedSceneRoot() : nullptr;
    const bool can_build = scene_root != nullptr && selected_info.available;
    if (!can_build) {
        ImGui::BeginDisabled();
    }
    if (ImGui::Button("Build World")) {
        simulation->SetBackendType(selected_backend_);
        simulation->BuildWorldFromScene(scene_root);
    }
    if (!can_build) {
        ImGui::EndDisabled();
    }

    ImGui::SameLine();
    if (!has_world) {
        ImGui::BeginDisabled();
    }
    if (ImGui::Button("Step")) {
        simulation->StepOnce();
    }
    ImGui::SameLine();
    if (ImGui::Button("Reset")) {
        simulation->Reset();
    }
    ImGui::SameLine();
    if (ImGui::Button("Clear")) {
        simulation->ClearWorld();
    }
    if (!has_world) {
        ImGui::EndDisabled();
    }

    if (scene_root == nullptr) {
        ImGui::TextColored(ImVec4(0.95f, 0.65f, 0.25f, 1.0f), "No edited scene root.");
    }

    const std::string& last_error = simulation->GetLastError();
    if (!last_error.empty()) {
        ImGui::Separator();
        ImGui::TextColored(ImVec4(0.95f, 0.35f, 0.35f, 1.0f), "Last error:");
        ImGui::TextWrapped("%s", last_error.c_str());
    }
}

} // namespace gobot

GOBOT_REGISTRATION {
    Class_<PhysicsPanel>("PhysicsPanel")
            .constructor()(CtorAsRawPtr);
}
