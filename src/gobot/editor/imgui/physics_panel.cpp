/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
*/

#include "gobot/editor/imgui/physics_panel.hpp"

#include <algorithm>
#include <cmath>
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
        case PhysicsBackendType::MuJoCoWarp:
            return "MuJoCo Warp";
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

void DrawTimingControls(SimulationServer* simulation) {
    const RealType fixed_time_step = simulation->GetFixedTimeStep();
    double physics_hz = fixed_time_step > 0.0 ? 1.0 / static_cast<double>(fixed_time_step) : 0.0;
    physics_hz = std::clamp(physics_hz, 1.0, 2000.0);
    if (ImGui::InputDouble("Physics Hz", &physics_hz, 1.0, 10.0, "%.3f")) {
        if (physics_hz > 0.0 && std::isfinite(physics_hz)) {
            simulation->SetFixedTimeStep(static_cast<RealType>(1.0 / physics_hz));
        }
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Fixed physics tick rate. Higher values run more simulation ticks per second.");
    }

    double fixed_dt = static_cast<double>(simulation->GetFixedTimeStep());
    if (ImGui::InputDouble("Fixed dt", &fixed_dt, 0.0001, 0.001, "%.6f")) {
        if (fixed_dt > 0.0 && std::isfinite(fixed_dt)) {
            simulation->SetFixedTimeStep(static_cast<RealType>(fixed_dt));
        }
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Seconds per physics tick. This is the inverse of Physics Hz.");
    }

    double time_scale = static_cast<double>(simulation->GetTimeScale());
    if (ImGui::InputDouble("Time scale", &time_scale, 0.05, 0.25, "%.3f")) {
        if (time_scale >= 0.0 && std::isfinite(time_scale)) {
            simulation->SetTimeScale(static_cast<RealType>(time_scale));
        }
    }

    int max_sub_steps = simulation->GetMaxSubSteps();
    if (ImGui::InputInt("Max substeps", &max_sub_steps, 1, 4)) {
        simulation->SetMaxSubSteps(std::max(1, max_sub_steps));
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Maximum physics ticks allowed during one editor frame.");
    }

    const float render_fps = ImGui::GetIO().Framerate;
    const double expected_ticks_per_render =
            render_fps > 0.0f && simulation->GetFixedTimeStep() > 0.0
                    ? static_cast<double>(simulation->GetTimeScale()) /
                              (static_cast<double>(render_fps) * static_cast<double>(simulation->GetFixedTimeStep()))
                    : 0.0;
    ImGui::Text("Render FPS: %.1f", static_cast<double>(render_fps));
    ImGui::Text("Physics ticks / render: last %d, expected %.2f",
                simulation->GetLastStepCount(),
                expected_ticks_per_render);
    ImGui::Text("Accumulator: %.6f", static_cast<double>(simulation->GetAccumulator()));
}

const char* JointControlModeLabel(PhysicsJointControlMode mode) {
    switch (mode) {
        case PhysicsJointControlMode::Passive:
            return "Passive";
        case PhysicsJointControlMode::Position:
            return "Position";
        case PhysicsJointControlMode::Velocity:
            return "Velocity";
        case PhysicsJointControlMode::Effort:
            return "Effort";
    }
    return "Unknown";
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
    Editor* editor = Editor::GetInstanceOrNull();
    const bool script_session_running = editor != nullptr && editor->IsScenePlaySessionRunning();

    if (script_session_running) {
        ImGui::BeginDisabled();
    }
    if (ImGui::BeginCombo("Backend", BackendTypeLabel(selected_backend_))) {
        for (const PhysicsBackendInfo& info : backend_infos) {
            const bool selected = info.type == selected_backend_;
            std::string label = info.name;
            if (!info.available) {
                label += " (Unavailable)";
            }

            if (ImGui::Selectable(label.c_str(), selected)) {
                if (Editor* editor = Editor::GetInstanceOrNull(); editor != nullptr) {
                    editor->StopScenePlaySession();
                }
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
    if (script_session_running) {
        ImGui::EndDisabled();
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
            ImGui::SetTooltip("Stop scene playback before changing physics backend");
        }
    }

    ImGui::SameLine();
    DrawStatusText(selected_info.available, "Available", "Unavailable");

    ImGui::TextWrapped("%s", selected_info.status.c_str());
    ImGui::Separator();

    const bool has_world = simulation->HasWorld();
    DrawStatusText(has_world, "World built", "No world");

    ImGui::Text("Time: %.6f", static_cast<double>(simulation->GetSimulationTime()));
    ImGui::Text("Frame: %llu", static_cast<unsigned long long>(simulation->GetFrameCount()));
    if (ImGui::CollapsingHeader("Timing", ImGuiTreeNodeFlags_DefaultOpen)) {
        DrawTimingControls(simulation);
    }
    const Vector3& gravity = simulation->GetPhysicsWorldSettings().gravity;
    ImGui::Text("Gravity: %.3f, %.3f, %.3f m/s^2",
                static_cast<double>(gravity.x()),
                static_cast<double>(gravity.y()),
                static_cast<double>(gravity.z()));

    Node* edited_scene_root = editor != nullptr ? editor->GetEditedSceneRoot() : nullptr;
    Node* scene_root = editor != nullptr ? editor->GetActiveSceneRoot() : nullptr;
    const bool can_build = scene_root != nullptr && selected_info.available;
    if (!can_build || script_session_running) {
        ImGui::BeginDisabled();
    }
    if (ImGui::Button("Build World")) {
        if (editor != nullptr) {
            editor->StopScene();
            scene_root = editor->GetEditedSceneRoot();
        } else {
            simulation->SetPaused(true);
        }
        simulation->SetBackendType(selected_backend_);
        simulation->BuildWorldFromScene(scene_root);
    }
    if (!can_build || script_session_running) {
        ImGui::EndDisabled();
    }

    ImGui::SameLine();
    const bool can_play = selected_info.available && scene_root != nullptr;
    if (!can_play) {
        ImGui::BeginDisabled();
    }
    if (simulation->IsPaused()) {
        if (ImGui::Button(ICON_MDI_PLAY " Play")) {
            simulation->SetBackendType(selected_backend_);
            if (editor != nullptr) {
                editor->PlayScene();
            } else if (scene_root != nullptr) {
                if (!simulation->HasWorld()) {
                    simulation->BuildWorldFromScene(scene_root);
                }
                simulation->SetPaused(!simulation->HasWorld());
            }
        }
    } else {
        if (ImGui::Button(ICON_MDI_STOP " Stop")) {
            if (editor != nullptr) {
                editor->StopScene();
            } else {
                simulation->SetPaused(true);
            }
        }
    }
    if (!can_play) {
        ImGui::EndDisabled();
    }

    ImGui::SameLine();
    if (!has_world) {
        ImGui::BeginDisabled();
    }
    if (ImGui::Button("Reset")) {
        if (editor != nullptr && editor->IsScenePlaySessionRunning()) {
            editor->ResetScenePlaySession();
        } else {
            simulation->Reset();
        }
    }
    ImGui::SameLine();
    if (ImGui::Button("Clear")) {
        if (editor != nullptr) {
            editor->StopScene();
        } else {
            simulation->ClearWorld();
        }
    }
    if (!has_world) {
        ImGui::EndDisabled();
    }

    if (edited_scene_root == nullptr) {
        ImGui::TextColored(ImVec4(0.95f, 0.65f, 0.25f, 1.0f), "No edited scene root.");
    }

    if (editor != nullptr) {
        ImGui::Separator();
        DrawStatusText(editor->IsScenePlaySessionRunning(), "Scene session running", "Scene session stopped");
        ImGui::Text("Active scripts: %zu", editor->GetActiveSceneScriptCount());
        if (editor->IsScenePlaySessionRunning()) {
            ImGui::TextWrapped("Backend and world build controls are locked while scripts are driving the runtime scene.");
        }
        const std::string session_error = editor->GetScenePlaySessionLastError();
        if (!session_error.empty()) {
            ImGui::TextColored(ImVec4(0.95f, 0.35f, 0.35f, 1.0f), "Script error:");
            ImGui::TextWrapped("%s", session_error.c_str());
        }
    }

    if (has_world && simulation->GetWorld().IsValid()) {
        const PhysicsSceneState& scene_state = simulation->GetWorld()->GetSceneState();
        ImGui::Separator();
        ImGui::Text("Robots: %zu", scene_state.robots.size());
        ImGui::Text("Links: %zu", scene_state.total_link_count);
        ImGui::Text("Joints: %zu", scene_state.total_joint_count);

        const float table_height = std::max(260.0f, ImGui::GetContentRegionAvail().y);
        if (ImGui::BeginTable("PhysicsJointStateTable",
                              6,
                              ImGuiTableFlags_Borders |
                                      ImGuiTableFlags_RowBg |
                                      ImGuiTableFlags_Resizable |
                                      ImGuiTableFlags_ScrollY,
                              ImVec2(0.0f, table_height))) {
            ImGui::TableSetupColumn("Robot");
            ImGui::TableSetupColumn("Joint");
            ImGui::TableSetupColumn("Position");
            ImGui::TableSetupColumn("Velocity");
            ImGui::TableSetupColumn("Control");
            ImGui::TableSetupColumn("Target");
            ImGui::TableHeadersRow();

            for (const PhysicsRobotState& robot_state : scene_state.robots) {
                for (const PhysicsJointState& joint_state : robot_state.joints) {
                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0);
                    ImGui::TextUnformatted(robot_state.name.c_str());
                    ImGui::TableSetColumnIndex(1);
                    ImGui::TextUnformatted(joint_state.joint_name.c_str());
                    ImGui::TableSetColumnIndex(2);
                    ImGui::Text("%.6f", static_cast<double>(joint_state.position));
                    ImGui::TableSetColumnIndex(3);
                    ImGui::Text("%.6f", static_cast<double>(joint_state.velocity));
                    ImGui::TableSetColumnIndex(4);
                    ImGui::TextUnformatted(JointControlModeLabel(joint_state.control_mode));
                    ImGui::TableSetColumnIndex(5);
                    switch (joint_state.control_mode) {
                        case PhysicsJointControlMode::Position:
                            ImGui::Text("%.6f", static_cast<double>(joint_state.target_position));
                            break;
                        case PhysicsJointControlMode::Velocity:
                            ImGui::Text("%.6f", static_cast<double>(joint_state.target_velocity));
                            break;
                        case PhysicsJointControlMode::Effort:
                            ImGui::Text("%.6f", static_cast<double>(joint_state.target_effort));
                            break;
                        case PhysicsJointControlMode::Passive:
                            ImGui::TextUnformatted("-");
                            break;
                    }
                }
            }

            ImGui::EndTable();
        }
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
