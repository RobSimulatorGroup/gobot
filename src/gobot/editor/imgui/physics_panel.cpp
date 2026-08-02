/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
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
    }
    return "Unknown";
}

std::vector<PhysicsBackendInfo> GetBackendInfos() {
    if (PhysicsServer::HasInstance()) {
        return PhysicsServer::GetInstance()->GetBackendInfos();
    }
    return PhysicsServer::GetBackendInfos();
}

PhysicsBackendInfo GetBackendInfo(PhysicsBackendType backend_type) {
    if (PhysicsServer::HasInstance()) {
        return PhysicsServer::GetInstance()->GetBackendInfo(backend_type);
    }
    return PhysicsServer::GetBackendInfo(backend_type);
}

void DrawStatusText(bool ok, const char* available_text, const char* unavailable_text) {
    const ImVec4 color = ok ? ImVec4(0.35f, 0.85f, 0.35f, 1.0f)
                            : ImVec4(0.95f, 0.35f, 0.35f, 1.0f);
    ImGui::TextColored(color, "%s", ok ? available_text : unavailable_text);
}

void DrawTimingControls(SimulationServer* simulation, bool lock_provider_timing) {
    if (lock_provider_timing) {
        ImGui::BeginDisabled();
    }
    const RealType fixed_time_step = simulation->GetFixedTimeStep();
    double physics_hz = fixed_time_step > 0.0 ? 1.0 / static_cast<double>(fixed_time_step) : 0.0;
    physics_hz = std::clamp(physics_hz, 1.0, 2000.0);
    if (ImGui::InputDouble("Physics Hz", &physics_hz, 1.0, 10.0, "%.3f")) {
        if (physics_hz > 0.0 && std::isfinite(physics_hz)) {
            simulation->SetFixedTimeStep(static_cast<RealType>(1.0 / physics_hz));
        }
    }
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
        ImGui::SetTooltip(lock_provider_timing
                                  ? "Physics Hz is owned by the active external provider"
                                  : "Fixed physics tick rate. Higher values run more simulation ticks per second.");
    }

    double fixed_dt = static_cast<double>(simulation->GetFixedTimeStep());
    if (ImGui::InputDouble("Fixed dt", &fixed_dt, 0.0001, 0.001, "%.6f")) {
        if (fixed_dt > 0.0 && std::isfinite(fixed_dt)) {
            simulation->SetFixedTimeStep(static_cast<RealType>(fixed_dt));
        }
    }
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
        ImGui::SetTooltip(lock_provider_timing
                                  ? "Fixed dt is owned by the active external provider"
                                  : "Seconds per physics tick. This is the inverse of Physics Hz.");
    }
    if (lock_provider_timing) {
        ImGui::EndDisabled();
    }

    double time_scale = static_cast<double>(simulation->GetTimeScale());
    if (ImGui::InputDouble("Time scale", &time_scale, 0.05, 0.25, "%.3f")) {
        if (time_scale >= 0.0 && std::isfinite(time_scale)) {
            simulation->SetTimeScale(static_cast<RealType>(time_scale));
        }
    }

    if (lock_provider_timing) {
        ImGui::BeginDisabled();
    }
    int max_sub_steps = simulation->GetMaxSubSteps();
    if (ImGui::InputInt("Max substeps", &max_sub_steps, 1, 4)) {
        simulation->SetMaxSubSteps(std::max(1, max_sub_steps));
    }
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
        ImGui::SetTooltip(lock_provider_timing
                                  ? "Max substeps is owned by the active external provider"
                                  : "Maximum physics ticks allowed during one editor frame.");
    }
    if (lock_provider_timing) {
        ImGui::EndDisabled();
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

void DrawDebugVisualizationControls(SimulationServer* simulation) {
    PhysicsWorldSettings settings = simulation->GetPhysicsWorldSettings();
    bool changed = false;

    bool draw_contacts = settings.debug_draw_contacts;
    if (ImGui::Checkbox(ICON_MDI_CROSSHAIRS " Contact points / normals", &draw_contacts)) {
        settings.debug_draw_contacts = draw_contacts;
        changed = true;
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Draw contact crosses and blue normal directions. Normals are geometric directions, not force magnitudes.");
    }

    bool draw_contact_forces = settings.debug_draw_contact_forces;
    if (ImGui::Checkbox(ICON_MDI_AXIS_ARROW " Contact force arrows", &draw_contact_forces)) {
        settings.debug_draw_contact_forces = draw_contact_forces;
        changed = true;
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Draw red solver force arrows. Force arrows encode direction and magnitude, including friction components when available.");
    }

    double force_scale = static_cast<double>(settings.debug_contact_force_scale);
    if (ImGui::InputDouble("Force scale", &force_scale, 0.005, 0.02, "%.4f")) {
        if (force_scale >= 0.0 && std::isfinite(force_scale)) {
            settings.debug_contact_force_scale = static_cast<RealType>(force_scale);
            changed = true;
        }
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Scales the red contact force arrows only; it does not change simulation forces.");
    }

    double max_force_length = static_cast<double>(settings.debug_contact_force_max_length);
    if (ImGui::InputDouble("Max force length", &max_force_length, 0.05, 0.2, "%.3f")) {
        if (max_force_length >= 0.0 && std::isfinite(max_force_length)) {
            settings.debug_contact_force_max_length = static_cast<RealType>(max_force_length);
            changed = true;
        }
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Caps the red contact force arrow length so large impulses remain readable.");
    }

    if (changed) {
        simulation->SetPhysicsWorldSettings(settings);
    }
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

    if (!simulation->HasActiveSession() && simulation->GetBackendType() != selected_backend_) {
        simulation->SetBackendType(selected_backend_);
    }

    const std::vector<PhysicsBackendInfo> backend_infos = GetBackendInfos();
    const PhysicsBackendInfo selected_info = GetBackendInfo(selected_backend_);
    Editor* editor = Editor::GetInstanceOrNull();
    const bool script_session_running = editor != nullptr && editor->IsScenePlaySessionRunning();
    const bool external_session = simulation->HasExternalSession();
    const ExternalSessionDiagnostics& external = simulation->GetExternalSessionDiagnostics();
    const char* backend_label =
            external_session && !external.provider_name.empty()
                    ? external.provider_name.c_str()
                    : BackendTypeLabel(selected_backend_);

    if (script_session_running) {
        ImGui::BeginDisabled();
    }
    if (ImGui::BeginCombo("Backend", backend_label)) {
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
    DrawStatusText(external_session || selected_info.available, "Available", "Unavailable");

    if (external_session) {
        ImGui::TextWrapped("%s", external.status.empty() ? "External provider active" : external.status.c_str());
    } else {
        ImGui::TextWrapped("%s", selected_info.status.c_str());
    }
    ImGui::Separator();

    const bool has_world = simulation->HasWorld();
    const bool has_active_session = simulation->HasActiveSession();
    DrawStatusText(has_active_session,
                   simulation->HasExternalSession() ? "External provider active" : "World built",
                   "No simulation session");
    if (external_session) {
        ImGui::Text("Provider: %s", external.provider_name.empty() ? "External" : external.provider_name.c_str());
        ImGui::Text("Device: %s", external.device.empty() ? "-" : external.device.c_str());
        ImGui::Text("Environments: %zu", external.environment_count);
        ImGui::Text("Controlled joints: %zu", external.controlled_joint_count);
        ImGui::Text("Capacities: %s", external.capacities.empty() ? "{}" : external.capacities.c_str());
        ImGui::Text("CUDA graph: %s", external.graph_status.empty() ? "Unknown" : external.graph_status.c_str());
        ImGui::Text("Step latency: %.3f ms last, %.3f ms average",
                    external.last_step_latency_ms,
                    external.average_step_latency_ms);
    }
    if (simulation->IsFaulted()) {
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(0.95f, 0.35f, 0.35f, 1.0f), "Reset required");
    }

    ImGui::Text("Time: %.6f", static_cast<double>(simulation->GetSimulationTime()));
    ImGui::Text("Frame: %llu", static_cast<unsigned long long>(simulation->GetFrameCount()));
    if (ImGui::CollapsingHeader("Timing", ImGuiTreeNodeFlags_DefaultOpen)) {
        DrawTimingControls(simulation, simulation->HasExternalSession());
    }
    const Vector3& gravity = simulation->GetPhysicsWorldSettings().gravity;
    ImGui::Text("Gravity: %.3f, %.3f, %.3f m/s^2",
                static_cast<double>(gravity.x()),
                static_cast<double>(gravity.y()),
                static_cast<double>(gravity.z()));
    if (ImGui::CollapsingHeader("Debug Visualization")) {
        DrawDebugVisualizationControls(simulation);
    }

    Node* edited_scene_root = editor != nullptr ? editor->GetEditedSceneRoot() : nullptr;
    Node* scene_root = editor != nullptr ? editor->GetActiveSceneRoot() : nullptr;
    const bool can_build = scene_root != nullptr && selected_info.available;
    if (!can_build || script_session_running) {
        ImGui::BeginDisabled();
    }
    if (ImGui::Button(ICON_MDI_CUBE_SCAN " Build World")) {
        if (editor != nullptr) {
            editor->StopScene();
            scene_root = editor->GetEditedSceneRoot();
        } else {
            simulation->SetPaused(true);
        }
        simulation->SetBackendType(selected_backend_);
        simulation->BuildWorldFromScene(scene_root);
    }
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
        ImGui::SetTooltip("Build a physics world from the active scene");
    }
    if (!can_build || script_session_running) {
        ImGui::EndDisabled();
    }

    ImGui::SameLine();
    const bool can_play = scene_root != nullptr && !simulation->IsFaulted() &&
                          (editor != nullptr || has_active_session || selected_info.available);
    if (!can_play) {
        ImGui::BeginDisabled();
    }
    if (simulation->IsPaused()) {
        if (ImGui::Button(ICON_MDI_PLAY " Play")) {
            if (!simulation->HasActiveSession()) {
                simulation->SetBackendType(selected_backend_);
            }
            if (editor != nullptr) {
                editor->PlayScene();
            } else if (scene_root != nullptr) {
                if (!simulation->HasActiveSession()) {
                    simulation->BuildWorldFromScene(scene_root);
                }
                simulation->SetPaused(!simulation->HasActiveSession());
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
    if (!has_active_session) {
        ImGui::BeginDisabled();
    }
    if (ImGui::Button(ICON_MDI_RESTART " Reset")) {
        if (editor != nullptr && editor->IsScenePlaySessionRunning()) {
            editor->ResetScenePlaySession();
        } else {
            simulation->Reset();
        }
    }
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
        ImGui::SetTooltip("Reset the current physics world or scene playback session");
    }
    ImGui::SameLine();
    if (ImGui::Button(ICON_MDI_DELETE " Clear")) {
        if (editor != nullptr) {
            editor->StopScene();
        } else {
            simulation->ClearWorld();
        }
    }
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
        ImGui::SetTooltip("Stop playback and remove the current physics world");
    }
    if (!has_active_session) {
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
