/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/editor/imgui/motion_planning_panel.hpp"

#include "gobot/core/registration.hpp"
#include "gobot/editor/editor.hpp"
#include "gobot/editor/ik_ghost_session.hpp"
#include "gobot/editor/motion_planning_session.hpp"
#include "gobot/editor/planning_scene_bridge.hpp"
#include "gobot/scene/node.hpp"
#include "gobot/scene/robot_3d.hpp"

#include <imgui.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <unordered_set>
#include <vector>

namespace gobot {

namespace {

void CollectRobotsRecursive(Node* node, std::vector<Robot3D*>& out) {
    if (node == nullptr) {
        return;
    }
    if (auto* robot = Object::PointerCastTo<Robot3D>(node)) {
        out.push_back(robot);
    }
    for (std::size_t i = 0; i < node->GetChildCount(); ++i) {
        CollectRobotsRecursive(node->GetChild(static_cast<int>(i)), out);
    }
}

const char* StatusLabel(MotionPlanningStatus status) {
    switch (status) {
        case MotionPlanningStatus::Idle: return "Idle";
        case MotionPlanningStatus::Planning: return "Planning...";
        case MotionPlanningStatus::Ready: return "Ready";
        case MotionPlanningStatus::Stale: return "Stale";
        case MotionPlanningStatus::Cancelled: return "Cancelled";
        case MotionPlanningStatus::TimedOut: return "Timed Out";
        case MotionPlanningStatus::Failed: return "Failed";
        case MotionPlanningStatus::SyncError: return "Sync Error";
        case MotionPlanningStatus::Unavailable: return "Unavailable";
    }
    return "Unknown";
}

ImVec4 StatusColor(MotionPlanningStatus status) {
    switch (status) {
        case MotionPlanningStatus::Ready: return ImVec4(0.22f, 0.88f, 0.44f, 1.0f);
        case MotionPlanningStatus::Planning: return ImVec4(0.32f, 0.72f, 0.98f, 1.0f);
        case MotionPlanningStatus::Stale: return ImVec4(0.98f, 0.62f, 0.16f, 1.0f);
        case MotionPlanningStatus::Cancelled: return ImVec4(0.76f, 0.76f, 0.76f, 1.0f);
        case MotionPlanningStatus::TimedOut:
        case MotionPlanningStatus::Failed:
        case MotionPlanningStatus::SyncError: return ImVec4(0.95f, 0.30f, 0.30f, 1.0f);
        case MotionPlanningStatus::Idle:
        case MotionPlanningStatus::Unavailable:
        default: return ImVec4(0.70f, 0.74f, 0.80f, 1.0f);
    }
}

const char* StrictCCDLabel(StrictCCDStatus status) {
    switch (status) {
        case StrictCCDStatus::NotRun: return "Not run";
        case StrictCCDStatus::Passed: return "Passed";
        case StrictCCDStatus::Failed: return "Failed";
        case StrictCCDStatus::Disabled:
        default: return "Disabled";
    }
}

ImVec4 StrictCCDColor(StrictCCDStatus status) {
    switch (status) {
        case StrictCCDStatus::Passed: return ImVec4(0.22f, 0.88f, 0.44f, 1.0f);
        case StrictCCDStatus::Failed: return ImVec4(0.95f, 0.30f, 0.30f, 1.0f);
        case StrictCCDStatus::Disabled:
        default: return ImVec4(0.70f, 0.74f, 0.80f, 1.0f);
    }
}

} // namespace

MotionPlanningPanel::MotionPlanningPanel() {
    SetName("MotionPlanningPanel");
    SetImGuiWindow("Motion Planning", "motion_planning");
}

void MotionPlanningPanel::OnImGuiContent() {
    auto* editor = Editor::GetInstanceOrNull();
    if (editor == nullptr) {
        ImGui::TextDisabled("Editor unavailable.");
        return;
    }

    if (!MotionPlanningSession::IsAvailable()) {
        ImGui::TextColored(ImVec4(0.95f, 0.65f, 0.25f, 1.0f), "Motion Planning Unavailable");
        ImGui::TextWrapped("Rebuild Gobot with GOB_BUILD_COAL=ON to enable C++ motion planning.");
        return;
    }

    auto* session = editor->GetMotionPlanningSession();
    auto* bridge = editor->GetPlanningSceneBridge();
    auto* ik_session = editor->GetIKGhostSession();
    if (session == nullptr || bridge == nullptr) {
        ImGui::TextDisabled("Motion planning session unavailable.");
        return;
    }

    const bool is_play_mode = editor->IsScenePlaySessionRunning();
    const MotionPlanningDiagnostics diag = session->GetDiagnostics();

    // Status Banner
    ImGui::TextColored(StatusColor(diag.status), "Status: %s", StatusLabel(diag.status));
    if (!diag.status_message.empty() && diag.status != MotionPlanningStatus::Ready) {
        ImGui::SameLine();
        ImGui::TextDisabled("- %s", diag.status_message.c_str());
    }
    if (is_play_mode) {
        ImGui::TextColored(ImVec4(0.98f, 0.62f, 0.16f, 1.0f),
                           "Editing and planning are disabled during Play Mode.");
        return;
    }

    // Robot Selection (explicit ObjectID pinning for multi-robot scenes)
    std::vector<Robot3D*> robots;
    CollectRobotsRecursive(editor->GetActiveSceneRoot(), robots);
    if (robots.empty()) {
        ImGui::Separator();
        ImGui::TextWrapped("No Robot3D found in the active scene.");
        return;
    }

    const ObjectID pinned_id = session->GetPinnedRobotId();
    std::string current_robot_label = "Auto";
    if (pinned_id.IsValid()) {
        for (Robot3D* r : robots) {
            if (r->GetInstanceId() == pinned_id) {
                current_robot_label = r->GetName();
                break;
            }
        }
    } else if (bridge->IsReady()) {
        current_robot_label = "Auto (" + bridge->GetRobotName() + ")";
    }

    if (ImGui::BeginCombo("Robot", current_robot_label.c_str())) {
        const bool is_auto = !pinned_id.IsValid();
        if (ImGui::Selectable("Auto (Follow Selection)", is_auto)) {
            session->SetPinnedRobotId(ObjectID{});
        }
        for (Robot3D* r : robots) {
            const bool selected = (pinned_id == r->GetInstanceId());
            const std::string item_id = r->GetName() + "##robot_" +
                                        std::to_string(static_cast<std::uint64_t>(r->GetInstanceId()));
            if (ImGui::Selectable(item_id.c_str(), selected)) {
                session->SetPinnedRobotId(r->GetInstanceId());
            }
        }
        ImGui::EndCombo();
    }

    if (!bridge->IsReady()) {
        ImGui::Separator();
        if (bridge->HasSyncError()) {
            ImGui::TextColored(ImVec4(0.95f, 0.30f, 0.30f, 1.0f), "Sync Error:");
            ImGui::TextWrapped("%s", bridge->GetLastSyncError().c_str());
        } else {
            ImGui::TextDisabled("Waiting for robot synchronization...");
        }
        return;
    }

    MotionPlanningConfig cfg = session->GetConfig();
    bool cfg_changed = false;

    // End-Effector Link selector
    const auto& link_names = bridge->GetLinkNames();
    std::string active_ee = cfg.end_effector_link;
    if (active_ee.empty()) {
        const int def_id = bridge->GetDefaultEndEffectorLinkId();
        if (def_id >= 0 && static_cast<std::size_t>(def_id) < link_names.size()) {
            active_ee = link_names[static_cast<std::size_t>(def_id)];
        }
    }
    if (ImGui::BeginCombo("End Effector", active_ee.empty() ? "<none>" : active_ee.c_str())) {
        for (const std::string& lname : link_names) {
            const bool sel = (lname == active_ee);
            if (ImGui::Selectable(lname.c_str(), sel)) {
                cfg.end_effector_link = lname;
                cfg_changed = true;
            }
        }
        ImGui::EndCombo();
    }

    // Active Joints / Planning Group selector (maps selected joint names -> active_mask on snapshot)
    const auto& joint_names = bridge->GetJointNames();
    if (ImGui::Checkbox("All Joints Active", &cfg.use_all_joints)) {
        if (!cfg.use_all_joints && cfg.selected_joint_names.empty()) {
            cfg.selected_joint_names = joint_names;
        }
        cfg_changed = true;
    }
    if (!cfg.use_all_joints && !joint_names.empty()) {
        if (ImGui::TreeNode("Active Joint Selection")) {
            std::unordered_set<std::string> selected_set(cfg.selected_joint_names.begin(),
                                                         cfg.selected_joint_names.end());
            bool list_changed = false;
            for (const std::string& jname : joint_names) {
                bool checked = selected_set.contains(jname);
                if (ImGui::Checkbox(jname.c_str(), &checked)) {
                    if (checked) {
                        selected_set.insert(jname);
                    } else {
                        selected_set.erase(jname);
                    }
                    list_changed = true;
                }
            }
            if (list_changed) {
                cfg.selected_joint_names.clear();
                for (const std::string& jname : joint_names) {
                    if (selected_set.contains(jname)) {
                        cfg.selected_joint_names.push_back(jname);
                    }
                }
                cfg_changed = true;
            }
            ImGui::TreePop();
        }
    }

    ImGui::Separator();

    // Start & Goal Capture Controls
    ImGui::Text("Start / Goal");
    if (ImGui::Button("Start = Current")) {
        if (session->CaptureStartFromCurrentRobot()) {
            last_action_note_.clear();
        }
    }
    ImGui::SameLine();
    if (ImGui::Button("Goal = Current")) {
        if (session->CaptureGoalFromCurrentRobot()) {
            last_action_note_.clear();
        }
    }

    if (ik_session != nullptr) {
        std::string ghost_reason;
        const bool ghost_exact_ok = ik_session->TryGetExactSolvedGoal(
                bridge->GetCurrentSnapshotIdentity(), active_ee, nullptr, &ghost_reason);
        ImGui::BeginDisabled(!ghost_exact_ok);
        if (ImGui::Button("Use IK Ghost")) {
            std::string reason;
            if (session->CaptureGoalFromIKGhost(*ik_session, &reason)) {
                last_action_note_.clear();
            } else {
                last_action_note_ = reason;
            }
        }
        ImGui::EndDisabled();
        if (!ghost_exact_ok && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
            ImGui::SetTooltip("%s", ghost_reason.c_str());
        }
        ImGui::SameLine();
        ImGui::BeginDisabled(!session->HasGoalJointPositions());
        if (ImGui::Button("Send Goal -> IK Ghost")) {
            if (session->PushGoalToIKGhost(ik_session)) {
                last_action_note_.clear();
            }
        }
        ImGui::EndDisabled();
    }

    if (session->HasStartJointPositions() && session->HasGoalJointPositions() &&
        session->GetStartJointPositions().size() == session->GetGoalJointPositions().size()) {
        const double dq_norm = (session->GetGoalJointPositions() - session->GetStartJointPositions()).norm();
        ImGui::TextDisabled("DoF: %d | ||q_goal - q_start|| = %.3f",
                            static_cast<int>(session->GetStartJointPositions().size()),
                            dq_norm);
    }
    if (!last_action_note_.empty()) {
        ImGui::TextColored(ImVec4(0.98f, 0.62f, 0.16f, 1.0f), "%s", last_action_note_.c_str());
    }

    ImGui::Separator();

    // Action Buttons: Plan / Re-Plan / Stop / Preview
    const bool is_planning = (diag.status == MotionPlanningStatus::Planning);
    const bool is_stale = (diag.status == MotionPlanningStatus::Stale);

    ImGui::BeginDisabled(is_planning || !session->HasGoalJointPositions());
    if (ImGui::Button("Plan")) {
        last_action_note_.clear();
        session->StartPlan();
    }
    ImGui::EndDisabled();

    ImGui::SameLine();
    if (is_stale) {
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.86f, 0.45f, 0.08f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.96f, 0.54f, 0.14f, 1.0f));
    }
    ImGui::BeginDisabled(is_planning || !session->HasGoalJointPositions());
    if (ImGui::Button("Re-Plan")) {
        last_action_note_.clear();
        session->RePlan();
    }
    ImGui::EndDisabled();
    if (is_stale) {
        ImGui::PopStyleColor(2);
    }

    ImGui::SameLine();
    ImGui::BeginDisabled(!is_planning);
    if (ImGui::Button("Stop")) {
        session->Stop();
    }
    ImGui::EndDisabled();

    ImGui::SameLine();
    ImGui::BeginDisabled(!session->HasTrajectory());
    if (session->IsPreviewActive()) {
        if (ImGui::Button("Stop Preview")) {
            session->StopPreview();
        }
    } else {
        if (ImGui::Button("Preview")) {
            session->StartPreview();
        }
    }
    ImGui::EndDisabled();

    // Debug Overlay Checkboxes
    ImGui::Separator();
    ImGui::Text("Viewport Debug");
    if (ImGui::Checkbox("EE Path", &cfg.show_ee_path)) cfg_changed = true;
    ImGui::SameLine();
    if (ImGui::Checkbox("Witnesses", &cfg.show_collision_witnesses)) cfg_changed = true;
    ImGui::SameLine();
    if (ImGui::Checkbox("Link Spheres", &cfg.show_link_spheres)) cfg_changed = true;

    // Solver Parameters
    if (ImGui::CollapsingHeader("Solver Settings", ImGuiTreeNodeFlags_DefaultOpen)) {
        float margin_f = static_cast<float>(cfg.margin);
        if (ImGui::DragFloat("Self Margin (m)", &margin_f, 0.0005f, 0.0f, 0.10f, "%.4f")) {
            cfg.margin = static_cast<double>(margin_f);
            cfg_changed = true;
        }
        float world_margin_f = static_cast<float>(cfg.world_margin);
        if (ImGui::DragFloat("World Margin (m)", &world_margin_f, 0.001f, 0.0f, 0.20f, "%.4f")) {
            cfg.world_margin = static_cast<double>(world_margin_f);
            cfg_changed = true;
        }
        float vel_f = static_cast<float>(cfg.velocity_scale);
        if (ImGui::SliderFloat("Velocity Scale", &vel_f, 0.05f, 1.0f, "%.2f")) {
            cfg.velocity_scale = static_cast<double>(vel_f);
            cfg_changed = true;
        }
        float acc_f = static_cast<float>(cfg.acceleration_scale);
        if (ImGui::SliderFloat("Accel Scale", &acc_f, 0.05f, 1.0f, "%.2f")) {
            cfg.acceleration_scale = static_cast<double>(acc_f);
            cfg_changed = true;
        }
        float jerk_f = static_cast<float>(cfg.jerk_scale);
        if (ImGui::SliderFloat("Jerk Scale", &jerk_f, 0.05f, 1.0f, "%.2f")) {
            cfg.jerk_scale = static_cast<double>(jerk_f);
            cfg_changed = true;
        }

        int spline_idx = (cfg.spline_order == 5) ? 1 : 0;
        if (ImGui::Combo("Spline Order", &spline_idx, "Cubic (3)\0Quintic (5)\0")) {
            cfg.spline_order = (spline_idx == 1) ? 5 : 3;
            cfg_changed = true;
        }
        if (ImGui::DragInt("Waypoints", &cfg.waypoints, 1.0f, 4, 64)) {
            cfg_changed = true;
        }
        if (ImGui::DragInt("Seeds", &cfg.num_seeds, 1.0f, 1, 8)) {
            cfg_changed = true;
        }
        float timeout_f = static_cast<float>(cfg.timeout_s);
        if (ImGui::DragFloat("Timeout (s)", &timeout_f, 0.1f, 0.2f, 60.0f, "%.1f")) {
            cfg.timeout_s = static_cast<double>(timeout_f);
            cfg_changed = true;
        }
        if (ImGui::Checkbox("Strict CCD", &cfg.strict_ccd)) {
            cfg_changed = true;
        }
        ImGui::SameLine();
        if (ImGui::Checkbox("Global Seeding (BiRRT)", &cfg.use_global_seeding)) {
            cfg_changed = true;
        }
    }

    if (cfg_changed) {
        session->SetConfig(cfg);
    }

    // Diagnostics (explicitly separating Sampled Min Clearance and Strict CCD)
    if (ImGui::CollapsingHeader("Diagnostics", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Text("Solve Time: %.1f ms", diag.total_ms);
        if (session->HasTrajectory()) {
            ImGui::Text("Duration: %.3f s  |  Waypoints: %d (order %d)",
                        diag.trajectory_duration_s,
                        diag.num_waypoints,
                        diag.spline_order);
            if (!diag.seed_strategy.empty()) {
                ImGui::Text("Seed Strategy: %s", diag.seed_strategy.c_str());
            }
            if (session->IsPreviewActive()) {
                ImGui::Text("Preview Time: %.2f / %.2f s",
                            session->GetPreviewTime(),
                            session->GetTrajectoryDuration());
            }
        }

        ImGui::Text("Strict CCD:");
        ImGui::SameLine();
        ImGui::TextColored(StrictCCDColor(diag.strict_ccd_status), "%s", StrictCCDLabel(diag.strict_ccd_status));

        if (diag.has_sampled_clearance) {
            if (diag.sampled_mesh_intersection) {
                ImGui::TextColored(ImVec4(0.95f, 0.30f, 0.30f, 1.0f),
                                   "Sampled Min Clearance: mesh intersection (depth unknown) @ t=%.3fs",
                                   diag.sampled_worst_time);
            } else {
                ImGui::Text("Sampled Min Clearance: %.4f m @ t=%.3fs",
                            diag.sampled_min_distance,
                            diag.sampled_worst_time);
            }
            if (!diag.sampled_shape_pair.empty()) {
                ImGui::TextWrapped("Closest Pair: %s", diag.sampled_shape_pair.c_str());
            }
            if (!diag.has_witness_points) {
                ImGui::TextDisabled("(no witness points available for this pair)");
            }
        } else {
            ImGui::TextDisabled("Sampled Min Clearance: n/a");
        }
    }
}

} // namespace gobot

GOBOT_REGISTRATION {
    Class_<MotionPlanningPanel>("MotionPlanningPanel")
        .constructor()(CtorAsRawPtr);
};
