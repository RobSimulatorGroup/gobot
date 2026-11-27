/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "gobot/core/math/geometry.hpp"
#include "gobot/core/object_id.hpp"
#include "gobot/editor/planning_scene_bridge.hpp"
#include "gobot/rendering/scene_render_items.hpp"
#include "gobot_export.h"

#include <Eigen/Core>

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace gobot {

class Node;
class Robot3D;

enum class IKGhostStatus {
    Inactive = 0,
    Solving = 1,
    Ok = 2,            // Pose solved and collision/margin free (green)
    Collision = 3,     // Pose solved but violates collision/margin/ESDF (orange)
    PoseNotSolved = 4, // Local IK did not converge within tolerance (red)
    Unavailable = 5,
};

struct IKGhostPreviewConfig {
    double pos_tol = 1e-3;
    double rot_tol = 1e-2;
    double margin = 0.002;
    double world_margin = 0.01;
    int ik_iterations = 40;
    double ik_damping = 1e-3;
    bool avoid_collisions = true;
    bool verify_original = true;
    bool check_rotation = true;
};

struct IKGhostDiagnostics {
    IKGhostStatus status = IKGhostStatus::Inactive;
    std::uint64_t scene_epoch = 0;
    std::uint64_t world_version = 0;
    std::uint64_t target_seq = 0;
    std::uint64_t displayed_seq = 0;
    double pos_err = 0.0;
    double rot_err = 0.0;
    double min_distance = 0.0;
    double last_solve_ms = 0.0;
    double last_input_to_display_ms = 0.0;
    double p95_input_to_display_ms = 0.0;
    std::string collision_pair;
    std::string end_effector_link;
    std::string status_message;
    bool has_witness = false;
    Vector3 witness_a = Vector3::Zero();
    Vector3 witness_b = Vector3::Zero();
};

class GOBOT_EXPORT InteractiveIKGhostSession {
public:
    InteractiveIKGhostSession();
    explicit InteractiveIKGhostSession(PlanningSceneBridge* shared_bridge);
    ~InteractiveIKGhostSession();

    InteractiveIKGhostSession(const InteractiveIKGhostSession&) = delete;
    InteractiveIKGhostSession& operator=(const InteractiveIKGhostSession&) = delete;

    [[nodiscard]] static bool IsAvailable() noexcept;

    void SetEnabled(bool enabled);
    [[nodiscard]] bool IsEnabled() const noexcept { return enabled_; }

    void SetPreviewConfig(const IKGhostPreviewConfig& config);
    [[nodiscard]] const IKGhostPreviewConfig& GetPreviewConfig() const noexcept { return config_; }

    void SetExtraIgnoredPairs(std::vector<std::pair<std::string, std::string>> pairs);
    bool SetEndEffectorLink(const std::string& link_name);

    // Per-frame main-thread update. Synchronizes the planning world at most once per frame when
    // scene_epoch or scene_change_version advances, polls latest async IK results, and updates
    // ghost overlay meshes. Never mutates SceneTree or SceneCommandStack.
    void Update(const Node* scene_root,
                const Robot3D* active_robot,
                std::uint64_t scene_epoch,
                std::uint64_t scene_change_version,
                bool is_play_mode);

    // Submit a new interactive TCP target pose from gizmo dragging (latest-wins, non-blocking).
    void SetTargetPose(const Affine3& world_pose);

    // Reset target pose back to the robot's current end-effector pose.
    void ResetTargetToCurrentPose();

    [[nodiscard]] bool HasActiveTarget() const noexcept { return enabled_ && has_target_pose_ && GetBridge().IsReady(); }
    [[nodiscard]] const Affine3& GetTargetPose() const noexcept { return target_pose_; }
    [[nodiscard]] const std::vector<VisualMeshRenderItem>& GetOverlayMeshes() const noexcept { return overlay_meshes_; }
    [[nodiscard]] const Eigen::VectorXd& GetGhostJointPositions() const noexcept { return ghost_q_; }
    [[nodiscard]] IKGhostDiagnostics GetDiagnostics() const;
    [[nodiscard]] const PlanningSceneBridge& GetBridge() const noexcept { return shared_bridge_ ? *shared_bridge_ : owned_bridge_; }
    [[nodiscard]] PlanningSceneBridge& GetBridge() noexcept { return shared_bridge_ ? *shared_bridge_ : owned_bridge_; }

    // Returns true and writes ghost joint positions ONLY if the currently displayed IK solution is Ok,
    // matches the expected snapshot identity and EE link, and displayed_seq == target_seq (no mid-drag lag).
    bool TryGetExactSolvedGoal(const PlanningSnapshotIdentity& expected_snapshot,
                               const std::string& expected_ee_link,
                               Eigen::VectorXd* out_q,
                               std::string* out_reason = nullptr) const;

    // Wait until the background worker finishes the latest submitted target_seq (for deterministic tests).
    bool WaitForIdle(std::chrono::milliseconds timeout = std::chrono::milliseconds(5000));

private:
    struct PendingRequest {
        PlanningSnapshotIdentity snapshot_id;
        std::uint64_t target_seq = 0;
        int link_id = -1;
        std::string ee_link_name;
        Affine3 target_pose = Affine3::Identity();
        Eigen::VectorXd seed_q;
        std::vector<Eigen::VectorXd> fallback_seeds;
        IKGhostPreviewConfig config;
        std::chrono::steady_clock::time_point submitted_at;
#if defined(GOBOT_HAS_PLANNING)
        std::shared_ptr<const planning::WorldSnapshot> snapshot;
#endif
    };

    struct CompletedResult {
        PlanningSnapshotIdentity snapshot_id;
        std::uint64_t target_seq = 0;
        std::string ee_link_name;
        IKGhostStatus status = IKGhostStatus::Inactive;
        Eigen::VectorXd q;
        double pos_err = 0.0;
        double rot_err = 0.0;
        double min_distance = 0.0;
        double solve_ms = 0.0;
        std::string collision_pair;
        std::string reason;
        bool has_witness = false;
        Vector3 witness_a = Vector3::Zero();
        Vector3 witness_b = Vector3::Zero();
        std::chrono::steady_clock::time_point submitted_at;
#if defined(GOBOT_HAS_PLANNING)
        std::shared_ptr<const planning::WorldSnapshot> snapshot;
#endif
    };

    void StartWorker();
    void StopWorker();
    void WorkerLoop();
    void SubmitCurrentTarget();
    void InvalidateDisplayedResult();
    void RebuildOverlayMeshesFromResult(const CompletedResult& res);
    void RecordLatencySample(double ms);
    void ClearActiveState();

    PlanningSceneBridge owned_bridge_;
    PlanningSceneBridge* shared_bridge_ = nullptr;
    IKGhostPreviewConfig config_;
    std::string preferred_ee_link_;

    bool enabled_ = false;
    bool has_target_pose_ = false;
    int ee_link_id_ = -1;
    PlanningSnapshotIdentity bound_snapshot_id_;
    PlanningSnapshotIdentity displayed_snapshot_id_;
    std::string displayed_ee_link_;
    std::uint64_t next_target_seq_ = 0;
    std::uint64_t first_valid_target_seq_ = 0;
    std::uint64_t displayed_target_seq_ = 0;

    Affine3 target_pose_ = Affine3::Identity();
    Eigen::VectorXd ghost_q_;
    std::vector<VisualMeshRenderItem> overlay_meshes_;
    IKGhostStatus status_ = IKGhostStatus::Inactive;
    double last_pos_err_ = 0.0;
    double last_rot_err_ = 0.0;
    double last_min_distance_ = 0.0;
    double last_solve_ms_ = 0.0;
    double last_input_to_display_ms_ = 0.0;
    std::deque<double> latency_samples_ms_;
    std::string last_collision_pair_;
    std::string last_status_message_;
    bool last_has_witness_ = false;
    Vector3 last_witness_a_ = Vector3::Zero();
    Vector3 last_witness_b_ = Vector3::Zero();

    std::mutex mutex_;
    std::condition_variable cv_;
    std::condition_variable idle_cv_;
    bool worker_stop_ = false;
    bool worker_busy_ = false;
    bool has_pending_ = false;
    bool has_completed_ = false;
    PendingRequest pending_;
    CompletedResult completed_;
    std::thread worker_;
};

} // namespace gobot
