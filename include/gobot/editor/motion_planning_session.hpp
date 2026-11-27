/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "gobot/core/math/geometry.hpp"
#include "gobot/core/object_id.hpp"
#include "gobot/editor/ik_ghost_session.hpp"
#include "gobot/editor/planning_scene_bridge.hpp"
#include "gobot/rendering/scene_render_items.hpp"
#include "gobot_export.h"

#if defined(GOBOT_HAS_PLANNING)
#include "planning_planner.hpp"
#endif

#include <Eigen/Core>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace gobot {

class Node;
class Robot3D;

enum class MotionPlanningStatus {
    Idle = 0,
    Planning = 1,
    Ready = 2,
    Stale = 3,
    Cancelled = 4,
    TimedOut = 5,
    Failed = 6,
    SyncError = 7,
    Unavailable = 8,
};

enum class StrictCCDStatus {
    Disabled = 0,
    Passed = 1,
    Failed = 2,
    NotRun = 3,
};

struct MotionPlanningConfig {
    std::string end_effector_link;
    bool use_all_joints = true;
    std::vector<std::string> selected_joint_names;

    double margin = 0.002;
    double world_margin = 0.01;
    double velocity_scale = 1.0;
    double acceleration_scale = 1.0;
    double jerk_scale = 1.0;
    int spline_order = 3;
    bool strict_ccd = false;
    bool use_global_seeding = true;
    int waypoints = 16;
    int num_seeds = 1;
    double timeout_s = 5.0;

    // Viewport debug toggles (changing visual toggles does not invalidate Ready trajectories).
    bool show_ee_path = true;
    bool show_collision_witnesses = true;
    bool show_link_spheres = false;
};

struct TrajectoryDebugOverlayData {
    bool valid = false;
    bool is_stale = false;
    std::vector<Vector3> ee_path_points;

    // Sampled minimum clearance along the final retimed spline on the planned snapshot.
    bool has_sampled_clearance = false;
    bool sampled_mesh_intersection = false;
    double sampled_min_distance = std::numeric_limits<double>::infinity();
    double sampled_worst_time = 0.0;
    std::string sampled_shape_pair;
    bool has_witness_points = false;
    Vector3 witness_a = Vector3::Zero();
    Vector3 witness_b = Vector3::Zero();
};

struct MotionPlanningDiagnostics {
    MotionPlanningStatus status = MotionPlanningStatus::Idle;
    StrictCCDStatus strict_ccd_status = StrictCCDStatus::Disabled;
    PlanningSnapshotIdentity current_snapshot_id;
    PlanningSnapshotIdentity planned_snapshot_id;
    std::uint64_t request_revision = 0;
    std::uint64_t planned_request_revision = 0;
    std::uint64_t active_request_seq = 0;
    std::uint64_t last_completed_request_seq = 0;

    double total_ms = 0.0;
    double trajectory_duration_s = 0.0;
    int num_waypoints = 0;
    int spline_order = 3;
    std::string seed_strategy;
    std::string status_message;

    // Sampled minimum clearance along the final retimed spline (separate from strict_ccd_status).
    bool has_sampled_clearance = false;
    bool sampled_mesh_intersection = false;
    double sampled_min_distance = std::numeric_limits<double>::infinity();
    double sampled_worst_time = 0.0;
    std::string sampled_shape_pair;
    bool has_witness_points = false;
};

class GOBOT_EXPORT MotionPlanningSession {
public:
    using WorkerHook = std::function<void(std::uint64_t request_seq, const std::atomic<bool>& cancel_flag)>;

    MotionPlanningSession();
    explicit MotionPlanningSession(PlanningSceneBridge* shared_bridge);
    ~MotionPlanningSession();

    MotionPlanningSession(const MotionPlanningSession&) = delete;
    MotionPlanningSession& operator=(const MotionPlanningSession&) = delete;

    [[nodiscard]] static bool IsAvailable() noexcept;

    // Per-frame main-thread update. Consumes the shared bridge (coalescing SyncFromScene at most once per
    // (scene_epoch, robot_id, scene_change_version)), checks snapshot identity & request revision, polls
    // async worker results, and advances read-only trajectory overlay preview if active.
    void Update(const Node* scene_root,
                const Robot3D* active_robot,
                std::uint64_t scene_epoch,
                std::uint64_t scene_change_version,
                bool is_play_mode,
                double dt_seconds = 0.0);

    // Explicit robot pinning by ObjectID for multi-robot scenes (invalid ObjectID = auto-follow active robot).
    void SetPinnedRobotId(ObjectID robot_id);
    [[nodiscard]] ObjectID GetPinnedRobotId() const noexcept { return pinned_robot_id_; }

    // Solver/planning configuration. Changing any non-visual field bumps request_revision_ and marks Ready -> Stale.
    void SetConfig(const MotionPlanningConfig& config);
    [[nodiscard]] const MotionPlanningConfig& GetConfig() const noexcept { return config_; }

    // Start & Goal joint states.
    bool CaptureStartFromCurrentRobot();
    bool CaptureGoalFromCurrentRobot();
    bool CaptureGoalFromIKGhost(const InteractiveIKGhostSession& ghost, std::string* out_reason = nullptr);
    bool PushGoalToIKGhost(InteractiveIKGhostSession* ghost) const;
    void SetStartJointPositions(const Eigen::VectorXd& q);
    void SetGoalJointPositions(const Eigen::VectorXd& q);
    [[nodiscard]] bool HasStartJointPositions() const noexcept { return has_start_q_; }
    [[nodiscard]] bool HasGoalJointPositions() const noexcept { return has_goal_q_; }
    [[nodiscard]] const Eigen::VectorXd& GetStartJointPositions() const noexcept { return start_q_; }
    [[nodiscard]] const Eigen::VectorXd& GetGoalJointPositions() const noexcept { return goal_q_; }

    // Async planning lifecycle.
    bool StartPlan();
    bool RePlan();
    void Stop();

    // Read-only trajectory overlay preview (zero SceneTree mutations).
    bool StartPreview();
    void StopPreview();
    [[nodiscard]] bool IsPreviewActive() const noexcept { return preview_active_ && has_trajectory_; }
    [[nodiscard]] double GetPreviewTime() const noexcept { return preview_time_; }
    [[nodiscard]] double GetTrajectoryDuration() const noexcept { return trajectory_duration_s_; }
    [[nodiscard]] bool HasTrajectory() const noexcept { return has_trajectory_; }

    [[nodiscard]] const std::vector<VisualMeshRenderItem>& GetPreviewOverlayMeshes() const noexcept {
        return preview_overlay_meshes_;
    }
    [[nodiscard]] const TrajectoryDebugOverlayData& GetDebugOverlayData() const noexcept {
        return debug_overlay_;
    }
    [[nodiscard]] std::vector<LinkSphereOverlay> CollectCurrentLinkSpheres() const;

    [[nodiscard]] MotionPlanningStatus GetStatus() const noexcept { return status_; }
    [[nodiscard]] MotionPlanningDiagnostics GetDiagnostics() const;
    [[nodiscard]] const PlanningSceneBridge& GetBridge() const noexcept {
        return shared_bridge_ ? *shared_bridge_ : owned_bridge_;
    }
    [[nodiscard]] PlanningSceneBridge& GetBridge() noexcept {
        return shared_bridge_ ? *shared_bridge_ : owned_bridge_;
    }

    // Deterministic test hooks & idle wait.
    void SetTestWorkerHooks(WorkerHook before_plan, WorkerHook after_plan);
    bool WaitForIdle(std::chrono::milliseconds timeout = std::chrono::milliseconds(10000));

private:
    struct PendingPlanRequest {
        std::chrono::steady_clock::time_point submitted_at;
        std::uint64_t request_seq = 0;
        std::uint64_t request_revision = 0;
        PlanningSnapshotIdentity snapshot_id;
        std::shared_ptr<std::atomic<bool>> cancel_flag;
        int ee_link_id = -1;
        Eigen::VectorXd start_q;
        Eigen::VectorXd goal_q;
        std::vector<std::uint8_t> active_mask;
        MotionPlanningConfig config;
#if defined(GOBOT_HAS_PLANNING)
        std::shared_ptr<const planning::WorldSnapshot> snapshot;
#endif
    };

    struct CompletedPlanResult {
        std::uint64_t request_seq = 0;
        std::uint64_t request_revision = 0;
        PlanningSnapshotIdentity snapshot_id;
        bool success = false;
        bool cancelled = false;
        bool timed_out = false;
        StrictCCDStatus strict_ccd = StrictCCDStatus::Disabled;
        double total_ms = 0.0;
        std::string seed_strategy;
        std::string reason;
        TrajectoryDebugOverlayData debug_overlay;
#if defined(GOBOT_HAS_PLANNING)
        planning::TrajectoryNative trajectory;
        std::shared_ptr<const planning::WorldSnapshot> snapshot;
#endif
    };

    void StartWorker();
    void StopWorker();
    void WorkerLoop();
    void MarkConfigMutated();
    void MarkTrajectoryStale();
    void ClearTrajectoryAndPreview();
    void InvalidateActiveRequest();
    void RebuildPreviewOverlayAtCurrentTime();
    [[nodiscard]] int ResolveEndEffectorLinkId() const;
    [[nodiscard]] std::vector<std::uint8_t> BuildActiveMask(const PlanningSceneBridge& bridge) const;
    static bool SolverConfigChanged(const MotionPlanningConfig& a, const MotionPlanningConfig& b) noexcept;

    PlanningSceneBridge owned_bridge_;
    PlanningSceneBridge* shared_bridge_ = nullptr;
    ObjectID pinned_robot_id_{};

    MotionPlanningConfig config_;
    std::uint64_t request_revision_ = 1;
    std::uint64_t next_request_seq_ = 0;
    std::uint64_t active_request_seq_ = 0;
    std::uint64_t disqualified_before_seq_ = 0;
    std::uint64_t last_completed_request_seq_ = 0;
    std::shared_ptr<std::atomic<bool>> active_cancel_flag_;

    PlanningSnapshotIdentity bound_snapshot_id_;
    PlanningSnapshotIdentity planned_snapshot_id_;
    std::uint64_t planned_request_revision_ = 0;

    bool has_start_q_ = false;
    bool has_goal_q_ = false;
    Eigen::VectorXd start_q_;
    Eigen::VectorXd goal_q_;

    MotionPlanningStatus status_ = MotionPlanningStatus::Idle;
    StrictCCDStatus strict_ccd_status_ = StrictCCDStatus::Disabled;
    double last_total_ms_ = 0.0;
    double trajectory_duration_s_ = 0.0;
    int trajectory_waypoints_ = 0;
    int trajectory_spline_order_ = 3;
    std::string last_seed_strategy_;
    std::string last_status_message_;

    bool has_trajectory_ = false;
    bool preview_active_ = false;
    double preview_time_ = 0.0;
    std::vector<VisualMeshRenderItem> preview_overlay_meshes_;
    TrajectoryDebugOverlayData debug_overlay_;

#if defined(GOBOT_HAS_PLANNING)
    planning::TrajectoryNative trajectory_;
    std::shared_ptr<const planning::WorldSnapshot> planned_snapshot_;
#endif

    std::mutex mutex_;
    std::condition_variable cv_;
    std::condition_variable idle_cv_;
    bool worker_stop_ = false;
    bool worker_busy_ = false;
    bool has_pending_ = false;
    PendingPlanRequest pending_;
    std::deque<CompletedPlanResult> completed_queue_;
#if defined(GOBOT_HAS_PLANNING)
    struct SampledClearanceUpdate {
        std::uint64_t request_seq;
        planning::TrajectorySampledClearanceNative clearance;
    };
    std::optional<SampledClearanceUpdate> completed_clearance_;
#endif
    WorkerHook before_plan_hook_;
    WorkerHook after_plan_hook_;
    std::thread worker_;
};

} // namespace gobot
