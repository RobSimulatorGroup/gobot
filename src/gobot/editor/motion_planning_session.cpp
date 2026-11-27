/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/editor/motion_planning_session.hpp"

#include "gobot/scene/node.hpp"
#include "gobot/scene/robot_3d.hpp"

#include <algorithm>
#include <cmath>
#include <unordered_set>
#include <utility>

namespace gobot {

namespace {

void ColorsForPlanPreview(bool is_stale, Color* albedo, Color* emissive) {
    if (is_stale) {
        *albedo = Color(0.98f, 0.56f, 0.10f, 0.48f);
        *emissive = Color(0.28f, 0.12f, 0.02f, 1.0f);
    } else {
        *albedo = Color(0.16f, 0.78f, 0.96f, 0.44f);
        *emissive = Color(0.04f, 0.20f, 0.28f, 1.0f);
    }
}

bool DoubleDiffers(double a, double b) noexcept {
    return std::abs(a - b) > 1e-12;
}

} // namespace

MotionPlanningSession::MotionPlanningSession()
    : shared_bridge_(nullptr) {
    StartWorker();
}

MotionPlanningSession::MotionPlanningSession(PlanningSceneBridge* shared_bridge)
    : shared_bridge_(shared_bridge) {
    StartWorker();
}

MotionPlanningSession::~MotionPlanningSession() {
    StopWorker();
}

bool MotionPlanningSession::IsAvailable() noexcept {
    return PlanningSceneBridge::IsPlanningAvailable();
}

void MotionPlanningSession::StartWorker() {
#if defined(GOBOT_HAS_PLANNING)
    if (worker_.joinable()) {
        return;
    }
    worker_stop_ = false;
    worker_ = std::thread(&MotionPlanningSession::WorkerLoop, this);
#endif
}

void MotionPlanningSession::StopWorker() {
#if defined(GOBOT_HAS_PLANNING)
    if (active_cancel_flag_) {
        active_cancel_flag_->store(true, std::memory_order_relaxed);
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        worker_stop_ = true;
        if (has_pending_ && pending_.cancel_flag) {
            pending_.cancel_flag->store(true, std::memory_order_relaxed);
        }
        has_pending_ = false;
    }
    cv_.notify_all();
    if (worker_.joinable()) {
        worker_.join();
    }
#endif
}

bool MotionPlanningSession::SolverConfigChanged(const MotionPlanningConfig& a,
                                                const MotionPlanningConfig& b) noexcept {
    return a.end_effector_link != b.end_effector_link ||
           a.use_all_joints != b.use_all_joints ||
           a.selected_joint_names != b.selected_joint_names ||
           DoubleDiffers(a.margin, b.margin) ||
           DoubleDiffers(a.world_margin, b.world_margin) ||
           DoubleDiffers(a.velocity_scale, b.velocity_scale) ||
           DoubleDiffers(a.acceleration_scale, b.acceleration_scale) ||
           DoubleDiffers(a.jerk_scale, b.jerk_scale) ||
           a.spline_order != b.spline_order ||
           a.strict_ccd != b.strict_ccd ||
           a.use_global_seeding != b.use_global_seeding ||
           a.waypoints != b.waypoints ||
           a.num_seeds != b.num_seeds ||
           DoubleDiffers(a.timeout_s, b.timeout_s);
}

void MotionPlanningSession::MarkConfigMutated() {
    ++request_revision_;
    MarkTrajectoryStale();
}

void MotionPlanningSession::MarkTrajectoryStale() {
    if (has_trajectory_) {
        if (status_ == MotionPlanningStatus::Ready) {
            status_ = MotionPlanningStatus::Stale;
            last_status_message_ = "Scene or parameters changed; Re-Plan required.";
        }
        debug_overlay_.is_stale = true;
        if (preview_active_) {
            RebuildPreviewOverlayAtCurrentTime();
        }
    }
}

void MotionPlanningSession::ClearTrajectoryAndPreview() {
    has_trajectory_ = false;
    preview_active_ = false;
    preview_time_ = 0.0;
    trajectory_duration_s_ = 0.0;
    trajectory_waypoints_ = 0;
    preview_overlay_meshes_.clear();
    debug_overlay_ = TrajectoryDebugOverlayData{};
    planned_snapshot_id_ = PlanningSnapshotIdentity{};
    planned_request_revision_ = 0;
    strict_ccd_status_ = StrictCCDStatus::Disabled;
#if defined(GOBOT_HAS_PLANNING)
    trajectory_ = planning::TrajectoryNative{};
    planned_snapshot_.reset();
#endif
}

void MotionPlanningSession::InvalidateActiveRequest() {
    if (active_cancel_flag_) {
        active_cancel_flag_->store(true, std::memory_order_relaxed);
        active_cancel_flag_.reset();
    }
    if (active_request_seq_ > 0) {
        disqualified_before_seq_ = std::max(disqualified_before_seq_, active_request_seq_ + 1);
        active_request_seq_ = 0;
    } else {
        disqualified_before_seq_ = std::max(disqualified_before_seq_, next_request_seq_ + 1);
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (has_pending_ && pending_.cancel_flag) {
        pending_.cancel_flag->store(true, std::memory_order_relaxed);
    }
    has_pending_ = false;
}

void MotionPlanningSession::SetPinnedRobotId(ObjectID robot_id) {
    if (pinned_robot_id_ == robot_id) {
        return;
    }
    pinned_robot_id_ = robot_id;
}

void MotionPlanningSession::SetConfig(const MotionPlanningConfig& config) {
    const bool solver_changed = SolverConfigChanged(config_, config);
    config_ = config;
    if (config_.spline_order != 3 && config_.spline_order != 5) {
        config_.spline_order = 3;
    }
    config_.waypoints = std::max(4, config_.waypoints);
    config_.num_seeds = std::max(1, config_.num_seeds);
    config_.velocity_scale = std::clamp(config_.velocity_scale, 0.05, 1.0);
    config_.acceleration_scale = std::clamp(config_.acceleration_scale, 0.05, 1.0);
    config_.jerk_scale = std::clamp(config_.jerk_scale, 0.05, 1.0);
    config_.margin = std::max(0.0, config_.margin);
    config_.world_margin = std::max(0.0, config_.world_margin);
    config_.timeout_s = std::max(0.1, config_.timeout_s);
    if (solver_changed) {
        MarkConfigMutated();
    }
}

int MotionPlanningSession::ResolveEndEffectorLinkId() const {
    const PlanningSceneBridge& bridge = GetBridge();
    if (!config_.end_effector_link.empty()) {
        if (const int id = bridge.FindLinkId(config_.end_effector_link); id >= 0) {
            return id;
        }
    }
    return bridge.GetDefaultEndEffectorLinkId();
}

std::vector<std::uint8_t> MotionPlanningSession::BuildActiveMask(const PlanningSceneBridge& bridge) const {
    const auto& joint_names = bridge.GetJointNames();
    const std::size_t J = joint_names.size();
    if (config_.use_all_joints) {
        return std::vector<std::uint8_t>(J, 1);
    }
    std::unordered_set<std::string> selected(config_.selected_joint_names.begin(),
                                             config_.selected_joint_names.end());
    std::vector<std::uint8_t> mask(J, 0);
    for (std::size_t j = 0; j < J; ++j) {
        if (selected.contains(joint_names[j])) {
            mask[j] = 1;
        }
    }
    return mask;
}

bool MotionPlanningSession::CaptureStartFromCurrentRobot() {
    const PlanningSceneBridge& bridge = GetBridge();
    if (!bridge.IsReady()) {
        return false;
    }
    const Eigen::VectorXd& cur = bridge.GetCurrentJointPositions();
    if (cur.size() == 0) {
        return false;
    }
    start_q_ = cur;
    has_start_q_ = true;
    MarkConfigMutated();
    return true;
}

bool MotionPlanningSession::CaptureGoalFromCurrentRobot() {
    const PlanningSceneBridge& bridge = GetBridge();
    if (!bridge.IsReady()) {
        return false;
    }
    const Eigen::VectorXd& cur = bridge.GetCurrentJointPositions();
    if (cur.size() == 0) {
        return false;
    }
    goal_q_ = cur;
    has_goal_q_ = true;
    MarkConfigMutated();
    return true;
}

bool MotionPlanningSession::CaptureGoalFromIKGhost(const InteractiveIKGhostSession& ghost,
                                                   std::string* out_reason) {
    const PlanningSceneBridge& bridge = GetBridge();
    if (!bridge.IsReady()) {
        if (out_reason != nullptr) *out_reason = "Planning scene is not synchronized.";
        return false;
    }
    const PlanningSnapshotIdentity cur_id = bridge.GetCurrentSnapshotIdentity();
    std::string expected_ee = config_.end_effector_link;
    if (expected_ee.empty()) {
        const int ee_id = bridge.GetDefaultEndEffectorLinkId();
        if (ee_id >= 0 && static_cast<std::size_t>(ee_id) < bridge.GetLinkNames().size()) {
            expected_ee = bridge.GetLinkNames()[static_cast<std::size_t>(ee_id)];
        }
    }
    Eigen::VectorXd q;
    if (!ghost.TryGetExactSolvedGoal(cur_id, expected_ee, &q, out_reason)) {
        return false;
    }
    goal_q_ = std::move(q);
    has_goal_q_ = true;
    MarkConfigMutated();
    return true;
}

bool MotionPlanningSession::PushGoalToIKGhost(InteractiveIKGhostSession* ghost) const {
    if (ghost == nullptr || !has_goal_q_) {
        return false;
    }
    const PlanningSceneBridge& bridge = GetBridge();
    if (!bridge.IsReady()) {
        return false;
    }
    const int ee_id = ResolveEndEffectorLinkId();
    if (ee_id < 0) {
        return false;
    }
    Affine3 ee_pose = Affine3::Identity();
    if (!bridge.ComputeLinkWorldTransform(goal_q_, ee_id, &ee_pose)) {
        return false;
    }
    if (!ghost->IsEnabled()) {
        ghost->SetEnabled(true);
    }
    if (!ghost->GetBridge().GetCurrentSnapshotIdentity().MatchesSnapshot(bridge.GetCurrentSnapshotIdentity()) ||
        !ghost->SetEndEffectorLink(bridge.GetLinkNames().at(ee_id)) || !ghost->HasActiveTarget()) {
        return false;
    }
    ghost->SetTargetPose(ee_pose);
    return true;
}

void MotionPlanningSession::SetStartJointPositions(const Eigen::VectorXd& q) {
    start_q_ = q;
    has_start_q_ = (q.size() > 0);
    MarkConfigMutated();
}

void MotionPlanningSession::SetGoalJointPositions(const Eigen::VectorXd& q) {
    goal_q_ = q;
    has_goal_q_ = (q.size() > 0);
    MarkConfigMutated();
}

bool MotionPlanningSession::StartPlan() {
#if !defined(GOBOT_HAS_PLANNING)
    status_ = MotionPlanningStatus::Unavailable;
    last_status_message_ = "Motion planning requires GOB_BUILD_COAL=ON.";
    return false;
#else
    const PlanningSceneBridge& bridge = GetBridge();
    if (!bridge.IsReady()) {
        status_ = bridge.HasSyncError() ? MotionPlanningStatus::SyncError : MotionPlanningStatus::Unavailable;
        last_status_message_ = bridge.HasSyncError() ? bridge.GetLastSyncError() : "No active robot synchronized.";
        return false;
    }

    const auto snap = bridge.CaptureSnapshot();
    const PlanningSnapshotIdentity snap_id = bridge.GetCurrentSnapshotIdentity();
    if (!snap || !snap_id.IsValid()) {
        status_ = MotionPlanningStatus::SyncError;
        last_status_message_ = "Failed to capture planning snapshot.";
        return false;
    }

    InvalidateActiveRequest();
    strict_ccd_status_ = config_.strict_ccd ? StrictCCDStatus::NotRun : StrictCCDStatus::Disabled;
    const int J = snap->kin.num_joints();
    if (!has_start_q_ || start_q_.size() != J) {
        start_q_ = bridge.GetCurrentJointPositions();
        has_start_q_ = (start_q_.size() == J);
    }
    if (!has_goal_q_ || goal_q_.size() != J) {
        status_ = MotionPlanningStatus::Failed;
        last_status_message_ = "Goal joint state is not set for the current robot.";
        return false;
    }

    PendingPlanRequest req;
    req.active_mask = BuildActiveMask(bridge);
    if (std::none_of(req.active_mask.begin(), req.active_mask.end(), [](auto active) { return active != 0; })) {
        status_ = MotionPlanningStatus::Failed;
        last_status_message_ = "Select at least one active joint on the current robot.";
        return false;
    }
    req.submitted_at = std::chrono::steady_clock::now();
    req.request_seq = ++next_request_seq_;
    req.request_revision = request_revision_;
    req.snapshot_id = snap_id;
    req.cancel_flag = std::make_shared<std::atomic<bool>>(false);
    req.ee_link_id = ResolveEndEffectorLinkId();
    req.start_q = start_q_;
    req.goal_q = goal_q_;
    req.config = config_;
    req.snapshot = snap;

    active_request_seq_ = req.request_seq;
    active_cancel_flag_ = req.cancel_flag;
    status_ = MotionPlanningStatus::Planning;
    last_status_message_ = "Planning...";

    {
        std::lock_guard<std::mutex> lock(mutex_);
        pending_ = std::move(req);
        has_pending_ = true;
    }
    cv_.notify_one();
    return true;
#endif
}

bool MotionPlanningSession::RePlan() {
    // In edit mode, Re-Plan directly re-plans from saved Start, Goal, and the latest WorldSnapshot.
    return StartPlan();
}

void MotionPlanningSession::Stop() {
    const bool was_planning = (active_request_seq_ != 0) || (status_ == MotionPlanningStatus::Planning);
    InvalidateActiveRequest();
    if (was_planning) {
        status_ = MotionPlanningStatus::Cancelled;
        last_status_message_ = "Planning stopped by user.";
    }
}

bool MotionPlanningSession::StartPreview() {
    if (!has_trajectory_) {
        return false;
    }
    preview_active_ = true;
    preview_time_ = 0.0;
    RebuildPreviewOverlayAtCurrentTime();
    return true;
}

void MotionPlanningSession::StopPreview() {
    preview_active_ = false;
    preview_overlay_meshes_.clear();
}

void MotionPlanningSession::RebuildPreviewOverlayAtCurrentTime() {
#if defined(GOBOT_HAS_PLANNING)
    if (!preview_active_ || !has_trajectory_ || !planned_snapshot_) {
        preview_overlay_meshes_.clear();
        return;
    }
    const int J = trajectory_.num_joints();
    if (J <= 0 || trajectory_.num_waypoints() < 2) {
        preview_overlay_meshes_.clear();
        return;
    }
    Eigen::VectorXd q(J);
    trajectory_.sample_single(preview_time_, q, nullptr, nullptr);
    Color albedo;
    Color emissive;
    const bool is_stale = (status_ == MotionPlanningStatus::Stale) || debug_overlay_.is_stale;
    ColorsForPlanPreview(is_stale, &albedo, &emissive);
    preview_overlay_meshes_ = GetBridge().BuildGhostOverlayMeshes(*planned_snapshot_, q, albedo, emissive);
#else
    preview_overlay_meshes_.clear();
#endif
}

std::vector<LinkSphereOverlay> MotionPlanningSession::CollectCurrentLinkSpheres() const {
    const PlanningSceneBridge& bridge = GetBridge();
    if (!bridge.IsReady()) {
        return {};
    }
#if defined(GOBOT_HAS_PLANNING)
    if (preview_active_ && has_trajectory_ && planned_snapshot_ && trajectory_.num_waypoints() >= 2) {
        Eigen::VectorXd q(trajectory_.num_joints());
        trajectory_.sample_single(preview_time_, q, nullptr, nullptr);
        return bridge.CollectLinkSpheres(*planned_snapshot_, q);
    }
#endif
    return bridge.CollectLinkSpheres(bridge.GetCurrentJointPositions());
}

void MotionPlanningSession::Update(const Node* scene_root,
                                   const Robot3D* active_robot,
                                   std::uint64_t scene_epoch,
                                   std::uint64_t scene_change_version,
                                   bool is_play_mode,
                                   double dt_seconds) {
#if !defined(GOBOT_HAS_PLANNING)
    (void)scene_root;
    (void)active_robot;
    (void)scene_epoch;
    (void)scene_change_version;
    (void)is_play_mode;
    (void)dt_seconds;
    status_ = MotionPlanningStatus::Unavailable;
    last_status_message_ = "Motion planning requires GOB_BUILD_COAL=ON.";
    ClearTrajectoryAndPreview();
    return;
#else
    if (is_play_mode || scene_root == nullptr || active_robot == nullptr) {
        InvalidateActiveRequest();
        ClearTrajectoryAndPreview();
        bound_snapshot_id_ = PlanningSnapshotIdentity{};
        has_start_q_ = false;
        has_goal_q_ = false;
        start_q_.resize(0);
        goal_q_.resize(0);
        status_ = MotionPlanningStatus::Unavailable;
        last_status_message_ = is_play_mode ? "Planning disabled during Play Mode." : "No active robot in scene.";
        return;
    }

    PlanningSceneBridge& bridge = GetBridge();
    if (!bridge.EnsureSynced(scene_root, active_robot, scene_epoch, scene_change_version)) {
        InvalidateActiveRequest();
        ClearTrajectoryAndPreview();
        bound_snapshot_id_ = PlanningSnapshotIdentity{};
        status_ = MotionPlanningStatus::SyncError;
        last_status_message_ = bridge.GetLastSyncError().empty()
                ? "Planning scene synchronization failed."
                : bridge.GetLastSyncError();
        return;
    }

    const PlanningSnapshotIdentity cur_id = bridge.GetCurrentSnapshotIdentity();
    if (!cur_id.IsValid()) {
        InvalidateActiveRequest();
        ClearTrajectoryAndPreview();
        bound_snapshot_id_ = PlanningSnapshotIdentity{};
        status_ = MotionPlanningStatus::SyncError;
        last_status_message_ = "Planning snapshot unavailable.";
        return;
    }

    // Detect topology rebuild or robot/scene switch (different scene_epoch, robot_id, or world_generation).
    if (!bound_snapshot_id_.IsValid() || !bound_snapshot_id_.SameTopology(cur_id)) {
        InvalidateActiveRequest();
        ClearTrajectoryAndPreview();
        bound_snapshot_id_ = cur_id;
        start_q_ = bridge.GetCurrentJointPositions();
        has_start_q_ = (start_q_.size() > 0);
        goal_q_.resize(0);
        has_goal_q_ = false;
        status_ = MotionPlanningStatus::Idle;
        last_status_message_.clear();
    } else if (!bound_snapshot_id_.MatchesSnapshot(cur_id)) {
        // Same robot topology, but world_version / snapshot changed (e.g. obstacle or base moved).
        bound_snapshot_id_ = cur_id;
        MarkTrajectoryStale();
    } else if (status_ == MotionPlanningStatus::SyncError || status_ == MotionPlanningStatus::Unavailable) {
        status_ = MotionPlanningStatus::Idle;
        last_status_message_.clear();
    }

    // Drain completed results from worker thread.
    std::deque<CompletedPlanResult> ready_results;
    std::optional<SampledClearanceUpdate> sampled;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        ready_results.swap(completed_queue_);
        sampled.swap(completed_clearance_);
    }

    for (CompletedPlanResult& res : ready_results) {
        // Disqualify any result from a stopped or superseded request (UI-frame immediate disqualification).
        if (res.request_seq < disqualified_before_seq_ || res.request_seq != active_request_seq_) {
            continue;
        }

        active_request_seq_ = 0;
        strict_ccd_status_ = res.strict_ccd;
        last_completed_request_seq_ = res.request_seq;
        last_total_ms_ = res.total_ms;
        last_seed_strategy_ = res.seed_strategy;

        if (res.cancelled) {
            status_ = MotionPlanningStatus::Cancelled;
            last_status_message_ = "Planning stopped by user.";
            continue;
        }

        if (res.timed_out) {
            status_ = MotionPlanningStatus::TimedOut;
            last_status_message_ = res.reason.empty() ? "Planning timed out." : res.reason;
            continue;
        }

        if (!res.success) {
            status_ = MotionPlanningStatus::Failed;
            last_status_message_ = res.reason.empty() ? "Planning failed." : res.reason;
            continue;
        }

        // Reject success if robot topology changed while planning was in flight.
        if (!res.snapshot_id.SameTopology(cur_id)) {
            ClearTrajectoryAndPreview();
            status_ = MotionPlanningStatus::Idle;
            continue;
        }

        trajectory_ = std::move(res.trajectory);
        planned_snapshot_ = std::move(res.snapshot);
        planned_snapshot_id_ = res.snapshot_id;
        planned_request_revision_ = res.request_revision;
        debug_overlay_ = std::move(res.debug_overlay);
        has_trajectory_ = (trajectory_.num_waypoints() >= 2);
        trajectory_waypoints_ = trajectory_.num_waypoints();
        trajectory_spline_order_ = trajectory_.spline_order;
        trajectory_duration_s_ = (trajectory_.times.size() >= 2)
                ? static_cast<double>(trajectory_.times(trajectory_.times.size() - 1) - trajectory_.times(0))
                : 0.0;

        // Ready requires BOTH exact current snapshot identity AND current request revision.
        const bool exact_match = planned_snapshot_id_.MatchesSnapshot(cur_id) &&
                                 planned_request_revision_ == request_revision_;
        if (exact_match) {
            status_ = MotionPlanningStatus::Ready;
            debug_overlay_.is_stale = false;
            last_status_message_ = "Ready";
        } else {
            status_ = MotionPlanningStatus::Stale;
            debug_overlay_.is_stale = true;
            last_status_message_ = "Scene or parameters changed during planning; Re-Plan required.";
        }
        if (preview_active_) {
            preview_time_ = 0.0;
            RebuildPreviewOverlayAtCurrentTime();
        }
    }

    if (sampled && has_trajectory_ && sampled->request_seq == last_completed_request_seq_) {
        const auto& clearance = sampled->clearance;
        debug_overlay_.has_sampled_clearance = clearance.valid;
        debug_overlay_.sampled_mesh_intersection = clearance.mesh_intersection;
        debug_overlay_.sampled_min_distance = clearance.min_distance;
        debug_overlay_.sampled_worst_time = clearance.worst_time;
        debug_overlay_.sampled_shape_pair = std::move(sampled->clearance.shape_pair);
        debug_overlay_.has_witness_points = clearance.has_witness;
        debug_overlay_.witness_a = clearance.witness_a.template cast<RealType>();
        debug_overlay_.witness_b = clearance.witness_b.template cast<RealType>();
    }

    // Advance read-only preview playback if active.
    if (preview_active_ && has_trajectory_ && dt_seconds > 0.0) {
        if (trajectory_duration_s_ > 1e-6) {
            preview_time_ = std::fmod(preview_time_ + dt_seconds, trajectory_duration_s_);
            if (preview_time_ < 0.0) preview_time_ = 0.0;
        } else {
            preview_time_ = 0.0;
        }
        RebuildPreviewOverlayAtCurrentTime();
    }
#endif
}

void MotionPlanningSession::WorkerLoop() {
#if defined(GOBOT_HAS_PLANNING)
    while (true) {
        PendingPlanRequest req;
        WorkerHook before_hook;
        WorkerHook after_hook;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this] { return worker_stop_ || has_pending_; });
            if (worker_stop_) {
                return;
            }
            req = std::move(pending_);
            has_pending_ = false;
            worker_busy_ = true;
            before_hook = before_plan_hook_;
            after_hook = after_plan_hook_;
        }

        CompletedPlanResult out;
        out.request_seq = req.request_seq;
        out.request_revision = req.request_revision;
        out.snapshot_id = req.snapshot_id;
        out.strict_ccd = req.config.strict_ccd ? StrictCCDStatus::NotRun : StrictCCDStatus::Disabled;
        out.snapshot = req.snapshot;

        using Clock = std::chrono::steady_clock;
        const auto deadline = req.submitted_at + std::chrono::duration_cast<Clock::duration>(
                std::chrono::duration<double>(req.config.timeout_s));
        const std::atomic<bool>* cancel_ptr = req.cancel_flag.get();
        if (before_hook) {
            before_hook(req.request_seq, *req.cancel_flag);
        }

        if (cancel_ptr != nullptr && cancel_ptr->load(std::memory_order_relaxed)) {
            out.cancelled = true;
            out.reason = "planning cancelled";
        } else if (Clock::now() >= deadline) {
            out.timed_out = true;
            out.reason = "Planning timed out.";
        } else if (!req.snapshot) {
            out.success = false;
            out.reason = "Invalid planning snapshot";
        } else {
            try {
                planning::MotionPlanRequestNative native_req;
                native_req.start_is_pose = false;
                native_req.start_q = req.start_q;
                native_req.goal_is_pose = false;
                native_req.goal_qs = {req.goal_q};
                native_req.active_mask = req.active_mask;
                native_req.config.waypoints = req.config.waypoints;
                native_req.config.margin = req.config.margin;
                native_req.config.has_world_margin = true;
                native_req.config.world_margin = req.config.world_margin;
                native_req.config.velocity_scale = req.config.velocity_scale;
                native_req.config.acceleration_scale = req.config.acceleration_scale;
                native_req.config.jerk_scale = req.config.jerk_scale;
                native_req.config.spline_order = req.config.spline_order;
                native_req.config.strict_ccd = req.config.strict_ccd;
                native_req.config.use_global_seeding = req.config.use_global_seeding;
                native_req.config.num_seeds = req.config.num_seeds;
                native_req.config.retime = true;

                // Plan directly against the immutable snapshot (no PlanningWorld mutation needed).
                auto dummy_world = std::make_shared<planning::PlanningWorld>();
                planning::MotionPlannerNative planner(dummy_world);
                planning::PlanResultNative res = planner.plan_with_snapshot(
                        *req.snapshot, native_req, &deadline, cancel_ptr);

                out.success = res.success;
                out.cancelled = res.diagnostics.cancelled;
                out.timed_out = res.diagnostics.timed_out;
                if (res.diagnostics.strict_ccd == planning::StrictCCDResult::Passed) {
                    out.strict_ccd = StrictCCDStatus::Passed;
                } else if (res.diagnostics.strict_ccd == planning::StrictCCDResult::Failed) {
                    out.strict_ccd = StrictCCDStatus::Failed;
                }
                out.seed_strategy = res.diagnostics.seed_strategy;
                out.reason = res.reason;

                if (res.success && !out.cancelled &&
                    !(cancel_ptr != nullptr && cancel_ptr->load(std::memory_order_relaxed))) {
                    out.trajectory = std::move(res.trajectory);

                    // Compute debug overlay data from the FINAL retimed spline + planned snapshot.
                    TrajectoryDebugOverlayData dbg;
                    dbg.valid = true;
                    dbg.is_stale = false;

                    const int T = out.trajectory.num_waypoints();
                    const int J = out.trajectory.num_joints();
                    const int ee_id = (req.ee_link_id >= 0 && req.ee_link_id < req.snapshot->kin.num_bodies())
                            ? req.ee_link_id
                            : (req.snapshot->kin.num_bodies() - 1);
                    const int ee_substeps = 8;
                    if (T >= 2 && J == req.snapshot->kin.num_joints() && ee_id >= 0) {
                        std::vector<double> samples;
                        out.trajectory.sample_segments(ee_substeps, samples);
                        const int64_t S = static_cast<int64_t>((T - 1) * ee_substeps + 1);
                        std::vector<planning::Vec3> bpos(static_cast<std::size_t>(req.snapshot->kin.num_bodies()));
                        std::vector<planning::Mat3> brot(static_cast<std::size_t>(req.snapshot->kin.num_bodies()));
                        dbg.ee_path_points.reserve(static_cast<std::size_t>(S));
                        for (int64_t s = 0; s < S; ++s) {
                            if (cancel_ptr != nullptr && cancel_ptr->load(std::memory_order_relaxed)) {
                                out.cancelled = true;
                                out.success = false;
                                out.reason = "planning cancelled";
                                break;
                            }
                            req.snapshot->kin.fk_single(samples.data() + s * J, bpos.data(), brot.data());
                            dbg.ee_path_points.push_back(bpos[static_cast<std::size_t>(ee_id)].template cast<RealType>());
                        }
                    }

                    if (out.success && !out.cancelled) {
                        out.debug_overlay = std::move(dbg);
                    }
                }
            } catch (const std::exception& ex) {
                out.success = false;
                out.reason = ex.what();
            }
        }

        const auto finished = Clock::now();
        out.total_ms = std::chrono::duration<double, std::milli>(finished - req.submitted_at).count();
        if (!out.cancelled && finished >= deadline) {
            out.success = false;
            out.timed_out = true;
            out.reason = "Planning timed out.";
        }
        // Make the validated plan visible before the optional mesh-distance sweep.
        planning::TrajectoryNative diagnostic_trajectory;
        if (out.success) diagnostic_trajectory = out.trajectory;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            completed_queue_.push_back(std::move(out));
        }
        if (after_hook) {
            after_hook(req.request_seq, *req.cancel_flag);
        }
        if (diagnostic_trajectory.num_waypoints() >= 2 && !cancel_ptr->load(std::memory_order_relaxed)) {
            try {
                auto clearance = planning::evaluate_trajectory_sampled_clearance(
                        *req.snapshot, diagnostic_trajectory, req.config.margin, req.config.world_margin,
                        8, true, cancel_ptr);
                if (clearance.valid) {
                    std::lock_guard<std::mutex> lock(mutex_);
                    completed_clearance_ = SampledClearanceUpdate{req.request_seq, std::move(clearance)};
                }
            } catch (const std::exception&) {
                // Missing distance diagnostics leave the independently validated trajectory usable.
            }
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            worker_busy_ = false;
        }
        idle_cv_.notify_all();
    }
#endif
}

void MotionPlanningSession::SetTestWorkerHooks(WorkerHook before_plan, WorkerHook after_plan) {
    std::lock_guard<std::mutex> lock(mutex_);
    before_plan_hook_ = std::move(before_plan);
    after_plan_hook_ = std::move(after_plan);
}

bool MotionPlanningSession::WaitForIdle(std::chrono::milliseconds timeout) {
#if !defined(GOBOT_HAS_PLANNING)
    (void)timeout;
    return true;
#else
    std::unique_lock<std::mutex> lock(mutex_);
    return idle_cv_.wait_for(lock, timeout, [this] { return !has_pending_ && !worker_busy_; });
#endif
}

MotionPlanningDiagnostics MotionPlanningSession::GetDiagnostics() const {
    MotionPlanningDiagnostics d;
    d.status = status_;
    d.strict_ccd_status = strict_ccd_status_;
    d.current_snapshot_id = bound_snapshot_id_;
    d.planned_snapshot_id = planned_snapshot_id_;
    d.request_revision = request_revision_;
    d.planned_request_revision = planned_request_revision_;
    d.active_request_seq = active_request_seq_;
    d.last_completed_request_seq = last_completed_request_seq_;
    d.total_ms = last_total_ms_;
    d.trajectory_duration_s = trajectory_duration_s_;
    d.num_waypoints = trajectory_waypoints_;
    d.spline_order = trajectory_spline_order_;
    d.seed_strategy = last_seed_strategy_;
    d.status_message = last_status_message_;
    d.has_sampled_clearance = debug_overlay_.has_sampled_clearance;
    d.sampled_mesh_intersection = debug_overlay_.sampled_mesh_intersection;
    d.sampled_min_distance = debug_overlay_.sampled_min_distance;
    d.sampled_worst_time = debug_overlay_.sampled_worst_time;
    d.sampled_shape_pair = debug_overlay_.sampled_shape_pair;
    d.has_witness_points = debug_overlay_.has_witness_points;
    return d;
}

} // namespace gobot
