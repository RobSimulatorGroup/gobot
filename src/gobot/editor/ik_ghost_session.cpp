/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/editor/ik_ghost_session.hpp"

#include "gobot/scene/node.hpp"
#include "gobot/scene/robot_3d.hpp"

#if defined(GOBOT_HAS_PLANNING)
#include "planning_cartesian.hpp"
#endif

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace gobot {

namespace {

constexpr std::size_t kMaxLatencySamples = 256;

void ColorsForGhostStatus(IKGhostStatus status, Color* albedo, Color* emissive) {
    switch (status) {
        case IKGhostStatus::Ok:
            *albedo = Color(0.18f, 0.88f, 0.42f, 0.46f);
            *emissive = Color(0.05f, 0.25f, 0.10f, 1.0f);
            break;
        case IKGhostStatus::Collision:
            *albedo = Color(0.98f, 0.56f, 0.10f, 0.54f);
            *emissive = Color(0.32f, 0.14f, 0.02f, 1.0f);
            break;
        case IKGhostStatus::PoseNotSolved:
            *albedo = Color(0.94f, 0.22f, 0.22f, 0.50f);
            *emissive = Color(0.32f, 0.05f, 0.05f, 1.0f);
            break;
        default:
            *albedo = Color(0.50f, 0.72f, 0.96f, 0.36f);
            *emissive = Color(0.08f, 0.14f, 0.22f, 1.0f);
            break;
    }
}

} // namespace

InteractiveIKGhostSession::InteractiveIKGhostSession() {
    StartWorker();
}

InteractiveIKGhostSession::~InteractiveIKGhostSession() {
    StopWorker();
}

bool InteractiveIKGhostSession::IsAvailable() noexcept {
    return PlanningSceneBridge::IsPlanningAvailable();
}

void InteractiveIKGhostSession::StartWorker() {
#if defined(GOBOT_HAS_PLANNING)
    if (worker_.joinable()) {
        return;
    }
    worker_stop_ = false;
    worker_ = std::thread(&InteractiveIKGhostSession::WorkerLoop, this);
#endif
}

void InteractiveIKGhostSession::StopWorker() {
#if defined(GOBOT_HAS_PLANNING)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        worker_stop_ = true;
        has_pending_ = false;
    }
    cv_.notify_all();
    if (worker_.joinable()) {
        worker_.join();
    }
#endif
}

void InteractiveIKGhostSession::SetEnabled(bool enabled) {
    if (!IsAvailable()) {
        enabled_ = false;
        status_ = IKGhostStatus::Unavailable;
        last_status_message_ = "Motion planning requires GOB_BUILD_COAL=ON.";
        return;
    }
    if (enabled_ == enabled) {
        return;
    }
    enabled_ = enabled;
    if (!enabled_) {
        ClearActiveState();
    }
}

void InteractiveIKGhostSession::SetPreviewConfig(const IKGhostPreviewConfig& config) {
    config_ = config;
    if (HasActiveTarget()) {
        InvalidateDisplayedResult();
        SubmitCurrentTarget();
    }
}

void InteractiveIKGhostSession::SetExtraIgnoredPairs(std::vector<std::pair<std::string, std::string>> pairs) {
    extra_ignored_pairs_ = std::move(pairs);
    last_synced_change_version_ = std::numeric_limits<std::uint64_t>::max();
}

bool InteractiveIKGhostSession::SetEndEffectorLink(const std::string& link_name) {
    preferred_ee_link_ = link_name;
    if (!HasActiveTarget()) {
        return false;
    }
    const int id = bridge_.FindLinkId(link_name);
    if (id < 0) {
        return false;
    }
    ee_link_id_ = id;
    ResetTargetToCurrentPose();
    return true;
}

void InteractiveIKGhostSession::InvalidateDisplayedResult() {
    // A reset or policy change can invalidate an in-flight solve even on the same snapshot.
    first_valid_target_seq_ = next_target_seq_ + 1;
    displayed_target_seq_ = 0;
    status_ = IKGhostStatus::Solving;
    overlay_meshes_.clear();
    last_collision_pair_.clear();
    last_status_message_.clear();
    last_pos_err_ = 0.0;
    last_rot_err_ = 0.0;
    last_min_distance_ = std::numeric_limits<double>::infinity();
}

void InteractiveIKGhostSession::ClearActiveState() {
    InvalidateDisplayedResult();
    has_target_pose_ = false;
    ghost_q_.resize(0);
    status_ = IsAvailable() ? IKGhostStatus::Inactive : IKGhostStatus::Unavailable;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        has_pending_ = false;
        has_completed_ = false;
    }
}

void InteractiveIKGhostSession::Update(const Node* scene_root,
                                       const Robot3D* active_robot,
                                       std::uint64_t scene_epoch,
                                       std::uint64_t scene_change_version,
                                       bool is_play_mode) {
#if !defined(GOBOT_HAS_PLANNING)
    (void)scene_root;
    (void)active_robot;
    (void)scene_epoch;
    (void)scene_change_version;
    (void)is_play_mode;
    status_ = IKGhostStatus::Unavailable;
    overlay_meshes_.clear();
    return;
#else
    if (!enabled_) return;
    if (is_play_mode || scene_root == nullptr || active_robot == nullptr) {
        if (bound_robot_id_.IsValid()) {
            ClearActiveState();
            bridge_.Clear();
            bound_robot_id_ = ObjectID{};
            ee_link_id_ = -1;
            last_synced_world_version_ = 0;
        }
        bound_scene_epoch_ = scene_epoch;
        last_status_message_ = is_play_mode ? "Paused during Play Mode." : "No active robot.";
        return;
    }

    const ObjectID robot_id = active_robot->GetInstanceId();
    const bool context_switched = (bound_scene_epoch_ != scene_epoch) || (bound_robot_id_ != robot_id);
    const bool needs_sync = context_switched || !bridge_.IsReady() ||
                            (last_synced_change_version_ != scene_change_version);

    if (needs_sync) {
        if (context_switched) {
            ClearActiveState();
            bridge_.Clear();
            bound_scene_epoch_ = scene_epoch;
            bound_robot_id_ = robot_id;
            ee_link_id_ = -1;
            last_synced_world_version_ = 0;
        }
        std::string sync_error;
        if (!bridge_.SyncFromScene(scene_root, active_robot, "", extra_ignored_pairs_, &sync_error)) {
            ClearActiveState();
            status_ = IKGhostStatus::Unavailable;
            last_status_message_ = sync_error;
            return;
        }
        last_synced_change_version_ = scene_change_version;

        int resolved_ee = bridge_.GetDefaultEndEffectorLinkId();
        if (!preferred_ee_link_.empty()) {
            if (const int idx = bridge_.FindLinkId(preferred_ee_link_); idx >= 0) {
                resolved_ee = idx;
            }
        }
        ee_link_id_ = resolved_ee;

        const std::uint64_t new_world_ver = bridge_.GetWorldVersion();
        const bool world_changed = (new_world_ver != last_synced_world_version_);
        last_synced_world_version_ = new_world_ver;

        if (bridge_.GetLastSyncStats().rebuilt_world) {
            // A replacement world may have the same local version and joint count.
            ClearActiveState();
        } else if (world_changed && has_target_pose_) {
            InvalidateDisplayedResult();
            SubmitCurrentTarget();
        }
    }
    if (!has_target_pose_) {
        ResetTargetToCurrentPose();
    }

    CompletedResult res;
    bool have_result = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (has_completed_) {
            res = std::move(completed_);
            has_completed_ = false;
            have_result = true;
        }
    }

    if (have_result &&
        res.scene_epoch == bound_scene_epoch_ &&
        res.robot_id == bound_robot_id_ &&
        res.target_seq >= first_valid_target_seq_ &&
        res.target_seq > displayed_target_seq_ &&
        res.world_version == last_synced_world_version_ &&
        res.snapshot == bridge_.CaptureSnapshot()) {
        displayed_target_seq_ = res.target_seq;
        status_ = res.status;
        last_pos_err_ = res.pos_err;
        last_rot_err_ = res.rot_err;
        last_min_distance_ = res.min_distance;
        last_solve_ms_ = res.solve_ms;
        last_collision_pair_ = res.collision_pair;
        last_status_message_ = res.reason;

        if (res.q.size() == bridge_.GetCurrentJointPositions().size()) {
            ghost_q_ = res.q;
            RebuildOverlayMeshesFromResult(res);
        }

        const auto now = std::chrono::steady_clock::now();
        const double e2e_ms = std::chrono::duration<double, std::milli>(now - res.submitted_at).count();
        RecordLatencySample(e2e_ms);
    }
#endif
}

void InteractiveIKGhostSession::ResetTargetToCurrentPose() {
#if !defined(GOBOT_HAS_PLANNING)
    return;
#else
    if (!enabled_ || !bridge_.IsReady() || ee_link_id_ < 0) {
        return;
    }
    ghost_q_ = bridge_.GetCurrentJointPositions();
    Affine3 ee_tf = Affine3::Identity();
    if (!bridge_.ComputeLinkWorldTransform(ghost_q_, ee_link_id_, &ee_tf)) {
        return;
    }
    InvalidateDisplayedResult();
    target_pose_ = ee_tf;
    has_target_pose_ = true;
    SubmitCurrentTarget();
#endif
}

void InteractiveIKGhostSession::SetTargetPose(const Affine3& world_pose) {
    if (!HasActiveTarget()) {
        return;
    }
    target_pose_ = world_pose;
    has_target_pose_ = true;
    SubmitCurrentTarget();
}

void InteractiveIKGhostSession::SubmitCurrentTarget() {
#if !defined(GOBOT_HAS_PLANNING)
    return;
#else
    const auto snap = bridge_.CaptureSnapshot();
    if (!snap || ee_link_id_ < 0 || ee_link_id_ >= snap->kin.num_bodies()) {
        return;
    }

    const int J = snap->kin.num_joints();
    Eigen::VectorXd seed = (ghost_q_.size() == J) ? ghost_q_ : bridge_.GetCurrentJointPositions();
    if (seed.size() != J) {
        seed = Eigen::VectorXd::Zero(J);
    }

    std::vector<Eigen::VectorXd> fallbacks;
    const Eigen::VectorXd& cur_q = bridge_.GetCurrentJointPositions();
    if (cur_q.size() == J && (cur_q - seed).cwiseAbs().maxCoeff() > 1e-3) {
        fallbacks.push_back(cur_q);
    }
    Eigen::VectorXd mid_q(J);
    const auto& lo = snap->kin.lower();
    const auto& hi = snap->kin.upper();
    for (int j = 0; j < J; ++j) {
        if (std::isfinite(lo(j)) && std::isfinite(hi(j))) {
            mid_q(j) = 0.5 * (lo(j) + hi(j));
        } else {
            mid_q(j) = 0.0;
        }
    }
    if ((mid_q - seed).cwiseAbs().maxCoeff() > 1e-3) {
        fallbacks.push_back(std::move(mid_q));
    }

    PendingRequest req;
    req.scene_epoch = bound_scene_epoch_;
    req.world_version = snap->version;
    req.target_seq = ++next_target_seq_;
    req.robot_id = bound_robot_id_;
    req.link_id = ee_link_id_;
    req.target_pose = target_pose_;
    req.seed_q = std::move(seed);
    req.fallback_seeds = std::move(fallbacks);
    req.config = config_;
    req.submitted_at = std::chrono::steady_clock::now();
    req.snapshot = snap;

    {
        std::lock_guard<std::mutex> lock(mutex_);
        pending_ = std::move(req);  // latest-wins: overwrite any older unstarted drag pose
        has_pending_ = true;
    }
    cv_.notify_one();
#endif
}

void InteractiveIKGhostSession::WorkerLoop() {
#if defined(GOBOT_HAS_PLANNING)
    while (true) {
        PendingRequest req;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this] { return worker_stop_ || has_pending_; });
            if (worker_stop_) {
                return;
            }
            req = std::move(pending_);
            has_pending_ = false;
            worker_busy_ = true;
        }

        const auto t0 = std::chrono::steady_clock::now();
        CompletedResult out;
        out.scene_epoch = req.scene_epoch;
        out.world_version = req.world_version;
        out.target_seq = req.target_seq;
        out.robot_id = req.robot_id;
        out.submitted_at = req.submitted_at;
        out.snapshot = req.snapshot;

        if (req.snapshot && req.link_id >= 0 && req.link_id < req.snapshot->kin.num_bodies()) {
            const int J = req.snapshot->kin.num_joints();
            std::vector<uint8_t> active_mask(static_cast<std::size_t>(J), 1);

            planning::PreviewIKConfigNative pcfg;
            pcfg.pos_tol = req.config.pos_tol;
            pcfg.rot_tol = req.config.rot_tol;
            pcfg.margin = req.config.margin;
            pcfg.world_margin = req.config.world_margin;
            pcfg.ik_iterations = req.config.ik_iterations;
            pcfg.ik_damping = req.config.ik_damping;
            pcfg.avoid_collisions = req.config.avoid_collisions;
            pcfg.verify_original = req.config.verify_original;
            pcfg.check_rotation = req.config.check_rotation;

            const planning::Vec3 target_pos = req.target_pose.translation().template cast<double>();
            const planning::Mat3 target_rot = req.target_pose.linear().template cast<double>();

            const planning::PreviewIKResultNative ik = planning::solve_ik_preview_on_snapshot(
                    *req.snapshot,
                    req.link_id,
                    planning::Vec3::Zero(),
                    planning::Mat3::Identity(),
                    target_pos,
                    target_rot,
                    req.seed_q,
                    active_mask,
                    pcfg,
                    req.fallback_seeds);

            out.q = ik.q;
            out.pos_err = ik.pos_err;
            out.rot_err = ik.rot_err;
            out.min_distance = ik.min_distance;
            out.collision_pair = ik.collision_pair;
            out.reason = ik.reason;
            if (ik.status == planning::PreviewIKStatus::Ok) {
                out.status = IKGhostStatus::Ok;
            } else if (ik.status == planning::PreviewIKStatus::Collision) {
                out.status = IKGhostStatus::Collision;
            } else {
                out.status = IKGhostStatus::PoseNotSolved;
            }
        } else {
            out.status = IKGhostStatus::PoseNotSolved;
            out.reason = "Invalid planning snapshot";
        }
        const auto t1 = std::chrono::steady_clock::now();
        out.solve_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        {
            std::lock_guard<std::mutex> lock(mutex_);
            // Only publish if newer than any previously completed unread result.
            if (!has_completed_ || out.target_seq >= completed_.target_seq) {
                completed_ = std::move(out);
                has_completed_ = true;
            }
            worker_busy_ = false;
        }
        idle_cv_.notify_all();
    }
#endif
}

void InteractiveIKGhostSession::RebuildOverlayMeshesFromResult(const CompletedResult& res) {
#if defined(GOBOT_HAS_PLANNING)
    Color albedo;
    Color emissive;
    ColorsForGhostStatus(res.status, &albedo, &emissive);
    if (res.snapshot) {
        overlay_meshes_ = bridge_.BuildGhostOverlayMeshes(*res.snapshot, res.q, albedo, emissive);
    } else {
        overlay_meshes_ = bridge_.BuildGhostOverlayMeshes(res.q, albedo, emissive);
    }
#else
    (void)res;
    overlay_meshes_.clear();
#endif
}

void InteractiveIKGhostSession::RecordLatencySample(double ms) {
    last_input_to_display_ms_ = ms;
    latency_samples_ms_.push_back(ms);
    if (latency_samples_ms_.size() > kMaxLatencySamples) {
        latency_samples_ms_.pop_front();
    }
}

bool InteractiveIKGhostSession::WaitForIdle(std::chrono::milliseconds timeout) {
#if !defined(GOBOT_HAS_PLANNING)
    (void)timeout;
    return true;
#else
    std::unique_lock<std::mutex> lock(mutex_);
    return idle_cv_.wait_for(lock, timeout, [this] { return !has_pending_ && !worker_busy_; });
#endif
}

IKGhostDiagnostics InteractiveIKGhostSession::GetDiagnostics() const {
    IKGhostDiagnostics d;
    d.status = status_;
    d.scene_epoch = bound_scene_epoch_;
    d.world_version = last_synced_world_version_;
    d.target_seq = next_target_seq_;
    d.displayed_seq = displayed_target_seq_;
    d.pos_err = last_pos_err_;
    d.rot_err = last_rot_err_;
    d.min_distance = last_min_distance_;
    d.last_solve_ms = last_solve_ms_;
    d.last_input_to_display_ms = last_input_to_display_ms_;
    d.collision_pair = last_collision_pair_;
    d.status_message = last_status_message_;
    if (ee_link_id_ >= 0 && static_cast<std::size_t>(ee_link_id_) < bridge_.GetLinkNames().size()) {
        d.end_effector_link = bridge_.GetLinkNames()[static_cast<std::size_t>(ee_link_id_)];
    }
    if (!latency_samples_ms_.empty()) {
        std::vector<double> sorted(latency_samples_ms_.begin(), latency_samples_ms_.end());
        std::sort(sorted.begin(), sorted.end());
        const std::size_t idx = std::min(sorted.size() - 1,
                                         static_cast<std::size_t>(std::ceil(0.95 * static_cast<double>(sorted.size()))) - 1);
        d.p95_input_to_display_ms = sorted[idx];
    }
    return d;
}

} // namespace gobot
