#pragma once

#include "planning_cartesian.hpp"
#include "planning_manifold.hpp"
#include "planning_retime.hpp"
#include "planning_sqp.hpp"
#include "planning_world.hpp"

#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace gobot::planning {

struct SolverConfigNative {
    int waypoints = 16;
    double margin = 0.002;
    bool has_world_margin = false;
    double world_margin = 0.002;
    int collision_substeps = 8;
    double max_segment_length = 0.0;  // <= 0 means disabled
    int iterations = 40;
    bool stop_at_endpoints = false;
    bool first_feasible = true;
    bool use_global_seeding = true;
    int num_seeds = 1;
    double rrt_timeout_s = 0.35;
    double rrt_step_size = 0.25;
    int rrt_max_iterations = 4000;
    int rrt_shortcut_iters = 64;
    double velocity_scale = 1.0;
    double acceleration_scale = 1.0;
    double jerk_scale = 1.0;
    int spline_order = 3;
    bool strict_ccd = false;
    bool has_dt = false;
    double dt = 0.0;
    bool retime = true;

    double effective_world_margin() const {
        return has_world_margin ? world_margin : margin;
    }
};

struct MotionPlanRequestNative {
    // Start: either joint config or pose constraint targets.
    bool start_is_pose = false;
    Eigen::VectorXd start_q;
    std::vector<PoseConstraintSpec> start_poses;
    bool has_start_seed = false;
    Eigen::VectorXd start_seed;

    // Goal: one or more joint configs, or one or more pose constraint target sets (GoalSet).
    bool goal_is_pose = false;
    std::vector<Eigen::VectorXd> goal_qs;
    std::vector<std::vector<PoseConstraintSpec>> goal_pose_sets;
    bool has_goal_seed = false;
    Eigen::VectorXd goal_seed;

    // Optional waypoint constraints, global manifold constraints, initial trajectory seed, and active joint mask.
    std::vector<std::pair<int, std::vector<PoseConstraintSpec>>> pose_constraints;
    std::vector<PoseConstraintSpec> manifold_constraints;
    std::vector<std::pair<int, Eigen::VectorXd>> joint_targets;
    bool has_seed = false;
    RowMatrixXd seed;
    std::string seed_strategy_label;  // optional label override for user seed (e.g. "warm_start")
    std::vector<uint8_t> active_mask;  // empty => all joints active

    // Optional initial velocity / quintic acceleration boundary constraints at t = 0.
    bool has_start_velocity = false;
    Eigen::VectorXd start_velocity;
    bool has_start_acceleration = false;
    Eigen::VectorXd start_acceleration;
    double dt_start_smooth = 0.0;  // optional virtual-predecessor dt hint for SQP smoothness

    SolverConfigNative config;
};

struct ReplanRequestNative {
    Eigen::VectorXd q_now;
    Eigen::VectorXd dq_now;
    bool has_ddq_now = false;
    Eigen::VectorXd ddq_now;

    bool has_prev_traj = false;
    TrajectoryNative prev_traj;
    bool has_t_on_prev = false;
    double t_on_prev = 0.0;
    bool reuse_if_valid = true;

    // Optional new goal. When omitted (!has_goal) and has_prev_traj is true, keeps prev_traj's final joint config.
    bool has_goal = false;
    bool goal_is_pose = false;
    std::vector<Eigen::VectorXd> goal_qs;
    std::vector<std::vector<PoseConstraintSpec>> goal_pose_sets;
    bool has_goal_seed = false;
    Eigen::VectorXd goal_seed;

    std::vector<std::pair<int, std::vector<PoseConstraintSpec>>> pose_constraints;
    std::vector<PoseConstraintSpec> manifold_constraints;
    std::vector<std::pair<int, Eigen::VectorXd>> joint_targets;
    std::vector<uint8_t> active_mask;  // empty => all joints active

    SolverConfigNative config;
    double time_budget_s = 0.0;  // <= 0 disables total deadline
};

struct PlanDiagnosticsNative {
    double total_ms = 0.0;
    double ik_ms = 0.0;
    double seed_ms = 0.0;
    double sqp_ms = 0.0;
    double validate_ms = 0.0;
    double retime_ms = 0.0;
    int qp_calls = 0;
    int sqp_iterations = 0;
    bool timed_out = false;
    std::string seed_strategy = "straight";
    std::vector<double> witnesses;  // flat [K * 6]
};

struct PlanResultNative {
    bool success = false;
    TrajectoryNative trajectory;
    bool has_initial_positions = false;
    RowMatrixXd initial_positions;
    std::vector<int> waypoint_indices;  // maps original waypoint index k -> row index in trajectory.positions
    int goal_index = 0;
    uint64_t world_version = 0;
    std::string reason;
    PlanDiagnosticsNative diagnostics;
};

struct FinalizeResultNative {
    bool success = false;
    TrajectoryNative trajectory;
    std::vector<int> waypoint_indices;  // maps original positions_in row k -> row index in trajectory.positions
    double retime_ms = 0.0;
    std::string reason;
};

namespace planner_detail {

// A raw BiRRT detour has no interior request targets. Associate the free request
// slots with the nearest dense knots at uniform joint-space arc length.
inline std::vector<int> dense_waypoint_indices(const RowMatrixXd& path, int count, int fixed_start, int fixed_end,
                                               const std::vector<int>& retimed_indices) {
    const int last = static_cast<int>(path.rows()) - 1;
    std::vector<double> length(last + 1, 0.0);
    for (int i = 1; i <= last; ++i) {
        length[i] = length[i - 1] + (path.row(i) - path.row(i - 1)).norm();
    }
    std::vector<int> result(count, retimed_indices.front());
    std::fill(result.end() - fixed_end, result.end(), retimed_indices.back());
    const int free_count = count - fixed_start - fixed_end;
    for (int k = fixed_start; k < count - fixed_end; ++k) {
        const double target = length.back() * (k - fixed_start + 1) / (free_count + 1);
        int i = static_cast<int>(std::lower_bound(length.begin(), length.end(), target) - length.begin());
        if (i > 0 && target - length[i - 1] < length[i] - target) --i;
        result[k] = retimed_indices[i];
    }
    return result;
}

inline RowMatrixXd linear_seed(const Eigen::VectorXd& start, const Eigen::VectorXd& goal,
                               int waypoints, int fixed_start, int fixed_end,
                               const std::vector<std::pair<int, Eigen::VectorXd>>& joint_targets) {
    const int J = static_cast<int>(start.size());
    std::vector<std::pair<int, Eigen::VectorXd>> anchors;
    anchors.emplace_back(fixed_start - 1, start);
    for (const auto& [idx, jt] : joint_targets) {
        if (idx >= fixed_start && idx < waypoints - fixed_end) anchors.emplace_back(idx, jt);
    }
    std::sort(anchors.begin(), anchors.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    anchors.emplace_back(waypoints - fixed_end, goal);

    RowMatrixXd out(waypoints, J);
    for (int t = 0; t < fixed_start; ++t) out.row(t) = start.transpose();
    for (int t = waypoints - fixed_end; t < waypoints; ++t) out.row(t) = goal.transpose();
    for (size_t k = 0; k + 1 < anchors.size(); ++k) {
        const int i0 = anchors[k].first;
        const int i1 = anchors[k + 1].first;
        const Eigen::VectorXd& a = anchors[k].second;
        const Eigen::VectorXd& b = anchors[k + 1].second;
        const int span = std::max(1, i1 - i0);
        for (int t = i0; t <= i1; ++t) {
            if (t >= fixed_start && t < waypoints - fixed_end) {
                const double frac = static_cast<double>(t - i0) / static_cast<double>(span);
                out.row(t) = ((1.0 - frac) * a + frac * b).transpose();
            }
        }
    }
    return out;
}

struct TrajectoryValidationResult {
    bool ok = false;
    std::string reason;
    std::set<int> bad_segments;
    std::set<int> manifold_bad_segments;
};

// Unified verification pipeline applied identically to retimed trajectories and reused suffixes:
// joint position/velocity/acceleration/(quintic jerk) bounds, discrete convex/mesh/ESDF clearance,
// kinematic-chain continuous CCD (convex + ESDF), and continuous manifold checks.
inline TrajectoryValidationResult validate_trajectory_atomic(
    const WorldSnapshot& snap,
    const TrajectoryNative& traj,
    const SolverConfigNative& cfg,
    const Eigen::VectorXd& vmax,
    const Eigen::VectorXd& amax,
    const Eigen::VectorXd& jmax,
    const Eigen::VectorXd& lo,
    const Eigen::VectorXd& hi,
    const std::vector<PoseConstraintSpec>* manifold_constraints,
    std::vector<double>& samples,
    std::vector<double>& frames,
    std::vector<int64_t>& screen_rows,
    std::vector<int64_t>& screen_pairs,
    std::vector<int64_t>& viol_rows,
    std::vector<int64_t>& viol_pairs) {
    TrajectoryValidationResult res;
    const int T = traj.num_waypoints();
    const int J = traj.num_joints();
    if (T < 2 || J != snap.kin.num_joints()) {
        res.reason = "invalid trajectory dimensions";
        return res;
    }

    // 1. Exact/certified continuous position, velocity, acceleration & quintic jerk verification.
    auto bchk = check_trajectory_bounds(
        traj, vmax, amax, true, true, 1e-5, &lo, &hi, &jmax, traj.spline_order == 5);
    if (!bchk.ok) {
        res.reason = bchk.reason;
        res.bad_segments.insert(bchk.bad_segments.begin(), bchk.bad_segments.end());
    }

    // 2. Sample actual spline curve and check discrete & continuous CCD.
    int substeps = std::max(2 * cfg.collision_substeps, 4);
    if (cfg.max_segment_length > 0.0 && T >= 2) {
        double longest = 0.0;
        for (int i = 0; i + 1 < T; ++i) {
            longest = std::max(longest, (traj.positions.row(i + 1) - traj.positions.row(i)).norm());
        }
        substeps = std::max(substeps, static_cast<int>(std::ceil(longest / cfg.max_segment_length)));
    }

    traj.sample_segments(substeps, samples);
    const int64_t S = static_cast<int64_t>((T - 1) * substeps + 1);
    for (int64_t s = 0; s < S; ++s) {
        const double* qs = samples.data() + s * J;
        for (int j = 0; j < J; ++j) {
            if (!std::isfinite(qs[j]) || qs[j] < lo(j) - 1e-6 || qs[j] > hi(j) + 1e-6) {
                res.bad_segments.insert(std::min(static_cast<int>(s / substeps), T - 2));
                if (res.reason.empty()) res.reason = "retimed spline curve violates joint limits";
                break;
            }
        }
    }

    const int P = static_cast<int>(snap.pairs.size());
    const bool use_esdf = snap.has_esdf();
    const bool has_col = (P > 0) || use_esdf;
    const double eff_wm = cfg.effective_world_margin();
    const int N_shapes = static_cast<int>(snap.shapes.size());
    const Eigen::VectorXd margins = (P > 0) ? snap.pair_margins(cfg.margin, eff_wm) : Eigen::VectorXd();

    if (res.bad_segments.empty() && has_col) {
        frames.resize(static_cast<size_t>(S * N_shapes * 12));
        snap.compute_frames(samples.data(), S, frames.data());

        const int n_passes = (P > 0 && snap.has_mesh) ? 2 : 1;
        for (int pass = 0; pass < n_passes; ++pass) {
            const bool verify_orig = (pass == 1);
            const char* kind = verify_orig ? "mesh" : "convex";
            if (P > 0) {
                snap.screen_aabbs(frames.data(), S, margins.data(), verify_orig, substeps, screen_rows, screen_pairs);
                if (!screen_rows.empty()) {
                    snap.violating_pairs_indexed(frames.data(), S, screen_rows.data(), screen_pairs.data(),
                                                 static_cast<int64_t>(screen_rows.size()), margins.data(), verify_orig,
                                                 viol_rows, viol_pairs);
                    if (!viol_rows.empty()) {
                        const int seg0 = std::min(static_cast<int>(viol_rows[0] / substeps), T - 2);
                        res.reason = std::string("retimed Hermite ") + kind + " clearance failed on segment " +
                                     std::to_string(seg0) + ", pair " + snap.pair_display_name(viol_pairs[0]);
                        for (const int64_t r : viol_rows) {
                            res.bad_segments.insert(std::min(static_cast<int>(r / substeps), T - 2));
                        }
                        break;
                    }
                }
            }
            if (!verify_orig && use_esdf) {
                snap.violating_esdf_samples(samples.data(), frames.data(), S, eff_wm, viol_rows, viol_pairs);
                if (!viol_rows.empty()) {
                    const int seg0 = std::min(static_cast<int>(viol_rows[0] / substeps), T - 2);
                    res.reason = std::string("retimed Hermite ESDF clearance failed on segment ") +
                                 std::to_string(seg0) + ", pair " + snap.pair_display_name(viol_pairs[0]);
                    for (const int64_t r : viol_rows) {
                        res.bad_segments.insert(std::min(static_cast<int>(r / substeps), T - 2));
                    }
                    break;
                }
            }
            if (!verify_orig) {
                auto ccd_hit = snap.first_violation_continuous(
                    traj.positions.data(), T, substeps, frames.data(), margins.data(),
                    false, 1e-3, traj.times.data(), traj.velocities.data(),
                    cfg.strict_ccd, traj.accelerations.data(), traj.spline_order, eff_wm);
                if (ccd_hit.has_value()) {
                    const char* ckind = (ccd_hit->pair_id < 0) ? "ESDF" : kind;
                    res.reason = std::string("retimed Hermite continuous ") + ckind + " clearance failed on segment " +
                                 std::to_string(ccd_hit->seg) + ", pair " + snap.pair_display_name(ccd_hit->pair_id);
                    res.bad_segments.insert(ccd_hit->seg);
                    break;
                }
            }
        }
    }

    // 3. Certified continuous manifold verification along the actual played cubic/quintic spline.
    const bool has_manifold = (manifold_constraints != nullptr && !manifold_constraints->empty());
    if (res.bad_segments.empty() && has_manifold) {
        const auto mchk = verify_spline_manifold_constraints(snap, traj, *manifold_constraints, false);
        if (!mchk.ok) {
            res.reason = mchk.reason;
            for (const int s : mchk.bad_segments) {
                res.bad_segments.insert(s);
                res.manifold_bad_segments.insert(s);
            }
        }
    }

    res.ok = res.reason.empty() && res.bad_segments.empty();
    return res;
}

}  // namespace planner_detail

// Unified Hermite retiming + continuous collision, bounds & manifold verification on an immutable WorldSnapshot.
// Used identically by single plan(), batch plan_batch(), replan(), and Python's MuJoCo Warp path.
inline FinalizeResultNative finalize_trajectory(
    const WorldSnapshot& snap,
    const RowMatrixXd& positions_in,
    const SolverConfigNative& cfg,
    const std::vector<PoseConstraintSpec>* manifold_constraints = nullptr,
    const std::vector<uint8_t>* active_mask = nullptr,
    const Eigen::VectorXd* start_velocity = nullptr,
    const Eigen::VectorXd* start_acceleration = nullptr,
    const std::chrono::steady_clock::time_point* deadline = nullptr) {
    using Clock = std::chrono::steady_clock;
    const auto t0 = Clock::now();
    FinalizeResultNative out;

    const int J = static_cast<int>(positions_in.cols());
    if (!std::isfinite(cfg.velocity_scale) || cfg.velocity_scale <= 0.0) {
        throw std::invalid_argument("velocity_scale must be finite and positive");
    }
    if (!std::isfinite(cfg.acceleration_scale) || cfg.acceleration_scale <= 0.0) {
        throw std::invalid_argument("acceleration_scale must be finite and positive");
    }
    if (!std::isfinite(cfg.jerk_scale) || cfg.jerk_scale <= 0.0) {
        throw std::invalid_argument("jerk_scale must be finite and positive");
    }
    const bool has_manifold = (manifold_constraints != nullptr && !manifold_constraints->empty());
    if (has_manifold) {
        manifold_detail::validate_manifold_specs(snap, *manifold_constraints);
    }
    if (cfg.spline_order != 3 && cfg.spline_order != 5) {
        throw std::invalid_argument("spline_order must be 3 (cubic) or 5 (quintic)");
    }
    if (start_acceleration != nullptr && cfg.spline_order != 5) {
        throw std::invalid_argument("start_acceleration is only supported for spline_order == 5 (quintic)");
    }
    const Eigen::VectorXd vmax = snap.kin.velocity_limit() * cfg.velocity_scale;
    const Eigen::VectorXd amax = snap.kin.acceleration_limit() * cfg.acceleration_scale;
    const Eigen::VectorXd jmax = snap.kin.jerk_limit() * cfg.jerk_scale;
    const Eigen::VectorXd& lo = snap.kin.lower();
    const Eigen::VectorXd& hi = snap.kin.upper();
    const bool has_dt = cfg.has_dt || !cfg.retime;
    const double dt_val = cfg.has_dt ? cfg.dt : 0.1;
    const int spline_order = cfg.spline_order;
    const bool lock_knot0 = (start_velocity != nullptr) || (start_acceleration != nullptr);
    std::vector<uint8_t> mask(static_cast<size_t>(J), 1);
    if (active_mask != nullptr && static_cast<int>(active_mask->size()) == J) {
        mask = *active_mask;
    }

    const int P = static_cast<int>(snap.pairs.size());
    const bool use_esdf = snap.has_esdf();
    const bool has_col = (P > 0) || use_esdf;
    const double eff_wm = cfg.effective_world_margin();
    const Eigen::VectorXd margins = (P > 0) ? snap.pair_margins(cfg.margin, eff_wm) : Eigen::VectorXd();

    RowMatrixXd positions = positions_in;
    std::vector<int> wp_indices(static_cast<size_t>(positions_in.rows()));
    for (int k = 0; k < static_cast<int>(positions_in.rows()); ++k) {
        wp_indices[static_cast<size_t>(k)] = k;
    }
    std::string last_error;
    std::vector<double> samples, frames;
    std::vector<int64_t> screen_rows, screen_pairs, viol_rows, viol_pairs;

    const int max_refine_rounds = (!has_dt && (has_manifold || lock_knot0)) ? 4 : 0;
    bool tried_seg0_brake = false;
    for (int refine = 0; refine <= max_refine_rounds; ++refine) {
        const int T = static_cast<int>(positions.rows());
        Eigen::VectorXd knot_scales = Eigen::VectorXd::Ones(T);
        std::set<int> manifold_bad_segs;
        std::set<int> last_bad_segs;

        for (int attempt = 0; attempt < 8; ++attempt) {
            if (deadline != nullptr && Clock::now() >= *deadline) {
                out.retime_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
                out.reason = "replan time budget exceeded";
                return out;
            }
            TrajectoryNative traj = retime_detail::retime_candidate(
                positions, vmax, amax, cfg.stop_at_endpoints, has_dt, dt_val, &knot_scales, &jmax, spline_order,
                start_velocity, start_acceleration, &lo, &hi);

            auto vres = planner_detail::validate_trajectory_atomic(
                snap, traj, cfg, vmax, amax, jmax, lo, hi, manifold_constraints,
                samples, frames, screen_rows, screen_pairs, viol_rows, viol_pairs);

            // Verify fixed start boundary preservation (never accept a dropped boundary).
            if (vres.ok && start_velocity != nullptr &&
                (traj.velocities.row(0).transpose() - *start_velocity).lpNorm<Eigen::Infinity>() > 1e-8) {
                vres.ok = false;
                vres.reason = "retimed trajectory failed to preserve start_velocity";
                vres.bad_segments.insert(0);
            }
            if (vres.ok && start_acceleration != nullptr &&
                (traj.accelerations.row(0).transpose() - *start_acceleration).lpNorm<Eigen::Infinity>() > 1e-8) {
                vres.ok = false;
                vres.reason = "retimed trajectory failed to preserve start_acceleration";
                vres.bad_segments.insert(0);
            }

            if (vres.ok) {
                out.success = true;
                out.trajectory = std::move(traj);
                out.waypoint_indices = std::move(wp_indices);
                out.retime_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
                return out;
            }
            last_error = vres.reason;
            manifold_bad_segs = std::move(vres.manifold_bad_segments);
            last_bad_segs = vres.bad_segments;
            if (has_dt) {
                if (vres.bad_segments.empty()) {
                    out.retime_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
                    out.reason = last_error;
                    return out;
                }
            }

            // Damp knot velocities around offending segments (never damp knot 0 when start boundary is fixed).
            bool changed = false;
            for (const int seg : vres.bad_segments) {
                int knots_pair[2] = {seg, seg + 1};
                int count = 2;
                if (has_dt && !cfg.stop_at_endpoints) {
                    int filtered[2];
                    int fc = 0;
                    for (int k = 0; k < 2; ++k) {
                        if (knots_pair[k] > 0 && knots_pair[k] < T - 1) filtered[fc++] = knots_pair[k];
                    }
                    if (fc > 0) {
                        count = fc;
                        for (int k = 0; k < fc; ++k) knots_pair[k] = filtered[k];
                    }
                }
                for (int k = 0; k < count; ++k) {
                    const int knot = knots_pair[k];
                    if (lock_knot0 && knot == 0) continue;
                    const double old = knot_scales(knot);
                    if (old > 0.0) {
                        knot_scales(knot) = (attempt >= 2 || old <= 0.25) ? 0.0 : (old * 0.5);
                        changed = true;
                    }
                }
            }
            if (!changed) {
                const int start_i = lock_knot0 ? 1 : 0;
                if (!has_dt && start_i < T && (knot_scales.tail(T - start_i).array() > 0.0).any()) {
                    knot_scales.tail(T - start_i).setZero();
                } else {
                    break;
                }
            }
        }

        // If segment 0 failed under a fixed non-zero start_velocity due to Hermite overshoot,
        // try inserting a minimum-distance constant-deceleration braking knot right after q0
        // (optionally blended laterally toward q1) so segment 0 stops cleanly and segment 1 turns from v=0.
        if (lock_knot0 && !tried_seg0_brake && !has_dt && !has_manifold &&
            start_velocity != nullptr && start_velocity->squaredNorm() > 1e-12 &&
            last_bad_segs.count(0) > 0 && T < 512 && refine < max_refine_rounds) {
            tried_seg0_brake = true;
            double t_brake = 0.0;
            for (int j = 0; j < J; ++j) {
                if (std::isfinite(amax(j)) && amax(j) > 1e-9) {
                    t_brake = std::max(t_brake, std::abs((*start_velocity)(j)) / amax(j));
                }
            }
            if (t_brake > 1e-4) {
                const Eigen::VectorXd q0 = positions.row(0).transpose();
                const Eigen::VectorXd q1 = positions.row(1).transpose();
                const Eigen::VectorXd d_brake = 0.5 * t_brake * (*start_velocity);
                const double betas[4] = {0.25, 0.5, 0.0, 0.75};
                for (double beta : betas) {
                    Eigen::VectorXd q_mid = q0 + d_brake + beta * (q1 - q0);
                    for (int j = 0; j < J; ++j) {
                        if (!mask[static_cast<size_t>(j)]) q_mid(j) = q0(j);
                        q_mid(j) = std::clamp(q_mid(j), lo(j), hi(j));
                    }
                    if (!has_col || snap.is_config_valid(q_mid.data(), margins, true, eff_wm)) {
                        RowMatrixXd next_pos(T + 1, J);
                        next_pos.row(0) = positions.row(0);
                        next_pos.row(1) = q_mid.transpose();
                        next_pos.bottomRows(T - 1) = positions.bottomRows(T - 1);
                        for (int& idx : wp_indices) {
                            if (idx >= 1) idx += 1;
                        }
                        positions = std::move(next_pos);
                        goto continue_refine_loop;
                    }
                }
            }
        }

        // If continuous manifold verification still failed at zero knot velocity (nonlinear FK chord bow-out),
        // insert projected midpoint knots on offending segments in free-timing mode and re-retime!
        if (!manifold_bad_segs.empty() && !has_dt && refine < max_refine_rounds && T < 512) {
            std::vector<Eigen::VectorXd> next_rows;
            next_rows.reserve(static_cast<size_t>(T) + manifold_bad_segs.size());
            std::vector<int> old_to_new(static_cast<size_t>(T), 0);
            bool inserted = false;
            for (int k = 0; k < T - 1; ++k) {
                old_to_new[static_cast<size_t>(k)] = static_cast<int>(next_rows.size());
                next_rows.push_back(positions.row(k).transpose());
                if (manifold_bad_segs.count(k) > 0) {
                    const Eigen::VectorXd qa = positions.row(k).transpose();
                    const Eigen::VectorXd qb = positions.row(k + 1).transpose();
                    const Eigen::VectorXd q_seed = 0.5 * (qa + qb);
                    Eigen::VectorXd q_mid;
                    const double max_d = std::max(0.25, (qb - qa).norm());
                    if (manifold_detail::project_onto_manifold(
                            snap, *manifold_constraints, mask, q_seed, q_mid, max_d, 0.25, 25) &&
                        (!has_col || snap.is_config_valid(q_mid.data(), margins, true, eff_wm))) {
                        next_rows.push_back(std::move(q_mid));
                        inserted = true;
                    }
                }
            }
            old_to_new[static_cast<size_t>(T - 1)] = static_cast<int>(next_rows.size());
            next_rows.push_back(positions.row(T - 1).transpose());
            if (inserted) {
                RowMatrixXd next_pos(static_cast<Eigen::Index>(next_rows.size()), J);
                for (size_t i = 0; i < next_rows.size(); ++i) {
                    next_pos.row(static_cast<Eigen::Index>(i)) = next_rows[i].transpose();
                }
                for (int& idx : wp_indices) {
                    idx = old_to_new[static_cast<size_t>(idx)];
                }
                positions = std::move(next_pos);
                continue;
            }
        }
        break;
        continue_refine_loop:;
    }

    out.retime_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    out.reason = last_error.empty() ? "retimed Hermite validation failed" : last_error;
    return out;
}

// Standalone C++ MotionPlanner operating on thread-safe immutable PlanningWorld snapshots.
// Can be called directly from C++ (e.g. gobot_editor, C++ tests) or via Python bindings.
class MotionPlannerNative {
public:
    explicit MotionPlannerNative(std::shared_ptr<PlanningWorld> world) : world_(std::move(world)) {
        if (!world_) throw std::invalid_argument("MotionPlannerNative requires a non-null PlanningWorld");
    }

    std::shared_ptr<PlanningWorld> world() const { return world_; }

    IKResultNative solve_ik(const std::vector<PoseConstraintSpec>& targets,
                            const Eigen::VectorXd* seed = nullptr,
                            const std::vector<uint8_t>* active_mask = nullptr,
                            double margin = 0.002,
                            std::optional<double> world_margin = std::nullopt,
                            int iterations = 40,
                            int num_seeds = 1,
                            bool verify_original = true,
                            uint64_t rng_seed = 0) const {
        auto snap = world_->snapshot();
        return solve_ik_on_snapshot(*snap, targets, seed, active_mask, margin,
                                    world_margin.value_or(margin), iterations, num_seeds, verify_original, rng_seed);
    }

    PlanResultNative plan(const MotionPlanRequestNative& req) const {
        auto snap = world_->snapshot();
        return plan_on_snapshot(*snap, req);
    }

    PlanResultNative plan_with_snapshot(const WorldSnapshot& snap, const MotionPlanRequestNative& req) const {
        return plan_on_snapshot(snap, req);
    }

    PlanResultNative replan(const ReplanRequestNative& req) const {
        auto snap = world_->snapshot();
        return replan_on_snapshot(*snap, req);
    }

    PlanResultNative replan_with_snapshot(const WorldSnapshot& snap, const ReplanRequestNative& req) const {
        return replan_on_snapshot(snap, req);
    }

    CartesianPlanResultNative plan_cartesian_path(const CartesianPlanRequestNative& req) const {
        auto snap = world_->snapshot();
        return plan_cartesian_path_on_snapshot(*snap, req);
    }

    std::vector<PlanResultNative> plan_batch(const std::vector<MotionPlanRequestNative>& requests,
                                             int chunk_size = 64) const {
        if (requests.empty()) throw std::invalid_argument("provide at least one MotionPlanRequest");
        if (chunk_size < 1) throw std::invalid_argument("chunk_size must be >= 1");
        auto snap = world_->snapshot();
        const int B = static_cast<int>(requests.size());
        std::vector<PlanResultNative> results(static_cast<size_t>(B));

        for (int offset = 0; offset < B; offset += chunk_size) {
            const int end = std::min(offset + chunk_size, B);
            const int count = end - offset;
            // Solve all items in this chunk concurrently with dynamic work-stealing across TBB/threads.
            std::atomic<int> next_idx{0};
            const int workers = std::min<int>(count, static_cast<int>(ThreadPool::instance().max_threads()));
            ThreadPool::instance().parallel_for(workers, 1, [&](int64_t, int64_t) {
                while (true) {
                    const int loc = next_idx.fetch_add(1, std::memory_order_relaxed);
                    if (loc >= count) break;
                    results[static_cast<size_t>(offset + loc)] = plan_on_snapshot(*snap, requests[static_cast<size_t>(offset + loc)]);
                }
            });
        }
        return results;
    }

private:
    static std::vector<uint8_t> resolve_mask(const WorldSnapshot& snap, const std::vector<uint8_t>* mask_in) {
        const int J = snap.kin.num_joints();
        if (mask_in != nullptr && static_cast<int>(mask_in->size()) == J) return *mask_in;
        return std::vector<uint8_t>(static_cast<size_t>(J), 1);
    }

    static IKResultNative solve_ik_on_snapshot(const WorldSnapshot& snap,
                                               const std::vector<PoseConstraintSpec>& targets,
                                               const Eigen::VectorXd* seed,
                                               const std::vector<uint8_t>* active_mask,
                                               double margin, double world_margin,
                                               int iterations, int num_seeds,
                                               bool verify_original, uint64_t rng_seed) {
        if (targets.empty()) return IKResultNative{{}, false, "provide at least one pose constraint"};
        const int J = snap.kin.num_joints();
        const auto mask = resolve_mask(snap, active_mask);
        const Eigen::VectorXd base_seed = (seed != nullptr && seed->size() == J) ? *seed : snap.kin.initial_q();
        std::vector<IKProblem> problems;
        problems.reserve(static_cast<size_t>(std::max(1, num_seeds)));
        problems.push_back({targets, base_seed});
        if (num_seeds > 1) {
            std::mt19937_64 rng(rng_seed);
            const auto& lo = snap.kin.lower();
            const auto& hi = snap.kin.upper();
            for (int k = 1; k < num_seeds; ++k) {
                Eigen::VectorXd s = base_seed;
                for (int j = 0; j < J; ++j) {
                    if (!mask[static_cast<size_t>(j)]) continue;
                    const double lj = std::isfinite(lo(j)) ? lo(j) : (base_seed(j) - M_PI);
                    const double hj = std::isfinite(hi(j)) ? hi(j) : (base_seed(j) + M_PI);
                    std::uniform_real_distribution<double> dist(lj, hj);
                    s(j) = dist(rng);
                }
                problems.push_back({targets, std::move(s)});
            }
        }
        auto batch = snap.solve_ik_batch(problems, mask, margin, world_margin, iterations, verify_original);
        int best = -1;
        double best_d2 = std::numeric_limits<double>::infinity();
        for (int i = 0; i < static_cast<int>(batch.size()); ++i) {
            if (!batch[static_cast<size_t>(i)].success) continue;
            const double d2 = (batch[static_cast<size_t>(i)].q - base_seed).squaredNorm();
            if (d2 < best_d2) { best_d2 = d2; best = i; }
        }
        if (best < 0) return IKResultNative{{}, false, batch.empty() ? "IK failed" : batch[0].reason};
        return batch[static_cast<size_t>(best)];
    }

    static std::string validate_start_boundary(const WorldSnapshot& snap,
                                               const SolverConfigNative& cfg,
                                               const std::vector<uint8_t>& mask,
                                               const Eigen::VectorXd* start_velocity,
                                               const Eigen::VectorXd* start_acceleration) {
        const int J = snap.kin.num_joints();
        if (start_acceleration != nullptr && cfg.spline_order != 5) {
            throw std::invalid_argument("start_acceleration (ddq_now) is only supported for spline_order == 5 (quintic)");
        }
        const Eigen::VectorXd vmax = snap.kin.velocity_limit() * cfg.velocity_scale;
        const Eigen::VectorXd amax = snap.kin.acceleration_limit() * cfg.acceleration_scale;
        if (start_velocity != nullptr) {
            if (static_cast<int>(start_velocity->size()) != J) {
                throw std::invalid_argument("start_velocity size must match num_joints");
            }
            for (int j = 0; j < J; ++j) {
                const double vj = (*start_velocity)(j);
                if (!std::isfinite(vj)) {
                    return "start_velocity contains non-finite value on joint " + std::to_string(j);
                }
                if (!mask[static_cast<size_t>(j)] && std::abs(vj) > 1e-9) {
                    return "boundary conflict: non-zero start_velocity on locked joint " + std::to_string(j);
                }
                if (std::isfinite(vmax(j)) && std::abs(vj) > vmax(j) * (1.0 + 1e-6)) {
                    return "start_velocity exceeds joint velocity limit on joint " + std::to_string(j);
                }
            }
        }
        if (start_acceleration != nullptr) {
            if (static_cast<int>(start_acceleration->size()) != J) {
                throw std::invalid_argument("start_acceleration size must match num_joints");
            }
            for (int j = 0; j < J; ++j) {
                const double aj = (*start_acceleration)(j);
                if (!std::isfinite(aj)) {
                    return "start_acceleration contains non-finite value on joint " + std::to_string(j);
                }
                if (!mask[static_cast<size_t>(j)] && std::abs(aj) > 1e-9) {
                    return "boundary conflict: non-zero start_acceleration on locked joint " + std::to_string(j);
                }
                if (std::isfinite(amax(j)) && std::abs(aj) > amax(j) * (1.0 + 1e-6)) {
                    return "start_acceleration exceeds joint acceleration limit on joint " + std::to_string(j);
                }
                if (start_velocity != nullptr && std::isfinite(vmax(j))) {
                    const double vj = (*start_velocity)(j);
                    if (std::abs(vj) >= vmax(j) * (1.0 - 1e-9) && vj * aj > 1e-9) {
                        return "infeasible start boundary: start_acceleration exceeds velocity limit at t=0 on joint " +
                               std::to_string(j);
                    }
                }
            }
        }
        return {};
    }

    static std::string validate_endpoints(const WorldSnapshot& snap,
                                          const Eigen::VectorXd& start_q,
                                          const Eigen::VectorXd& goal_q,
                                          const SolverConfigNative& cfg, int fixed_start, int fixed_end,
                                          const std::vector<PoseConstraintSpec>& manifold_constraints) {
        const int J = snap.kin.num_joints();
        const auto& lo = snap.kin.lower();
        const auto& hi = snap.kin.upper();
        for (int j = 0; j < J; ++j) {
            if (!std::isfinite(start_q(j)) || start_q(j) < lo(j) || start_q(j) > hi(j) ||
                !std::isfinite(goal_q(j)) || goal_q(j) < lo(j) || goal_q(j) > hi(j)) {
                return "start or goal violates joint limits";
            }
        }
        if (!manifold_constraints.empty()) {
            if (!manifold_detail::is_on_manifold(snap, start_q.data(), manifold_constraints, 1.0)) {
                return "start configuration violates manifold constraint";
            }
            if (!manifold_detail::is_on_manifold(snap, goal_q.data(), manifold_constraints, 1.0)) {
                return "goal configuration violates manifold constraint";
            }
        }
        if (cfg.has_dt) {
            const int moving = std::max(1, cfg.waypoints - fixed_start - fixed_end + 1);
            const Eigen::VectorXd vmax = snap.kin.velocity_limit() * cfg.velocity_scale;
            for (int j = 0; j < J; ++j) {
                if (std::isfinite(vmax(j)) &&
                    std::abs(goal_q(j) - start_q(j)) / (static_cast<double>(moving) * cfg.dt) > vmax(j) * (1.0 + 1e-6)) {
                    return "joint velocity limits require a longer duration";
                }
            }
        }
        if (snap.has_collision_checks()) {
            const double eff_wm = cfg.effective_world_margin();
            const Eigen::VectorXd margins = snap.pair_margins(cfg.margin, eff_wm);
            std::vector<double> ends(static_cast<size_t>(2 * J));
            std::copy(start_q.data(), start_q.data() + J, ends.data());
            std::copy(goal_q.data(), goal_q.data() + J, ends.data() + J);
            std::vector<double> frames(static_cast<size_t>(2 * snap.shapes.size() * 12));
            snap.compute_frames(ends.data(), 2, frames.data());
            if (!snap.pairs.empty()) {
                std::vector<int64_t> rows, pids;
                const int n_passes = snap.has_mesh ? 2 : 1;
                for (int pass = 0; pass < n_passes; ++pass) {
                    const bool verify = (pass == 1);
                    snap.screen_aabbs(frames.data(), 2, margins.data(), verify, 1, rows, pids);
                    if (!rows.empty()) {
                        auto hit = snap.first_violation_indexed(frames.data(), 2, rows.data(), pids.data(),
                                                                static_cast<int64_t>(rows.size()), margins.data(), verify);
                        if (hit.has_value()) {
                            const int seg = (hit->first == 0) ? 0 : (cfg.waypoints - 2);
                            return "endpoint collision on segment " + std::to_string(seg) + ", pair " + snap.pair_display_name(hit->second);
                        }
                    }
                }
            }
            if (snap.has_esdf()) {
                auto ehit = snap.first_esdf_violation_samples(ends.data(), frames.data(), 2, eff_wm);
                if (ehit.has_value()) {
                    const int seg = (ehit->first == 0) ? 0 : (cfg.waypoints - 2);
                    return "endpoint ESDF collision on segment " + std::to_string(seg) + ", pair " + snap.pair_display_name(ehit->second);
                }
            }
        }
        return {};
    }

    static std::vector<SQPJobSpecNative> build_seed_jobs(
        const WorldSnapshot& snap,
        const Eigen::VectorXd& start_q,
        const std::vector<std::pair<int, Eigen::VectorXd>>& goal_candidates,
        const MotionPlanRequestNative& req,
        const std::vector<uint8_t>& mask,
        int fixed_start,
        int fixed_end,
        bool force_rrt,
        double& out_seed_ms,
        std::vector<std::pair<int, RowMatrixXd>>* out_dense_rrt = nullptr,
        const std::chrono::steady_clock::time_point* deadline = nullptr) {
        using Clock = std::chrono::steady_clock;
        const auto t0 = Clock::now();
        const auto& cfg = req.config;
        const int J = snap.kin.num_joints();
        const double dt_smooth = cfg.has_dt ? cfg.dt : 0.25;
        bool has_interior_anchors = false;
        for (const auto& [idx, _] : req.joint_targets) {
            if (idx >= fixed_start && idx < cfg.waypoints - fixed_end) { has_interior_anchors = true; break; }
        }

        auto make_spec = [&](RowMatrixXd init_q, const std::string& strategy, int goal_idx) {
            SQPJobSpecNative sp;
            sp.initial = std::move(init_q);
            sp.fixed = fixed_end;
            sp.fixed_start = fixed_start;
            sp.fixed_end = fixed_end;
            sp.has_dt_vel = cfg.has_dt;
            sp.dt_vel = cfg.dt;
            sp.dt_smooth = dt_smooth;
            sp.dt_start_smooth = req.dt_start_smooth;
            sp.has_start_velocity = req.has_start_velocity;
            if (req.has_start_velocity) sp.start_velocity = req.start_velocity;
            sp.active_mask = mask;
            sp.pose_constraints = req.pose_constraints;
            if (!req.manifold_constraints.empty()) {
                for (int t = fixed_start; t < cfg.waypoints - fixed_end; ++t) {
                    bool found = false;
                    for (auto& [wt, vec] : sp.pose_constraints) {
                        if (wt == t) {
                            vec.insert(vec.end(), req.manifold_constraints.begin(), req.manifold_constraints.end());
                            found = true;
                            break;
                        }
                    }
                    if (!found) {
                        sp.pose_constraints.emplace_back(t, req.manifold_constraints);
                    }
                }
            }
            sp.joint_targets = req.joint_targets;
            sp.margin = cfg.margin;
            sp.world_margin = cfg.effective_world_margin();
            sp.collision_substeps = cfg.collision_substeps;
            sp.max_segment_length = cfg.max_segment_length;
            sp.iterations = cfg.iterations;
            sp.first_feasible = cfg.first_feasible;
            sp.strict_ccd = cfg.strict_ccd;
            sp.velocity_scale = cfg.velocity_scale;
            sp.acceleration_scale = cfg.acceleration_scale;
            if (deadline != nullptr) {
                sp.has_deadline = true;
                sp.deadline = *deadline;
            }
            sp.seed_strategy = strategy;
            sp.goal_index = goal_idx;
            return sp;
        };

        std::vector<SQPJobSpecNative> specs;
        if (req.has_seed && !force_rrt) {
            RowMatrixXd q0 = req.seed;
            const auto& lo = snap.kin.lower();
            const auto& hi = snap.kin.upper();
            for (int t = 0; t < fixed_start; ++t) q0.row(t) = start_q.transpose();
            for (int t = cfg.waypoints - fixed_end; t < cfg.waypoints; ++t) q0.row(t) = goal_candidates[0].second.transpose();
            for (int t = fixed_start; t < cfg.waypoints - fixed_end; ++t) {
                for (int j = 0; j < J; ++j) q0(t, j) = std::clamp(q0(t, j), lo(j), hi(j));
            }
            for (int j = 0; j < J; ++j) {
                if (!mask[static_cast<size_t>(j)]) q0.col(j).setConstant(start_q(j));
            }
            const std::string label = req.seed_strategy_label.empty() ? "user_seed" : req.seed_strategy_label;
            specs.push_back(make_spec(std::move(q0), label, goal_candidates[0].first));
            out_seed_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
            return specs;
        }

        if (!force_rrt) {
            const int n_lin = std::min(static_cast<int>(goal_candidates.size()), std::max(1, cfg.num_seeds));
            for (int i = 0; i < n_lin; ++i) {
                auto q0 = planner_detail::linear_seed(start_q, goal_candidates[static_cast<size_t>(i)].second,
                                                      cfg.waypoints, fixed_start, fixed_end, req.joint_targets);
                specs.push_back(make_spec(std::move(q0), "straight", goal_candidates[static_cast<size_t>(i)].first));
            }
        }

        const bool need_rrt = force_rrt || (cfg.use_global_seeding && cfg.num_seeds > 1);
        if (need_rrt && !has_interior_anchors) {
            double rrt_budget_s = cfg.rrt_timeout_s;
            if (deadline != nullptr) {
                const double rem_s = std::chrono::duration<double>(*deadline - Clock::now()).count();
                if (rem_s <= 0.0) {
                    out_seed_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
                    return specs;
                }
                // Reserve ~35% of remaining budget for SQP polishing & Hermite verification.
                rrt_budget_s = std::min(rrt_budget_s, std::max(0.001, rem_s * 0.65));
            }
            const int n_rrt = force_rrt ? std::max(1, cfg.num_seeds) : std::max(1, cfg.num_seeds - static_cast<int>(specs.size()));
            const int start_pad = fixed_start - 1;
            const int end_pad = fixed_end - 1;
            const int inner_wp = cfg.waypoints - start_pad - end_pad;
            std::vector<RRTResultNative> rrt_out(static_cast<size_t>(n_rrt));
            const bool has_manifold = !req.manifold_constraints.empty();
            ThreadPool::instance().parallel_for(n_rrt, 1, [&](int64_t b, int64_t e) {
                for (int64_t i = b; i < e; ++i) {
                    const auto& [gid, gq] = goal_candidates[static_cast<size_t>(i) % goal_candidates.size()];
                    const uint64_t sd = 1001ULL + static_cast<uint64_t>(i) * 7919ULL;
                    rrt_out[static_cast<size_t>(i)] = has_manifold
                        ? plan_constrained_birrt(
                              snap, start_q, gq, req.manifold_constraints, mask,
                              cfg.margin, cfg.effective_world_margin(),
                              cfg.rrt_step_size, cfg.rrt_max_iterations, rrt_budget_s,
                              sd, snap.has_mesh, cfg.rrt_shortcut_iters, inner_wp)
                        : snap.plan_birrt(
                              start_q, gq, mask, cfg.margin, cfg.effective_world_margin(),
                              cfg.rrt_step_size, cfg.rrt_max_iterations, rrt_budget_s,
                              sd, snap.has_mesh, cfg.rrt_shortcut_iters, inner_wp);
                }
            });
            for (int i = 0; i < n_rrt; ++i) {
                const bool have_dense = !rrt_out[static_cast<size_t>(i)].dense_waypoints.empty();
                if (!rrt_out[static_cast<size_t>(i)].success && !have_dense) continue;
                const auto& [gid, gq] = goal_candidates[static_cast<size_t>(i) % goal_candidates.size()];
                if (out_dense_rrt != nullptr && rrt_out[static_cast<size_t>(i)].dense_waypoints.size() >= 2) {
                    const auto& dw = rrt_out[static_cast<size_t>(i)].dense_waypoints;
                    RowMatrixXd dense_mat(static_cast<Eigen::Index>(dw.size()), J);
                    for (size_t k = 0; k < dw.size(); ++k) {
                        dense_mat.row(static_cast<Eigen::Index>(k)) = dw[k].transpose();
                    }
                    out_dense_rrt->emplace_back(gid, std::move(dense_mat));
                }
                if (static_cast<int>(rrt_out[static_cast<size_t>(i)].waypoints.size()) != inner_wp) continue;
                RowMatrixXd path(cfg.waypoints, J);
                for (int t = 0; t < start_pad; ++t) path.row(t) = start_q.transpose();
                for (int t = 0; t < inner_wp; ++t) {
                    path.row(start_pad + t) = rrt_out[static_cast<size_t>(i)].waypoints[static_cast<size_t>(t)].transpose();
                }
                for (int t = cfg.waypoints - end_pad; t < cfg.waypoints; ++t) path.row(t) = gq.transpose();
                specs.push_back(make_spec(std::move(path), has_manifold ? "constrained_birrt" : "birrt", gid));
            }
        }

        if (!force_rrt && static_cast<int>(specs.size()) < cfg.num_seeds && !has_interior_anchors) {
            std::mt19937_64 rng(42);
            std::normal_distribution<double> norm(0.0, 0.35);
            const auto& lo = snap.kin.lower();
            const auto& hi = snap.kin.upper();
            const int mid_idx = cfg.waypoints / 2;
            while (static_cast<int>(specs.size()) < cfg.num_seeds) {
                const auto& [gid, gq] = goal_candidates[specs.size() % goal_candidates.size()];
                Eigen::VectorXd mid = 0.5 * (start_q + gq);
                for (int j = 0; j < J; ++j) {
                    if (!mask[static_cast<size_t>(j)]) continue;
                    const double lj = std::isfinite(lo(j)) ? lo(j) : (start_q(j) - 1.5);
                    const double hj = std::isfinite(hi(j)) ? hi(j) : (start_q(j) + 1.5);
                    mid(j) = std::clamp(mid(j) + norm(rng), lj, hj);
                }
                std::vector<std::pair<int, Eigen::VectorXd>> mid_anchor{{mid_idx, mid}};
                auto q0 = planner_detail::linear_seed(start_q, gq, cfg.waypoints, fixed_start, fixed_end, mid_anchor);
                specs.push_back(make_spec(std::move(q0), "perturbed", gid));
            }
        }

        out_seed_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        return specs;
    }

    static PlanResultNative plan_on_snapshot(
        const WorldSnapshot& snap,
        const MotionPlanRequestNative& req,
        const std::chrono::steady_clock::time_point* deadline = nullptr) {
        using Clock = std::chrono::steady_clock;
        const auto t_start = Clock::now();
        PlanResultNative res;
        res.world_version = snap.version;
        const auto& cfg = req.config;
        if (cfg.spline_order != 3 && cfg.spline_order != 5) {
            throw std::invalid_argument("spline_order must be 3 (cubic) or 5 (quintic)");
        }
        if (!std::isfinite(cfg.velocity_scale) || cfg.velocity_scale <= 0.0) {
            throw std::invalid_argument("velocity_scale must be finite and positive");
        }
        if (!std::isfinite(cfg.acceleration_scale) || cfg.acceleration_scale <= 0.0) {
            throw std::invalid_argument("acceleration_scale must be finite and positive");
        }
        if (!std::isfinite(cfg.jerk_scale) || cfg.jerk_scale <= 0.0) {
            throw std::invalid_argument("jerk_scale must be finite and positive");
        }
        const bool has_start_bc = req.has_start_velocity || req.has_start_acceleration;
        const int fixed_start = has_start_bc ? 1 : (cfg.stop_at_endpoints ? 2 : 1);
        const int fixed_end = cfg.stop_at_endpoints ? 2 : 1;
        if (cfg.waypoints < fixed_start + fixed_end) {
            throw std::invalid_argument("waypoints must be >= fixed_start + fixed_end");
        }

        const int J = snap.kin.num_joints();
        const auto mask = resolve_mask(snap, req.active_mask.empty() ? nullptr : &req.active_mask);
        const double wm = cfg.effective_world_margin();
        const Eigen::VectorXd* v0_ptr = req.has_start_velocity ? &req.start_velocity : nullptr;
        const Eigen::VectorXd* a0_ptr = req.has_start_acceleration ? &req.start_acceleration : nullptr;

        {
            const std::string bc_err = validate_start_boundary(snap, cfg, mask, v0_ptr, a0_ptr);
            if (!bc_err.empty()) {
                res.reason = bc_err;
                res.diagnostics.total_ms = std::chrono::duration<double, std::milli>(Clock::now() - t_start).count();
                return res;
            }
        }

        if (!req.manifold_constraints.empty()) {
            manifold_detail::validate_manifold_specs(snap, req.manifold_constraints);
        }

        auto deadline_exceeded = [&]() -> bool {
            return deadline != nullptr && Clock::now() >= *deadline;
        };

        // 1. Resolve start & goal endpoints (with C++ IK / parallel GoalSet IK, including manifold constraints).
        const auto t_ik0 = Clock::now();
        Eigen::VectorXd start_q;
        if (req.start_is_pose) {
            auto start_targets = req.start_poses;
            start_targets.insert(start_targets.end(), req.manifold_constraints.begin(), req.manifold_constraints.end());
            auto s_ik = solve_ik_on_snapshot(snap, start_targets, req.has_start_seed ? &req.start_seed : nullptr,
                                             &mask, cfg.margin, wm, 40, std::max(1, cfg.num_seeds), true, 0);
            if (!s_ik.success) {
                res.reason = "start IK failed: " + s_ik.reason;
                res.diagnostics.ik_ms = std::chrono::duration<double, std::milli>(Clock::now() - t_ik0).count();
                res.diagnostics.total_ms = std::chrono::duration<double, std::milli>(Clock::now() - t_start).count();
                return res;
            }
            start_q = std::move(s_ik.q);
        } else {
            start_q = req.start_q;
        }

        std::vector<std::pair<int, Eigen::VectorXd>> goal_candidates;
        if (req.goal_is_pose) {
            const Eigen::VectorXd& g_seed = req.has_goal_seed ? req.goal_seed : start_q;
            if (req.goal_pose_sets.size() == 1) {
                auto goal_targets = req.goal_pose_sets[0];
                goal_targets.insert(goal_targets.end(), req.manifold_constraints.begin(), req.manifold_constraints.end());
                auto g_ik = solve_ik_on_snapshot(snap, goal_targets, &g_seed, &mask,
                                                 cfg.margin, wm, 40, std::max(1, cfg.num_seeds), true, 0);
                if (!g_ik.success) {
                    res.reason = "goal IK failed: " + g_ik.reason;
                    res.diagnostics.ik_ms = std::chrono::duration<double, std::milli>(Clock::now() - t_ik0).count();
                    res.diagnostics.total_ms = std::chrono::duration<double, std::milli>(Clock::now() - t_start).count();
                    return res;
                }
                goal_candidates.emplace_back(0, std::move(g_ik.q));
            } else {
                std::vector<IKProblem> probs;
                probs.reserve(req.goal_pose_sets.size());
                for (const auto& ts : req.goal_pose_sets) {
                    auto merged = ts;
                    merged.insert(merged.end(), req.manifold_constraints.begin(), req.manifold_constraints.end());
                    probs.push_back({std::move(merged), g_seed});
                }
                auto ik_batch = snap.solve_ik_batch(probs, mask, cfg.margin, wm, 40, true);
                for (int i = 0; i < static_cast<int>(ik_batch.size()); ++i) {
                    if (ik_batch[static_cast<size_t>(i)].success) {
                        goal_candidates.emplace_back(i, std::move(ik_batch[static_cast<size_t>(i)].q));
                    }
                }
                if (goal_candidates.empty()) {
                    res.reason = "all " + std::to_string(req.goal_pose_sets.size()) + " GoalSet candidates failed IK";
                    res.diagnostics.ik_ms = std::chrono::duration<double, std::milli>(Clock::now() - t_ik0).count();
                    res.diagnostics.total_ms = std::chrono::duration<double, std::milli>(Clock::now() - t_start).count();
                    return res;
                }
                std::sort(goal_candidates.begin(), goal_candidates.end(), [&](const auto& a, const auto& b) {
                    return (a.second - start_q).squaredNorm() < (b.second - start_q).squaredNorm();
                });
            }
        } else {
            for (int i = 0; i < static_cast<int>(req.goal_qs.size()); ++i) {
                goal_candidates.emplace_back(i, req.goal_qs[static_cast<size_t>(i)]);
            }
            if (goal_candidates.size() > 1) {
                std::sort(goal_candidates.begin(), goal_candidates.end(), [&](const auto& a, const auto& b) {
                    return (a.second - start_q).squaredNorm() < (b.second - start_q).squaredNorm();
                });
            }
        }
        for (auto& [gid, gq] : goal_candidates) {
            for (int j = 0; j < J; ++j) if (!mask[static_cast<size_t>(j)]) gq(j) = start_q(j);
        }
        const double ik_ms = std::chrono::duration<double, std::milli>(Clock::now() - t_ik0).count();

        // 2. Filter valid endpoints.
        std::vector<std::pair<int, Eigen::VectorXd>> valid_goals;
        std::string last_ep_err;
        for (const auto& [gid, gq] : goal_candidates) {
            std::string err = validate_endpoints(snap, start_q, gq, cfg, fixed_start, fixed_end, req.manifold_constraints);
            if (err.empty()) valid_goals.emplace_back(gid, gq);
            else last_ep_err = std::move(err);
        }
        if (valid_goals.empty()) {
            res.reason = last_ep_err.empty() ? "no valid goal endpoints" : last_ep_err;
            res.diagnostics.ik_ms = ik_ms;
            res.diagnostics.total_ms = std::chrono::duration<double, std::milli>(Clock::now() - t_start).count();
            return res;
        }

        if (deadline_exceeded()) {
            res.diagnostics.timed_out = true;
            res.diagnostics.ik_ms = ik_ms;
            res.diagnostics.total_ms = std::chrono::duration<double, std::milli>(Clock::now() - t_start).count();
            res.reason = "replan time budget exceeded";
            return res;
        }

        // 3. Build seed jobs and run SQP (with optional multi-seed racing).
        double seed_ms = 0.0;
        std::vector<std::pair<int, RowMatrixXd>> initial_dense_rrt;
        auto specs = build_seed_jobs(snap, start_q, valid_goals, req, mask, fixed_start, fixed_end,
                                     false, seed_ms, &initial_dense_rrt, deadline);
        auto sqp_results = solve_sqp_batch(snap, specs, specs.size() > 1 && cfg.first_feasible);

        int winner_idx = -1;
        double winner_cost = std::numeric_limits<double>::infinity();
        bool any_timed_out = false;
        for (int i = 0; i < static_cast<int>(sqp_results.size()); ++i) {
            if (sqp_results[static_cast<size_t>(i)].timed_out) any_timed_out = true;
            if (sqp_results[static_cast<size_t>(i)].success && sqp_results[static_cast<size_t>(i)].cost < winner_cost) {
                winner_cost = sqp_results[static_cast<size_t>(i)].cost;
                winner_idx = i;
            }
        }

        const bool can_raw_rrt_fallback =
            (!req.manifold_constraints.empty() || has_start_bc) &&
            req.pose_constraints.empty() && req.joint_targets.empty();
        const bool fixed_timing = cfg.has_dt || !cfg.retime;
        const auto* mc_ptr = req.manifold_constraints.empty() ? nullptr : &req.manifold_constraints;

        auto try_raw_rrt_fallback = [&](const std::vector<std::pair<int, RowMatrixXd>>& dense_list,
                                        const std::vector<SQPJobSpecNative>& spec_list) -> bool {
            if (!can_raw_rrt_fallback || deadline_exceeded()) return false;
            const std::string fb_strat = req.manifold_constraints.empty() ? "birrt" : "constrained_birrt";
            // In fixed-timing mode, never change waypoint count/duration via variable-length dense_rrt.
            if (!fixed_timing) {
                for (const auto& [dgid, dmat] : dense_list) {
                    if (deadline_exceeded()) return false;
                    auto fin_rrt = finalize_trajectory(snap, dmat, cfg, mc_ptr, &mask, v0_ptr, a0_ptr, deadline);
                    if (fin_rrt.success) {
                        res.has_initial_positions = true;
                        res.initial_positions = dmat;
                        res.waypoint_indices = planner_detail::dense_waypoint_indices(
                            dmat, cfg.waypoints, fixed_start, fixed_end, fin_rrt.waypoint_indices);
                        res.goal_index = dgid;
                        res.diagnostics.total_ms = std::chrono::duration<double, std::milli>(Clock::now() - t_start).count();
                        res.diagnostics.ik_ms = ik_ms;
                        res.diagnostics.seed_ms = seed_ms;
                        res.diagnostics.retime_ms = fin_rrt.retime_ms;
                        res.diagnostics.seed_strategy = fb_strat;
                        res.success = true;
                        res.trajectory = std::move(fin_rrt.trajectory);
                        return true;
                    }
                }
            }
            for (const auto& sp : spec_list) {
                if (deadline_exceeded()) return false;
                if (sp.seed_strategy != "constrained_birrt" && sp.seed_strategy != "birrt") continue;
                auto fin_rrt = finalize_trajectory(snap, sp.initial, cfg, mc_ptr, &mask, v0_ptr, a0_ptr, deadline);
                if (fin_rrt.success) {
                    res.has_initial_positions = true;
                    res.initial_positions = sp.initial;
                    res.waypoint_indices = std::move(fin_rrt.waypoint_indices);
                    res.goal_index = sp.goal_index;
                    res.diagnostics.total_ms = std::chrono::duration<double, std::milli>(Clock::now() - t_start).count();
                    res.diagnostics.ik_ms = ik_ms;
                    res.diagnostics.seed_ms = seed_ms;
                    res.diagnostics.retime_ms = fin_rrt.retime_ms;
                    res.diagnostics.seed_strategy = sp.seed_strategy;
                    res.success = true;
                    res.trajectory = std::move(fin_rrt.trajectory);
                    return true;
                }
            }
            return false;
        };

        // 4. Global BiRRT (or Manifold-Projected CBiRRT) fallback when local SQP gets stuck.
        if (winner_idx < 0 && cfg.use_global_seeding && !req.has_seed && !deadline_exceeded()) {
            if (cfg.num_seeds == 1) {
                double rrt_ms = 0.0;
                std::vector<std::pair<int, RowMatrixXd>> dense_rrt;
                auto rrt_specs = build_seed_jobs(snap, start_q, valid_goals, req, mask, fixed_start, fixed_end,
                                                 true, rrt_ms, &dense_rrt, deadline);
                seed_ms += rrt_ms;
                if (!rrt_specs.empty() && !deadline_exceeded()) {
                    auto rrt_res = solve_sqp_batch(snap, rrt_specs, cfg.first_feasible);
                    for (int i = 0; i < static_cast<int>(rrt_res.size()); ++i) {
                        if (rrt_res[static_cast<size_t>(i)].timed_out) any_timed_out = true;
                        if (rrt_res[static_cast<size_t>(i)].success && rrt_res[static_cast<size_t>(i)].cost < winner_cost) {
                            winner_cost = rrt_res[static_cast<size_t>(i)].cost;
                            winner_idx = i;
                        }
                    }
                    if (winner_idx >= 0) {
                        specs = std::move(rrt_specs);
                        sqp_results = std::move(rrt_res);
                        initial_dense_rrt = std::move(dense_rrt);
                    }
                }
                if (winner_idx < 0 && try_raw_rrt_fallback(dense_rrt, rrt_specs)) {
                    return res;
                }
            } else if (try_raw_rrt_fallback(initial_dense_rrt, specs)) {
                return res;
            }
        }

        if (winner_idx < 0) {
            const bool has_first = !sqp_results.empty();
            res.diagnostics.total_ms = std::chrono::duration<double, std::milli>(Clock::now() - t_start).count();
            res.diagnostics.ik_ms = ik_ms;
            res.diagnostics.seed_ms = seed_ms;
            res.diagnostics.timed_out = any_timed_out || deadline_exceeded();
            if (has_first) {
                res.diagnostics.sqp_ms = sqp_results[0].sqp_ms;
                res.diagnostics.validate_ms = sqp_results[0].validate_ms;
                res.diagnostics.qp_calls = sqp_results[0].qp_calls;
                res.diagnostics.sqp_iterations = sqp_results[0].sqp_iterations;
                res.diagnostics.seed_strategy = sqp_results[0].seed_strategy;
                res.diagnostics.witnesses = sqp_results[0].witnesses;
                res.initial_positions = specs[0].initial;
                res.has_initial_positions = true;
                res.reason = res.diagnostics.timed_out
                    ? "replan time budget exceeded"
                    : (sqp_results[0].reason.empty() ? "trajectory optimization failed" : sqp_results[0].reason);
            } else {
                res.diagnostics.seed_strategy = "none";
                res.reason = res.diagnostics.timed_out ? "replan time budget exceeded" : "trajectory optimization failed";
            }
            return res;
        }

        // 5. Unified Hermite retiming, continuous CCD & continuous manifold verification.
        const auto& win = sqp_results[static_cast<size_t>(winner_idx)];
        auto fin = finalize_trajectory(snap, win.positions, cfg, mc_ptr, &mask, v0_ptr, a0_ptr, deadline);

        // If SQP winner failed finalization (e.g. manifold or start-boundary Hermite overshoot) on a global plan,
        // try existing dense RRT fallback first, then global BiRRT/CBiRRT + finalize before giving up.
        if (!fin.success && (!req.manifold_constraints.empty() || has_start_bc) &&
            cfg.use_global_seeding && !req.has_seed && !deadline_exceeded()) {
            if (!initial_dense_rrt.empty() && try_raw_rrt_fallback(initial_dense_rrt, specs)) {
                return res;
            }
            if (win.seed_strategy != "constrained_birrt" && win.seed_strategy != "birrt") {
                double rrt_ms = 0.0;
                std::vector<std::pair<int, RowMatrixXd>> dense_rrt;
                auto rrt_specs = build_seed_jobs(snap, start_q, valid_goals, req, mask, fixed_start, fixed_end,
                                                 true, rrt_ms, &dense_rrt, deadline);
                seed_ms += rrt_ms;
                if (!rrt_specs.empty() && !deadline_exceeded()) {
                    auto rrt_res = solve_sqp_batch(snap, rrt_specs, cfg.first_feasible);
                    for (size_t i = 0; i < rrt_res.size(); ++i) {
                        if (rrt_res[i].success && !deadline_exceeded()) {
                            auto fin2 = finalize_trajectory(
                                snap, rrt_res[i].positions, cfg, mc_ptr, &mask, v0_ptr, a0_ptr, deadline);
                            if (fin2.success) {
                                res.has_initial_positions = true;
                                res.initial_positions = rrt_specs[i].initial;
                                res.waypoint_indices = std::move(fin2.waypoint_indices);
                                res.goal_index = rrt_res[i].goal_index;
                                res.diagnostics.total_ms = std::chrono::duration<double, std::milli>(Clock::now() - t_start).count();
                                res.diagnostics.ik_ms = ik_ms;
                                res.diagnostics.seed_ms = seed_ms;
                                res.diagnostics.sqp_ms = rrt_res[i].sqp_ms;
                                res.diagnostics.validate_ms = rrt_res[i].validate_ms;
                                res.diagnostics.retime_ms = fin2.retime_ms;
                                res.diagnostics.qp_calls = rrt_res[i].qp_calls;
                                res.diagnostics.sqp_iterations = rrt_res[i].sqp_iterations;
                                res.diagnostics.seed_strategy = rrt_res[i].seed_strategy;
                                res.diagnostics.witnesses = rrt_res[i].witnesses;
                                res.success = true;
                                res.trajectory = std::move(fin2.trajectory);
                                return res;
                            }
                        }
                    }
                }
                if (try_raw_rrt_fallback(dense_rrt, rrt_specs)) {
                    return res;
                }
            }
        }

        res.has_initial_positions = true;
        res.initial_positions = specs[static_cast<size_t>(winner_idx)].initial;
        res.goal_index = win.goal_index;
        res.diagnostics.total_ms = std::chrono::duration<double, std::milli>(Clock::now() - t_start).count();
        res.diagnostics.ik_ms = ik_ms;
        res.diagnostics.seed_ms = seed_ms;
        res.diagnostics.sqp_ms = win.sqp_ms;
        res.diagnostics.validate_ms = win.validate_ms;
        res.diagnostics.retime_ms = fin.retime_ms;
        res.diagnostics.qp_calls = win.qp_calls;
        res.diagnostics.sqp_iterations = win.sqp_iterations;
        res.diagnostics.seed_strategy = win.seed_strategy;
        res.diagnostics.witnesses = win.witnesses;
        if (!fin.success) {
            res.diagnostics.timed_out = deadline_exceeded() || (fin.reason == "replan time budget exceeded");
            res.reason = res.diagnostics.timed_out ? "replan time budget exceeded" : fin.reason;
            return res;
        }
        res.success = true;
        res.waypoint_indices = std::move(fin.waypoint_indices);
        res.trajectory = std::move(fin.trajectory);
        return res;
    }

    static PlanResultNative replan_on_snapshot(const WorldSnapshot& snap, const ReplanRequestNative& req) {
        using Clock = std::chrono::steady_clock;
        const auto t_start = Clock::now();
        if (!std::isfinite(req.time_budget_s) || req.time_budget_s < 0.0)
            throw std::invalid_argument("time_budget_s must be finite and >= 0");
        const auto max_budget = std::chrono::duration<double>(Clock::time_point::max() - t_start).count();
        const bool limited = req.time_budget_s > 0.0 && req.time_budget_s < max_budget;
        const auto deadline = limited
            ? t_start + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(req.time_budget_s))
            : Clock::time_point::max();
        auto res = replan_impl(snap, req, t_start, limited ? &deadline : nullptr);
        const auto finished = Clock::now();
        res.diagnostics.total_ms = std::chrono::duration<double, std::milli>(finished - t_start).count();
        if (limited && finished >= deadline) {
            res.success = false;
            res.trajectory = {};
            res.waypoint_indices.clear();
            res.reason = "replan time budget exceeded";
            res.diagnostics.timed_out = true;
        }
        return res;
    }

    static PlanResultNative replan_impl(const WorldSnapshot& snap, const ReplanRequestNative& req,
                                        std::chrono::steady_clock::time_point t_start,
                                        const std::chrono::steady_clock::time_point* deadline_ptr) {
        using Clock = std::chrono::steady_clock;
        PlanResultNative res;
        res.world_version = snap.version;
        const auto& cfg = req.config;
        const int J = snap.kin.num_joints();
        if (cfg.waypoints < (cfg.stop_at_endpoints ? 3 : 2))
            throw std::invalid_argument("too few waypoints for replan");
        if (req.has_prev_traj) validate_slice_input(req.prev_traj);
        if (!req.q_now.allFinite()) throw std::invalid_argument("q_now must be finite");
        for (const auto& goal : req.goal_qs) {
            if (goal.size() != J || !goal.allFinite()) throw std::invalid_argument("invalid joint goal");
        }
        if (cfg.spline_order != 3 && cfg.spline_order != 5) {
            throw std::invalid_argument("spline_order must be 3 (cubic) or 5 (quintic)");
        }
        if (!std::isfinite(cfg.velocity_scale) || cfg.velocity_scale <= 0.0) {
            throw std::invalid_argument("velocity_scale must be finite and positive");
        }
        if (!std::isfinite(cfg.acceleration_scale) || cfg.acceleration_scale <= 0.0) {
            throw std::invalid_argument("acceleration_scale must be finite and positive");
        }
        if (!std::isfinite(cfg.jerk_scale) || cfg.jerk_scale <= 0.0) {
            throw std::invalid_argument("jerk_scale must be finite and positive");
        }
        if (static_cast<int>(req.q_now.size()) != J) {
            throw std::invalid_argument("q_now size must match num_joints");
        }
        if (static_cast<int>(req.dq_now.size()) != J) {
            throw std::invalid_argument("dq_now size must match num_joints");
        }
        if (req.has_ddq_now && cfg.spline_order != 5) {
            throw std::invalid_argument("ddq_now (start_acceleration) is only supported for spline_order == 5 (quintic)");
        }
        const bool has_valid_prev =
            req.has_prev_traj && req.prev_traj.num_waypoints() >= 2 && req.prev_traj.num_joints() == J;
        if (!req.has_goal && !has_valid_prev) {
            throw std::invalid_argument("replan requires either goal or a valid prev_traj");
        }

        auto deadline_exceeded = [&]() -> bool {
            return deadline_ptr && Clock::now() >= *deadline_ptr;
        };

        const auto mask = resolve_mask(snap, req.active_mask.empty() ? nullptr : &req.active_mask);
        const Eigen::VectorXd* v0_ptr = &req.dq_now;
        const Eigen::VectorXd* a0_ptr = req.has_ddq_now ? &req.ddq_now : nullptr;

        {
            const std::string bc_err = validate_start_boundary(snap, cfg, mask, v0_ptr, a0_ptr);
            if (!bc_err.empty()) {
                res.reason = bc_err;
                res.diagnostics.total_ms = std::chrono::duration<double, std::milli>(Clock::now() - t_start).count();
                return res;
            }
        }
        if (!req.manifold_constraints.empty()) {
            manifold_detail::validate_manifold_specs(snap, req.manifold_constraints);
        }

        // Stage 1: Locate anchor on prev_traj and attempt exact Hermite suffix reuse if strictly provable.
        double t_anchor = 0.0;
        bool anchor_ambiguous = false;
        bool t_out_of_range = false;
        if (has_valid_prev && !deadline_exceeded()) {
            const double t0 = req.prev_traj.times(0);
            const double t_end = req.prev_traj.times(req.prev_traj.num_waypoints() - 1);
            if (req.has_t_on_prev) {
                if (!std::isfinite(req.t_on_prev)) {
                    throw std::invalid_argument("t_on_prev must be finite");
                }
                t_out_of_range = req.t_on_prev < t0 || req.t_on_prev > t_end;
                t_anchor = std::clamp(req.t_on_prev, t0, t_end);
                anchor_ambiguous = false;
            } else {
                const auto match = find_anchor_on_trajectory(req.prev_traj, req.q_now, req.dq_now);
                t_anchor = match.t_anchor;
                anchor_ambiguous = match.ambiguous;
            }

            if (req.reuse_if_valid && !anchor_ambiguous && !t_out_of_range &&
                req.prev_traj.spline_order == cfg.spline_order &&
                req.pose_constraints.empty() && req.joint_targets.empty()) {
                constexpr double roundoff = 64 * std::numeric_limits<double>::epsilon();
                auto matches = [&](const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
                    return ((a - b).array().abs() <= roundoff * b.array().abs().max(1.0)).all();
                };
                const Eigen::VectorXd prev_end = req.prev_traj.positions.bottomRows<1>().transpose();
                const bool goal_matches = !req.has_goal ||
                    (!req.goal_is_pose && req.goal_qs.size() == 1 &&
                     matches(req.goal_qs[0], prev_end));

                if (goal_matches) {
                    Eigen::VectorXd p_anc(J), v_anc(J), a_anc(J);
                    req.prev_traj.sample_single(t_anchor, p_anc, &v_anc, &a_anc);
                    // Tight numerical rounding tolerances: any real measured state drift goes to warm-start SQP.
                    const bool state_matches =
                        matches(p_anc, req.q_now) &&
                        matches(v_anc, req.dq_now) &&
                        (!req.has_ddq_now || matches(a_anc, req.ddq_now));

                    if (state_matches) {
                        TrajectoryNative suffix = slice_trajectory_suffix(req.prev_traj, t_anchor);
                        bool compat = true;
                        if (cfg.has_dt || !cfg.retime) {
                            const double dt = cfg.has_dt ? cfg.dt : 0.1;
                            compat = suffix.has_dt &&
                                     (std::abs(suffix.dt - dt) <= roundoff * dt) &&
                                     (suffix.num_waypoints() == cfg.waypoints);
                        }
                        if (compat && cfg.stop_at_endpoints) {
                            if (suffix.velocities.bottomRows<1>().lpNorm<Eigen::Infinity>() > 1e-6) compat = false;
                            if (cfg.spline_order == 5 &&
                                suffix.accelerations.bottomRows<1>().lpNorm<Eigen::Infinity>() > 1e-5) {
                                compat = false;
                            }
                        }
                        if (compat) {
                            for (int j = 0; j < J && compat; ++j) {
                                if (!mask[static_cast<size_t>(j)]) {
                                    if ((suffix.positions.col(j).array() - req.q_now(j)).abs().maxCoeff() > roundoff ||
                                        !suffix.velocities.col(j).isZero(0.0) ||
                                        (suffix.spline_order == 5 && !suffix.accelerations.col(j).isZero(0.0))) {
                                        compat = false;
                                    }
                                }
                            }
                        }

                        if (compat && !deadline_exceeded()) {
                            const auto tv0 = Clock::now();
                            const Eigen::VectorXd vmax = snap.kin.velocity_limit() * cfg.velocity_scale;
                            const Eigen::VectorXd amax = snap.kin.acceleration_limit() * cfg.acceleration_scale;
                            const Eigen::VectorXd jmax = snap.kin.jerk_limit() * cfg.jerk_scale;
                            const auto* mc_ptr = req.manifold_constraints.empty() ? nullptr : &req.manifold_constraints;
                            std::vector<double> samples, frames;
                            std::vector<int64_t> screen_rows, screen_pairs, viol_rows, viol_pairs;
                            auto vres = planner_detail::validate_trajectory_atomic(
                                snap, suffix, cfg, vmax, amax, jmax, snap.kin.lower(), snap.kin.upper(),
                                mc_ptr, samples, frames, screen_rows, screen_pairs, viol_rows, viol_pairs);
                            const double val_ms = std::chrono::duration<double, std::milli>(Clock::now() - tv0).count();
                            if (vres.ok) {
                                res.success = true;
                                res.world_version = snap.version;
                                const int M = suffix.num_waypoints();
                                res.waypoint_indices.resize(static_cast<size_t>(M));
                                for (int k = 0; k < M; ++k) res.waypoint_indices[static_cast<size_t>(k)] = k;
                                if (M != cfg.waypoints) {
                                    res.waypoint_indices = planner_detail::dense_waypoint_indices(
                                        suffix.positions, cfg.waypoints, 1, cfg.stop_at_endpoints ? 2 : 1,
                                        res.waypoint_indices);
                                }
                                res.trajectory = std::move(suffix);
                                res.diagnostics.seed_strategy = "reused_prev";
                                res.diagnostics.validate_ms = val_ms;
                                res.diagnostics.total_ms =
                                    std::chrono::duration<double, std::milli>(Clock::now() - t_start).count();
                                return res;
                            }
                        }
                    }
                }
            }
        }

        if (deadline_exceeded()) {
            res.diagnostics.timed_out = true;
            res.diagnostics.total_ms = std::chrono::duration<double, std::milli>(Clock::now() - t_start).count();
            res.reason = "replan time budget exceeded";
            return res;
        }

        // Build base MotionPlanRequestNative for warm-start SQP and optional global fallback.
        MotionPlanRequestNative base_req;
        base_req.start_is_pose = false;
        base_req.start_q = req.q_now;
        base_req.has_start_velocity = true;
        base_req.start_velocity = req.dq_now;
        base_req.has_start_acceleration = req.has_ddq_now;
        if (req.has_ddq_now) base_req.start_acceleration = req.ddq_now;

        if (req.has_goal) {
            base_req.goal_is_pose = req.goal_is_pose;
            base_req.goal_qs = req.goal_qs;
            base_req.goal_pose_sets = req.goal_pose_sets;
            base_req.has_goal_seed = req.has_goal_seed;
            base_req.goal_seed = req.goal_seed;
        } else {
            base_req.goal_is_pose = false;
            base_req.goal_qs = {req.prev_traj.positions.bottomRows<1>().transpose()};
        }
        base_req.pose_constraints = req.pose_constraints;
        base_req.manifold_constraints = req.manifold_constraints;
        base_req.joint_targets = req.joint_targets;
        base_req.active_mask = req.active_mask;
        base_req.config = cfg;

        // Stage 2: Shifted warm-start SQP seeded from the remaining prev_traj suffix.
        PlanDiagnosticsNative accum_diag;
        PlanResultNative last_res;
        bool tried_warm = false;
        if (has_valid_prev) {
            tried_warm = true;
            MotionPlanRequestNative warm_req = base_req;
            const int W = cfg.waypoints;
            const int fixed_end = cfg.stop_at_endpoints ? 2 : 1;
            const int sample_n = W - (fixed_end - 1);
            const double t_end = req.prev_traj.times(req.prev_traj.num_waypoints() - 1);
            const double rem_dur = std::max(0.0, t_end - t_anchor);

            RowMatrixXd seed(W, J);
            Eigen::VectorXd p_tmp(J);
            const Eigen::VectorXd prev_end = req.prev_traj.positions.bottomRows<1>().transpose();
            const bool have_single_joint_goal = (!base_req.goal_is_pose && base_req.goal_qs.size() == 1);
            const Eigen::VectorXd goal_shift = have_single_joint_goal
                ? Eigen::VectorXd(base_req.goal_qs[0] - prev_end)
                : Eigen::VectorXd::Zero(J);

            for (int k = 0; k < sample_n; ++k) {
                const double alpha = (sample_n > 1) ? (static_cast<double>(k) / static_cast<double>(sample_n - 1)) : 1.0;
                const double ts = t_anchor + alpha * rem_dur;
                req.prev_traj.sample_single(ts, p_tmp, nullptr, nullptr);
                if (have_single_joint_goal) {
                    p_tmp += alpha * goal_shift;
                }
                seed.row(k) = p_tmp.transpose();
            }
            seed.row(0) = req.q_now.transpose();
            if (fixed_end == 2 && sample_n < W) {
                seed.row(W - 1) = seed.row(sample_n - 1);
            }
            // Prefer the resampled suffix's first-segment duration over a large fixed dt_smooth.
            if (cfg.has_dt) {
                warm_req.dt_start_smooth = cfg.dt;
            } else {
                const int n_segs = std::max(1, sample_n - 1);
                const double dt_est = (rem_dur > 1e-4) ? (rem_dur / static_cast<double>(n_segs)) : 0.05;
                warm_req.dt_start_smooth = std::clamp(dt_est, 0.02, 0.25);
            }
            warm_req.has_seed = true;
            warm_req.seed = std::move(seed);
            warm_req.seed_strategy_label = "warm_start";

            last_res = plan_on_snapshot(snap, warm_req, deadline_ptr);
            if (last_res.success) {
                last_res.diagnostics.total_ms =
                    std::chrono::duration<double, std::milli>(Clock::now() - t_start).count();
                return last_res;
            }
            accum_diag = last_res.diagnostics;
            if (last_res.diagnostics.timed_out || deadline_exceeded()) {
                last_res.diagnostics.timed_out = true;
                last_res.reason = "replan time budget exceeded";
                last_res.diagnostics.total_ms =
                    std::chrono::duration<double, std::milli>(Clock::now() - t_start).count();
                return last_res;
            }
        }

        // Stage 3: Time-budgeted global fallback (straight + BiRRT/CBiRRT) preserving exact start boundary.
        if (!tried_warm || cfg.use_global_seeding) {
            if (deadline_exceeded()) {
                res.diagnostics = accum_diag;
                res.diagnostics.timed_out = true;
                res.diagnostics.total_ms = std::chrono::duration<double, std::milli>(Clock::now() - t_start).count();
                res.reason = "replan time budget exceeded";
                return res;
            }
            if (!cfg.has_dt && has_valid_prev) {
                const double t_end = req.prev_traj.times(req.prev_traj.num_waypoints() - 1);
                const double rem_dur = std::max(0.0, t_end - t_anchor);
                const int n_segs = std::max(1, cfg.waypoints - (cfg.stop_at_endpoints ? 2 : 1));
                const double dt_est = (rem_dur > 1e-4) ? (rem_dur / static_cast<double>(n_segs)) : 0.05;
                base_req.dt_start_smooth = std::clamp(dt_est, 0.02, 0.25);
            }
            PlanResultNative fb = plan_on_snapshot(snap, base_req, deadline_ptr);
            fb.diagnostics.ik_ms += accum_diag.ik_ms;
            fb.diagnostics.seed_ms += accum_diag.seed_ms;
            fb.diagnostics.sqp_ms += accum_diag.sqp_ms;
            fb.diagnostics.validate_ms += accum_diag.validate_ms;
            fb.diagnostics.retime_ms += accum_diag.retime_ms;
            fb.diagnostics.qp_calls += accum_diag.qp_calls;
            fb.diagnostics.sqp_iterations += accum_diag.sqp_iterations;
            fb.diagnostics.total_ms = std::chrono::duration<double, std::milli>(Clock::now() - t_start).count();
            if (!fb.success && (fb.diagnostics.timed_out || deadline_exceeded())) {
                fb.diagnostics.timed_out = true;
                fb.reason = "replan time budget exceeded";
            }
            return fb;
        }

        last_res.diagnostics.total_ms = std::chrono::duration<double, std::milli>(Clock::now() - t_start).count();
        return last_res;
    }

    std::shared_ptr<PlanningWorld> world_;
};

}  // namespace gobot::planning
