#pragma once

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

    // Optional waypoint constraints, initial trajectory seed, and active joint mask.
    std::vector<std::pair<int, std::vector<PoseConstraintSpec>>> pose_constraints;
    std::vector<std::pair<int, Eigen::VectorXd>> joint_targets;
    bool has_seed = false;
    RowMatrixXd seed;
    std::vector<uint8_t> active_mask;  // empty => all joints active

    SolverConfigNative config;
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
    std::string seed_strategy = "straight";
    std::vector<double> witnesses;  // flat [K * 6]
};

struct PlanResultNative {
    bool success = false;
    TrajectoryNative trajectory;
    bool has_initial_positions = false;
    RowMatrixXd initial_positions;
    int goal_index = 0;
    std::string reason;
    PlanDiagnosticsNative diagnostics;
};

struct FinalizeResultNative {
    bool success = false;
    TrajectoryNative trajectory;
    double retime_ms = 0.0;
    std::string reason;
};

namespace planner_detail {

inline RowMatrixXd linear_seed(const Eigen::VectorXd& start, const Eigen::VectorXd& goal,
                               int waypoints, int fixed,
                               const std::vector<std::pair<int, Eigen::VectorXd>>& joint_targets) {
    const int J = static_cast<int>(start.size());
    std::vector<std::pair<int, Eigen::VectorXd>> anchors;
    anchors.emplace_back(fixed - 1, start);
    for (const auto& [idx, jt] : joint_targets) {
        if (idx >= fixed && idx < waypoints - fixed) anchors.emplace_back(idx, jt);
    }
    std::sort(anchors.begin(), anchors.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    anchors.emplace_back(waypoints - fixed, goal);

    RowMatrixXd out(waypoints, J);
    for (int t = 0; t < fixed; ++t) out.row(t) = start.transpose();
    for (int t = waypoints - fixed; t < waypoints; ++t) out.row(t) = goal.transpose();
    for (size_t k = 0; k + 1 < anchors.size(); ++k) {
        const int i0 = anchors[k].first;
        const int i1 = anchors[k + 1].first;
        const Eigen::VectorXd& a = anchors[k].second;
        const Eigen::VectorXd& b = anchors[k + 1].second;
        const int span = std::max(1, i1 - i0);
        for (int t = i0; t <= i1; ++t) {
            if (t >= fixed && t < waypoints - fixed) {
                const double frac = static_cast<double>(t - i0) / static_cast<double>(span);
                out.row(t) = ((1.0 - frac) * a + frac * b).transpose();
            }
        }
    }
    return out;
}

}  // namespace planner_detail

// Unified Hermite retiming + continuous collision & bounds verification on an immutable WorldSnapshot.
// Used identically by single plan(), batch plan_batch(), and Python's MuJoCo Warp path.
inline FinalizeResultNative finalize_trajectory(const WorldSnapshot& snap,
                                                const RowMatrixXd& positions,
                                                const SolverConfigNative& cfg) {
    using Clock = std::chrono::steady_clock;
    const auto t0 = Clock::now();
    FinalizeResultNative out;

    const int T = static_cast<int>(positions.rows());
    const int J = static_cast<int>(positions.cols());
    const Eigen::VectorXd vmax = snap.kin.velocity_limit() * cfg.velocity_scale;
    const Eigen::VectorXd amax = snap.kin.acceleration_limit() * cfg.acceleration_scale;
    const Eigen::VectorXd& lo = snap.kin.lower();
    const Eigen::VectorXd& hi = snap.kin.upper();
    const bool has_dt = cfg.has_dt || !cfg.retime;
    const double dt_val = cfg.has_dt ? cfg.dt : 0.1;

    int substeps = std::max(2 * cfg.collision_substeps, 4);
    if (cfg.max_segment_length > 0.0 && T >= 2) {
        double longest = 0.0;
        for (int i = 0; i + 1 < T; ++i) longest = std::max(longest, (positions.row(i + 1) - positions.row(i)).norm());
        substeps = std::max(substeps, static_cast<int>(std::ceil(longest / cfg.max_segment_length)));
    }
    const int P = static_cast<int>(snap.pairs.size());
    const int N_shapes = static_cast<int>(snap.shapes.size());
    const Eigen::VectorXd margins = (P > 0) ? snap.pair_margins(cfg.margin, cfg.effective_world_margin()) : Eigen::VectorXd();

    Eigen::VectorXd knot_scales = Eigen::VectorXd::Ones(T);
    std::string last_error;
    std::vector<double> samples, frames;
    std::vector<int64_t> screen_rows, screen_pairs, viol_rows, viol_pairs;

    for (int attempt = 0; attempt < 8; ++attempt) {
        TrajectoryNative traj = retime_trajectory(
            positions, vmax, amax, cfg.stop_at_endpoints, has_dt, dt_val, &knot_scales);

        // 1. Exact continuous cubic Hermite velocity & acceleration verification.
        auto bchk = check_trajectory_bounds(traj, vmax, amax, true, true, 1e-5);
        if (!bchk.ok) {
            last_error = bchk.reason;
            if (has_dt) {
                out.retime_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
                out.reason = last_error;
                return out;
            }
        }

        // 2. Sample actual cubic Hermite curve and check joint limits + discrete & continuous CCD.
        traj.sample_segments(substeps, samples);
        const int64_t S = static_cast<int64_t>((T - 1) * substeps + 1);
        std::set<int> bad_segs(bchk.bad_segments.begin(), bchk.bad_segments.end());
        for (int64_t s = 0; s < S; ++s) {
            const double* qs = samples.data() + s * J;
            for (int j = 0; j < J; ++j) {
                if (!std::isfinite(qs[j]) || qs[j] < lo(j) - 1e-6 || qs[j] > hi(j) + 1e-6) {
                    bad_segs.insert(std::min(static_cast<int>(s / substeps), T - 2));
                    if (last_error.empty()) last_error = "retimed Hermite curve violates joint limits";
                    break;
                }
            }
        }

        if (bad_segs.empty() && P > 0) {
            frames.resize(static_cast<size_t>(S * N_shapes * 12));
            snap.compute_frames(samples.data(), S, frames.data());

            const int n_passes = snap.has_mesh ? 2 : 1;
            for (int pass = 0; pass < n_passes; ++pass) {
                const bool verify_orig = (pass == 1);
                const char* kind = verify_orig ? "mesh" : "convex";
                // Discrete check across all samples (collects all violating segments in one pass).
                snap.screen_aabbs(frames.data(), S, margins.data(), verify_orig, substeps, screen_rows, screen_pairs);
                if (!screen_rows.empty()) {
                    snap.violating_pairs_indexed(frames.data(), S, screen_rows.data(), screen_pairs.data(),
                                                 static_cast<int64_t>(screen_rows.size()), margins.data(), verify_orig,
                                                 viol_rows, viol_pairs);
                    if (!viol_rows.empty()) {
                        const int seg0 = std::min(static_cast<int>(viol_rows[0] / substeps), T - 2);
                        last_error = std::string("retimed Hermite ") + kind + " clearance failed on segment " +
                                     std::to_string(seg0) + ", pair " + snap.pair_display_name(viol_pairs[0]);
                        for (const int64_t r : viol_rows) {
                            bad_segs.insert(std::min(static_cast<int>(r / substeps), T - 2));
                        }
                        break;
                    }
                }
                // Sound kinematic-chain Conservative Advancement CCD on the actual played Hermite curve.
                // (For triangle meshes, d_orig >= d_convex everywhere, so convex CCD proves mesh clearance too.)
                if (!verify_orig) {
                    auto ccd_hit = snap.first_violation_continuous(
                        traj.positions.data(), T, substeps, frames.data(), margins.data(),
                        false, 1e-3, traj.times.data(), traj.velocities.data());
                    if (ccd_hit.has_value()) {
                        last_error = std::string("retimed Hermite continuous ") + kind + " clearance failed on segment " +
                                     std::to_string(ccd_hit->seg) + ", pair " + snap.pair_display_name(ccd_hit->pair_id);
                        bad_segs.insert(ccd_hit->seg);
                        break;
                    }
                }
            }
        }

        if (bad_segs.empty()) {
            out.success = true;
            out.trajectory = std::move(traj);
            out.retime_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
            return out;
        }

        // Damp knot velocities around offending segments (at 0 velocity Hermite collapses onto the verified chord).
        bool changed = false;
        for (const int seg : bad_segs) {
            int knots[2] = {seg, seg + 1};
            int count = 2;
            if (has_dt && !cfg.stop_at_endpoints) {
                int filtered[2];
                int fc = 0;
                for (int k = 0; k < 2; ++k) {
                    if (knots[k] > 0 && knots[k] < T - 1) filtered[fc++] = knots[k];
                }
                if (fc > 0) {
                    count = fc;
                    for (int k = 0; k < fc; ++k) knots[k] = filtered[k];
                }
            }
            for (int k = 0; k < count; ++k) {
                const int knot = knots[k];
                const double old = knot_scales(knot);
                if (old > 0.0) {
                    knot_scales(knot) = (attempt >= 2 || old <= 0.25) ? 0.0 : (old * 0.5);
                    changed = true;
                }
            }
        }
        if (!changed) {
            if (!has_dt && (knot_scales.array() > 0.0).any()) {
                knot_scales.setZero();
            } else {
                out.retime_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
                out.reason = last_error;
                return out;
            }
        }
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

    static std::string validate_endpoints(const WorldSnapshot& snap,
                                          const Eigen::VectorXd& start_q,
                                          const Eigen::VectorXd& goal_q,
                                          const SolverConfigNative& cfg, int fixed) {
        const int J = snap.kin.num_joints();
        const auto& lo = snap.kin.lower();
        const auto& hi = snap.kin.upper();
        for (int j = 0; j < J; ++j) {
            if (!std::isfinite(start_q(j)) || start_q(j) < lo(j) || start_q(j) > hi(j) ||
                !std::isfinite(goal_q(j)) || goal_q(j) < lo(j) || goal_q(j) > hi(j)) {
                return "start or goal violates joint limits";
            }
        }
        if (cfg.has_dt) {
            const int moving = cfg.waypoints - 2 * fixed + 1;
            const Eigen::VectorXd vmax = snap.kin.velocity_limit() * cfg.velocity_scale;
            for (int j = 0; j < J; ++j) {
                if (std::isfinite(vmax(j)) &&
                    std::abs(goal_q(j) - start_q(j)) / (static_cast<double>(moving) * cfg.dt) > vmax(j) * (1.0 + 1e-6)) {
                    return "joint velocity limits require a longer duration";
                }
            }
        }
        if (!snap.pairs.empty()) {
            const Eigen::VectorXd margins = snap.pair_margins(cfg.margin, cfg.effective_world_margin());
            std::vector<double> ends(static_cast<size_t>(2 * J));
            std::copy(start_q.data(), start_q.data() + J, ends.data());
            std::copy(goal_q.data(), goal_q.data() + J, ends.data() + J);
            std::vector<double> frames(static_cast<size_t>(2 * snap.shapes.size() * 12));
            snap.compute_frames(ends.data(), 2, frames.data());
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
        return {};
    }

    static std::vector<SQPJobSpecNative> build_seed_jobs(
        const WorldSnapshot& snap,
        const Eigen::VectorXd& start_q,
        const std::vector<std::pair<int, Eigen::VectorXd>>& goal_candidates,
        const MotionPlanRequestNative& req,
        const std::vector<uint8_t>& mask,
        bool force_rrt,
        double& out_seed_ms) {
        using Clock = std::chrono::steady_clock;
        const auto t0 = Clock::now();
        const auto& cfg = req.config;
        const int J = snap.kin.num_joints();
        const int fixed = cfg.stop_at_endpoints ? 2 : 1;
        const double dt_smooth = cfg.has_dt ? cfg.dt : 0.25;
        bool has_interior_anchors = false;
        for (const auto& [idx, _] : req.joint_targets) {
            if (idx >= fixed && idx < cfg.waypoints - fixed) { has_interior_anchors = true; break; }
        }

        auto make_spec = [&](RowMatrixXd init_q, const std::string& strategy, int goal_idx) {
            SQPJobSpecNative sp;
            sp.initial = std::move(init_q);
            sp.fixed = fixed;
            sp.has_dt_vel = cfg.has_dt;
            sp.dt_vel = cfg.dt;
            sp.dt_smooth = dt_smooth;
            sp.active_mask = mask;
            sp.pose_constraints = req.pose_constraints;
            sp.joint_targets = req.joint_targets;
            sp.margin = cfg.margin;
            sp.world_margin = cfg.effective_world_margin();
            sp.collision_substeps = cfg.collision_substeps;
            sp.max_segment_length = cfg.max_segment_length;
            sp.iterations = cfg.iterations;
            sp.first_feasible = cfg.first_feasible;
            sp.velocity_scale = cfg.velocity_scale;
            sp.seed_strategy = strategy;
            sp.goal_index = goal_idx;
            return sp;
        };

        std::vector<SQPJobSpecNative> specs;
        if (req.has_seed && !force_rrt) {
            RowMatrixXd q0 = req.seed;
            const auto& lo = snap.kin.lower();
            const auto& hi = snap.kin.upper();
            for (int t = 0; t < fixed; ++t) q0.row(t) = start_q.transpose();
            for (int t = cfg.waypoints - fixed; t < cfg.waypoints; ++t) q0.row(t) = goal_candidates[0].second.transpose();
            for (int t = fixed; t < cfg.waypoints - fixed; ++t) {
                for (int j = 0; j < J; ++j) q0(t, j) = std::clamp(q0(t, j), lo(j), hi(j));
            }
            for (int j = 0; j < J; ++j) {
                if (!mask[static_cast<size_t>(j)]) q0.col(j).setConstant(start_q(j));
            }
            specs.push_back(make_spec(std::move(q0), "user_seed", goal_candidates[0].first));
            out_seed_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
            return specs;
        }

        if (!force_rrt) {
            const int n_lin = std::min(static_cast<int>(goal_candidates.size()), std::max(1, cfg.num_seeds));
            for (int i = 0; i < n_lin; ++i) {
                auto q0 = planner_detail::linear_seed(start_q, goal_candidates[static_cast<size_t>(i)].second,
                                                      cfg.waypoints, fixed, req.joint_targets);
                specs.push_back(make_spec(std::move(q0), "straight", goal_candidates[static_cast<size_t>(i)].first));
            }
        }

        const bool need_rrt = force_rrt || (cfg.use_global_seeding && cfg.num_seeds > 1);
        if (need_rrt && !has_interior_anchors) {
            const int n_rrt = force_rrt ? std::max(1, cfg.num_seeds) : std::max(1, cfg.num_seeds - static_cast<int>(specs.size()));
            const int inner_wp = cfg.waypoints - 2 * (fixed - 1);
            std::vector<RRTResultNative> rrt_out(static_cast<size_t>(n_rrt));
            ThreadPool::instance().parallel_for(n_rrt, 1, [&](int64_t b, int64_t e) {
                for (int64_t i = b; i < e; ++i) {
                    const auto& [gid, gq] = goal_candidates[static_cast<size_t>(i) % goal_candidates.size()];
                    const uint64_t sd = 1001ULL + static_cast<uint64_t>(i) * 7919ULL;
                    rrt_out[static_cast<size_t>(i)] = snap.plan_birrt(
                        start_q, gq, mask, cfg.margin, cfg.effective_world_margin(),
                        cfg.rrt_step_size, cfg.rrt_max_iterations, cfg.rrt_timeout_s,
                        sd, snap.has_mesh, cfg.rrt_shortcut_iters, inner_wp);
                }
            });
            for (int i = 0; i < n_rrt; ++i) {
                if (!rrt_out[static_cast<size_t>(i)].success) continue;
                const auto& [gid, gq] = goal_candidates[static_cast<size_t>(i) % goal_candidates.size()];
                RowMatrixXd path(cfg.waypoints, J);
                if (fixed == 2) {
                    path.row(0) = start_q.transpose();
                    for (int t = 0; t < inner_wp; ++t) path.row(t + 1) = rrt_out[static_cast<size_t>(i)].waypoints[static_cast<size_t>(t)].transpose();
                    path.row(cfg.waypoints - 1) = gq.transpose();
                } else {
                    for (int t = 0; t < cfg.waypoints; ++t) path.row(t) = rrt_out[static_cast<size_t>(i)].waypoints[static_cast<size_t>(t)].transpose();
                }
                specs.push_back(make_spec(std::move(path), "birrt", gid));
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
                auto q0 = planner_detail::linear_seed(start_q, gq, cfg.waypoints, fixed, mid_anchor);
                specs.push_back(make_spec(std::move(q0), "perturbed", gid));
            }
        }

        out_seed_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        return specs;
    }

    static PlanResultNative plan_on_snapshot(const WorldSnapshot& snap, const MotionPlanRequestNative& req) {
        using Clock = std::chrono::steady_clock;
        const auto t_start = Clock::now();
        PlanResultNative res;
        const auto& cfg = req.config;
        const int fixed = cfg.stop_at_endpoints ? 2 : 1;
        if (cfg.waypoints < 2 * fixed) throw std::invalid_argument("waypoints must be >= 2 * fixed");

        const int J = snap.kin.num_joints();
        const auto mask = resolve_mask(snap, req.active_mask.empty() ? nullptr : &req.active_mask);
        const double wm = cfg.effective_world_margin();

        // 1. Resolve start & goal endpoints (with C++ IK / parallel GoalSet IK).
        const auto t_ik0 = Clock::now();
        Eigen::VectorXd start_q;
        if (req.start_is_pose) {
            auto s_ik = solve_ik_on_snapshot(snap, req.start_poses, req.has_start_seed ? &req.start_seed : nullptr,
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
                auto g_ik = solve_ik_on_snapshot(snap, req.goal_pose_sets[0], &g_seed, &mask,
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
                for (const auto& ts : req.goal_pose_sets) probs.push_back({ts, g_seed});
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
            std::string err = validate_endpoints(snap, start_q, gq, cfg, fixed);
            if (err.empty()) valid_goals.emplace_back(gid, gq);
            else last_ep_err = std::move(err);
        }
        if (valid_goals.empty()) {
            res.reason = last_ep_err.empty() ? "no valid goal endpoints" : last_ep_err;
            res.diagnostics.ik_ms = ik_ms;
            res.diagnostics.total_ms = std::chrono::duration<double, std::milli>(Clock::now() - t_start).count();
            return res;
        }

        // 3. Build seed jobs and run SQP (with optional multi-seed racing).
        double seed_ms = 0.0;
        auto specs = build_seed_jobs(snap, start_q, valid_goals, req, mask, false, seed_ms);
        auto sqp_results = solve_sqp_batch(snap, specs, specs.size() > 1 && cfg.first_feasible);

        int winner_idx = -1;
        double winner_cost = std::numeric_limits<double>::infinity();
        for (int i = 0; i < static_cast<int>(sqp_results.size()); ++i) {
            if (sqp_results[static_cast<size_t>(i)].success && sqp_results[static_cast<size_t>(i)].cost < winner_cost) {
                winner_cost = sqp_results[static_cast<size_t>(i)].cost;
                winner_idx = i;
            }
        }

        // 4. Global BiRRT fallback when local single-seed SQP gets stuck.
        if (winner_idx < 0 && cfg.use_global_seeding && cfg.num_seeds == 1 && !req.has_seed) {
            double rrt_ms = 0.0;
            auto rrt_specs = build_seed_jobs(snap, start_q, valid_goals, req, mask, true, rrt_ms);
            seed_ms += rrt_ms;
            if (!rrt_specs.empty()) {
                auto rrt_res = solve_sqp_batch(snap, rrt_specs, cfg.first_feasible);
                for (int i = 0; i < static_cast<int>(rrt_res.size()); ++i) {
                    if (rrt_res[static_cast<size_t>(i)].success && rrt_res[static_cast<size_t>(i)].cost < winner_cost) {
                        winner_cost = rrt_res[static_cast<size_t>(i)].cost;
                        winner_idx = i;
                    }
                }
                if (winner_idx >= 0) {
                    specs = std::move(rrt_specs);
                    sqp_results = std::move(rrt_res);
                }
            }
        }

        if (winner_idx < 0) {
            const bool has_first = !sqp_results.empty();
            res.diagnostics.total_ms = std::chrono::duration<double, std::milli>(Clock::now() - t_start).count();
            res.diagnostics.ik_ms = ik_ms;
            res.diagnostics.seed_ms = seed_ms;
            if (has_first) {
                res.diagnostics.sqp_ms = sqp_results[0].sqp_ms;
                res.diagnostics.validate_ms = sqp_results[0].validate_ms;
                res.diagnostics.qp_calls = sqp_results[0].qp_calls;
                res.diagnostics.sqp_iterations = sqp_results[0].sqp_iterations;
                res.diagnostics.seed_strategy = sqp_results[0].seed_strategy;
                res.diagnostics.witnesses = sqp_results[0].witnesses;
                res.initial_positions = specs[0].initial;
                res.has_initial_positions = true;
                res.reason = sqp_results[0].reason.empty() ? "trajectory optimization failed" : sqp_results[0].reason;
            } else {
                res.diagnostics.seed_strategy = "none";
                res.reason = "trajectory optimization failed";
            }
            return res;
        }

        // 5. Unified Hermite retiming & continuous CCD verification.
        const auto& win = sqp_results[static_cast<size_t>(winner_idx)];
        auto fin = finalize_trajectory(snap, win.positions, cfg);

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
            res.reason = fin.reason;
            return res;
        }
        res.success = true;
        res.trajectory = std::move(fin.trajectory);
        return res;
    }

    std::shared_ptr<PlanningWorld> world_;
};

}  // namespace gobot::planning
