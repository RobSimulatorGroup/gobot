#pragma once

#include "planning_qp.hpp"
#include "planning_world.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace gobot::planning {

struct SQPJobSpecNative {
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> initial;  // (T, J)
    int fixed = 1;                  // legacy symmetric endpoint count (1 or 2)
    int fixed_start = -1;           // if >= 0, overrides fixed at start
    int fixed_end = -1;             // if >= 0, overrides fixed at end
    bool has_dt_vel = false;
    double dt_vel = 0.0;
    double dt_smooth = 0.25;
    double dt_start_smooth = 0.0;   // if > 0, overrides virtual-predecessor dt in smoothness cost
    bool has_start_velocity = false;
    Eigen::VectorXd start_velocity; // (J,)
    std::vector<uint8_t> active_mask;  // (J,)
    std::vector<std::pair<int, std::vector<PoseConstraintSpec>>> pose_constraints;
    std::vector<std::pair<int, Eigen::VectorXd>> joint_targets;
    double margin = 0.002;
    double world_margin = 0.002;
    int collision_substeps = 8;
    double max_segment_length = 0.0;  // <= 0 means disabled
    int iterations = 40;
    bool first_feasible = true;
    bool strict_ccd = false;
    double velocity_scale = 1.0;
    double acceleration_scale = 1.0;
    bool has_deadline = false;
    std::chrono::steady_clock::time_point deadline;
    std::string seed_strategy = "straight";
    int goal_index = 0;
};

struct SQPJobResultNative {
    bool success = false;
    bool timed_out = false;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> positions;
    std::string reason;
    int qp_calls = 0;
    int sqp_iterations = 0;
    double sqp_ms = 0.0;
    double validate_ms = 0.0;
    std::vector<double> witnesses;  // flat [K * 6]
    double cost = std::numeric_limits<double>::infinity();
    std::string seed_strategy;
    int goal_index = 0;
};

namespace detail {

constexpr double kCollisionLinearizationBufferM = 0.05;

inline int grid_substeps(const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& q,
                         int minimum, double max_length) {
    if (max_length <= 0.0 || q.rows() < 2) return minimum;
    double longest = 0.0;
    for (Eigen::Index i = 0; i + 1 < q.rows(); ++i) {
        longest = std::max(longest, (q.row(i + 1) - q.row(i)).norm());
    }
    return std::max(minimum, static_cast<int>(std::ceil(longest / max_length)));
}

inline void build_segment_samples(
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& q,
    int substeps,
    bool include_goal,
    std::vector<double>& out_samples) {
    const int T = static_cast<int>(q.rows());
    const int J = static_cast<int>(q.cols());
    const int n_segs = T - 1;
    const int total = n_segs * substeps + (include_goal ? 1 : 0);
    out_samples.resize(static_cast<size_t>(total * J));
    const double inv_sub = 1.0 / static_cast<double>(substeps);
    for (int seg = 0; seg < n_segs; ++seg) {
        const double* qa = q.data() + seg * J;
        const double* qb = q.data() + (seg + 1) * J;
        for (int k = 0; k < substeps; ++k) {
            const double u = static_cast<double>(k) * inv_sub;
            const double w = 1.0 - u;
            double* dst = out_samples.data() + (seg * substeps + k) * J;
            for (int j = 0; j < J; ++j) {
                dst[j] = w * qa[j] + u * qb[j];
            }
        }
    }
    if (include_goal) {
        std::copy(q.data() + (T - 1) * J, q.data() + T * J, out_samples.data() + n_segs * substeps * J);
    }
}

inline double estimate_seg0_hermite_dt(
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& q,
    const Eigen::VectorXd& v0,
    const Eigen::VectorXd& vmax,
    const Eigen::VectorXd& amax,
    bool has_dt_vel,
    double dt_vel) {
    if (has_dt_vel) return dt_vel;
    const int J = static_cast<int>(q.cols());
    auto seg0_ok = [&](double h) -> bool {
        if (h <= 1e-9) return false;
        for (int j = 0; j < J; ++j) {
            const double dqj = q(1, j) - q(0, j);
            const double v0j = v0(j);
            const double a0 = (6.0 * dqj / h - 4.0 * v0j) / h;
            const double a1 = (-6.0 * dqj / h + 2.0 * v0j) / h;
            if (std::isfinite(amax(j)) && std::max(std::abs(a0), std::abs(a1)) > amax(j) * (1.0 + 1e-5)) {
                return false;
            }
            if (std::isfinite(vmax(j))) {
                double mv = std::abs(v0j);
                const double denom = a1 - a0;
                if (std::abs(denom) > 1e-12) {
                    const double u_star = -a0 / denom;
                    if (u_star > 0.0 && u_star < 1.0) {
                        const double u2 = u_star * u_star;
                        const double vs = (6.0 * u_star - 6.0 * u2) * (dqj / h) + (3.0 * u2 - 4.0 * u_star + 1.0) * v0j;
                        mv = std::max(mv, std::abs(vs));
                    }
                }
                if (mv > vmax(j) * (1.0 + 1e-5)) return false;
            }
        }
        return true;
    };
    double h_min = 1e-4;
    for (int j = 0; j < J; ++j) {
        const double dqj = std::abs(q(1, j) - q(0, j));
        const double v0j = std::abs(v0(j));
        if (std::isfinite(vmax(j)) && vmax(j) > 1e-9) h_min = std::max(h_min, dqj / vmax(j));
        if (std::isfinite(amax(j)) && amax(j) > 1e-9) h_min = std::max(h_min, v0j / amax(j));
    }
    if (seg0_ok(h_min)) return h_min;
    // Scan log-spaced factors to find the smallest feasible braking/transition duration.
    double prev = h_min;
    for (int k = 1; k <= 36; ++k) {
        const double h = h_min * std::pow(1.22, static_cast<double>(k));
        if (seg0_ok(h)) {
            double lo = prev, hi = h;
            for (int it = 0; it < 12; ++it) {
                const double mid = 0.5 * (lo + hi);
                if (seg0_ok(mid)) hi = mid;
                else lo = mid;
            }
            return hi;
        }
        prev = h;
    }
    return prev;
}

inline void build_seg0_hermite_samples(
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& q,
    const Eigen::VectorXd& v0,
    double h0,
    std::vector<double>& out_q,
    double out_h01[3]) {
    constexpr double kUs[3] = {0.25, 0.50, 0.75};
    const int J = static_cast<int>(q.cols());
    out_q.resize(static_cast<size_t>(3 * J));
    for (int k = 0; k < 3; ++k) {
        const double u = kUs[k];
        const double u2 = u * u;
        const double u3 = u2 * u;
        const double h00 = 2.0 * u3 - 3.0 * u2 + 1.0;
        const double h10 = u3 - 2.0 * u2 + u;
        const double h01 = -2.0 * u3 + 3.0 * u2;
        out_h01[k] = h01;
        for (int j = 0; j < J; ++j) {
            out_q[static_cast<size_t>(k * J + j)] = h00 * q(0, j) + h01 * q(1, j) + h10 * h0 * v0(j);
        }
    }
}

}  // namespace detail

inline SQPJobResultNative solve_sqp_job(const WorldSnapshot& snap,
                                        const SQPJobSpecNative& spec,
                                        const std::atomic<bool>* cancel_flag = nullptr) {
    using Clock = std::chrono::steady_clock;
    const auto t_job_start = Clock::now();

    SQPJobResultNative out;
    out.seed_strategy = spec.seed_strategy;
    out.goal_index = spec.goal_index;

    const auto& q_init = spec.initial;
    const int T = static_cast<int>(q_init.rows());
    const int J = static_cast<int>(q_init.cols());
    const int fixed_start = (spec.fixed_start >= 0) ? spec.fixed_start : spec.fixed;
    const int fixed_end = (spec.fixed_end >= 0) ? spec.fixed_end : spec.fixed;
    const int n_free_wp = T - fixed_start - fixed_end;
    const int n_vars = std::max(0, n_free_wp * J);
    const bool has_v0 = spec.has_start_velocity && (static_cast<int>(spec.start_velocity.size()) == J) && (T >= 2);
    const int N_shapes = static_cast<int>(snap.shapes.size());
    const int B_links = snap.kin.num_bodies();
    const int P = static_cast<int>(snap.pairs.size());
    const bool use_esdf = snap.has_esdf();
    const bool has_col = (P > 0) || use_esdf;
    const bool has_mesh = snap.has_mesh;

    const Eigen::VectorXd& lower = snap.kin.lower();
    const Eigen::VectorXd& upper = snap.kin.upper();
    const Eigen::VectorXd vmax = snap.kin.velocity_limit() * spec.velocity_scale;
    const Eigen::VectorXd amax = snap.kin.acceleration_limit() * spec.acceleration_scale;
    const auto deadline_exceeded = [&]() -> bool {
        return spec.has_deadline && Clock::now() >= spec.deadline;
    };
    const Eigen::VectorXd margins = snap.pair_margins(spec.margin, spec.world_margin);
    const double cushion = spec.strict_ccd ? 1.0e-3 : 5e-4;
    const double extra_cushion = spec.strict_ccd ? 1.5e-3 : 8e-4;
    const Eigen::VectorXd opt_margins_cushion = margins.array() + cushion;
    const Eigen::VectorXd opt_margins_extra = margins.array() + extra_cushion;
    const double esdf_margin_cushion = spec.world_margin + cushion;
    const double esdf_margin_extra = spec.world_margin + extra_cushion;

    // Precompute fixed goal frame once for fast validation reuse.
    std::vector<double> goal_frame(static_cast<size_t>(N_shapes * 12));
    snap.compute_frames(q_init.data() + (T - 1) * J, 1, goal_frame.data());

    // Per-job failure caches for convex and original mesh checks.
    bool has_prev_convex = false, has_prev_orig = false;
    int64_t prev_c_row = 0, prev_c_pair = 0, prev_c_sub = 0;
    int64_t prev_o_row = 0, prev_o_pair = 0, prev_o_sub = 0;
    std::string failure;
    double validate_ms = 0.0;
    const int validation_substeps = 2 * spec.collision_substeps;

    std::vector<int64_t> screen_rows, screen_pairs;
    std::vector<double> val_samples, val_frames;
    std::vector<Vec3> bpos(static_cast<size_t>(B_links)), jorig(static_cast<size_t>(J)), jaxis(static_cast<size_t>(J));
    std::vector<Mat3> brot(static_cast<size_t>(B_links));

    auto check_frames = [&](const double* frames_ptr, int64_t S, int substeps, bool verify_orig)
        -> std::optional<std::pair<int64_t, int64_t>> {
        if (verify_orig ? has_prev_orig : has_prev_convex) {
            const int64_t ps = verify_orig ? prev_o_row : prev_c_row;
            const int64_t pp = verify_orig ? prev_o_pair : prev_c_pair;
            const int64_t psub = verify_orig ? prev_o_sub : prev_c_sub;
            if (psub == substeps && ps < S) {
                auto hit = snap.first_violation_indexed(frames_ptr, S, &ps, &pp, 1, margins.data(), verify_orig);
                if (hit.has_value()) return hit;
            }
        }
        snap.screen_aabbs(frames_ptr, S, margins.data(), verify_orig, substeps, screen_rows, screen_pairs);
        std::optional<std::pair<int64_t, int64_t>> hit;
        if (!screen_rows.empty()) {
            hit = snap.first_violation_indexed(frames_ptr, S, screen_rows.data(), screen_pairs.data(),
                                               static_cast<int64_t>(screen_rows.size()), margins.data(), verify_orig);
        }
        if (verify_orig) {
            has_prev_orig = hit.has_value();
            if (hit.has_value()) { prev_o_row = hit->first; prev_o_pair = hit->second; prev_o_sub = substeps; }
        } else {
            has_prev_convex = hit.has_value();
            if (hit.has_value()) { prev_c_row = hit->first; prev_c_pair = hit->second; prev_c_sub = substeps; }
        }
        return hit;
    };

    auto check_targets = [&](const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& pos) -> bool {
        for (const auto& [idx, tgt] : spec.joint_targets) {
            if ((pos.row(idx).transpose() - tgt).lpNorm<Eigen::Infinity>() > 1e-4) {
                failure = "joint position tolerance at waypoint " + std::to_string(idx);
                return false;
            }
        }
        Eigen::VectorXd r_tmp;
        for (const auto& [idx, specs] : spec.pose_constraints) {
            snap.kin.fk_single(pos.data() + idx * J, bpos.data(), brot.data());
            for (const auto& cs : specs) {
                if (!snap.kin.eval_pose_constraint(cs, bpos.data(), brot.data(), nullptr, nullptr, r_tmp, nullptr)) {
                    failure = "pose tolerance at waypoint " + std::to_string(idx);
                    return false;
                }
            }
        }
        return true;
    };

    auto pair_name = [&](int64_t pid) -> std::string {
        return snap.pair_display_name(pid);
    };

    // Per-job adaptive inter-sample points injected whenever Conservative Advancement catches tunneling.
    std::vector<std::pair<int, double>> extra_samples;

    auto feasible = [&](const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& pos,
                        const CollisionEvalResult* seg_eval = nullptr,
                        int seg_eval_substeps = 0,
                        bool seg_eval_convex_ok = false,
                        bool record_ccd_sample = false) -> bool {
        const auto tv0 = Clock::now();
        bool ok = false;
        do {
            if (!pos.allFinite()) {
                failure = "trajectory violates joint limits";
                break;
            }
            bool lim_ok = true;
            for (int t = 0; t < T && lim_ok; ++t) {
                for (int j = 0; j < J; ++j) {
                    const double v = pos(t, j);
                    if (v < lower(j) || v > upper(j)) { lim_ok = false; break; }
                }
            }
            if (!lim_ok) {
                failure = "trajectory violates joint limits";
                break;
            }
            if (spec.has_dt_vel) {
                bool vel_ok = true;
                for (int t = 0; t + 1 < T && vel_ok; ++t) {
                    for (int j = 0; j < J; ++j) {
                        if (std::isfinite(vmax(j)) &&
                            std::abs(pos(t + 1, j) - pos(t, j)) / spec.dt_vel > vmax(j) * (1.0 + 1e-6)) {
                            vel_ok = false;
                            break;
                        }
                    }
                }
                if (!vel_ok) {
                    failure = "trajectory violates joint velocity limits";
                    break;
                }
                if (has_v0) {
                    // From |q''(t)| <= a_max integrated twice over [0, h]: |q1 - q0 - h*v0| <= 0.5 * a_max * h^2.
                    const double h = spec.dt_vel;
                    bool acc_ok = true;
                    for (int j = 0; j < J; ++j) {
                        if (std::isfinite(amax(j))) {
                            const double diff = std::abs(pos(1, j) - pos(0, j) - h * spec.start_velocity(j));
                            if (diff > 0.5 * amax(j) * h * h * (1.0 + 1e-6)) {
                                acc_ok = false;
                                break;
                            }
                        }
                    }
                    if (!acc_ok) {
                        failure = "trajectory violates first-segment start acceleration limit";
                        break;
                    }
                }
            }
            if (!check_targets(pos)) break;
            if (!has_col) {
                failure.clear();
                ok = true;
                break;
            }
            const int substeps = detail::grid_substeps(pos, validation_substeps, spec.max_segment_length);
            const int64_t S_total = static_cast<int64_t>((T - 1) * substeps + 1);
            bool skip_convex = false;
            bool have_val_samples = false;
            if (seg_eval != nullptr && seg_eval_substeps == substeps &&
                static_cast<int64_t>(seg_eval->frames.size()) == (S_total - 1) * N_shapes * 12) {
                val_frames.resize(static_cast<size_t>(S_total * N_shapes * 12));
                std::copy(seg_eval->frames.begin(), seg_eval->frames.end(), val_frames.begin());
                std::copy(goal_frame.begin(), goal_frame.end(), val_frames.begin() + (S_total - 1) * N_shapes * 12);
                skip_convex = seg_eval_convex_ok;
            } else {
                detail::build_segment_samples(pos, substeps, true, val_samples);
                have_val_samples = true;
                val_frames.resize(static_cast<size_t>(S_total * N_shapes * 12));
                snap.compute_frames(val_samples.data(), S_total, val_frames.data());
            }
            if (!skip_convex && P > 0) {
                auto hit = check_frames(val_frames.data(), S_total, substeps, false);
                if (hit.has_value()) {
                    const int seg = std::min(static_cast<int>(hit->first / substeps), T - 2);
                    failure = "collision clearance on segment " + std::to_string(seg) + ", pair " + pair_name(hit->second);
                    break;
                }
            }
            if (!skip_convex && use_esdf) {
                if (!have_val_samples) {
                    detail::build_segment_samples(pos, substeps, true, val_samples);
                    have_val_samples = true;
                }
                auto ehit = snap.first_esdf_violation_samples(val_samples.data(), val_frames.data(), S_total, spec.world_margin);
                if (ehit.has_value()) {
                    const int seg = std::min(static_cast<int>(ehit->first / substeps), T - 2);
                    failure = "ESDF clearance on segment " + std::to_string(seg) + ", pair " + pair_name(ehit->second);
                    break;
                }
            }
            // Sound kinematic-chain Conservative Advancement check (convex hull + ESDF sphere-cover).
            {
                auto ccd_hit = snap.first_violation_continuous(
                    pos.data(), T, substeps, val_frames.data(), margins.data(), false, 1e-3,
                    nullptr, nullptr, spec.strict_ccd, nullptr, 3, spec.world_margin);
                if (ccd_hit.has_value()) {
                    failure = "continuous collision clearance on segment " + std::to_string(ccd_hit->seg) +
                              ", pair " + pair_name(ccd_hit->pair_id);
                    if (record_ccd_sample && extra_samples.size() < 32) {
                        bool dup = false;
                        for (const auto& [es, eu] : extra_samples) {
                            if (es == ccd_hit->seg && std::abs(eu - ccd_hit->u) < 0.005) { dup = true; break; }
                        }
                        if (!dup) extra_samples.emplace_back(ccd_hit->seg, ccd_hit->u);
                    }
                    break;
                }
            }
            if (P > 0 && has_mesh) {
                auto hit = check_frames(val_frames.data(), S_total, substeps, true);
                if (hit.has_value()) {
                    const int seg = std::min(static_cast<int>(hit->first / substeps), T - 2);
                    failure = "original mesh clearance on segment " + std::to_string(seg) + ", pair " + pair_name(hit->second);
                    break;
                }
            }
            // When start_velocity is non-zero, also check the segment-0 cubic Hermite braking arc so SQP
            // never terminates early on a linear chord whose Hermite tangent bows into an obstacle.
            if (has_v0 && spec.start_velocity.squaredNorm() > 1e-12) {
                const double h0_est = detail::estimate_seg0_hermite_dt(
                    pos, spec.start_velocity, vmax, amax, spec.has_dt_vel, spec.dt_vel);
                std::vector<double> h_q;
                double h01_unused[3];
                detail::build_seg0_hermite_samples(pos, spec.start_velocity, h0_est, h_q, h01_unused);
                std::vector<double> h_frames(static_cast<size_t>(3 * N_shapes * 12));
                snap.compute_frames(h_q.data(), 3, h_frames.data());
                if (P > 0) {
                    std::vector<int64_t> hr, hp;
                    snap.screen_aabbs(h_frames.data(), 3, margins.data(), false, 1, hr, hp);
                    if (!hr.empty()) {
                        auto hhit = snap.first_violation_indexed(
                            h_frames.data(), 3, hr.data(), hp.data(),
                            static_cast<int64_t>(hr.size()), margins.data(), false);
                        if (hhit.has_value()) {
                            failure = "segment-0 Hermite tangent clearance, pair " + pair_name(hhit->second);
                            break;
                        }
                    }
                }
                if (use_esdf) {
                    auto ehit = snap.first_esdf_violation_samples(h_q.data(), h_frames.data(), 3, spec.world_margin);
                    if (ehit.has_value()) {
                        failure = "segment-0 Hermite tangent ESDF clearance, pair " + pair_name(ehit->second);
                        break;
                    }
                }
            }
            failure.clear();
            ok = true;
        } while (false);
        validate_ms += std::chrono::duration<double, std::milli>(Clock::now() - tv0).count();
        return ok;
    };

    // Zero interior free waypoints fast path.
    if (n_free_wp <= 0) {
        const bool ok = feasible(q_init);
        const double total_ms = std::chrono::duration<double, std::milli>(Clock::now() - t_job_start).count();
        out.success = ok;
        if (ok) {
            out.positions = q_init;
            out.cost = 0.0;
        } else {
            out.reason = failure;
        }
        out.sqp_ms = std::max(0.0, total_ms - validate_ms);
        out.validate_ms = validate_ms;
        return out;
    }

    // Build smoothness matrix M_s (n_smooth_rows x T) and offset b_s (n_smooth_rows x J)
    // for first and second waypoint differences plus optional start-velocity virtual predecessor.
    const int n_d1 = T - 1;
    const int n_d2 = std::max(0, T - 2);
    const int n_v0_row = has_v0 ? 1 : 0;
    const int n_srows = n_d1 + n_d2 + n_v0_row;
    Eigen::MatrixXd M_s = Eigen::MatrixXd::Zero(n_srows, T);
    Eigen::MatrixXd b_s = Eigen::MatrixXd::Zero(n_srows, J);
    for (int i = 0; i < n_d1; ++i) {
        M_s(i, i) = -1.0;
        M_s(i, i + 1) = 1.0;
    }
    const double d2_scale = 0.1 / (spec.dt_smooth * spec.dt_smooth);
    for (int i = 0; i < n_d2; ++i) {
        M_s(n_d1 + i, i) = d2_scale;
        M_s(n_d1 + i, i + 1) = -2.0 * d2_scale;
        M_s(n_d1 + i, i + 2) = d2_scale;
    }
    if (has_v0) {
        const double dt_start = (spec.dt_start_smooth > 0.0)
            ? spec.dt_start_smooth
            : (spec.has_dt_vel ? spec.dt_vel : spec.dt_smooth);
        const double d2_start_scale = 0.1 / (dt_start * dt_start);
        const int r0 = n_d1 + n_d2;
        M_s(r0, 0) = -d2_start_scale;
        M_s(r0, 1) = d2_start_scale;
        b_s.row(r0) = (d2_start_scale * dt_start) * spec.start_velocity.transpose();
    }
    // M_free^T * M_free is (n_free_wp x n_free_wp); H = kron(M_free^T * M_free, I_J).
    const Eigen::MatrixXd M_free = M_s.middleCols(fixed_start, n_free_wp);
    const Eigen::MatrixXd H_wp = M_free.transpose() * M_free;
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(n_vars, n_vars);
    for (int r = 0; r < n_free_wp; ++r) {
        for (int c = 0; c < n_free_wp; ++c) {
            const double val = H_wp(r, c);
            if (val != 0.0) {
                for (int j = 0; j < J; ++j) {
                    H(r * J + j, c * J + j) = val;
                }
            }
        }
    }

    auto compute_smooth_cost_and_grad = [&](
        const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& q,
        Eigen::VectorXd* grad_out) -> double {
        const Eigen::MatrixXd res = M_s * q - b_s;  // (n_srows, J)
        if (grad_out != nullptr) {
            const Eigen::MatrixXd g_mat = M_free.transpose() * res;  // (n_free_wp, J)
            grad_out->resize(n_vars);
            for (int w = 0; w < n_free_wp; ++w) {
                for (int j = 0; j < J; ++j) (*grad_out)(w * J + j) = g_mat(w, j);
            }
        }
        return 0.5 * res.squaredNorm();
    };

    // Build pinned mask for inactive joints and matched interior joint targets.
    std::vector<uint8_t> pinned(static_cast<size_t>(n_vars), 0);
    bool any_pinned = false;
    if (static_cast<int>(spec.active_mask.size()) == J) {
        for (int j = 0; j < J; ++j) {
            if (!spec.active_mask[static_cast<size_t>(j)]) {
                any_pinned = true;
                for (int w = 0; w < n_free_wp; ++w) pinned[static_cast<size_t>(w * J + j)] = 1;
            }
        }
    }
    std::vector<std::pair<int, Eigen::VectorXd>> unpinned_joint_targets;
    for (const auto& [idx, tgt] : spec.joint_targets) {
        if (idx >= fixed_start && idx < T - fixed_end &&
            (q_init.row(idx).transpose() - tgt).lpNorm<Eigen::Infinity>() == 0.0) {
            any_pinned = true;
            for (int j = 0; j < J; ++j) pinned[static_cast<size_t>((idx - fixed_start) * J + j)] = 1;
        } else {
            unpinned_joint_targets.emplace_back(idx, tgt);
        }
    }

    // Evaluate segment collision + velocity + target constraints and optionally assemble dense (A, v).
    std::vector<double> eval_samples;
    struct EvalPack {
        CollisionEvalResult col;
        Eigen::VectorXd v;
        Eigen::MatrixXd A;
        bool seg_ok = true;
        bool cart_ok = true;
        double total_viol = 0.0;
    };

    auto eval_all = [&](const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& q,
                        int substeps, bool segment_minimum, bool use_cushion, bool linearize, EvalPack& ep) {
        detail::build_segment_samples(q, substeps, false, eval_samples);
        const int64_t S = static_cast<int64_t>((T - 1) * substeps);
        const Eigen::VectorXd& opt_m = use_cushion ? opt_margins_cushion : margins;
        const double opt_esdf_m = use_cushion ? esdf_margin_cushion : spec.world_margin;
        const double buffer = linearize ? detail::kCollisionLinearizationBufferM : 0.0;
        ep.col = snap.eval_collision_rows(eval_samples.data(), S, opt_m.data(), margins.data(),
                                          buffer, linearize, substeps, segment_minimum,
                                          opt_esdf_m, spec.world_margin);

        bool all_seg_ok = true;
        for (uint8_t b : ep.col.seg_ok) if (!b) { all_seg_ok = false; break; }

        // Count velocity & start-acceleration constraint rows when hard dt_vel is set.
        int n_vel_rows = 0;
        int n_acc0_rows = 0;
        if (spec.has_dt_vel) {
            for (int j = 0; j < J; ++j) if (std::isfinite(vmax(j))) n_vel_rows += 2 * (T - 1);
            if (has_v0) {
                for (int j = 0; j < J; ++j) if (std::isfinite(amax(j))) n_acc0_rows += 2;
            }
        }
        const int n_col_rows = static_cast<int>(ep.col.rows.size());
        // Evaluate any adaptive CCD extra sample points injected for tunneling segments.
        CollisionEvalResult extra_col;
        int n_extra_rows = 0;
        if (!extra_samples.empty() && has_col) {
            const int64_t E = static_cast<int64_t>(extra_samples.size());
            std::vector<double> extra_q(static_cast<size_t>(E * J));
            for (int64_t e = 0; e < E; ++e) {
                const int seg = extra_samples[static_cast<size_t>(e)].first;
                const double u = extra_samples[static_cast<size_t>(e)].second;
                const double w = 1.0 - u;
                for (int j = 0; j < J; ++j) extra_q[static_cast<size_t>(e * J + j)] = w * q(seg, j) + u * q(seg + 1, j);
            }
            const Eigen::VectorXd& extra_opt = use_cushion ? opt_margins_extra : opt_margins_cushion;
            const double extra_esdf_opt = use_cushion ? esdf_margin_extra : esdf_margin_cushion;
            extra_col = snap.eval_collision_rows(extra_q.data(), E, extra_opt.data(), margins.data(),
                                                 buffer, linearize, 1, false,
                                                 extra_esdf_opt, spec.world_margin);
            for (uint8_t b : extra_col.seg_ok) if (!b) { all_seg_ok = false; break; }
            n_extra_rows = static_cast<int>(extra_col.rows.size());
        }

        // Evaluate segment-0 Hermite bow-out samples when start_velocity is non-zero so SQP steers q[1]
        // to keep the actual curved tangent arc clear of obstacles. Use a q-independent reference duration
        // h0_ref so q_H(u) is strictly affine in q[1] with exact Jacobian d q_H(u) / d q[1] = h01(u) * I.
        CollisionEvalResult hermite0_col;
        double hermite0_h01[3] = {0.0, 0.0, 0.0};
        int n_hermite0_rows = 0;
        if (has_v0 && has_col && spec.start_velocity.squaredNorm() > 1e-12) {
            double t_brake_ref = 0.0;
            for (int j = 0; j < J; ++j) {
                if (std::isfinite(amax(j)) && amax(j) > 1e-9) {
                    t_brake_ref = std::max(t_brake_ref, std::abs(spec.start_velocity(j)) / amax(j));
                }
            }
            const double dt_nom = (spec.dt_start_smooth > 0.0) ? spec.dt_start_smooth : spec.dt_smooth;
            const double h0_ref = spec.has_dt_vel ? spec.dt_vel : std::clamp(std::max(t_brake_ref, dt_nom), 0.02, 0.35);
            std::vector<double> h_q;
            detail::build_seg0_hermite_samples(q, spec.start_velocity, h0_ref, h_q, hermite0_h01);
            const Eigen::VectorXd& h_opt = use_cushion ? opt_margins_cushion : margins;
            const double h_esdf_opt = use_cushion ? esdf_margin_cushion : spec.world_margin;
            hermite0_col = snap.eval_collision_rows(h_q.data(), 3, h_opt.data(), margins.data(),
                                                    buffer, linearize, 1, false,
                                                    h_esdf_opt, spec.world_margin);
            for (uint8_t b : hermite0_col.seg_ok) if (!b) { all_seg_ok = false; break; }
            n_hermite0_rows = static_cast<int>(hermite0_col.rows.size());
        }

        // Evaluate target constraints (joint + pose).
        bool cart_ok = true;
        int n_cart_dim = 0;
        std::vector<std::tuple<int, Eigen::VectorXd, Eigen::MatrixXd>> cart_items;
        for (const auto& [idx, tgt] : unpinned_joint_targets) {
            Eigen::VectorXd r = q.row(idx).transpose() - tgt;
            if (r.lpNorm<Eigen::Infinity>() > 1e-4) cart_ok = false;
            n_cart_dim += J;
            if (linearize) {
                cart_items.emplace_back(idx, std::move(r), Eigen::MatrixXd::Identity(J, J));
            } else {
                cart_items.emplace_back(idx, std::move(r), Eigen::MatrixXd());
            }
        }
        for (const auto& [idx, specs] : spec.pose_constraints) {
            snap.kin.fk_single(q.data() + idx * J, bpos.data(), brot.data(),
                               linearize ? jorig.data() : nullptr, linearize ? jaxis.data() : nullptr);
            for (const auto& cs : specs) {
                Eigen::VectorXd r;
                Eigen::MatrixXd jc;
                const bool ok_c = snap.kin.eval_pose_constraint(
                    cs, bpos.data(), brot.data(),
                    linearize ? jorig.data() : nullptr, linearize ? jaxis.data() : nullptr,
                    r, linearize ? &jc : nullptr);
                if (!ok_c) cart_ok = false;
                n_cart_dim += static_cast<int>(r.size());
                cart_items.emplace_back(idx, std::move(r), std::move(jc));
            }
        }

        const int total_rows = n_vel_rows + n_acc0_rows + n_col_rows + n_extra_rows + n_hermite0_rows + 2 * n_cart_dim;
        ep.v.resize(total_rows);
        if (linearize) ep.A.setZero(total_rows, n_vars);
        else ep.A.resize(0, 0);

        int row = 0;
        double viol = 0.0;

        // 1. Hard velocity limits (only when fixed dt is specified).
        if (spec.has_dt_vel) {
            const double inv_dt = 1.0 / spec.dt_vel;
            for (int seg = 0; seg < T - 1; ++seg) {
                const int w0 = seg - fixed_start;
                const int w1 = seg + 1 - fixed_start;
                for (int j = 0; j < J; ++j) {
                    if (!std::isfinite(vmax(j))) continue;
                    const double spd = (q(seg + 1, j) - q(seg, j)) * inv_dt;
                    if (std::abs(spd) > vmax(j) * (1.0 + 1e-6)) all_seg_ok = false;
                    const double r_pos = spd - vmax(j);
                    const double r_neg = -spd - vmax(j);
                    ep.v(row) = r_pos;
                    if (r_pos > 0.0) viol += r_pos;
                    if (linearize) {
                        if (w0 >= 0 && w0 < n_free_wp) ep.A(row, w0 * J + j) = -inv_dt;
                        if (w1 >= 0 && w1 < n_free_wp) ep.A(row, w1 * J + j) = inv_dt;
                    }
                    ++row;
                    ep.v(row) = r_neg;
                    if (r_neg > 0.0) viol += r_neg;
                    if (linearize) {
                        if (w0 >= 0 && w0 < n_free_wp) ep.A(row, w0 * J + j) = inv_dt;
                        if (w1 >= 0 && w1 < n_free_wp) ep.A(row, w1 * J + j) = -inv_dt;
                    }
                    ++row;
                }
            }
            // 1b. First-segment start-velocity acceleration bound: |q1 - q0 - h*v0| <= 0.5 * a_max * h^2
            // Written in rad/s units as |(q1 - q0)/h - v0| <= 0.5 * a_max * h.
            if (has_v0) {
                const double h = spec.dt_vel;
                const int w1 = 1 - fixed_start;
                for (int j = 0; j < J; ++j) {
                    if (!std::isfinite(amax(j))) continue;
                    const double dv = (q(1, j) - q(0, j)) * inv_dt - spec.start_velocity(j);
                    const double lim_v = 0.5 * amax(j) * h;
                    if (std::abs(dv) > lim_v * (1.0 + 1e-6)) all_seg_ok = false;
                    const double r_pos = dv - lim_v;
                    const double r_neg = -dv - lim_v;
                    ep.v(row) = r_pos;
                    if (r_pos > 0.0) viol += r_pos;
                    if (linearize && w1 >= 0 && w1 < n_free_wp) ep.A(row, w1 * J + j) = inv_dt;
                    ++row;
                    ep.v(row) = r_neg;
                    if (r_neg > 0.0) viol += r_neg;
                    if (linearize && w1 >= 0 && w1 < n_free_wp) ep.A(row, w1 * J + j) = -inv_dt;
                    ++row;
                }
            }
        }

        // 2. Collision rows.
        const double inv_sub = 1.0 / static_cast<double>(substeps);
        for (int k = 0; k < n_col_rows; ++k) {
            const double rv = ep.col.residuals[static_cast<size_t>(k)];
            ep.v(row) = rv;
            if (rv > 0.0) viol += rv;
            if (linearize) {
                const int64_t sample_idx = ep.col.rows[static_cast<size_t>(k)];
                const int seg = static_cast<int>(sample_idx / substeps);
                const double t_frac = static_cast<double>(sample_idx % substeps) * inv_sub;
                const double w0_coef = 1.0 - t_frac;
                const double w1_coef = t_frac;
                const int w0 = seg - fixed_start;
                const int w1 = seg + 1 - fixed_start;
                const double* gptr = ep.col.gradients.data() + k * J;
                if (w0 >= 0 && w0 < n_free_wp) {
                    for (int j = 0; j < J; ++j) ep.A(row, w0 * J + j) = w0_coef * gptr[j];
                }
                if (w1 >= 0 && w1 < n_free_wp) {
                    for (int j = 0; j < J; ++j) ep.A(row, w1 * J + j) = w1_coef * gptr[j];
                }
            }
            ++row;
        }

        // 2b. Adaptive CCD extra collision rows.
        for (int k = 0; k < n_extra_rows; ++k) {
            const double rv = extra_col.residuals[static_cast<size_t>(k)];
            ep.v(row) = rv;
            if (rv > 0.0) viol += rv;
            if (linearize) {
                const int64_t e_idx = extra_col.rows[static_cast<size_t>(k)];
                const int seg = extra_samples[static_cast<size_t>(e_idx)].first;
                const double t_frac = extra_samples[static_cast<size_t>(e_idx)].second;
                const double w0_coef = 1.0 - t_frac;
                const double w1_coef = t_frac;
                const int w0 = seg - fixed_start;
                const int w1 = seg + 1 - fixed_start;
                const double* gptr = extra_col.gradients.data() + k * J;
                if (w0 >= 0 && w0 < n_free_wp) {
                    for (int j = 0; j < J; ++j) ep.A(row, w0 * J + j) = w0_coef * gptr[j];
                }
                if (w1 >= 0 && w1 < n_free_wp) {
                    for (int j = 0; j < J; ++j) ep.A(row, w1 * J + j) = w1_coef * gptr[j];
                }
            }
            ++row;
        }

        // 2c. Segment-0 Hermite tangent-arc collision rows (d q_H(u) / d q[1] = h01(u) * I).
        for (int k = 0; k < n_hermite0_rows; ++k) {
            const double rv = hermite0_col.residuals[static_cast<size_t>(k)];
            ep.v(row) = rv;
            if (rv > 0.0) viol += rv;
            if (linearize) {
                const int64_t h_idx = hermite0_col.rows[static_cast<size_t>(k)];
                const double w1_coef = hermite0_h01[static_cast<size_t>(h_idx)];
                const int w1 = 1 - fixed_start;
                const double* gptr = hermite0_col.gradients.data() + k * J;
                if (w1 >= 0 && w1 < n_free_wp) {
                    for (int j = 0; j < J; ++j) ep.A(row, w1 * J + j) = w1_coef * gptr[j];
                }
            }
            ++row;
        }

        // 3. Equality targets (r <= 0 and -r <= 0 via L1 hinge).
        for (const auto& [idx, r, jc] : cart_items) {
            const int w = idx - fixed_start;
            const int dim = static_cast<int>(r.size());
            for (int d = 0; d < dim; ++d) {
                viol += std::abs(r(d));
                ep.v(row) = r(d);
                if (linearize && w >= 0 && w < n_free_wp) {
                    for (int j = 0; j < J; ++j) ep.A(row, w * J + j) = jc(d, j);
                }
                ++row;
                ep.v(row) = -r(d);
                if (linearize && w >= 0 && w < n_free_wp) {
                    for (int j = 0; j < J; ++j) ep.A(row, w * J + j) = -jc(d, j);
                }
                ++row;
            }
        }

        ep.seg_ok = all_seg_ok;
        ep.cart_ok = cart_ok;
        ep.total_viol = viol;
    };

    struct AttemptResult {
        bool has_sol = false;
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> sol;
        double cost = std::numeric_limits<double>::infinity();
        bool has_warm = false;
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> warm;
    };

    L1BoxQPSolver qp_solver;
    Eigen::VectorXd grad(n_vars), lo_box(n_vars), hi_box(n_vars);
    EvalPack cur_eval, trial_eval;

    auto run_attempt = [&](const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& seed_q,
                           int base_substeps, bool segment_minimum, bool use_cushion) -> AttemptResult {
        AttemptResult ar;
        auto q = seed_q;
        const bool initial_feasible = feasible(q);
        if (spec.first_feasible && initial_feasible) {
            ar.has_sol = true;
            ar.sol = q;
            ar.cost = compute_smooth_cost_and_grad(q, nullptr);
            return ar;
        }
        if (initial_feasible && fixed_start == 1 && fixed_end == 1 && !has_v0 &&
            spec.pose_constraints.empty() && spec.joint_targets.empty()) {
            bool is_straight = true;
            for (int t = 1; t < T - 1 && is_straight; ++t) {
                const double frac = static_cast<double>(t) / static_cast<double>(T - 1);
                for (int j = 0; j < J; ++j) {
                    const double interp = (1.0 - frac) * q(0, j) + frac * q(T - 1, j);
                    if (std::abs(q(t, j) - interp) > 1e-12) { is_straight = false; break; }
                }
            }
            if (is_straight) {
                ar.has_sol = true;
                ar.sol = q;
                ar.cost = compute_smooth_cost_and_grad(q, nullptr);
                return ar;
            }
        }

        if (initial_feasible) {
            ar.has_sol = true;
            ar.sol = q;
            ar.cost = compute_smooth_cost_and_grad(q, nullptr);
        }

        bool has_init_viol = false;
        double init_viol = 0.0;
        double min_viol = std::numeric_limits<double>::infinity();
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> min_viol_q;
        double radius = 0.1;
        double penalty = 10.0;

        for (int it = 0; it < spec.iterations; ++it) {
            if (cancel_flag && cancel_flag->load(std::memory_order_relaxed)) return ar;
            if (deadline_exceeded()) {
                out.timed_out = true;
                failure = "replan time budget exceeded";
                return ar;
            }
            ++out.sqp_iterations;

            const int substeps = detail::grid_substeps(q, base_substeps, spec.max_segment_length);
            eval_all(q, substeps, segment_minimum, use_cushion, true, cur_eval);
            if (!cur_eval.col.witnesses.empty()) {
                out.witnesses = cur_eval.col.witnesses;
            }

            const double cost = compute_smooth_cost_and_grad(q, &grad);
            const double cur_viol = cur_eval.total_viol;
            if (!has_init_viol) {
                init_viol = cur_viol;
                has_init_viol = true;
            }
            if (cur_viol < min_viol) {
                min_viol = cur_viol;
                min_viol_q = q;
            }

            qp_solver.setup(H, grad, cur_eval.A, cur_eval.v, penalty);
            double predicted = 0.0;
            bool tr_exhausted = true;
            while (radius >= 1e-4) {
                if (cancel_flag && cancel_flag->load(std::memory_order_relaxed)) return ar;
                if (deadline_exceeded()) {
                    out.timed_out = true;
                    failure = "replan time budget exceeded";
                    return ar;
                }
                for (int w = 0; w < n_free_wp; ++w) {
                    const int t = w + fixed_start;
                    for (int j = 0; j < J; ++j) {
                        const int idx = w * J + j;
                        if (any_pinned && pinned[static_cast<size_t>(idx)]) {
                            lo_box(idx) = 0.0;
                            hi_box(idx) = 0.0;
                        } else {
                            lo_box(idx) = std::max(lower(j) - q(t, j), -radius);
                            hi_box(idx) = std::min(upper(j) - q(t, j), radius);
                        }
                    }
                }
                ++out.qp_calls;
                const auto qp_res = qp_solver.solve(lo_box, hi_box);
                if (!qp_res.solved) {
                    radius *= 0.25;
                    continue;
                }
                const Eigen::VectorXd& step = qp_res.x;
                double lin_viol = 0.0;
                if (cur_eval.v.size() > 0) {
                    const Eigen::VectorXd lin = cur_eval.v + cur_eval.A * step;
                    lin_viol = lin.cwiseMax(0.0).sum();
                }
                predicted = penalty * (cur_viol - lin_viol) - step.dot(grad) - 0.5 * step.dot(H * step);
                if (predicted <= 1e-9) {
                    tr_exhausted = false;
                    break;
                }
                auto trial = q;
                for (int w = 0; w < n_free_wp; ++w) {
                    const int t = w + fixed_start;
                    for (int j = 0; j < J; ++j) {
                        trial(t, j) = std::clamp(q(t, j) + step(w * J + j), lower(j), upper(j));
                    }
                }
                eval_all(trial, substeps, segment_minimum, use_cushion, false, trial_eval);
                const double trial_cost = compute_smooth_cost_and_grad(trial, nullptr);
                const double trial_viol = trial_eval.total_viol;
                const double actual = cost - trial_cost + penalty * (cur_viol - trial_viol);
                if (actual >= 0.25 * predicted) {
                    q = trial;
                    radius *= 1.5;
                    if (trial_viol < min_viol) {
                        min_viol = trial_viol;
                        min_viol_q = q;
                    }
                    if (trial_cost < ar.cost && trial_eval.seg_ok && trial_eval.cart_ok) {
                        if (feasible(q, &trial_eval.col, substeps, trial_eval.seg_ok, true)) {
                            ar.has_sol = true;
                            ar.sol = q;
                            ar.cost = trial_cost;
                            if (spec.first_feasible) return ar;
                        }
                    }
                    tr_exhausted = false;
                    break;
                }
                radius *= 0.25;
            }
            if (tr_exhausted) predicted = 0.0;

            if (predicted <= 1e-9) {
                if (ar.has_sol && (q - ar.sol).lpNorm<Eigen::Infinity>() == 0.0) break;
                const size_t prev_extra = extra_samples.size();
                if (cur_eval.seg_ok && cur_eval.cart_ok && feasible(q, &cur_eval.col, substeps, cur_eval.seg_ok, true)) {
                    if (cost < ar.cost) {
                        ar.has_sol = true;
                        ar.sol = q;
                        ar.cost = cost;
                    }
                    if (spec.first_feasible) return ar;
                    break;
                }
                if (extra_samples.size() > prev_extra) {
                    // A new continuous-collision sample was injected; continue SQP with fresh trust region.
                    radius = 0.1;
                    continue;
                }
                if (penalty >= 1e5) break;
                penalty *= 10.0;
                radius = 0.1;
            }
        }

        if (!ar.has_sol && has_init_viol && min_viol < init_viol && min_viol_q.rows() == T) {
            ar.has_warm = true;
            ar.warm = std::move(min_viol_q);
        }
        return ar;
    };

    struct AttemptCfg { int substeps; bool seg_min; bool cushion; };
    std::vector<AttemptCfg> attempts;
    if (spec.first_feasible && has_col) attempts.push_back({validation_substeps, true, true});
    attempts.push_back({spec.collision_substeps, false, true});
    attempts.push_back({validation_substeps, false, false});

    auto seed_cur = q_init;
    for (size_t ai = 0; ai < attempts.size(); ++ai) {
        if (cancel_flag && cancel_flag->load(std::memory_order_relaxed)) break;
        if (deadline_exceeded()) {
            out.timed_out = true;
            failure = "replan time budget exceeded";
            break;
        }
        const auto& ac = attempts[ai];
        extra_samples.clear();
        auto ar = run_attempt(seed_cur, ac.substeps, ac.seg_min, ac.cushion);
        if (ar.has_sol) {
            out.success = true;
            out.timed_out = false;
            out.positions = std::move(ar.sol);
            out.cost = ar.cost;
            break;
        }
        if (out.timed_out) break;
        if (!has_col) break;
        // Avoid propagating a trapped seg_min local minimum into full-sample attempts.
        if (ac.seg_min) {
            seed_cur = q_init;
        } else if (ar.has_warm) {
            seed_cur = std::move(ar.warm);
        }
    }

    const double total_ms = std::chrono::duration<double, std::milli>(Clock::now() - t_job_start).count();
    out.sqp_ms = std::max(0.0, total_ms - validate_ms);
    out.validate_ms = validate_ms;
    if (!out.success) {
        out.reason = failure.empty() ? "trajectory optimization found no feasible path" : failure;
    }
    return out;
}

inline std::vector<SQPJobResultNative> solve_sqp_batch(const WorldSnapshot& snap,
                                                       const std::vector<SQPJobSpecNative>& specs,
                                                       bool race = false) {
    const int64_t B = static_cast<int64_t>(specs.size());
    std::vector<SQPJobResultNative> results(static_cast<size_t>(B));
    if (B == 0) return results;
    if (B == 1) {
        results[0] = solve_sqp_job(snap, specs[0], nullptr);
        return results;
    }
    std::atomic<bool> race_won{false};
    std::atomic<int64_t> next_idx{0};
    const std::atomic<bool>* flag_ptr = race ? &race_won : nullptr;
    const int64_t workers = std::min<int64_t>(B, static_cast<int64_t>(ThreadPool::instance().max_threads()));
    ThreadPool::instance().parallel_for(workers, 1, [&](int64_t, int64_t) {
        while (true) {
            if (race && race_won.load(std::memory_order_relaxed)) break;
            const int64_t i = next_idx.fetch_add(1, std::memory_order_relaxed);
            if (i >= B) break;
            results[static_cast<size_t>(i)] = solve_sqp_job(snap, specs[static_cast<size_t>(i)], flag_ptr);
            if (race && results[static_cast<size_t>(i)].success) {
                race_won.store(true, std::memory_order_relaxed);
            }
        }
    });
    return results;
}

inline std::vector<SQPJobResultNative> solve_sqp_batch(const PlanningWorld& world,
                                                       const std::vector<SQPJobSpecNative>& specs,
                                                       bool race = false) {
    auto snap = world.snapshot();
    return solve_sqp_batch(*snap, specs, race);
}

}  // namespace gobot::planning
