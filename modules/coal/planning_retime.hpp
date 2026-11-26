#pragma once

#include "planning_spline.hpp"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace gobot::planning {

using RowMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

struct TrajectoryNative {
    Eigen::VectorXd times;        // (T,)
    RowMatrixXd positions;        // (T, J)
    RowMatrixXd velocities;       // (T, J)
    RowMatrixXd accelerations;    // (T, J)
    bool has_dt = false;
    double dt = 0.0;
    int spline_order = 3;         // 3 = cubic Hermite (C^1), 5 = quintic Hermite (C^2)

    int num_waypoints() const { return static_cast<int>(positions.rows()); }
    int num_joints() const { return static_cast<int>(positions.cols()); }
    double duration() const {
        return times.size() > 0 ? (times(times.size() - 1) - times(0)) : 0.0;
    }

    // Shared single-time evaluator supporting both cubic (order 3) and C^2 quintic (order 5) splines.
    void sample_single(double t, Eigen::VectorXd& pos,
                       Eigen::VectorXd* vel = nullptr,
                       Eigen::VectorXd* acc = nullptr,
                       Eigen::VectorXd* jerk = nullptr) const {
        const int T = num_waypoints();
        const int J = num_joints();
        const double tc = std::clamp(t, times(0), times(T - 1));
        int idx = static_cast<int>(std::upper_bound(times.data(), times.data() + T, tc) - times.data()) - 1;
        idx = std::clamp(idx, 0, T - 2);
        const double dt_seg = times(idx + 1) - times(idx);

        pos.resize(J);
        if (vel) vel->resize(J);
        if (acc) acc->resize(J);
        if (jerk) jerk->resize(J);

        if (dt_seg <= 0.0) {
            const int knot = (tc >= times(idx + 1)) ? (idx + 1) : idx;
            for (int j = 0; j < J; ++j) {
                pos(j) = positions(knot, j);
                if (vel) (*vel)(j) = velocities(knot, j);
                if (acc) (*acc)(j) = accelerations(knot, j);
                if (jerk) (*jerk)(j) = 0.0;
            }
            return;
        }
        const double u = std::clamp((tc - times(idx)) / dt_seg, 0.0, 1.0);

        if (spline_order == 5) {
            for (int j = 0; j < J; ++j) {
                eval_quintic_hermite_scalar(
                    positions(idx, j), velocities(idx, j), accelerations(idx, j),
                    positions(idx + 1, j), velocities(idx + 1, j), accelerations(idx + 1, j),
                    dt_seg, u,
                    &pos(j),
                    vel ? &(*vel)(j) : nullptr,
                    acc ? &(*acc)(j) : nullptr,
                    jerk ? &(*jerk)(j) : nullptr);
            }
        } else {
            for (int j = 0; j < J; ++j) {
                eval_cubic_hermite_scalar(
                    positions(idx, j), velocities(idx, j),
                    positions(idx + 1, j), velocities(idx + 1, j),
                    dt_seg, u,
                    &pos(j),
                    vel ? &(*vel)(j) : nullptr,
                    acc ? &(*acc)(j) : nullptr,
                    jerk ? &(*jerk)(j) : nullptr);
            }
        }
    }

    // Shared batch evaluator for arrays of query times ts[0..S-1].
    void sample_batch(const double* ts, int64_t S,
                      double* out_pos,
                      double* out_vel = nullptr,
                      double* out_acc = nullptr,
                      double* out_jerk = nullptr) const {
        const int T = num_waypoints();
        const int J = num_joints();
        const double t_lo = times(0);
        const double t_hi = times(T - 1);
        for (int64_t s = 0; s < S; ++s) {
            const double tc = std::clamp(ts[s], t_lo, t_hi);
            int idx = static_cast<int>(std::upper_bound(times.data(), times.data() + T, tc) - times.data()) - 1;
            idx = std::clamp(idx, 0, T - 2);
            const double dt_seg = times(idx + 1) - times(idx);
            double* dp = out_pos + s * J;
            double* dv = out_vel ? (out_vel + s * J) : nullptr;
            double* da = out_acc ? (out_acc + s * J) : nullptr;
            double* dj = out_jerk ? (out_jerk + s * J) : nullptr;
            if (dt_seg <= 0.0) {
                const int knot = (tc >= times(idx + 1)) ? (idx + 1) : idx;
                for (int j = 0; j < J; ++j) {
                    dp[j] = positions(knot, j);
                    if (dv) dv[j] = velocities(knot, j);
                    if (da) da[j] = accelerations(knot, j);
                    if (dj) dj[j] = 0.0;
                }
                continue;
            }
            const double u = std::clamp((tc - times(idx)) / dt_seg, 0.0, 1.0);
            if (spline_order == 5) {
                for (int j = 0; j < J; ++j) {
                    eval_quintic_hermite_scalar(
                        positions(idx, j), velocities(idx, j), accelerations(idx, j),
                        positions(idx + 1, j), velocities(idx + 1, j), accelerations(idx + 1, j),
                        dt_seg, u,
                        dp + j,
                        dv ? (dv + j) : nullptr,
                        da ? (da + j) : nullptr,
                        dj ? (dj + j) : nullptr);
                }
            } else {
                for (int j = 0; j < J; ++j) {
                    eval_cubic_hermite_scalar(
                        positions(idx, j), velocities(idx, j),
                        positions(idx + 1, j), velocities(idx + 1, j),
                        dt_seg, u,
                        dp + j,
                        dv ? (dv + j) : nullptr,
                        da ? (da + j) : nullptr,
                        dj ? (dj + j) : nullptr);
                }
            }
        }
    }

    // Build flat sample array [((T-1)*substeps + 1), J] uniformly spaced in time per segment.
    void sample_segments(int substeps, std::vector<double>& out_samples) const {
        const int T = num_waypoints();
        const int J = num_joints();
        const int n_segs = T - 1;
        const int total = n_segs * substeps + 1;
        out_samples.resize(static_cast<size_t>(total * J));
        const double inv_sub = 1.0 / static_cast<double>(substeps);
        for (int seg = 0; seg < n_segs; ++seg) {
            const double dt_seg = times(seg + 1) - times(seg);
            const double* p0 = positions.data() + seg * J;
            const double* p1 = positions.data() + (seg + 1) * J;
            const double* v0 = velocities.data() + seg * J;
            const double* v1 = velocities.data() + (seg + 1) * J;
            const double* a0 = accelerations.data() + seg * J;
            const double* a1 = accelerations.data() + (seg + 1) * J;
            for (int k = 0; k < substeps; ++k) {
                const double u = static_cast<double>(k) * inv_sub;
                double* dst = out_samples.data() + (seg * substeps + k) * J;
                if (dt_seg <= 0.0) {
                    std::copy(p0, p0 + J, dst);
                } else if (spline_order == 5) {
                    for (int j = 0; j < J; ++j) {
                        eval_quintic_hermite_scalar(p0[j], v0[j], a0[j], p1[j], v1[j], a1[j], dt_seg, u, dst + j);
                    }
                } else {
                    for (int j = 0; j < J; ++j) {
                        eval_cubic_hermite_scalar(p0[j], v0[j], p1[j], v1[j], dt_seg, u, dst + j);
                    }
                }
            }
        }
        std::copy(positions.data() + (T - 1) * J, positions.data() + T * J,
                  out_samples.data() + n_segs * substeps * J);
    }
};

struct BoundsCheckResult {
    bool ok = true;
    std::string reason;
    std::vector<int> bad_segments;
};

namespace retime_detail {

inline void hermite_segment_peaks(const double* d, const double* v0, const double* v1,
                                  int J, double h, double* max_v, double* max_a) {
    const double inv_h = 1.0 / h;
    for (int j = 0; j < J; ++j) {
        const double six_d_h = 6.0 * d[j] * inv_h;
        const double c1 = six_d_h - 4.0 * v0[j] - 2.0 * v1[j];
        const double c2 = -six_d_h + 3.0 * (v0[j] + v1[j]);
        const double a0 = c1 * inv_h;
        const double a1 = (c1 + 2.0 * c2) * inv_h;
        max_a[j] = std::max(std::abs(a0), std::abs(a1));

        double mv = std::max(std::abs(v0[j]), std::abs(v1[j]));
        if (std::abs(c2) > 1e-12) {
            const double u_star = -c1 / (2.0 * c2);
            if (u_star > 0.0 && u_star < 1.0) {
                const double v_star = v0[j] - (c1 * c1) / (4.0 * c2);
                mv = std::max(mv, std::abs(v_star));
            }
        }
        max_v[j] = mv;
    }
}

inline bool hermite_segment_feasible(const double* d, const double* v0, const double* v1,
                                     int J, double h, const Eigen::VectorXd& vmax,
                                     const Eigen::VectorXd& amax, double tol = 1e-9) {
    const double inv_h = 1.0 / h;
    const double scale = 1.0 + tol;
    for (int j = 0; j < J; ++j) {
        const double vm = vmax(j) * scale;
        const double am = amax(j) * scale;
        const double six_d_h = 6.0 * d[j] * inv_h;
        const double c1 = six_d_h - 4.0 * v0[j] - 2.0 * v1[j];
        const double c2 = -six_d_h + 3.0 * (v0[j] + v1[j]);
        if (std::abs(c1 * inv_h) > am || std::abs((c1 + 2.0 * c2) * inv_h) > am) return false;
        if (std::abs(v0[j]) > vm || std::abs(v1[j]) > vm) return false;
        if (std::abs(c2) > 1e-12) {
            const double u_star = -c1 / (2.0 * c2);
            if (u_star > 0.0 && u_star < 1.0) {
                const double v_star = v0[j] - (c1 * c1) / (4.0 * c2);
                if (std::abs(v_star) > vm) return false;
            }
        }
    }
    return true;
}

inline double zero_vel_hermite_dt(const double* d, int J,
                                  const Eigen::VectorXd& vmax, const Eigen::VectorXd& amax) {
    double max_abs = 0.0;
    double hv = 0.0;
    double ha = 0.0;
    for (int j = 0; j < J; ++j) {
        const double ad = std::abs(d[j]);
        if (ad > max_abs) max_abs = ad;
        hv = std::max(hv, 1.5 * ad / vmax(j));
        ha = std::max(ha, std::sqrt(6.0 * ad / amax(j)));
    }
    if (max_abs <= 1e-12) return 1e-4;
    return std::max({hv, ha, 1e-4});
}

inline double min_feasible_hermite_dt(const double* d, const double* v0, const double* v1,
                                      int J, const Eigen::VectorXd& vmax, const Eigen::VectorXd& amax) {
    double min_dt_vel = 0.0;
    for (int j = 0; j < J; ++j) min_dt_vel = std::max(min_dt_vel, std::abs(d[j]) / vmax(j));
    const double h_min = std::max(min_dt_vel, 1e-4);

    std::vector<std::pair<double, double>> forbidden;
    forbidden.reserve(static_cast<size_t>(4 * J));
    for (int j = 0; j < J; ++j) {
        const double six_d = 6.0 * d[j];
        const double v42 = 4.0 * v0[j] + 2.0 * v1[j];
        const double v24 = 2.0 * v0[j] + 4.0 * v1[j];
        const double am = amax(j) * (1.0 + 1e-9);
        const double two_a = 2.0 * am;
        const double four_a = 4.0 * am;
        const double b_vals[4] = {v42, -v42, v24, -v24};
        const double c_vals[4] = {-six_d, six_d, -six_d, six_d};
        for (int q = 0; q < 4; ++q) {
            const double disc = b_vals[q] * b_vals[q] - four_a * c_vals[q];
            if (disc > 0.0) {
                const double sq = std::sqrt(disc);
                const double r_lo = (-b_vals[q] - sq) / two_a;
                const double r_hi = (-b_vals[q] + sq) / two_a;
                if (r_hi > h_min) forbidden.emplace_back(r_lo, r_hi);
            }
        }
    }
    std::sort(forbidden.begin(), forbidden.end());

    auto next_acc_feasible = [&](double h) -> double {
        double cur = h;
        for (const auto& [lo_v, hi_v] : forbidden) {
            if (lo_v > cur) break;
            if (hi_v > cur) cur = hi_v * (1.0 + 1e-12);
        }
        return cur;
    };

    auto vel_feasible = [&](double h) -> bool {
        const double inv_h = 1.0 / h;
        for (int j = 0; j < J; ++j) {
            const double vm = vmax(j) * (1.0 + 1e-9);
            if (std::abs(v0[j]) > vm || std::abs(v1[j]) > vm) return false;
            const double sdh = 6.0 * d[j] * inv_h;
            const double c1 = sdh - (4.0 * v0[j] + 2.0 * v1[j]);
            const double c2 = 3.0 * (v0[j] + v1[j]) - sdh;
            if (std::abs(c2) > 1e-12) {
                const double u_star = -c1 / (2.0 * c2);
                if (u_star > 0.0 && u_star < 1.0) {
                    const double v_star = v0[j] - (c1 * c1) / (4.0 * c2);
                    if (std::abs(v_star) > vm) return false;
                }
            }
        }
        return true;
    };

    const double h_acc = next_acc_feasible(h_min);
    if (vel_feasible(h_acc)) return h_acc;

    double h_lo = h_acc;
    double h_hi = next_acc_feasible(std::max(h_lo * 1.5, zero_vel_hermite_dt(d, J, vmax, amax)));
    for (int i = 0; i < 32; ++i) {
        if (vel_feasible(h_hi)) break;
        h_lo = h_hi;
        h_hi = next_acc_feasible(h_hi * 2.0);
    }
    for (int i = 0; i < 12; ++i) {
        const double mid = 0.5 * (h_lo + h_hi);
        const double cand = next_acc_feasible(mid);
        if (cand < h_hi && vel_feasible(cand)) h_hi = cand;
        else h_lo = mid;
    }
    return h_hi;
}

// Certified feasible cubic Hermite segment duration search (returns nullopt if infeasible).
inline std::optional<double> find_feasible_hermite_dt(
    const double* d, const double* v0, const double* v1,
    int J, const Eigen::VectorXd& vmax, const Eigen::VectorXd& amax,
    const double* p0 = nullptr, const double* p1 = nullptr,
    const Eigen::VectorXd* lower_limits = nullptr, const Eigen::VectorXd* upper_limits = nullptr) {
    for (int j = 0; j < J; ++j) {
        if (std::isfinite(vmax(j)) &&
            (std::abs(v0[j]) > vmax(j) * (1.0 + 1e-6) || std::abs(v1[j]) > vmax(j) * (1.0 + 1e-6))) {
            return std::nullopt;
        }
    }
    auto check_h = [&](double h) -> bool {
        if (!std::isfinite(h) || h <= 0.0) return false;
        if (!hermite_segment_feasible(d, v0, v1, J, h, vmax, amax, 1e-6)) return false;
        if (p0 != nullptr && p1 != nullptr && (lower_limits != nullptr || upper_limits != nullptr)) {
            for (int j = 0; j < J; ++j) {
                const auto [lo_p, hi_p] = cubic_hermite_scalar_range(p0[j], p1[j], v0[j], v1[j], h);
                if (lower_limits != nullptr && std::isfinite((*lower_limits)(j)) && lo_p < (*lower_limits)(j) - 1e-6)
                    return false;
                if (upper_limits != nullptr && std::isfinite((*upper_limits)(j)) && hi_p > (*upper_limits)(j) + 1e-6)
                    return false;
            }
        }
        return true;
    };
    const double h_cand = min_feasible_hermite_dt(d, v0, v1, J, vmax, amax);
    if (check_h(h_cand)) return h_cand;
    // Fallback log scan in case position limits constrain large h differently from acceleration limits.
    const double h_lo = 1e-4;
    const double h_hi = std::max(20.0, std::min(1e4, h_cand * 8.0));
    const double log_lo = std::log(h_lo), log_hi = std::log(h_hi);
    constexpr int kSamples = 64;
    for (int i = 0; i <= kSamples; ++i) {
        const double h = std::exp(log_lo + (log_hi - log_lo) * (static_cast<double>(i) / static_cast<double>(kSamples)));
        if (check_h(h)) return h;
    }
    return std::nullopt;
}

// Certified feasible quintic Hermite segment duration search (handles non-zero fixed v0, a0).
// Checks velocity, acceleration, jerk, and optional joint position limits using exact/certified bounds.
inline std::optional<double> find_feasible_quintic_dt(
    const double* p0, const double* v0, const double* a0,
    const double* p1, const double* v1, const double* a1,
    int J, const Eigen::VectorXd& vmax, const Eigen::VectorXd& amax, const Eigen::VectorXd& jmax,
    double h_hint,
    const Eigen::VectorXd* lower_limits = nullptr, const Eigen::VectorXd* upper_limits = nullptr) {
    constexpr double kTol = 1e-6;
    const double scale = 1.0 + kTol;
    for (int j = 0; j < J; ++j) {
        if (std::isfinite(vmax(j))) {
            if (std::abs(v0[j]) > vmax(j) * scale || std::abs(v1[j]) > vmax(j) * scale) return std::nullopt;
            // Immediate right-derivative overshoot check at t = 0+ when |v0| == vmax and a0 pushes outward.
            if (std::abs(v0[j]) >= vmax(j) * (1.0 - 1e-9) && v0[j] * a0[j] > 1e-9) return std::nullopt;
        }
        if (std::isfinite(amax(j))) {
            if (std::abs(a0[j]) > amax(j) * scale || std::abs(a1[j]) > amax(j) * scale) return std::nullopt;
        }
    }

    auto check_h = [&](double h) -> bool {
        if (!std::isfinite(h) || h <= 0.0) return false;
        for (int j = 0; j < J; ++j) {
            const auto pk = quintic_hermite_scalar_peaks(p0[j], v0[j], a0[j], p1[j], v1[j], a1[j], h);
            if (std::isfinite(vmax(j)) && (!std::isfinite(pk.max_vel) || pk.max_vel > vmax(j) * scale)) return false;
            if (std::isfinite(amax(j)) && (!std::isfinite(pk.max_acc) || pk.max_acc > amax(j) * scale)) return false;
            if (std::isfinite(jmax(j)) && (!std::isfinite(pk.max_jerk) || pk.max_jerk > jmax(j) * scale)) return false;
            if (lower_limits != nullptr || upper_limits != nullptr) {
                const auto [lo_p, hi_p] = quintic_hermite_scalar_pos_range(p0[j], v0[j], a0[j], p1[j], v1[j], a1[j], h);
                if (lower_limits != nullptr && std::isfinite((*lower_limits)(j)) && lo_p < (*lower_limits)(j) - 1e-6)
                    return false;
                if (upper_limits != nullptr && std::isfinite((*upper_limits)(j)) && hi_p > (*upper_limits)(j) + 1e-6)
                    return false;
            }
        }
        return true;
    };

    const double base_h = (std::isfinite(h_hint) && h_hint > 1e-4) ? h_hint : 0.1;
    // Scan log-spaced window [1e-4, max(32.0, base_h * 64.0)] to find any feasible window even when a0 != 0.
    const double h_min = 1e-4;
    const double h_max = std::max(32.0, std::min(1e3, base_h * 64.0));
    const double log_min = std::log(h_min);
    const double log_max = std::log(h_max);
    constexpr int kGrid = 72;

    double best_h = -1.0;
    double prev_bad = h_min;
    if (check_h(base_h)) {
        best_h = base_h;
    }
    for (int i = 0; i <= kGrid; ++i) {
        const double h = std::exp(log_min + (log_max - log_min) * (static_cast<double>(i) / static_cast<double>(kGrid)));
        if (best_h > 0.0 && h >= best_h) break;
        if (check_h(h)) {
            best_h = h;
            break;
        }
        prev_bad = h;
    }
    if (best_h <= 0.0) return std::nullopt;

    // Refine downward toward the minimum feasible h in [prev_bad, best_h].
    double lo = (prev_bad < best_h) ? prev_bad : (0.5 * best_h);
    double hi = best_h;
    for (int it = 0; it < 10; ++it) {
        const double mid = 0.5 * (lo + hi);
        if (check_h(mid)) hi = mid;
        else lo = mid;
    }
    return hi;
}

}  // namespace retime_detail

inline BoundsCheckResult check_trajectory_bounds(const TrajectoryNative& traj,
                                                 const Eigen::VectorXd& velocity_limits,
                                                 const Eigen::VectorXd& acceleration_limits,
                                                 bool check_velocity = true,
                                                 bool check_acceleration = true,
                                                 double tol = 1e-5,
                                                 const Eigen::VectorXd* lower_limits = nullptr,
                                                 const Eigen::VectorXd* upper_limits = nullptr,
                                                 const Eigen::VectorXd* jerk_limits = nullptr,
                                                 bool check_jerk = true) {
    BoundsCheckResult out;
    const int T = traj.num_waypoints();
    const int J = traj.num_joints();
    if (T < 2 || J <= 0) {
        throw std::invalid_argument("trajectory must have shape [T>=2, J>=1]");
    }
    if (traj.spline_order != 3 && traj.spline_order != 5) {
        throw std::invalid_argument("spline_order must be 3 (cubic) or 5 (quintic)");
    }
    if (traj.times.size() != T || traj.velocities.rows() != T || traj.velocities.cols() != J ||
        traj.accelerations.rows() != T || traj.accelerations.cols() != J) {
        throw std::invalid_argument("trajectory array shapes are inconsistent");
    }
    if (velocity_limits.size() != J) {
        throw std::invalid_argument("velocity_limits length must match num_joints");
    }
    if (acceleration_limits.size() != J) {
        throw std::invalid_argument("acceleration_limits length must match num_joints");
    }
    if (lower_limits != nullptr && lower_limits->size() != J) {
        throw std::invalid_argument("lower_limits length must match num_joints");
    }
    if (upper_limits != nullptr && upper_limits->size() != J) {
        throw std::invalid_argument("upper_limits length must match num_joints");
    }
    if (jerk_limits != nullptr && jerk_limits->size() != J) {
        throw std::invalid_argument("jerk_limits length must match num_joints");
    }
    const bool check_lo = (lower_limits != nullptr);
    const bool check_hi = (upper_limits != nullptr);
    const bool check_pos = (check_lo || check_hi);
    const bool do_jerk = (check_jerk && traj.spline_order == 5 && jerk_limits != nullptr);

    Eigen::VectorXd safe_vmax(J), safe_amax(J), safe_jmax(J);
    for (int j = 0; j < J; ++j) {
        if (std::isnan(velocity_limits(j)) || velocity_limits(j) <= 0.0) {
            throw std::invalid_argument("velocity_limits must be positive (or +inf for unlimited)");
        }
        if (std::isnan(acceleration_limits(j)) || acceleration_limits(j) <= 0.0) {
            throw std::invalid_argument("acceleration_limits must be positive (or +inf for unlimited)");
        }
        if (jerk_limits != nullptr && (std::isnan((*jerk_limits)(j)) || (*jerk_limits)(j) <= 0.0)) {
            throw std::invalid_argument("jerk_limits must be positive (or +inf for unlimited)");
        }
        safe_vmax(j) = velocity_limits(j);
        safe_amax(j) = acceleration_limits(j);
        safe_jmax(j) = do_jerk ? (*jerk_limits)(j) : std::numeric_limits<double>::infinity();
    }

    std::vector<double> d(static_cast<size_t>(J)), mv(static_cast<size_t>(J)), ma(static_cast<size_t>(J));
    const double scale = 1.0 + tol;

    for (int k = 0; k < T - 1; ++k) {
        const double h = traj.times(k + 1) - traj.times(k);
        if (!std::isfinite(h) || h < 0.0) {
            out.ok = false;
            out.bad_segments.push_back(k);
            if (out.reason.empty()) {
                char buf[160];
                std::snprintf(buf, sizeof(buf),
                              "Trajectory has non-monotonic or non-finite timestamps on segment %d (dt=%.6g)", k, h);
                out.reason = buf;
            }
            continue;
        }
        if (h == 0.0) {
            // Exact zero-duration segment: only valid if strictly stationary with consistent states within limits.
            bool seg_bad = false;
            for (int j = 0; j < J; ++j) {
                const double p0 = traj.positions(k, j);
                const double p1 = traj.positions(k + 1, j);
                const double v0 = traj.velocities(k, j);
                const double v1 = traj.velocities(k + 1, j);
                const double a0 = traj.accelerations(k, j);
                const double a1 = traj.accelerations(k + 1, j);
                if (!std::isfinite(p0) || !std::isfinite(p1) || !std::isfinite(v0) || !std::isfinite(v1) ||
                    (traj.spline_order == 5 && (!std::isfinite(a0) || !std::isfinite(a1)))) {
                    seg_bad = true;
                    if (out.reason.empty()) {
                        char buf[160];
                        std::snprintf(buf, sizeof(buf),
                                      "Zero-duration segment %d has non-finite state on joint %d", k, j);
                        out.reason = buf;
                    }
                    break;
                }
                if (check_pos) {
                    const double lo = check_lo ? (*lower_limits)(j) : -std::numeric_limits<double>::infinity();
                    const double hi = check_hi ? (*upper_limits)(j) : std::numeric_limits<double>::infinity();
                    if ((std::isfinite(lo) && (p0 < lo - 1e-6 || p1 < lo - 1e-6)) ||
                        (std::isfinite(hi) && (p0 > hi + 1e-6 || p1 > hi + 1e-6))) {
                        seg_bad = true;
                        if (out.reason.empty()) {
                            char buf[180];
                            std::snprintf(buf, sizeof(buf),
                                          "Zero-duration segment %d violates joint position limits on joint %d", k, j);
                            out.reason = buf;
                        }
                        break;
                    }
                }
                if (p1 != p0) {
                    seg_bad = true;
                    if (out.reason.empty()) {
                        char buf[180];
                        std::snprintf(buf, sizeof(buf),
                                      "Zero-duration segment %d has non-zero displacement on joint %d (dq=%.6g)",
                                      k, j, p1 - p0);
                        out.reason = buf;
                    }
                    break;
                }
                if (check_velocity && std::isfinite(safe_vmax(j)) &&
                    (std::abs(v0) > safe_vmax(j) * scale || std::abs(v1) > safe_vmax(j) * scale)) {
                    seg_bad = true;
                    if (out.reason.empty()) {
                        char buf[180];
                        std::snprintf(buf, sizeof(buf),
                                      "Zero-duration segment %d violates joint velocity limits on joint %d", k, j);
                        out.reason = buf;
                    }
                    break;
                }
                if (check_acceleration && std::isfinite(safe_amax(j)) && std::abs(v1 - v0) > 1e-9) {
                    seg_bad = true;
                    if (out.reason.empty()) {
                        char buf[180];
                        std::snprintf(buf, sizeof(buf),
                                      "Zero-duration segment %d has velocity discontinuity on joint %d", k, j);
                        out.reason = buf;
                    }
                    break;
                }
                if (traj.spline_order == 5) {
                    if (check_acceleration && std::isfinite(safe_amax(j)) &&
                        (std::abs(a0) > safe_amax(j) * scale || std::abs(a1) > safe_amax(j) * scale)) {
                        seg_bad = true;
                        if (out.reason.empty()) {
                            char buf[180];
                            std::snprintf(buf, sizeof(buf),
                                          "Zero-duration segment %d violates joint acceleration limits on joint %d", k, j);
                            out.reason = buf;
                        }
                        break;
                    }
                    if (do_jerk && std::isfinite(safe_jmax(j)) && std::abs(a1 - a0) > 1e-9) {
                        seg_bad = true;
                        if (out.reason.empty()) {
                            char buf[180];
                            std::snprintf(buf, sizeof(buf),
                                          "Zero-duration segment %d has acceleration discontinuity on joint %d", k, j);
                            out.reason = buf;
                        }
                        break;
                    }
                }
            }
            if (seg_bad) {
                out.ok = false;
                out.bad_segments.push_back(k);
            }
            continue;
        }

        if (traj.spline_order == 5) {
            bool seg_bad = false;
            int bad_pos_j = -1;
            double bad_pos_val = 0.0;
            int bad_v_j = -1, bad_a_j = -1, bad_j_j = -1;
            double max_vr = 0.0, max_ar = 0.0, max_jr = 0.0;

            for (int j = 0; j < J; ++j) {
                const double p0 = traj.positions(k, j);
                const double p1 = traj.positions(k + 1, j);
                const double v0 = traj.velocities(k, j);
                const double v1 = traj.velocities(k + 1, j);
                const double a0 = traj.accelerations(k, j);
                const double a1 = traj.accelerations(k + 1, j);
                if (!std::isfinite(p0) || !std::isfinite(p1) || !std::isfinite(v0) ||
                    !std::isfinite(v1) || !std::isfinite(a0) || !std::isfinite(a1)) {
                    seg_bad = true;
                    if (out.reason.empty()) {
                        char buf[160];
                        std::snprintf(buf, sizeof(buf),
                                      "Quintic curve has non-finite state on segment %d, joint %d", k, j);
                        out.reason = buf;
                    }
                    break;
                }
                if (check_pos) {
                    const double lo = check_lo ? (*lower_limits)(j) : -std::numeric_limits<double>::infinity();
                    const double hi = check_hi ? (*upper_limits)(j) : std::numeric_limits<double>::infinity();
                    const auto [pmin, pmax] = quintic_hermite_scalar_pos_range(p0, v0, a0, p1, v1, a1, h);
                    if ((std::isfinite(lo) && pmin < lo - 1e-6) || (std::isfinite(hi) && pmax > hi + 1e-6)) {
                        if (bad_pos_j < 0) {
                            bad_pos_j = j;
                            bad_pos_val = (std::isfinite(lo) && pmin < lo - 1e-6) ? pmin : pmax;
                        }
                    }
                }
                const auto pk = quintic_hermite_scalar_peaks(p0, v0, a0, p1, v1, a1, h);
                if (check_velocity && std::isfinite(safe_vmax(j))) {
                    const double vr = std::isfinite(pk.max_vel) ? (pk.max_vel / safe_vmax(j))
                                                                : std::numeric_limits<double>::infinity();
                    if (vr > max_vr) { max_vr = vr; bad_v_j = j; }
                }
                if (check_acceleration && std::isfinite(safe_amax(j))) {
                    const double ar = std::isfinite(pk.max_acc) ? (pk.max_acc / safe_amax(j))
                                                                : std::numeric_limits<double>::infinity();
                    if (ar > max_ar) { max_ar = ar; bad_a_j = j; }
                }
                if (do_jerk && std::isfinite(safe_jmax(j))) {
                    const double jr = std::isfinite(pk.max_jerk) ? (pk.max_jerk / safe_jmax(j))
                                                                 : std::numeric_limits<double>::infinity();
                    if (jr > max_jr) { max_jr = jr; bad_j_j = j; }
                }
            }
            if (!seg_bad) {
                if (bad_pos_j >= 0) {
                    seg_bad = true;
                    if (out.reason.empty()) {
                        char buf[180];
                        std::snprintf(buf, sizeof(buf),
                                      "Quintic curve violates joint position limits on segment %d, joint %d (value %.6f)",
                                      k, bad_pos_j, bad_pos_val);
                        out.reason = buf;
                    }
                } else if (check_velocity && max_vr > scale) {
                    seg_bad = true;
                    if (out.reason.empty()) {
                        char buf[180];
                        std::snprintf(buf, sizeof(buf),
                                      "Quintic curve violates joint velocity limits on segment %d, joint %d (%.2fx limit)",
                                      k, bad_v_j, max_vr);
                        out.reason = buf;
                    }
                } else if (check_acceleration && max_ar > scale) {
                    seg_bad = true;
                    if (out.reason.empty()) {
                        char buf[180];
                        std::snprintf(buf, sizeof(buf),
                                      "Quintic curve violates joint acceleration limits on segment %d, joint %d (%.2fx limit)",
                                      k, bad_a_j, max_ar);
                        out.reason = buf;
                    }
                } else if (do_jerk && max_jr > scale) {
                    seg_bad = true;
                    if (out.reason.empty()) {
                        char buf[180];
                        std::snprintf(buf, sizeof(buf),
                                      "Quintic curve violates joint jerk limits on segment %d, joint %d (%.2fx limit)",
                                      k, bad_j_j, max_jr);
                        out.reason = buf;
                    }
                }
            }
            if (seg_bad) {
                out.ok = false;
                out.bad_segments.push_back(k);
            }
            continue;
        }

        // Cubic Hermite (order 3) exact continuous position, velocity & acceleration verification.
        int bad_pos_j = -1;
        double bad_pos_val = 0.0;
        for (int j = 0; j < J; ++j) {
            d[static_cast<size_t>(j)] = traj.positions(k + 1, j) - traj.positions(k, j);
            if (check_pos) {
                const double lo = check_lo ? (*lower_limits)(j) : -std::numeric_limits<double>::infinity();
                const double hi = check_hi ? (*upper_limits)(j) : std::numeric_limits<double>::infinity();
                const auto [pmin, pmax] = cubic_hermite_scalar_range(
                    traj.positions(k, j), traj.positions(k + 1, j),
                    traj.velocities(k, j), traj.velocities(k + 1, j), h);
                if ((std::isfinite(lo) && pmin < lo - 1e-6) || (std::isfinite(hi) && pmax > hi + 1e-6)) {
                    if (bad_pos_j < 0) {
                        bad_pos_j = j;
                        bad_pos_val = (std::isfinite(lo) && pmin < lo - 1e-6) ? pmin : pmax;
                    }
                }
            }
        }
        retime_detail::hermite_segment_peaks(
            d.data(), traj.velocities.data() + k * J, traj.velocities.data() + (k + 1) * J,
            J, h, mv.data(), ma.data());
        double max_vr = 0.0, max_ar = 0.0;
        int bad_v_j = -1, bad_a_j = -1;
        for (int j = 0; j < J; ++j) {
            if (check_velocity && std::isfinite(safe_vmax(j))) {
                const double vr = std::isfinite(mv[static_cast<size_t>(j)]) ? (mv[static_cast<size_t>(j)] / safe_vmax(j))
                                                                            : std::numeric_limits<double>::infinity();
                if (vr > max_vr) { max_vr = vr; bad_v_j = j; }
            }
            if (check_acceleration && std::isfinite(safe_amax(j))) {
                const double ar = std::isfinite(ma[static_cast<size_t>(j)]) ? (ma[static_cast<size_t>(j)] / safe_amax(j))
                                                                            : std::numeric_limits<double>::infinity();
                if (ar > max_ar) { max_ar = ar; bad_a_j = j; }
            }
        }
        if (bad_pos_j >= 0) {
            out.ok = false;
            out.bad_segments.push_back(k);
            if (out.reason.empty()) {
                char buf[180];
                std::snprintf(buf, sizeof(buf),
                              "Hermite curve violates joint position limits on segment %d, joint %d (value %.6f)",
                              k, bad_pos_j, bad_pos_val);
                out.reason = buf;
            }
        } else if (check_velocity && max_vr > scale) {
            out.ok = false;
            out.bad_segments.push_back(k);
            if (out.reason.empty()) {
                char buf[180];
                std::snprintf(buf, sizeof(buf),
                              "Hermite curve violates joint velocity limits on segment %d, joint %d (%.2fx limit)",
                              k, bad_v_j, max_vr);
                out.reason = buf;
            }
        } else if (check_acceleration && max_ar > scale) {
            out.ok = false;
            out.bad_segments.push_back(k);
            if (out.reason.empty()) {
                char buf[180];
                std::snprintf(buf, sizeof(buf),
                              "Hermite curve violates joint acceleration limits on segment %d, joint %d (%.2fx limit)",
                              k, bad_a_j, max_ar);
                out.reason = buf;
            }
        }
    }
    return out;
}

namespace retime_detail {

inline TrajectoryNative retime_candidate(const RowMatrixXd& positions,
                                          const Eigen::VectorXd& velocity_limits,
                                          const Eigen::VectorXd& acceleration_limits,
                                          bool stop_at_endpoints = true,
                                          bool has_dt = false,
                                          double dt = 0.0,
                                          const Eigen::VectorXd* knot_velocity_scales = nullptr,
                                          const Eigen::VectorXd* jerk_limits = nullptr,
                                          int spline_order = 3,
                                          const Eigen::VectorXd* start_velocity = nullptr,
                                          const Eigen::VectorXd* start_acceleration = nullptr,
                                          const Eigen::VectorXd* lower_limits = nullptr,
                                          const Eigen::VectorXd* upper_limits = nullptr) {
    if (spline_order != 3 && spline_order != 5) {
        throw std::invalid_argument("spline_order must be 3 (cubic) or 5 (quintic)");
    }
    const int T = static_cast<int>(positions.rows());
    const int J = static_cast<int>(positions.cols());
    if (T < 2 || J <= 0) throw std::invalid_argument("positions must have shape [T>=2, J>=1]");
    if (velocity_limits.size() != J) {
        throw std::invalid_argument("velocity_limits length must match num_joints");
    }
    if (acceleration_limits.size() != J) {
        throw std::invalid_argument("acceleration_limits length must match num_joints");
    }
    if (jerk_limits != nullptr && jerk_limits->size() != J) {
        throw std::invalid_argument("jerk_limits length must match num_joints");
    }
    if (start_velocity != nullptr && (start_velocity->size() != J || !start_velocity->allFinite())) {
        throw std::invalid_argument("start_velocity must be a finite vector of length num_joints");
    }
    if (start_acceleration != nullptr) {
        if (spline_order != 5) {
            throw std::invalid_argument("start_acceleration is only supported for quintic (spline_order=5) trajectories");
        }
        if (start_acceleration->size() != J || !start_acceleration->allFinite()) {
            throw std::invalid_argument("start_acceleration must be a finite vector of length num_joints");
        }
    }
    if (lower_limits != nullptr && lower_limits->size() != J) {
        throw std::invalid_argument("lower_limits length must match num_joints");
    }
    if (upper_limits != nullptr && upper_limits->size() != J) {
        throw std::invalid_argument("upper_limits length must match num_joints");
    }
    if (has_dt && (!std::isfinite(dt) || dt <= 0.0)) {
        throw std::invalid_argument("dt must be finite and positive");
    }

    const bool has_v0 = (start_velocity != nullptr);
    const bool has_a0 = (start_acceleration != nullptr);
    const Eigen::VectorXd v0_fixed = has_v0 ? *start_velocity : Eigen::VectorXd::Zero(J);
    const Eigen::VectorXd a0_fixed = has_a0 ? *start_acceleration : Eigen::VectorXd::Zero(J);
    const bool seg0_boundary_motion =
        (has_v0 && v0_fixed.cwiseAbs().maxCoeff() > 0.0) ||
        (has_a0 && a0_fixed.cwiseAbs().maxCoeff() > 0.0);

    Eigen::VectorXd safe_vmax(J), safe_amax(J), safe_jmax(J);
    for (int j = 0; j < J; ++j) {
        if (std::isnan(velocity_limits(j)) || velocity_limits(j) <= 0.0) {
            throw std::invalid_argument("velocity_limits must be positive (or +inf for unlimited)");
        }
        if (std::isnan(acceleration_limits(j)) || acceleration_limits(j) <= 0.0) {
            throw std::invalid_argument("acceleration_limits must be positive (or +inf for unlimited)");
        }
        if (jerk_limits != nullptr && (std::isnan((*jerk_limits)(j)) || (*jerk_limits)(j) <= 0.0)) {
            throw std::invalid_argument("jerk_limits must be positive (or +inf for unlimited)");
        }
        safe_vmax(j) = std::isfinite(velocity_limits(j)) ? velocity_limits(j) : 1e6;
        safe_amax(j) = std::isfinite(acceleration_limits(j)) ? acceleration_limits(j) : 1e6;
        safe_jmax(j) = (jerk_limits != nullptr) ? (*jerk_limits)(j) : std::numeric_limits<double>::infinity();
    }

    Eigen::VectorXd k_scales = Eigen::VectorXd::Ones(T);
    if (knot_velocity_scales != nullptr) {
        if (knot_velocity_scales->size() != T) throw std::invalid_argument("knot_velocity_scales size mismatch");
        k_scales = knot_velocity_scales->cwiseMax(0.0).cwiseMin(1.0);
    }

    RowMatrixXd dq(T - 1, J);
    Eigen::VectorXd ds(T - 1);
    std::vector<uint8_t> nonzero(static_cast<size_t>(T - 1), 0);
    for (int k = 0; k < T - 1; ++k) {
        dq.row(k) = positions.row(k + 1) - positions.row(k);
        ds(k) = dq.row(k).norm();
        nonzero[static_cast<size_t>(k)] = (ds(k) > 1e-12) ? 1 : 0;
    }

    // C^2 Quintic Hermite path (order 5) with global time scaling (free dt) or fixed-dt validation.
    if (spline_order == 5) {
        TrajectoryNative traj;
        traj.spline_order = 5;
        traj.positions = positions;
        traj.velocities = RowMatrixXd::Zero(T, J);
        traj.accelerations = RowMatrixXd::Zero(T, J);
        traj.times.resize(T);

        if (has_dt) {
            traj.has_dt = true;
            traj.dt = dt;
            for (int i = 0; i < T; ++i) traj.times(i) = static_cast<double>(i) * dt;
            for (int i = 1; i < T - 1; ++i) {
                if (nonzero[static_cast<size_t>(i - 1)] && nonzero[static_cast<size_t>(i)]) {
                    traj.velocities.row(i) = (positions.row(i + 1) - positions.row(i - 1)) / (2.0 * dt);
                }
            }
            if (!stop_at_endpoints) {
                if (!has_v0 && nonzero[0]) traj.velocities.row(0) = dq.row(0) / dt;
                if (nonzero[static_cast<size_t>(T - 2)]) traj.velocities.row(T - 1) = dq.row(T - 2) / dt;
            }
            for (int i = 0; i < T; ++i) traj.velocities.row(i) *= k_scales(i);
            for (int k = 0; k < T - 1; ++k) {
                if (!nonzero[static_cast<size_t>(k)]) {
                    if (!(k == 0 && has_v0)) traj.velocities.row(k).setZero();
                    traj.velocities.row(k + 1).setZero();
                }
            }
            // Clamp non-fixed knot velocities to velocity_limits.
            for (int i = (has_v0 ? 1 : 0); i < T; ++i) {
                double ratio = 0.0;
                for (int j = 0; j < J; ++j) ratio = std::max(ratio, std::abs(traj.velocities(i, j)) / safe_vmax(j));
                if (ratio > 1.0) traj.velocities.row(i) /= ratio;
            }
            if (has_v0) traj.velocities.row(0) = v0_fixed.transpose();

            // C^2 knot accelerations (explicitly zero at end when stop_at_endpoints == true).
            for (int i = 1; i < T - 1; ++i) {
                if ((nonzero[static_cast<size_t>(i - 1)] || (i == 1 && seg0_boundary_motion)) &&
                    nonzero[static_cast<size_t>(i)]) {
                    traj.accelerations.row(i) =
                        ((traj.velocities.row(i + 1) - traj.velocities.row(i - 1)) / (2.0 * dt)) * (k_scales(i) * k_scales(i));
                }
            }
            if (!stop_at_endpoints || has_v0) {
                if (!has_a0 && (nonzero[0] || seg0_boundary_motion)) {
                    traj.accelerations.row(0) = ((traj.velocities.row(1) - traj.velocities.row(0)) / dt) * (k_scales(0) * k_scales(0));
                }
            }
            if (!stop_at_endpoints && nonzero[static_cast<size_t>(T - 2)]) {
                traj.accelerations.row(T - 1) =
                    ((traj.velocities.row(T - 1) - traj.velocities.row(T - 2)) / dt) * (k_scales(T - 1) * k_scales(T - 1));
            }
            for (int k = 0; k < T - 1; ++k) {
                if (!nonzero[static_cast<size_t>(k)] && !(k == 0 && seg0_boundary_motion)) {
                    if (!(k == 0 && has_a0)) traj.accelerations.row(k).setZero();
                    traj.accelerations.row(k + 1).setZero();
                }
            }
            for (int i = (has_a0 ? 1 : 0); i < T; ++i) {
                for (int j = 0; j < J; ++j) {
                    traj.accelerations(i, j) = std::clamp(traj.accelerations(i, j), -safe_amax(j), safe_amax(j));
                }
            }
            if (has_a0) traj.accelerations.row(0) = a0_fixed.transpose();
            // Preserve exact fixed dt and fixed start boundary without rescaling; caller validates bounds.
            return traj;
        }

        // Free timing: build initial well-proportioned cubic TOPP timing and knot velocities,
        // derive C^2 knot accelerations, then enforce bounds while preserving exact start boundary.
        TrajectoryNative base = retime_candidate(
            positions, velocity_limits, acceleration_limits, stop_at_endpoints, false, 0.0,
            knot_velocity_scales, nullptr, 3, start_velocity, nullptr, lower_limits, upper_limits);
        traj.times = base.times;
        traj.velocities = base.velocities;
        if (has_v0) traj.velocities.row(0) = v0_fixed.transpose();

        for (int i = 1; i < T - 1; ++i) {
            if ((nonzero[static_cast<size_t>(i - 1)] || (i == 1 && seg0_boundary_motion)) &&
                nonzero[static_cast<size_t>(i)]) {
                const double h_sum = std::max((traj.times(i) - traj.times(i - 1)) + (traj.times(i + 1) - traj.times(i)), 1e-12);
                traj.accelerations.row(i) =
                    ((traj.velocities.row(i + 1) - traj.velocities.row(i - 1)) / h_sum) * (k_scales(i) * k_scales(i));
            }
        }
        if (stop_at_endpoints) {
            if (!has_v0) traj.velocities.row(0).setZero();
            traj.velocities.row(T - 1).setZero();
            if (!has_a0 && !has_v0) traj.accelerations.row(0).setZero();
            traj.accelerations.row(T - 1).setZero();
        }
        if (!has_a0 && (has_v0 || !stop_at_endpoints) && (nonzero[0] || seg0_boundary_motion)) {
            const double h0 = std::max(traj.times(1) - traj.times(0), 1e-12);
            traj.accelerations.row(0) = ((traj.velocities.row(1) - traj.velocities.row(0)) / h0) * (k_scales(0) * k_scales(0));
        }
        if (!stop_at_endpoints && nonzero[static_cast<size_t>(T - 2)]) {
            const double hlast = std::max(traj.times(T - 1) - traj.times(T - 2), 1e-12);
            traj.accelerations.row(T - 1) =
                ((traj.velocities.row(T - 1) - traj.velocities.row(T - 2)) / hlast) * (k_scales(T - 1) * k_scales(T - 1));
        }
        for (int k = 0; k < T - 1; ++k) {
            if (!nonzero[static_cast<size_t>(k)] && !(k == 0 && seg0_boundary_motion)) {
                if (!(k == 0 && has_v0)) traj.velocities.row(k).setZero();
                traj.velocities.row(k + 1).setZero();
                if (!(k == 0 && has_a0)) traj.accelerations.row(k).setZero();
                traj.accelerations.row(k + 1).setZero();
            }
        }
        for (int i = (has_a0 ? 1 : 0); i < T; ++i) {
            for (int j = 0; j < J; ++j) {
                traj.accelerations(i, j) = std::clamp(traj.accelerations(i, j), -safe_amax(j), safe_amax(j));
            }
        }
        if (has_a0) traj.accelerations.row(0) = a0_fixed.transpose();

        if (!has_v0 && !has_a0) {
            // Standard global time scaling (exact geometric path & C^2 continuity).
            double r_v = 0.0, r_a = 0.0, r_j = 0.0;
            std::vector<uint8_t> scale_seg(static_cast<size_t>(T - 1), 0);
            for (int k = 0; k < T - 1; ++k) {
                const bool has_motion = (dq.row(k).cwiseAbs().maxCoeff() > 0.0) ||
                                        (traj.velocities.row(k).cwiseAbs().maxCoeff() > 0.0) ||
                                        (traj.velocities.row(k + 1).cwiseAbs().maxCoeff() > 0.0) ||
                                        (traj.accelerations.row(k).cwiseAbs().maxCoeff() > 0.0) ||
                                        (traj.accelerations.row(k + 1).cwiseAbs().maxCoeff() > 0.0);
                if (!has_motion) continue;
                scale_seg[static_cast<size_t>(k)] = 1;
                const double h = traj.times(k + 1) - traj.times(k);
                if (h <= 0.0) continue;
                for (int j = 0; j < J; ++j) {
                    const auto pk = quintic_hermite_scalar_peaks(
                        traj.positions(k, j), traj.velocities(k, j), traj.accelerations(k, j),
                        traj.positions(k + 1, j), traj.velocities(k + 1, j), traj.accelerations(k + 1, j),
                        h);
                    if (std::isfinite(velocity_limits(j))) r_v = std::max(r_v, pk.max_vel / velocity_limits(j));
                    if (std::isfinite(acceleration_limits(j))) r_a = std::max(r_a, pk.max_acc / acceleration_limits(j));
                    if (std::isfinite(safe_jmax(j))) r_j = std::max(r_j, pk.max_jerk / safe_jmax(j));
                }
            }
            double alpha = std::max({1.0, r_v, std::sqrt(r_a), std::cbrt(r_j)});
            if (alpha > 1.0) {
                alpha *= (1.0 + 1e-7);
                const double inv_alpha = 1.0 / alpha;
                const double inv_alpha2 = inv_alpha * inv_alpha;
                Eigen::VectorXd new_times(T);
                new_times(0) = 0.0;
                for (int k = 0; k < T - 1; ++k) {
                    const double hk = traj.times(k + 1) - traj.times(k);
                    new_times(k + 1) = new_times(k) + (scale_seg[static_cast<size_t>(k)] ? (hk * alpha) : hk);
                }
                traj.times = std::move(new_times);
                traj.velocities *= inv_alpha;
                traj.accelerations *= inv_alpha2;
            }
            return traj;
        }

        // Fixed start boundary (v0 and/or a0): NEVER scale knot 0!
        // 1. Scale trailing segments 1..T-2 (and knots 1..T-1) cleanly to satisfy vel/acc/jerk limits.
        std::vector<double> seg_h(static_cast<size_t>(T - 1), 1e-4);
        for (int k = 0; k < T - 1; ++k) seg_h[static_cast<size_t>(k)] = std::max(traj.times(k + 1) - traj.times(k), 1e-4);

        if (T >= 3) {
            double r_v = 0.0, r_a = 0.0, r_j = 0.0;
            std::vector<uint8_t> scale_seg(static_cast<size_t>(T - 1), 0);
            for (int k = 1; k < T - 1; ++k) {
                const bool has_motion = (dq.row(k).cwiseAbs().maxCoeff() > 0.0) ||
                                        (traj.velocities.row(k).cwiseAbs().maxCoeff() > 0.0) ||
                                        (traj.velocities.row(k + 1).cwiseAbs().maxCoeff() > 0.0) ||
                                        (traj.accelerations.row(k).cwiseAbs().maxCoeff() > 0.0) ||
                                        (traj.accelerations.row(k + 1).cwiseAbs().maxCoeff() > 0.0);
                if (!has_motion) continue;
                scale_seg[static_cast<size_t>(k)] = 1;
                const double h = seg_h[static_cast<size_t>(k)];
                for (int j = 0; j < J; ++j) {
                    const auto pk = quintic_hermite_scalar_peaks(
                        traj.positions(k, j), traj.velocities(k, j), traj.accelerations(k, j),
                        traj.positions(k + 1, j), traj.velocities(k + 1, j), traj.accelerations(k + 1, j),
                        h);
                    if (std::isfinite(velocity_limits(j))) r_v = std::max(r_v, pk.max_vel / velocity_limits(j));
                    if (std::isfinite(acceleration_limits(j))) r_a = std::max(r_a, pk.max_acc / acceleration_limits(j));
                    if (std::isfinite(safe_jmax(j))) r_j = std::max(r_j, pk.max_jerk / safe_jmax(j));
                }
            }
            double alpha = std::max({1.0, r_v, std::sqrt(r_a), std::cbrt(r_j)});
            if (alpha > 1.0) {
                alpha *= (1.0 + 1e-7);
                const double inv_alpha = 1.0 / alpha;
                const double inv_alpha2 = inv_alpha * inv_alpha;
                for (int k = 1; k < T - 1; ++k) {
                    if (scale_seg[static_cast<size_t>(k)]) seg_h[static_cast<size_t>(k)] *= alpha;
                }
                for (int i = 1; i < T; ++i) {
                    traj.velocities.row(i) *= inv_alpha;
                    traj.accelerations.row(i) *= inv_alpha2;
                }
            }
        }

        // 2. Solve segment 0 (and re-verify shared knot 1 on segment 1 if T >= 3!) while keeping knot 0 strictly fixed.
        const Eigen::RowVectorXd v1_orig = traj.velocities.row(1);
        const Eigen::RowVectorXd a1_orig = traj.accelerations.row(1);
        const Eigen::RowVectorXd a0_orig = traj.accelerations.row(0);
        const double h0_hint = seg_h[0];
        const double h1_hint = (T >= 3) ? seg_h[1] : 0.0;
        const bool seg1_active = (T >= 3) &&
            (nonzero[1] || traj.velocities.row(2).cwiseAbs().maxCoeff() > 0.0 ||
             traj.accelerations.row(2).cwiseAbs().maxCoeff() > 0.0);

        const double v1_damp_factors[4] = {1.0, 0.5, 0.25, 0.0};
        const double a0_damp_factors[3] = {1.0, 0.5, 0.0};
        const int n_a0_tries = has_a0 ? 1 : 3;
        bool solved_head = false;
        for (int ai = 0; ai < n_a0_tries && !solved_head; ++ai) {
            const Eigen::RowVectorXd a0_try = has_a0 ? a0_orig : (a0_orig * a0_damp_factors[ai]);
            for (int vi = 0; vi < 4 && !solved_head; ++vi) {
                const double vf = v1_damp_factors[vi];
                const Eigen::RowVectorXd v1_try = v1_orig * vf;
                const Eigen::RowVectorXd a1_try = a1_orig * (vf * vf);
                auto h0_opt = retime_detail::find_feasible_quintic_dt(
                    traj.positions.data(), traj.velocities.data(), a0_try.data(),
                    traj.positions.data() + J, v1_try.data(), a1_try.data(),
                    J, velocity_limits, acceleration_limits, safe_jmax, h0_hint,
                    lower_limits, upper_limits);
                if (!h0_opt.has_value()) continue;
                if (T >= 3 && (seg1_active || v1_try.cwiseAbs().maxCoeff() > 0.0 || a1_try.cwiseAbs().maxCoeff() > 0.0)) {
                    auto h1_opt = retime_detail::find_feasible_quintic_dt(
                        traj.positions.data() + J, v1_try.data(), a1_try.data(),
                        traj.positions.data() + 2 * J, traj.velocities.data() + 2 * J, traj.accelerations.data() + 2 * J,
                        J, velocity_limits, acceleration_limits, safe_jmax, h1_hint,
                        lower_limits, upper_limits);
                    if (!h1_opt.has_value()) continue;
                    seg_h[1] = *h1_opt;
                }
                seg_h[0] = *h0_opt;
                traj.accelerations.row(0) = a0_try;
                traj.velocities.row(1) = v1_try;
                traj.accelerations.row(1) = a1_try;
                solved_head = true;
            }
        }
        // Always preserve exact knot-0 fixed boundaries (if infeasible, check_trajectory_bounds will reject!).
        if (has_v0) traj.velocities.row(0) = v0_fixed.transpose();
        if (has_a0) traj.accelerations.row(0) = a0_fixed.transpose();
        traj.times(0) = 0.0;
        for (int k = 0; k < T - 1; ++k) {
            traj.times(k + 1) = traj.times(k) + seg_h[static_cast<size_t>(k)];
        }
        return traj;
    }

    TrajectoryNative traj;
    traj.positions = positions;
    traj.velocities = RowMatrixXd::Zero(T, J);
    traj.accelerations = RowMatrixXd::Zero(T, J);
    traj.times.resize(T);

    if (has_dt) {
        traj.has_dt = true;
        traj.dt = dt;
        for (int i = 0; i < T; ++i) traj.times(i) = static_cast<double>(i) * dt;
        for (int i = 1; i < T - 1; ++i) {
            traj.velocities.row(i) = (positions.row(i + 1) - positions.row(i - 1)) / (2.0 * dt);
        }
        if (!stop_at_endpoints) {
            if (!has_v0) traj.velocities.row(0) = dq.row(0) / dt;
            traj.velocities.row(T - 1) = dq.row(T - 2) / dt;
        }
        for (int i = 0; i < T; ++i) traj.velocities.row(i) *= k_scales(i);
        for (int k = 0; k < T - 1; ++k) {
            if (!nonzero[static_cast<size_t>(k)]) {
                if (!(k == 0 && has_v0)) traj.velocities.row(k).setZero();
                traj.velocities.row(k + 1).setZero();
            }
        }
        for (int i = (has_v0 ? 1 : 0); i < T; ++i) {
            double ratio = 0.0;
            for (int j = 0; j < J; ++j) ratio = std::max(ratio, std::abs(traj.velocities(i, j)) / safe_vmax(j));
            if (ratio > 1.0) traj.velocities.row(i) /= ratio;
        }
        if (has_v0) traj.velocities.row(0) = v0_fixed.transpose();

        if ((k_scales.array() >= 1.0 - 1e-12).all()) {
            bool needs_proj = false;
            for (int k = 0; k < T - 1; ++k) {
                if ((nonzero[static_cast<size_t>(k)] || (k == 0 && seg0_boundary_motion)) &&
                    !retime_detail::hermite_segment_feasible(
                        dq.data() + k * J, traj.velocities.data() + k * J,
                        traj.velocities.data() + (k + 1) * J, J, dt, safe_vmax, safe_amax, 1e-5)) {
                    needs_proj = true;
                    break;
                }
            }
            if (needs_proj) {
                const int free_lo = (stop_at_endpoints || has_v0) ? 1 : 0;
                const int free_hi = stop_at_endpoints ? (T - 1) : T;
                const Eigen::VectorXd ah = safe_amax * dt;
                for (int pass = 0; pass < 8; ++pass) {
                    for (int i = free_lo; i < free_hi; ++i) {
                        if ((i > 0 && !nonzero[static_cast<size_t>(i - 1)] && !(i == 1 && seg0_boundary_motion)) ||
                            (i < T - 1 && !nonzero[static_cast<size_t>(i)])) continue;
                        for (int j = 0; j < J; ++j) {
                            double lo = -safe_vmax(j), hi = safe_vmax(j);
                            if (i > 0) {
                                const double c = 6.0 * dq(i - 1, j) / dt;
                                const double vp = traj.velocities(i - 1, j);
                                lo = std::max({lo, (c - 4.0 * vp - ah(j)) * 0.5, (c - 2.0 * vp - ah(j)) * 0.25});
                                hi = std::min({hi, (c - 4.0 * vp + ah(j)) * 0.5, (c - 2.0 * vp + ah(j)) * 0.25});
                            }
                            if (i < T - 1) {
                                const double c = 6.0 * dq(i, j) / dt;
                                const double vn = traj.velocities(i + 1, j);
                                lo = std::max({lo, (c - 2.0 * vn - ah(j)) * 0.25, (c - 4.0 * vn - ah(j)) * 0.5});
                                hi = std::min({hi, (c - 2.0 * vn + ah(j)) * 0.25, (c - 4.0 * vn + ah(j)) * 0.5});
                            }
                            if (lo <= hi) traj.velocities(i, j) = std::clamp(traj.velocities(i, j), lo, hi);
                        }
                    }
                }
            }
        }
        if (has_v0) traj.velocities.row(0) = v0_fixed.transpose();
        for (int k = 0; k < T - 1; ++k) {
            for (int j = 0; j < J; ++j) {
                const double a0 = (6.0 * dq(k, j) / dt - 4.0 * traj.velocities(k, j) - 2.0 * traj.velocities(k + 1, j)) / dt;
                const double a1 = (-6.0 * dq(k, j) / dt + 2.0 * traj.velocities(k, j) + 4.0 * traj.velocities(k + 1, j)) / dt;
                traj.accelerations(k, j) = (k == 0) ? a0 : 0.5 * (traj.accelerations(k, j) + a0);
                traj.accelerations(k + 1, j) = a1;
            }
        }
        return traj;
    }

    // TOPP parameterization with exact cubic Hermite segment durations.
    RowMatrixXd qp = RowMatrixXd::Zero(T - 1, J);
    Eigen::VectorXd seg_sdot_max = Eigen::VectorXd::Constant(T - 1, std::numeric_limits<double>::infinity());
    Eigen::VectorXd seg_sddot_max = Eigen::VectorXd::Constant(T - 1, 1e6);
    for (int k = 0; k < T - 1; ++k) {
        if (!nonzero[static_cast<size_t>(k)]) {
            seg_sdot_max(k) = 0.0;
            continue;
        }
        qp.row(k) = dq.row(k) / ds(k);
        double sv = std::numeric_limits<double>::infinity();
        double sa = 1e6;
        for (int j = 0; j < J; ++j) {
            const double a = std::abs(qp(k, j));
            if (a > 1e-12) {
                sv = std::min(sv, safe_vmax(j) / a);
                sa = std::min(sa, safe_amax(j) / a);
            }
        }
        seg_sdot_max(k) = sv;
        seg_sddot_max(k) = sa;
    }

    Eigen::VectorXd sdot = Eigen::VectorXd::Constant(T, std::numeric_limits<double>::infinity());
    if (has_v0) {
        sdot(0) = nonzero[0] ? std::min(v0_fixed.norm(), seg_sdot_max(0)) : 0.0;
    } else {
        sdot(0) = stop_at_endpoints ? 0.0 : seg_sdot_max(0);
    }
    sdot(T - 1) = stop_at_endpoints ? 0.0 : seg_sdot_max(T - 2);
    for (int i = 1; i < T - 1; ++i) {
        sdot(i) = std::min(seg_sdot_max(i - 1), seg_sdot_max(i));
        if (nonzero[static_cast<size_t>(i - 1)] && nonzero[static_cast<size_t>(i)]) {
            const double ds_avg = std::max(0.5 * (ds(i - 1) + ds(i)), 1e-9);
            double curv_bound = std::numeric_limits<double>::infinity();
            for (int j = 0; j < J; ++j) {
                const double qpp = std::abs(qp(i, j) - qp(i - 1, j)) / ds_avg;
                if (qpp > 1e-12) curv_bound = std::min(curv_bound, std::sqrt(safe_amax(j) / qpp));
            }
            sdot(i) = std::min(sdot(i), curv_bound);
        }
    }
    sdot = sdot.cwiseProduct(k_scales);

    for (int i = 0; i < T - 1; ++i) {
        if (!nonzero[static_cast<size_t>(i)]) sdot(i + 1) = std::min(sdot(i + 1), sdot(i));
        else sdot(i + 1) = std::min(sdot(i + 1), std::sqrt(sdot(i) * sdot(i) + 2.0 * seg_sddot_max(i) * ds(i)));
    }
    for (int i = T - 2; i >= 0; --i) {
        if (!nonzero[static_cast<size_t>(i)]) sdot(i) = std::min(sdot(i), sdot(i + 1));
        else sdot(i) = std::min(sdot(i), std::sqrt(sdot(i + 1) * sdot(i + 1) + 2.0 * seg_sddot_max(i) * ds(i)));
    }

    auto clamp_vel_row = [&](int i) {
        double ratio = 0.0;
        for (int j = 0; j < J; ++j) ratio = std::max(ratio, std::abs(traj.velocities(i, j)) / safe_vmax(j));
        if (ratio > 1.0) traj.velocities.row(i) /= ratio;
    };
    for (int i = 1; i < T - 1; ++i) {
        if (!nonzero[static_cast<size_t>(i - 1)] || !nonzero[static_cast<size_t>(i)]) continue;
        Eigen::RowVectorXd tang = 0.5 * (qp.row(i - 1) + qp.row(i));
        const double nrm = tang.norm();
        if (nrm > 1e-12) {
            traj.velocities.row(i) = (tang / nrm) * sdot(i);
            clamp_vel_row(i);
        }
    }
    if (has_v0) {
        traj.velocities.row(0) = v0_fixed.transpose();
    } else if (!stop_at_endpoints && nonzero[0]) {
        traj.velocities.row(0) = qp.row(0) * sdot(0);
        clamp_vel_row(0);
    }
    if (!stop_at_endpoints && nonzero[static_cast<size_t>(T - 2)]) {
        traj.velocities.row(T - 1) = qp.row(T - 2) * sdot(T - 1);
        clamp_vel_row(T - 1);
    }

    Eigen::VectorXd seg_dt = Eigen::VectorXd::Constant(T - 1, 1e-4);
    Eigen::VectorXd zero_dt(T - 1);
    for (int k = 0; k < T - 1; ++k) {
        zero_dt(k) = nonzero[static_cast<size_t>(k)]
            ? retime_detail::zero_vel_hermite_dt(dq.data() + k * J, J, safe_vmax, safe_amax)
            : 1e-4;
    }

    for (int attempt = 0; attempt < 6; ++attempt) {
        bool damped = false;
        for (int k = 0; k < T - 1; ++k) {
            const bool active = nonzero[static_cast<size_t>(k)] || (k == 0 && seg0_boundary_motion);
            if (!active) { seg_dt(k) = 1e-4; continue; }
            const double hk = retime_detail::min_feasible_hermite_dt(
                dq.data() + k * J, traj.velocities.data() + k * J,
                traj.velocities.data() + (k + 1) * J, J, safe_vmax, safe_amax);
            if (k == 0 && has_v0) {
                // Never damp knot 0 when start_velocity is fixed; damp knot 1 if seg 0 is overly stretched or infeasible.
                const bool seg0_ok = retime_detail::hermite_segment_feasible(
                    dq.data(), traj.velocities.data(), traj.velocities.data() + J, J, hk, safe_vmax, safe_amax, 1e-6);
                if ((!seg0_ok || (nonzero[0] && hk > 1.35 * zero_dt(0))) &&
                    traj.velocities.row(1).squaredNorm() > 0.0) {
                    const double factor = (attempt == 5) ? 0.0 : 0.5;
                    traj.velocities.row(1) *= factor;
                    damped = true;
                }
            } else if (hk > 1.15 * zero_dt(k) &&
                       (traj.velocities.row(k).squaredNorm() > 0.0 || traj.velocities.row(k + 1).squaredNorm() > 0.0)) {
                const double factor = (attempt == 5) ? 0.0 : 0.6;
                traj.velocities.row(k) *= factor;
                traj.velocities.row(k + 1) *= factor;
                damped = true;
            }
            seg_dt(k) = hk;
        }
        if (!damped) break;
    }

    for (int k = 0; k < T - 1; ++k) {
        const bool active = nonzero[static_cast<size_t>(k)] || (k == 0 && seg0_boundary_motion);
        if (!active) {
            if (!(k == 0 && has_v0)) traj.velocities.row(k).setZero();
            traj.velocities.row(k + 1).setZero();
            seg_dt(k) = 1e-4;
            continue;
        }
        if (k == 0 && has_v0) {
            // Certified duration search for segment 0 with fixed start_velocity (handles q1==q0 & position bounds).
            auto h0_opt = retime_detail::find_feasible_hermite_dt(
                dq.data(), traj.velocities.data(), traj.velocities.data() + J, J,
                velocity_limits, acceleration_limits,
                traj.positions.data(), traj.positions.data() + J, lower_limits, upper_limits);
            if (!h0_opt.has_value() && traj.velocities.row(1).squaredNorm() > 0.0) {
                traj.velocities.row(1).setZero();
                h0_opt = retime_detail::find_feasible_hermite_dt(
                    dq.data(), traj.velocities.data(), traj.velocities.data() + J, J,
                    velocity_limits, acceleration_limits,
                    traj.positions.data(), traj.positions.data() + J, lower_limits, upper_limits);
            }
            seg_dt(0) = h0_opt.value_or(retime_detail::min_feasible_hermite_dt(
                dq.data(), traj.velocities.data(), traj.velocities.data() + J, J, safe_vmax, safe_amax));
        } else {
            seg_dt(k) = retime_detail::min_feasible_hermite_dt(
                dq.data() + k * J, traj.velocities.data() + k * J,
                traj.velocities.data() + (k + 1) * J, J, safe_vmax, safe_amax);
        }
    }
    if (has_v0) traj.velocities.row(0) = v0_fixed.transpose();

    traj.times(0) = 0.0;
    for (int k = 0; k < T - 1; ++k) {
        traj.times(k + 1) = traj.times(k) + seg_dt(k);
        const double h = seg_dt(k);
        for (int j = 0; j < J; ++j) {
            const double a0 = (6.0 * dq(k, j) / h - 4.0 * traj.velocities(k, j) - 2.0 * traj.velocities(k + 1, j)) / h;
            const double a1 = (-6.0 * dq(k, j) / h + 2.0 * traj.velocities(k, j) + 4.0 * traj.velocities(k + 1, j)) / h;
            traj.accelerations(k, j) = (k == 0) ? a0 : 0.5 * (traj.accelerations(k, j) + a0);
            traj.accelerations(k + 1, j) = a1;
        }
    }
    return traj;
}

}  // namespace retime_detail

inline TrajectoryNative retime_trajectory(const RowMatrixXd& positions,
                                          const Eigen::VectorXd& velocity_limits,
                                          const Eigen::VectorXd& acceleration_limits,
                                          bool stop_at_endpoints = true,
                                          bool has_dt = false,
                                          double dt = 0.0,
                                          const Eigen::VectorXd* knot_velocity_scales = nullptr,
                                          const Eigen::VectorXd* jerk_limits = nullptr,
                                          int spline_order = 3,
                                          const Eigen::VectorXd* start_velocity = nullptr,
                                          const Eigen::VectorXd* start_acceleration = nullptr,
                                          const Eigen::VectorXd* lower_limits = nullptr,
                                          const Eigen::VectorXd* upper_limits = nullptr) {
    auto traj = retime_detail::retime_candidate(
        positions, velocity_limits, acceleration_limits, stop_at_endpoints, has_dt, dt, knot_velocity_scales,
        jerk_limits, spline_order, start_velocity, start_acceleration, lower_limits, upper_limits);
    if (start_velocity || start_acceleration || lower_limits || upper_limits) {
        if (start_velocity && start_acceleration) {
            for (int j = 0; j < start_velocity->size(); ++j) {
                const double v = (*start_velocity)(j);
                if (std::abs(v) >= velocity_limits(j) && v * (*start_acceleration)(j) > 0.0)
                    throw std::invalid_argument("infeasible start acceleration at velocity limit");
            }
        }
        const auto bounds = check_trajectory_bounds(traj, velocity_limits, acceleration_limits,
            true, true, 1e-6, lower_limits, upper_limits, jerk_limits);
        if (!bounds.ok) throw std::invalid_argument(bounds.reason);
    }
    return traj;
}

inline void validate_slice_input(const TrajectoryNative& traj) {
    const int T = traj.num_waypoints(), J = traj.num_joints();
    if (T < 2 || J < 1 || traj.times.size() != T ||
        traj.velocities.rows() != T || traj.velocities.cols() != J ||
        traj.accelerations.rows() != T || traj.accelerations.cols() != J ||
        (traj.spline_order != 3 && traj.spline_order != 5))
        throw std::invalid_argument("invalid spline dimensions or order");
    if (!traj.positions.allFinite() || !traj.velocities.allFinite() || !traj.accelerations.allFinite() ||
        !traj.times.allFinite() || (traj.times.tail(T-1).array() < traj.times.head(T-1).array()).any())
        throw std::invalid_argument("spline state must be finite and times must be non-decreasing");
}

// Exact Hermite suffix slicing at `t_anchor` in [times[0], times[T-1]].
// Restricting a cubic (order 3) or quintic (order 5) Hermite segment to [t_anchor, times[k+1]]
// is mathematically identical to the Hermite segment with endpoints sampled at t_anchor and times[k+1].
// Returned trajectory always starts at times[0] == 0.0 and has >= 2 knots.
inline TrajectoryNative slice_trajectory_suffix(const TrajectoryNative& traj, double t_anchor) {
    const int T = traj.num_waypoints();
    const int J = traj.num_joints();
    validate_slice_input(traj);
    if (!std::isfinite(t_anchor)) {
        throw std::invalid_argument("t_anchor must be finite");
    }
    const double t0 = traj.times(0);
    const double t_end = traj.times(T - 1);
    const double tc = std::clamp(t_anchor, t0, t_end);

    Eigen::VectorXd p_a(J), v_a(J), a_a(J);
    traj.sample_single(tc, p_a, &v_a, &a_a);

    const int next_idx = static_cast<int>(std::upper_bound(traj.times.data(), traj.times.data() + T, tc)
                                          - traj.times.data());

    TrajectoryNative out;
    out.spline_order = traj.spline_order;
    out.has_dt = false;
    out.dt = 0.0;

    if (next_idx >= T) {
        // At or beyond end of trajectory: return a 2-knot stationary suffix at the final state.
        out.times = Eigen::VectorXd::Zero(2);
        out.times(1) = 0.0;
        out.positions.resize(2, J);
        out.velocities.resize(2, J);
        out.accelerations.resize(2, J);
        out.positions.row(0) = p_a.transpose();
        out.positions.row(1) = p_a.transpose();
        out.velocities.row(0) = v_a.transpose();
        out.velocities.row(1) = v_a.transpose();
        out.accelerations.row(0) = a_a.transpose();
        out.accelerations.row(1) = a_a.transpose();
        return out;
    }

    const int M = 1 + (T - next_idx);
    out.times.resize(M);
    out.positions.resize(M, J);
    out.velocities.resize(M, J);
    out.accelerations.resize(M, J);

    out.times(0) = 0.0;
    out.positions.row(0) = p_a.transpose();
    out.velocities.row(0) = v_a.transpose();
    out.accelerations.row(0) = a_a.transpose();
    for (int i = next_idx; i < T; ++i) {
        const int dst = 1 + (i - next_idx);
        out.times(dst) = traj.times(i) - tc;
        out.positions.row(dst) = traj.positions.row(i);
        out.velocities.row(dst) = traj.velocities.row(i);
        out.accelerations.row(dst) = traj.accelerations.row(i);
    }
    if (traj.has_dt && traj.dt > 0.0) {
        bool uniform_dt = true;
        for (int k = 0; k < M - 1; ++k) {
            if (std::abs((out.times(k + 1) - out.times(k)) - traj.dt) >
                64 * std::numeric_limits<double>::epsilon() * std::max(std::abs(out.times(k + 1)), traj.dt)) {
                uniform_dt = false;
                break;
            }
        }
        if (uniform_dt) {
            out.has_dt = true;
            out.dt = traj.dt;
        }
    }
    return out;
}

struct TrajectoryAnchorMatch {
    double t_anchor = 0.0;
    double pos_err = 0.0;
    double vel_err = 0.0;
    bool ambiguous = false;  // true when multiple separated timestamps match (e.g. reversal/pause)
};

namespace retime_detail {

// The squared distance in (q, w*dq) is a polynomial on each Hermite segment.
// Its stationary points and the knots include every possible closest time.
constexpr double anchor_velocity_weight = 0.1;

constexpr int binomial(int n, int k) {
    int result = 1;
    for (int i = 1; i <= k; ++i) result = result * (n - i + 1) / i;
    return result;
}

// Coefficients of r * dr/du in Bernstein form, for a scalar residual r(u).
template <size_t N>
inline void add_anchor_derivative(const std::array<double, N>& r, std::array<double, 2 * (N - 1)>& sum) {
    constexpr int degree = N - 1;
    static constexpr auto weights = [] {
        std::array<std::array<double, degree>, N> w{};
        for (int i = 0; i <= degree; ++i)
            for (int j = 0; j < degree; ++j)
                w[i][j] = static_cast<double>(degree * binomial(degree, i) * binomial(degree - 1, j))
                          / binomial(2 * degree - 1, i + j);
        return w;
    }();
    for (int i = 0; i <= degree; ++i)
        for (int j = 0; j < degree; ++j)
            sum[i + j] += weights[i][j] * r[i] * (r[j + 1] - r[j]);
}

template <int Degree>
inline auto anchor_score_derivative(const TrajectoryNative& traj, int k,
                                    const Eigen::VectorXd& q, const Eigen::VectorXd& dq) {
    std::array<double, 2 * Degree> derivative{};
    const double h = traj.times(k + 1) - traj.times(k);
    for (int j = 0; j < traj.num_joints(); ++j) {
        const double p0 = traj.positions(k, j), p1 = traj.positions(k + 1, j);
        const double v0 = traj.velocities(k, j), v1 = traj.velocities(k + 1, j);
        std::array<double, Degree + 1> pos{};
        std::array<double, Degree> vel{};
        if constexpr (Degree == 3) {
            pos = {p0 - q(j), (p0 - q(j)) + h * v0 / 3.0, (p1 - q(j)) - h * v1 / 3.0, p1 - q(j)};
            vel = {v0, 3.0 * (p1 - p0) / h - v0 - v1, v1};
        } else {
            const double a0 = traj.accelerations(k, j), a1 = traj.accelerations(k + 1, j);
            QuinticPoly poly{};
            poly.h = h;
            pos = poly.pos_bernstein(p0 - q(j), v0, a0, p1 - q(j), v1, a1);
            vel = poly.vel_bernstein(p0, v0, a0, p1, v1, a1);
        }
        add_anchor_derivative(pos, derivative);
        // Elevate velocity by one degree so both derivatives use the same basis.
        for (double& v : vel) v = anchor_velocity_weight * (v - dq(j));
        std::array<double, Degree + 1> elevated{};
        elevated.front() = vel.front();
        elevated.back() = vel.back();
        for (int i = 1; i < Degree; ++i)
            elevated[i] = (i * vel[i - 1] + (Degree - i) * vel[i]) / Degree;
        add_anchor_derivative(elevated, derivative);
    }
    return derivative;
}

// A one-sided Bernstein hull excludes interior sign changes. Split the other
// intervals until their extrema can be located to floating-point precision.
template <size_t N>
inline void append_anchor_extrema(const std::array<double, N>& derivative,
                                  double lo, double hi, int depth, std::vector<double>& times) {
    if (depth == 0 && !std::all_of(derivative.begin(), derivative.end(), [](double x) { return std::isfinite(x); }))
        throw std::invalid_argument("previous trajectory exceeds numeric range during anchor search");
    const auto [min_value, max_value] = bernstein_detail::control_minmax(derivative);
    if (min_value >= 0.0 || max_value <= 0.0) return;
    const double mid = lo + 0.5 * (hi - lo);
    if (mid <= lo || mid >= hi || depth == 64) {
        times.push_back(mid);
        return;
    }
    std::array<double, N> left, right;
    bernstein_detail::split_at(derivative, 0.5, left, right);
    append_anchor_extrema(left, lo, mid, depth + 1, times);
    if (left.back() == 0.0) times.push_back(mid);
    append_anchor_extrema(right, mid, hi, depth + 1, times);
}

}  // namespace retime_detail

inline TrajectoryAnchorMatch find_anchor_on_trajectory(
    const TrajectoryNative& traj, const Eigen::VectorXd& q_now, const Eigen::VectorXd& dq_now) {
    std::vector<double> times{traj.times(0)};
    for (int k = 0; k + 1 < traj.num_waypoints(); ++k) {
        const double lo = traj.times(k), hi = traj.times(k + 1);
        if (lo == hi) continue;
        if (traj.spline_order == 5)
            retime_detail::append_anchor_extrema(retime_detail::anchor_score_derivative<5>(traj, k, q_now, dq_now),
                                                 lo, hi, 0, times);
        else
            retime_detail::append_anchor_extrema(retime_detail::anchor_score_derivative<3>(traj, k, q_now, dq_now),
                                                 lo, hi, 0, times);
        times.push_back(hi);
    }
    times.erase(std::unique(times.begin(), times.end()), times.end());

    TrajectoryAnchorMatch best;
    best.t_anchor = times.front();
    double best_score = std::numeric_limits<double>::infinity();
    std::vector<double> scores;
    scores.reserve(times.size());
    Eigen::VectorXd p, v;
    for (double t : times) {
        traj.sample_single(t, p, &v);
        p -= q_now;
        v -= dq_now;
        const double score = std::hypot(p.norm(), retime_detail::anchor_velocity_weight * v.norm());
        scores.push_back(score);
        if (score < best_score) {
            best_score = score;
            best.t_anchor = t;
            best.pos_err = p.lpNorm<Eigen::Infinity>();
            best.vel_err = v.lpNorm<Eigen::Infinity>();
        }
    }

    // Count distinct local minima, including both ends of a matching pause.
    // An unrelated long hold must not set the time resolution of a short loop.
    const double near = std::max(5e-3, 3.0 * best_score);
    int matches = 0;
    for (size_t i = 0; i < scores.size(); ++i) {
        if (scores[i] <= near &&
            (i == 0 || scores[i] <= scores[i - 1]) &&
            (i + 1 == scores.size() || scores[i] <= scores[i + 1])) {
            if (++matches == 2) {
                best.ambiguous = true;
                break;
            }
        }
    }
    return best;
}

}  // namespace gobot::planning
