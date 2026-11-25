#pragma once

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <limits>
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

    int num_waypoints() const { return static_cast<int>(positions.rows()); }
    int num_joints() const { return static_cast<int>(positions.cols()); }
    double duration() const {
        return times.size() > 0 ? (times(times.size() - 1) - times(0)) : 0.0;
    }

    void sample_single(double t, Eigen::VectorXd& pos,
                       Eigen::VectorXd* vel = nullptr,
                       Eigen::VectorXd* acc = nullptr) const {
        const int T = num_waypoints();
        const int J = num_joints();
        const double tc = std::clamp(t, times(0), times(T - 1));
        int idx = static_cast<int>(std::upper_bound(times.data(), times.data() + T, tc) - times.data()) - 1;
        idx = std::clamp(idx, 0, T - 2);
        const double h = std::max(times(idx + 1) - times(idx), 1e-12);
        const double u = (tc - times(idx)) / h;
        const double u2 = u * u;
        const double u3 = u2 * u;

        const double h00 = 2.0 * u3 - 3.0 * u2 + 1.0;
        const double h10 = (u3 - 2.0 * u2 + u) * h;
        const double h01 = -2.0 * u3 + 3.0 * u2;
        const double h11 = (u3 - u2) * h;

        pos.resize(J);
        for (int j = 0; j < J; ++j) {
            pos(j) = h00 * positions(idx, j) + h10 * velocities(idx, j)
                   + h01 * positions(idx + 1, j) + h11 * velocities(idx + 1, j);
        }
        if (vel != nullptr) {
            const double inv_h = 1.0 / h;
            const double dh00 = (6.0 * u2 - 6.0 * u) * inv_h;
            const double dh10 = 3.0 * u2 - 4.0 * u + 1.0;
            const double dh01 = (-6.0 * u2 + 6.0 * u) * inv_h;
            const double dh11 = 3.0 * u2 - 2.0 * u;
            vel->resize(J);
            for (int j = 0; j < J; ++j) {
                (*vel)(j) = dh00 * positions(idx, j) + dh10 * velocities(idx, j)
                          + dh01 * positions(idx + 1, j) + dh11 * velocities(idx + 1, j);
            }
        }
        if (acc != nullptr) {
            const double inv_h = 1.0 / h;
            const double inv_h2 = inv_h * inv_h;
            const double ddh00 = (12.0 * u - 6.0) * inv_h2;
            const double ddh10 = (6.0 * u - 4.0) * inv_h;
            const double ddh01 = (-12.0 * u + 6.0) * inv_h2;
            const double ddh11 = (6.0 * u - 2.0) * inv_h;
            acc->resize(J);
            for (int j = 0; j < J; ++j) {
                (*acc)(j) = ddh00 * positions(idx, j) + ddh10 * velocities(idx, j)
                          + ddh01 * positions(idx + 1, j) + ddh11 * velocities(idx + 1, j);
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
            const double h = std::max(times(seg + 1) - times(seg), 1e-12);
            const double* p0 = positions.data() + seg * J;
            const double* p1 = positions.data() + (seg + 1) * J;
            const double* v0 = velocities.data() + seg * J;
            const double* v1 = velocities.data() + (seg + 1) * J;
            for (int k = 0; k < substeps; ++k) {
                const double u = static_cast<double>(k) * inv_sub;
                const double u2 = u * u;
                const double u3 = u2 * u;
                const double h00 = 2.0 * u3 - 3.0 * u2 + 1.0;
                const double h10 = (u3 - 2.0 * u2 + u) * h;
                const double h01 = -2.0 * u3 + 3.0 * u2;
                const double h11 = (u3 - u2) * h;
                double* dst = out_samples.data() + (seg * substeps + k) * J;
                for (int j = 0; j < J; ++j) {
                    dst[j] = h00 * p0[j] + h10 * v0[j] + h01 * p1[j] + h11 * v1[j];
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
    for (int i = 0; i < 16; ++i) {
        if (vel_feasible(h_hi)) break;
        h_lo = h_hi;
        h_hi = next_acc_feasible(h_hi * 2.0);
    }
    for (int i = 0; i < 10; ++i) {
        const double mid = 0.5 * (h_lo + h_hi);
        const double cand = next_acc_feasible(mid);
        if (cand < h_hi && vel_feasible(cand)) h_hi = cand;
        else h_lo = mid;
    }
    return h_hi;
}

}  // namespace retime_detail

inline BoundsCheckResult check_trajectory_bounds(const TrajectoryNative& traj,
                                                 const Eigen::VectorXd& velocity_limits,
                                                 const Eigen::VectorXd& acceleration_limits,
                                                 bool check_velocity = true,
                                                 bool check_acceleration = true,
                                                 double tol = 1e-5) {
    BoundsCheckResult out;
    const int T = traj.num_waypoints();
    const int J = traj.num_joints();
    Eigen::VectorXd safe_vmax(J), safe_amax(J);
    for (int j = 0; j < J; ++j) {
        safe_vmax(j) = (std::isfinite(velocity_limits(j)) && velocity_limits(j) > 0.0)
            ? velocity_limits(j) : std::numeric_limits<double>::infinity();
        safe_amax(j) = (std::isfinite(acceleration_limits(j)) && acceleration_limits(j) > 0.0)
            ? acceleration_limits(j) : std::numeric_limits<double>::infinity();
    }
    std::vector<double> d(static_cast<size_t>(J)), mv(static_cast<size_t>(J)), ma(static_cast<size_t>(J));
    const double scale = 1.0 + tol;

    for (int k = 0; k < T - 1; ++k) {
        const double h = traj.times(k + 1) - traj.times(k);
        if (h <= 1e-12) continue;
        for (int j = 0; j < J; ++j) d[static_cast<size_t>(j)] = traj.positions(k + 1, j) - traj.positions(k, j);
        retime_detail::hermite_segment_peaks(
            d.data(), traj.velocities.data() + k * J, traj.velocities.data() + (k + 1) * J,
            J, h, mv.data(), ma.data());
        double max_vr = 0.0, max_ar = 0.0;
        for (int j = 0; j < J; ++j) {
            if (check_velocity && std::isfinite(safe_vmax(j))) max_vr = std::max(max_vr, mv[static_cast<size_t>(j)] / safe_vmax(j));
            if (check_acceleration && std::isfinite(safe_amax(j))) max_ar = std::max(max_ar, ma[static_cast<size_t>(j)] / safe_amax(j));
        }
        if (check_velocity && max_vr > scale) {
            out.ok = false;
            out.bad_segments.push_back(k);
            if (out.reason.empty()) {
                char buf[160];
                std::snprintf(buf, sizeof(buf), "Hermite curve violates joint velocity limits on segment %d (%.2fx limit)", k, max_vr);
                out.reason = buf;
            }
        } else if (check_acceleration && max_ar > scale) {
            out.ok = false;
            out.bad_segments.push_back(k);
            if (out.reason.empty()) {
                char buf[160];
                std::snprintf(buf, sizeof(buf), "Hermite curve violates joint acceleration limits on segment %d (%.2fx limit)", k, max_ar);
                out.reason = buf;
            }
        }
    }
    return out;
}

inline TrajectoryNative retime_trajectory(const RowMatrixXd& positions,
                                          const Eigen::VectorXd& velocity_limits,
                                          const Eigen::VectorXd& acceleration_limits,
                                          bool stop_at_endpoints = true,
                                          bool has_dt = false,
                                          double dt = 0.0,
                                          const Eigen::VectorXd* knot_velocity_scales = nullptr) {
    const int T = static_cast<int>(positions.rows());
    const int J = static_cast<int>(positions.cols());
    if (T < 2) throw std::invalid_argument("positions must have shape [T>=2, J]");

    Eigen::VectorXd safe_vmax(J), safe_amax(J);
    for (int j = 0; j < J; ++j) {
        safe_vmax(j) = (std::isfinite(velocity_limits(j)) && velocity_limits(j) > 0.0) ? velocity_limits(j) : 1e6;
        safe_amax(j) = (std::isfinite(acceleration_limits(j)) && acceleration_limits(j) > 0.0) ? acceleration_limits(j) : 1e6;
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
            traj.velocities.row(0) = dq.row(0) / dt;
            traj.velocities.row(T - 1) = dq.row(T - 2) / dt;
        }
        for (int i = 0; i < T; ++i) traj.velocities.row(i) *= k_scales(i);
        for (int k = 0; k < T - 1; ++k) {
            if (!nonzero[static_cast<size_t>(k)]) {
                traj.velocities.row(k).setZero();
                traj.velocities.row(k + 1).setZero();
            }
        }
        for (int i = 0; i < T; ++i) {
            double ratio = 0.0;
            for (int j = 0; j < J; ++j) ratio = std::max(ratio, std::abs(traj.velocities(i, j)) / safe_vmax(j));
            if (ratio > 1.0) traj.velocities.row(i) /= ratio;
        }
        if ((k_scales.array() >= 1.0 - 1e-12).all()) {
            bool needs_proj = false;
            for (int k = 0; k < T - 1; ++k) {
                if (nonzero[static_cast<size_t>(k)] &&
                    !retime_detail::hermite_segment_feasible(
                        dq.data() + k * J, traj.velocities.data() + k * J,
                        traj.velocities.data() + (k + 1) * J, J, dt, safe_vmax, safe_amax, 1e-5)) {
                    needs_proj = true;
                    break;
                }
            }
            if (needs_proj) {
                const int free_lo = stop_at_endpoints ? 1 : 0;
                const int free_hi = stop_at_endpoints ? (T - 1) : T;
                const Eigen::VectorXd ah = safe_amax * dt;
                for (int pass = 0; pass < 8; ++pass) {
                    for (int i = free_lo; i < free_hi; ++i) {
                        if ((i > 0 && !nonzero[static_cast<size_t>(i - 1)]) ||
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
    sdot(0) = stop_at_endpoints ? 0.0 : seg_sdot_max(0);
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
    if (!stop_at_endpoints) {
        if (nonzero[0]) { traj.velocities.row(0) = qp.row(0) * sdot(0); clamp_vel_row(0); }
        if (nonzero[static_cast<size_t>(T - 2)]) { traj.velocities.row(T - 1) = qp.row(T - 2) * sdot(T - 1); clamp_vel_row(T - 1); }
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
            if (!nonzero[static_cast<size_t>(k)]) { seg_dt(k) = 1e-4; continue; }
            const double hk = retime_detail::min_feasible_hermite_dt(
                dq.data() + k * J, traj.velocities.data() + k * J,
                traj.velocities.data() + (k + 1) * J, J, safe_vmax, safe_amax);
            if (hk > 1.15 * zero_dt(k) &&
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
        if (!nonzero[static_cast<size_t>(k)]) {
            traj.velocities.row(k).setZero();
            traj.velocities.row(k + 1).setZero();
            seg_dt(k) = 1e-4;
            continue;
        }
        seg_dt(k) = retime_detail::min_feasible_hermite_dt(
            dq.data() + k * J, traj.velocities.data() + k * J,
            traj.velocities.data() + (k + 1) * J, J, safe_vmax, safe_amax);
    }

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

}  // namespace gobot::planning
