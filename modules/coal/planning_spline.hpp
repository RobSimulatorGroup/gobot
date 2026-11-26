#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <utility>

namespace gobot::planning {

// Shared scalar spline evaluation and certified Bernstein-hull interval bounds for both
// C^1 cubic Hermite (order 3) and C^2 quintic Hermite (order 5) trajectories.
// Used identically by trajectory playback, joint limit verification, velocity/acceleration/jerk
// checks, continuous CCD, and Cartesian path deviation verification.

// ============================================================================
// Cubic Hermite (order 3) scalar evaluation & exact bounds
// ============================================================================

inline void eval_cubic_hermite_scalar(double p0, double v0, double p1, double v1,
                                      double h, double u,
                                      double* pos, double* vel = nullptr,
                                      double* acc = nullptr, double* jerk = nullptr) {
    const double u2 = u * u;
    const double u3 = u2 * u;
    if (pos) {
        const double h00 = 2.0 * u3 - 3.0 * u2 + 1.0;
        const double h10 = (u3 - 2.0 * u2 + u) * h;
        const double h01 = -2.0 * u3 + 3.0 * u2;
        const double h11 = (u3 - u2) * h;
        *pos = h00 * p0 + h10 * v0 + h01 * p1 + h11 * v1;
    }
    const double inv_h = 1.0 / ((h > 0.0) ? h : 1e-12);
    if (vel) {
        const double dh00 = (6.0 * u2 - 6.0 * u) * inv_h;
        const double dh10 = 3.0 * u2 - 4.0 * u + 1.0;
        const double dh01 = (-6.0 * u2 + 6.0 * u) * inv_h;
        const double dh11 = 3.0 * u2 - 2.0 * u;
        *vel = dh00 * p0 + dh10 * v0 + dh01 * p1 + dh11 * v1;
    }
    if (acc) {
        const double inv_h2 = inv_h * inv_h;
        const double ddh00 = (12.0 * u - 6.0) * inv_h2;
        const double ddh10 = (6.0 * u - 4.0) * inv_h;
        const double ddh01 = (-12.0 * u + 6.0) * inv_h2;
        const double ddh11 = (6.0 * u - 2.0) * inv_h;
        *acc = ddh00 * p0 + ddh10 * v0 + ddh01 * p1 + ddh11 * v1;
    }
    if (jerk) {
        const double inv_h2 = inv_h * inv_h;
        const double inv_h3 = inv_h2 * inv_h;
        *jerk = (12.0 * (p0 - p1)) * inv_h3 + (6.0 * (v0 + v1)) * inv_h2;
    }
}

// Cancellation-free solver for qa * u^2 + qb * u + qc = 0 returning real roots strictly in (u_lo, u_hi).
// Uses Citardauq's formula (q = -0.5 * (qb + copysign(sqrt(disc), qb)); r1 = q/qa; r2 = qc/q)
// and scale-relative linear fallback so tiny `qa` never suffers IEEE-754 catastrophic cancellation.
inline int solve_quadratic_in_interval(double qa, double qb, double qc,
                                       double u_lo, double u_hi, double out_roots[2]) {
    int count = 0;
    const double lin_scale = std::max(std::abs(qb), std::abs(qc));
    if (std::abs(qa) <= 1e-12 * std::max(1.0, lin_scale)) {
        if (std::abs(qb) > 1e-15 * std::max(1.0, std::abs(qc))) {
            const double r = -qc / qb;
            if (r > u_lo && r < u_hi) out_roots[count++] = r;
        }
        return count;
    }
    const double disc = qb * qb - 4.0 * qa * qc;
    if (disc < 0.0) return 0;
    const double sq = std::sqrt(std::max(0.0, disc));
    const double q = -0.5 * (qb + (qb >= 0.0 ? sq : -sq));
    double cand[2];
    int nc = 0;
    if (std::abs(q) > 0.0) {
        cand[nc++] = q / qa;
        cand[nc++] = qc / q;
    } else {
        cand[nc++] = -qb / (2.0 * qa);
    }
    if (nc == 2 && cand[0] > cand[1]) std::swap(cand[0], cand[1]);
    for (int i = 0; i < nc; ++i) {
        if (cand[i] > u_lo && cand[i] < u_hi) {
            if (count == 0 || std::abs(cand[i] - out_roots[count - 1]) > 1e-12) {
                out_roots[count++] = cand[i];
            }
        }
    }
    return count;
}

// Exact [min, max] of a scalar cubic Hermite segment p(u) over u in [0, 1].
inline std::pair<double, double> cubic_hermite_scalar_range(double p0, double p1,
                                                            double v0, double v1, double h) {
    double lo = std::min(p0, p1);
    double hi = std::max(p0, p1);
    if (h <= 0.0) return {lo, hi};
    const double d = p1 - p0;
    const double hv0 = h * v0;
    const double hv1 = h * v1;
    auto eval = [&](double u) {
        const double u2 = u * u;
        const double u3 = u2 * u;
        return p0 + (3.0 * u2 - 2.0 * u3) * d + (u3 - 2.0 * u2 + u) * hv0 + (u3 - u2) * hv1;
    };
    // Roots of dp/du = c2 * u^2 + c1 * u + hv0 = 0 strictly inside (0, 1).
    const double c1 = 6.0 * d - 4.0 * hv0 - 2.0 * hv1;
    const double c2 = -6.0 * d + 3.0 * (hv0 + hv1);
    double roots[2];
    const int nr = solve_quadratic_in_interval(c2, c1, hv0, 0.0, 1.0, roots);
    for (int i = 0; i < nr; ++i) {
        const double val = eval(roots[i]);
        lo = std::min(lo, val);
        hi = std::max(hi, val);
    }
    return {lo, hi};
}

// ============================================================================
// Quintic Hermite (order 5, C^2) scalar evaluation & Bernstein-hull bounds
// ============================================================================

struct QuinticPoly {
    // Power basis p(u) = c0 + c1*u + c2*u^2 + c3*u^3 + c4*u^4 + c5*u^5 on u in [0, 1].
    double c0, c1, c2, c3, c4, c5;
    double h;

    static QuinticPoly from_endpoints(double p0, double v0, double a0,
                                      double p1, double v1, double a1, double h_in) {
        const double h = (h_in > 0.0) ? h_in : 1e-12;
        const double h2 = h * h;
        const double d = p1 - p0;
        QuinticPoly q;
        q.h = h;
        q.c0 = p0;
        q.c1 = h * v0;
        q.c2 = 0.5 * h2 * a0;
        q.c3 = 10.0 * d - (6.0 * v0 + 4.0 * v1) * h - 0.5 * (3.0 * a0 - a1) * h2;
        q.c4 = -15.0 * d + (8.0 * v0 + 7.0 * v1) * h + 0.5 * (3.0 * a0 - 2.0 * a1) * h2;
        q.c5 = 6.0 * d - 3.0 * (v0 + v1) * h - 0.5 * (a0 - a1) * h2;
        return q;
    }

    double eval_pos(double u) const {
        return ((((c5 * u + c4) * u + c3) * u + c2) * u + c1) * u + c0;
    }

    double eval_vel(double u) const {
        const double dp_du = (((5.0 * c5 * u + 4.0 * c4) * u + 3.0 * c3) * u + 2.0 * c2) * u + c1;
        return dp_du / h;
    }

    double eval_acc(double u) const {
        const double d2p_du2 = ((20.0 * c5 * u + 12.0 * c4) * u + 6.0 * c3) * u + 2.0 * c2;
        return d2p_du2 / (h * h);
    }

    double eval_jerk(double u) const {
        const double d3p_du3 = (60.0 * c5 * u + 24.0 * c4) * u + 6.0 * c3;
        return d3p_du3 / (h * h * h);
    }

    // Exact degree-5 Bernstein control points b[0..5] for p(u) on u in [0, 1].
    std::array<double, 6> pos_bernstein(double p0, double v0, double a0,
                                        double p1, double v1, double a1) const {
        const double h2 = h * h;
        return {
            p0,
            p0 + 0.2 * h * v0,
            p0 + 0.4 * h * v0 + 0.05 * h2 * a0,
            p1 - 0.4 * h * v1 + 0.05 * h2 * a1,
            p1 - 0.2 * h * v1,
            p1,
        };
    }

    // Derivative controls from relative displacement and endpoint derivatives. With
    // normalized endpoint derivatives these give dp/du, with physical ones dp/dt.
    static std::array<double, 5> derivative_controls(double mean, double v0, double v1,
                                                      double scaled_a0, double scaled_a1) {
        return {v0, v0 + 0.25 * scaled_a0,
                mean + 2.0 * ((mean - v0) + (mean - v1)) + 0.25 * (scaled_a1 - scaled_a0),
                v1 - 0.25 * scaled_a1, v1};
    }

    // Work in derivatives directly: distinct position controls can round to the same
    // absolute joint position when h is tiny, erasing real velocity/acceleration peaks.
    std::array<double, 5> vel_bernstein(double p0, double v0, double a0,
                                        double p1, double v1, double a1) const {
        return derivative_controls((p1 - p0) / h, v0, v1, h * a0, h * a1);
    }

    // Exact degree-3 Bernstein control points for acceleration a(u) = d^2p/dt^2 on u in [0, 1].
    std::array<double, 4> acc_bernstein(double p0, double v0, double a0,
                                        double p1, double v1, double a1) const {
        const double mean = (p1 - p0) / h;
        const double dv0 = mean - v0, dv1 = mean - v1;
        return {a0, 4.0 * (3.0 * dv0 + 2.0 * dv1) / h + a1 - 2.0 * a0,
                -4.0 * (2.0 * dv0 + 3.0 * dv1) / h + a0 - 2.0 * a1, a1};
    }

    // Exact degree-2 Bernstein control points for jerk j(u) = d^3p/dt^3 on u in [0, 1].
    std::array<double, 3> jerk_bernstein(double p0, double v0, double a0,
                                         double p1, double v1, double a1) const {
        const auto ab = acc_bernstein(p0, v0, a0, p1, v1, a1);
        const double scale = 3.0 / h;
        return {
            scale * (ab[1] - ab[0]),
            scale * (ab[2] - ab[1]),
            scale * (ab[3] - ab[2]),
        };
    }
};

inline void eval_quintic_hermite_scalar(double p0, double v0, double a0,
                                        double p1, double v1, double a1,
                                        double h, double u,
                                        double* pos, double* vel = nullptr,
                                        double* acc = nullptr, double* jerk = nullptr) {
    const QuinticPoly qp = QuinticPoly::from_endpoints(p0, v0, a0, p1, v1, a1, h);
    if (pos) *pos = qp.eval_pos(u);
    if (vel) *vel = qp.eval_vel(u);
    if (acc) *acc = qp.eval_acc(u);
    if (jerk) *jerk = qp.eval_jerk(u);
}

namespace bernstein_detail {

// Exact de Casteljau split of a degree-D Bernstein polynomial on [0, 1] at parameter t in [0, 1].
template <size_t N>
inline void split_at(const std::array<double, N>& b, double t,
                     std::array<double, N>& left, std::array<double, N>& right) {
    std::array<double, N> tmp = b;
    const double s = 1.0 - t;
    left[0] = tmp[0];
    right[N - 1] = tmp[N - 1];
    for (size_t r = 1; r < N; ++r) {
        for (size_t i = 0; i < N - r; ++i) {
            tmp[i] = s * tmp[i] + t * tmp[i + 1];
        }
        left[r] = tmp[0];
        right[N - 1 - r] = tmp[N - 1 - r];
    }
}

// Extract exact Bernstein control points of a degree-D polynomial restricted to [u0, u1] subset [0, 1].
template <size_t N>
inline std::array<double, N> restrict_subinterval(const std::array<double, N>& b, double u0, double u1) {
    if (u0 <= 0.0 && u1 >= 1.0) return b;
    std::array<double, N> left, right;
    if (u1 >= 1.0) {
        split_at(b, u0, left, right);
        return right;
    }
    split_at(b, u1, left, right);
    if (u0 <= 0.0) return left;
    std::array<double, N> sub_left, sub_right;
    split_at(left, u0 / u1, sub_left, sub_right);
    return sub_right;
}

template <size_t N>
inline std::pair<double, double> control_minmax(const std::array<double, N>& b) {
    double lo = b[0], hi = b[0];
    for (size_t i = 1; i < N; ++i) {
        lo = std::min(lo, b[i]);
        hi = std::max(hi, b[i]);
    }
    return {lo, hi};
}

// Certified conservative outer range [cert_lo, cert_hi] of a degree-(N-1) Bernstein polynomial on [0, 1]
// via adaptive de Casteljau branch-and-bound. Guaranteed: cert_lo <= min_{u in [0,1]} P(u) <= max P(u) <= cert_hi.
// Converges to within `tol` of exact extrema without root-finding heuristics that could miss roots.
template <size_t N>
inline std::pair<double, double> certified_range(const std::array<double, N>& b0,
                                                 double tol = 1e-9, int max_nodes = 128) {
    struct Node {
        std::array<double, N> b;
        double lo;
        double hi;
    };
    const auto [lo0, hi0] = control_minmax(b0);
    const double abs_tol = std::max(tol, tol * std::max(std::abs(lo0), std::abs(hi0)));
    // Endpoints b[0] and b[N-1] are exact polynomial values at u=0 and u=1.
    double known_min = std::min(b0[0], b0[N - 1]);
    double known_max = std::max(b0[0], b0[N - 1]);
    if (lo0 >= known_min - abs_tol && hi0 <= known_max + abs_tol) {
        return {lo0, hi0};
    }

    std::array<Node, 160> stack;
    int sp = 0;
    stack[sp++] = {b0, lo0, hi0};
    double cert_lo = known_min;
    double cert_hi = known_max;
    int expanded = 0;

    while (sp > 0) {
        const Node cur = stack[--sp];
        const bool can_lower = (cur.lo < known_min - abs_tol);
        const bool can_raise = (cur.hi > known_max + abs_tol);
        if (!can_lower && !can_raise) {
            cert_lo = std::min(cert_lo, cur.lo);
            cert_hi = std::max(cert_hi, cur.hi);
            continue;
        }
        if (expanded >= max_nodes || sp + 2 >= static_cast<int>(stack.size())) {
            // Budget reached: retain conservative hull of this leaf so guarantee never weakens.
            cert_lo = std::min(cert_lo, cur.lo);
            cert_hi = std::max(cert_hi, cur.hi);
            continue;
        }
        std::array<double, N> left, right;
        split_at(cur.b, 0.5, left, right);
        ++expanded;
        // Midpoint value left[N-1] == right[0] is an exact point on the curve.
        const double mid_val = left[N - 1];
        known_min = std::min(known_min, mid_val);
        known_max = std::max(known_max, mid_val);

        const auto [llo, lhi] = control_minmax(left);
        const auto [rlo, rhi] = control_minmax(right);
        stack[sp++] = {left, llo, lhi};
        stack[sp++] = {right, rlo, rhi};
    }
    cert_lo = std::min(cert_lo, known_min);
    cert_hi = std::max(cert_hi, known_max);
    return {cert_lo, cert_hi};
}

}  // namespace bernstein_detail

// Certified conservative range [cert_min, cert_max] of quintic position p(u) on u in [0, 1].
// Guaranteed: cert_min <= p(u) <= cert_max for all u in [0, 1], tight to ~tol.
inline std::pair<double, double> quintic_hermite_scalar_pos_range(
    double p0, double v0, double a0, double p1, double v1, double a1,
    double h, double tol = 1e-9) {
    if (h <= 0.0) return {std::min(p0, p1), std::max(p0, p1)};
    const QuinticPoly qp = QuinticPoly::from_endpoints(p0, v0, a0, p1, v1, a1, h);
    const auto b = qp.pos_bernstein(p0, v0, a0, p1, v1, a1);
    return bernstein_detail::certified_range(b, tol);
}

// Certified upper bound on max_{u in [0, 1]} |p(u)| for quintic segment (used for prismatic CCD reach).
inline double quintic_hermite_scalar_abs_max(
    double p0, double v0, double a0, double p1, double v1, double a1,
    double h, double tol = 1e-7) {
    const auto [lo, hi] = quintic_hermite_scalar_pos_range(p0, v0, a0, p1, v1, a1, h, tol);
    return std::max(std::abs(lo), std::abs(hi));
}

// Certified conservative peaks (max |v|, max |a|, max |j|) of a scalar quintic Hermite segment on [0, 1].
// - Jerk (degree 2): exact closed-form (checks both knot sides u=0, u=1 and interior stationary point).
// - Acceleration (degree 3): exact closed-form (checks u=0, u=1 and all real roots of quadratic j(u)=0).
// - Velocity (degree 4): certified Bernstein branch-and-bound (guaranteed >= true max, tight to ~tol).
struct QuinticSegmentPeaks {
    double max_vel = 0.0;
    double max_acc = 0.0;
    double max_jerk = 0.0;
};

inline QuinticSegmentPeaks quintic_hermite_scalar_peaks(
    double p0, double v0, double a0, double p1, double v1, double a1,
    double h_in, double tol = 1e-9) {
    const double h = (h_in > 0.0) ? h_in : 1e-12;
    const QuinticPoly qp = QuinticPoly::from_endpoints(p0, v0, a0, p1, v1, a1, h);
    QuinticSegmentPeaks out;

    // 1. Velocity (degree 4) via certified Bernstein branch-and-bound.
    const auto w = qp.vel_bernstein(p0, v0, a0, p1, v1, a1);
    const auto [v_lo, v_hi] = bernstein_detail::certified_range(w, tol);
    out.max_vel = std::max(std::abs(v_lo), std::abs(v_hi));

    // 2. Acceleration (degree 3): combine cancellation-free Citardauq stationary points AND
    //    certified degree-3 Bernstein branch-and-bound so near-zero c5 never underestimates peaks.
    const auto ab = qp.acc_bernstein(p0, v0, a0, p1, v1, a1);
    const auto [a_lo, a_hi] = bernstein_detail::certified_range(ab, tol);
    out.max_acc = std::max({std::abs(a0), std::abs(a1), std::abs(a_lo), std::abs(a_hi)});
    double acc_roots[2];
    const int n_acc_roots = solve_quadratic_in_interval(10.0 * qp.c5, 4.0 * qp.c4, qp.c3, 0.0, 1.0, acc_roots);
    for (int i = 0; i < n_acc_roots; ++i) {
        out.max_acc = std::max(out.max_acc, std::abs(qp.eval_acc(acc_roots[i])));
    }

    // 3. Jerk (degree 2): endpoints (both knot sides), interior vertex, AND degree-2 Bernstein envelope.
    const auto jb = qp.jerk_bernstein(p0, v0, a0, p1, v1, a1);
    const auto [j_lo, j_hi] = bernstein_detail::certified_range(jb, tol);
    out.max_jerk = std::max({std::abs(qp.eval_jerk(0.0)), std::abs(qp.eval_jerk(1.0)),
                             std::abs(j_lo), std::abs(j_hi)});
    if (std::abs(qp.c5) > 1e-12 * std::max(1.0, std::abs(qp.c4))) {
        const double u_star = -qp.c4 / (5.0 * qp.c5);
        if (u_star > 0.0 && u_star < 1.0) {
            out.max_jerk = std::max(out.max_jerk, std::abs(qp.eval_jerk(u_star)));
        }
    }

    return out;
}

// Certified upper bound on total variation int_{u0}^{u1} |dp/du| du for a scalar quintic Hermite segment
// on subinterval [u0, u1] subset [0, 1].
// Uses exact de Casteljau restriction of degree-4 velocity Bernstein control points:
// - If all subinterval velocity control points share sign (monotonic), returns exact |p(u1) - p(u0)|.
// - Otherwise returns conservative h * (u1 - u0) * max_i |w_i^{[u0, u1]}|, which tightens proportionally
//   with subinterval duration h * (u1 - u0) under CCD bisection.
inline double quintic_hermite_scalar_total_variation_bound(
    double p0, double v0, double a0, double p1, double v1, double a1,
    double h_in, double u0, double u1) {
    if (u1 <= u0) return 0.0;
    if (h_in <= 0.0) return std::abs(p1 - p0);
    const auto dp_du_full = QuinticPoly::derivative_controls(
        p1 - p0, h_in * v0, h_in * v1, h_in * (h_in * a0), h_in * (h_in * a1));
    const auto dp_du_sub = bernstein_detail::restrict_subinterval(dp_du_full, u0, u1);
    const auto [d_lo, d_hi] = bernstein_detail::control_minmax(dp_du_sub);
    if (d_lo >= 0.0 || d_hi <= 0.0) {
        // Each degree-4 Bernstein basis function integrates to 1/5 on [0, 1].
        double mean = 0.0;
        for (double derivative : dp_du_sub) mean += derivative / dp_du_sub.size();
        return (u1 - u0) * std::abs(mean);
    }
    const double dmax_sub = std::max(std::abs(d_lo), std::abs(d_hi));
    return (u1 - u0) * dmax_sub;
}

}  // namespace gobot::planning
