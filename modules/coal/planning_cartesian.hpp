#pragma once

#include "planning_retime.hpp"
#include "planning_world.hpp"

#include <Eigen/SVD>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace gobot::planning {

enum class CartesianMotionType : int {
    Linear = 0,
    Circular = 1,
};

struct CartesianWaypointNative {
    CartesianMotionType motion_type = CartesianMotionType::Linear;
    Vec3 position = Vec3::Zero();
    bool has_rotation = false;
    Mat3 rotation = Mat3::Identity();
    bool has_via_position = false;
    Vec3 via_position = Vec3::Zero();
    double blend_radius = 0.0;
};

struct CartesianPlanConfigNative {
    double max_step = 0.01;
    double max_rot_step = 0.05;
    double max_pos_error = 1e-3;
    double max_rot_error = 1e-2;
    double jump_threshold_revolute = 0.5;
    double jump_threshold_prismatic = 0.2;
    int max_subdivision_depth = 4;
    int ik_iterations = 30;
    double ik_damping = 1e-3;
    bool avoid_collisions = true;
    double margin = 0.002;
    bool has_world_margin = false;
    double world_margin = 0.002;
    bool strict_ccd = false;
    int collision_substeps = 4;
    int spline_order = 3;
    double velocity_scale = 1.0;
    double acceleration_scale = 1.0;
    double jerk_scale = 1.0;
    bool stop_at_endpoints = true;
    bool hold_orientation = false;
    bool verify_original = true;

    double effective_world_margin() const {
        return has_world_margin ? world_margin : margin;
    }
};

struct CartesianPlanRequestNative {
    Eigen::VectorXd start_q;
    int link_id = -1;
    Vec3 tcp_offset_position = Vec3::Zero();
    Mat3 tcp_offset_rotation = Mat3::Identity();
    std::vector<CartesianWaypointNative> waypoints;
    std::vector<uint8_t> active_mask;  // empty => all joints active
    CartesianPlanConfigNative config;
};

struct CartesianPlanResultNative {
    bool success = false;
    bool has_trajectory = false;
    TrajectoryNative trajectory;
    double fraction = 0.0;
    int completed_waypoints = 0;
    double max_pos_error = 0.0;
    double max_rot_error = 0.0;
    uint64_t world_version = 0;
    std::string reason;
    // Timing breakdown matching PlanDiagnosticsNative
    double total_ms = 0.0;
    double ik_ms = 0.0;
    double retime_ms = 0.0;
    double validate_ms = 0.0;
};

namespace cartesian_detail {

inline Mat3 rodrigues_exp(const Vec3& phi) {
    const double theta = phi.norm();
    if (theta < 1e-7) {
        const Mat3 S = skew_matrix(phi);
        return Mat3::Identity() + S + 0.5 * (S * S);
    }
    const Vec3 u = phi / theta;
    const Mat3 S = skew_matrix(u);
    return Mat3::Identity() + std::sin(theta) * S + (1.0 - std::cos(theta)) * (S * S);
}

inline Mat3 validate_and_orthonormalize_rot(const Mat3& R, const char* context) {
    if (!R.allFinite()) {
        throw std::invalid_argument(std::string(context) + ": rotation matrix must be finite");
    }
    const double ortho_err = (R.transpose() * R - Mat3::Identity()).norm();
    const double det = R.determinant();
    if (ortho_err > 1e-3 || det < 0.99) {
        throw std::invalid_argument(std::string(context) + ": rotation matrix must be proper orthonormal (SO(3))");
    }
    Eigen::JacobiSVD<Mat3> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Mat3 U = svd.matrixU();
    const Mat3 Vt = svd.matrixV().transpose();
    Mat3 R_clean = U * Vt;
    if (R_clean.determinant() < 0.0) {
        U.col(2) *= -1.0;
        R_clean = U * Vt;
    }
    return R_clean;
}

inline double rot_geodesic_angle(const Mat3& R_a, const Mat3& R_b) {
    return rotation_log(R_a.transpose() * R_b).norm();
}

struct CartesianPrimitive {
    enum class Kind { Linear = 0, CircularArc = 1, BlendBezier = 2 };
    Kind kind = Kind::Linear;

    // Linear endpoints
    Vec3 p0 = Vec3::Zero();
    Vec3 p1 = Vec3::Zero();

    // Circular arc: p(tau) = arc_center + arc_radius * (cos(ang)*arc_u + sin(ang)*arc_v)
    // where ang = arc_angle0 + tau * arc_angle_span, arc_angle_span > 0.
    Vec3 arc_center = Vec3::Zero();
    Vec3 arc_u = Vec3::UnitX();
    Vec3 arc_v = Vec3::UnitY();
    double arc_radius = 0.0;
    double arc_angle0 = 0.0;
    double arc_angle_span = 0.0;
    bool has_via_tau = false;
    double arc_via_tau = 0.5;

    // Cubic Bezier blend
    Vec3 bez[4] = {Vec3::Zero(), Vec3::Zero(), Vec3::Zero(), Vec3::Zero()};

    double arc_length = 0.0;

    // Orientation interpolation: R(tau) = r0 * exp(tau * rot_phi)
    bool has_rot = false;
    bool dynamic_r0 = false;
    double dynamic_r1_frac = 1.0;
    Mat3 r0 = Mat3::Identity();
    Mat3 r1 = Mat3::Identity();
    Vec3 rot_phi = Vec3::Zero();
    double rot_angle = 0.0;

    // Progress & corner metadata
    double progress_start = 0.0;
    double progress_end = 0.0;
    bool stop_at_end = false;
    int orig_waypoint_idx = 0;
    bool is_blend = false;

    void set_rotations(const Mat3& R_start, const Mat3& R_end) {
        r0 = R_start;
        r1 = R_end;
        rot_phi = rotation_log(r0.transpose() * r1);
        rot_angle = rot_phi.norm();
    }

    Vec3 eval_pos(double tau) const {
        const double t = std::clamp(tau, 0.0, 1.0);
        if (kind == Kind::Linear) {
            return (1.0 - t) * p0 + t * p1;
        }
        if (kind == Kind::CircularArc) {
            const double ang = arc_angle0 + t * arc_angle_span;
            return arc_center + arc_radius * (std::cos(ang) * arc_u + std::sin(ang) * arc_v);
        }
        // Cubic Bezier
        const double s = 1.0 - t;
        const double s2 = s * s;
        const double t2 = t * t;
        return (s2 * s) * bez[0] + (3.0 * s2 * t) * bez[1] + (3.0 * s * t2) * bez[2] + (t2 * t) * bez[3];
    }

    Vec3 eval_tangent(double tau) const {
        const double t = std::clamp(tau, 0.0, 1.0);
        Vec3 d = Vec3::Zero();
        if (kind == Kind::Linear) {
            d = p1 - p0;
        } else if (kind == Kind::CircularArc) {
            const double ang = arc_angle0 + t * arc_angle_span;
            d = (-std::sin(ang) * arc_u + std::cos(ang) * arc_v) * (arc_radius * arc_angle_span);
        } else {
            const double s = 1.0 - t;
            d = 3.0 * s * s * (bez[1] - bez[0]) + 6.0 * s * t * (bez[2] - bez[1]) + 3.0 * t * t * (bez[3] - bez[2]);
        }
        const double n = d.norm();
        return (n > 1e-12) ? Vec3(d / n) : Vec3::Zero();
    }

    Mat3 eval_rot(double tau) const {
        if (!has_rot) return r0;
        const double t = std::clamp(tau, 0.0, 1.0);
        return r0 * rodrigues_exp(t * rot_phi);
    }

    double compute_bezier_length() const {
        // Composite Simpson quadrature (32 subintervals) for cubic Bezier arc length.
        constexpr int N = 32;
        const double h = 1.0 / static_cast<double>(N);
        auto speed = [&](double t) {
            const double s = 1.0 - t;
            const Vec3 d = 3.0 * s * s * (bez[1] - bez[0]) + 6.0 * s * t * (bez[2] - bez[1]) +
                           3.0 * t * t * (bez[3] - bez[2]);
            return d.norm();
        };
        double sum = speed(0.0) + speed(1.0);
        for (int i = 1; i < N; ++i) {
            sum += ((i % 2 == 0) ? 2.0 : 4.0) * speed(static_cast<double>(i) * h);
        }
        return sum * h / 3.0;
    }

    // Exact or high-accuracy closest-position parameter tau in [tau0, tau1].
    double closest_pos_tau(const Vec3& p, double tau0, double tau1) const {
        if (tau1 <= tau0 + 1e-15) return tau0;
        if (kind == Kind::Linear) {
            const Vec3 d = p1 - p0;
            const double d2 = d.squaredNorm();
            if (d2 <= 1e-24) return tau0;
            return std::clamp((p - p0).dot(d) / d2, tau0, tau1);
        }
        if (kind == Kind::CircularArc) {
            const Vec3 rel = p - arc_center;
            const double cx = rel.dot(arc_u);
            const double cy = rel.dot(arc_v);
            if (cx * cx + cy * cy <= 1e-24) {
                const double d0 = (p - eval_pos(tau0)).squaredNorm();
                const double d1 = (p - eval_pos(tau1)).squaredNorm();
                return (d0 <= d1) ? tau0 : tau1;
            }
            const double ang_a = arc_angle0 + tau0 * arc_angle_span;
            const double ang_b = arc_angle0 + tau1 * arc_angle_span;
            const double ang_mid = 0.5 * (ang_a + ang_b);
            double ang = std::atan2(cy, cx);
            // Wrap ang into [ang_mid - pi, ang_mid + pi]
            ang += 2.0 * M_PI * std::round((ang_mid - ang) / (2.0 * M_PI));
            double best_tau = tau0;
            double best_d2 = (p - eval_pos(tau0)).squaredNorm();
            const double d2_end = (p - eval_pos(tau1)).squaredNorm();
            if (d2_end < best_d2) {
                best_d2 = d2_end;
                best_tau = tau1;
            }
            if (ang >= ang_a && ang <= ang_b && arc_angle_span > 1e-15) {
                const double cand = std::clamp((ang - arc_angle0) / arc_angle_span, tau0, tau1);
                const double d2_c = (p - eval_pos(cand)).squaredNorm();
                if (d2_c < best_d2) best_tau = cand;
            }
            return best_tau;
        }
        // Cubic Bezier: coarse grid + golden section polish.
        double best_t = tau0;
        double best_d2 = (p - eval_pos(tau0)).squaredNorm();
        constexpr int G = 8;
        for (int i = 1; i <= G; ++i) {
            const double t = tau0 + (tau1 - tau0) * (static_cast<double>(i) / static_cast<double>(G));
            const double d2 = (p - eval_pos(t)).squaredNorm();
            if (d2 < best_d2) {
                best_d2 = d2;
                best_t = t;
            }
        }
        const double step = (tau1 - tau0) / static_cast<double>(G);
        double lo = std::max(tau0, best_t - step);
        double hi = std::min(tau1, best_t + step);
        constexpr double inv_phi = 0.6180339887498948482;
        double x1 = hi - inv_phi * (hi - lo);
        double x2 = lo + inv_phi * (hi - lo);
        double f1 = (p - eval_pos(x1)).squaredNorm();
        double f2 = (p - eval_pos(x2)).squaredNorm();
        for (int it = 0; it < 20; ++it) {
            if (f1 < f2) {
                hi = x2;
                x2 = x1;
                f2 = f1;
                x1 = hi - inv_phi * (hi - lo);
                f1 = (p - eval_pos(x1)).squaredNorm();
            } else {
                lo = x1;
                x1 = x2;
                f1 = f2;
                x2 = lo + inv_phi * (hi - lo);
                f2 = (p - eval_pos(x2)).squaredNorm();
            }
        }
        const double mid = 0.5 * (lo + hi);
        if ((p - eval_pos(mid)).squaredNorm() < best_d2) best_t = mid;
        return best_t;
    }

    // Closest-rotation parameter tau in [tau0, tau1] along R(tau) = r0 * exp(tau * rot_phi).
    double closest_rot_tau(const Mat3& R, double tau0, double tau1) const {
        if (!has_rot || rot_angle <= 1e-12 || tau1 <= tau0 + 1e-15) return tau0;
        const Vec3 psi = rotation_log(r0.transpose() * R);
        const double t_est = psi.dot(rot_phi) / (rot_angle * rot_angle);
        return std::clamp(t_est, tau0, tau1);
    }

    // Coupled normalized error Phi(q) = min_{tau in [tau0, tau1]} max(pos_err/eps_p, rot_err/eps_r).
    // Note: any valid upper bound on min_tau is sound for certifying Phi <= 1; we compute it tightly.
    double eval_coupled_error(const Vec3& p, const Mat3& R, double tau0, double tau1,
                              double eps_p, double eps_r,
                              double& out_pos_err, double& out_rot_err) const {
        auto eval_at = [&](double tau, double& pe, double& re) -> double {
            pe = (p - eval_pos(tau)).norm();
            re = has_rot ? rot_geodesic_angle(eval_rot(tau), R) : 0.0;
            const double np = pe / eps_p;
            const double nr = has_rot ? (re / eps_r) : 0.0;
            return std::max(np, nr);
        };

        if (!has_rot || rot_angle <= 1e-12) {
            const double t = closest_pos_tau(p, tau0, tau1);
            return eval_at(t, out_pos_err, out_rot_err);
        }
        if (arc_length <= 1e-12) {
            const double t = closest_rot_tau(R, tau0, tau1);
            return eval_at(t, out_pos_err, out_rot_err);
        }

        // Mixed translation + rotation: evaluate key candidates and polish with golden section.
        double best_pe = 0.0, best_re = 0.0;
        double best_phi = eval_at(tau0, best_pe, best_re);
        auto try_cand = [&](double t) {
            double pe = 0.0, re = 0.0;
            const double ph = eval_at(t, pe, re);
            if (ph < best_phi) {
                best_phi = ph;
                best_pe = pe;
                best_re = re;
            }
        };
        const double tp = closest_pos_tau(p, tau0, tau1);
        const double tr = closest_rot_tau(R, tau0, tau1);
        try_cand(tau1);
        try_cand(tp);
        try_cand(tr);
        try_cand(0.5 * (tp + tr));

        double lo = std::min(tp, tr);
        double hi = std::max(tp, tr);
        if (hi - lo > 1e-12) {
            constexpr double inv_phi = 0.6180339887498948482;
            double x1 = hi - inv_phi * (hi - lo);
            double x2 = lo + inv_phi * (hi - lo);
            double pe1, re1, pe2, re2;
            double f1 = eval_at(x1, pe1, re1);
            double f2 = eval_at(x2, pe2, re2);
            for (int it = 0; it < 18; ++it) {
                if (f1 < f2) {
                    hi = x2;
                    x2 = x1;
                    f2 = f1;
                    x1 = hi - inv_phi * (hi - lo);
                    f1 = eval_at(x1, pe1, re1);
                } else {
                    lo = x1;
                    x1 = x2;
                    f1 = f2;
                    x2 = lo + inv_phi * (hi - lo);
                    f2 = eval_at(x2, pe2, re2);
                }
            }
            try_cand(0.5 * (lo + hi));
        }
        out_pos_err = best_pe;
        out_rot_err = best_re;
        return best_phi;
    }
};

// Build directed circular arc primitive from p0 -> p_via -> p1.
// Explicitly rejects duplicate, near-duplicate, collinear, or near-collinear triplets.
inline CartesianPrimitive make_circular_arc_primitive(const Vec3& p0, const Vec3& p_via, const Vec3& p1) {
    if (!p0.allFinite() || !p_via.allFinite() || !p1.allFinite()) {
        throw std::invalid_argument("circular arc points must be finite");
    }
    const Vec3 a = p_via - p0;
    const Vec3 b = p1 - p0;
    const Vec3 c = p1 - p_via;
    const double la = a.norm();
    const double lb = b.norm();
    const double lc = c.norm();
    if (la <= 1e-7 || lb <= 1e-7 || lc <= 1e-7) {
        throw std::invalid_argument("circular arc requires three distinct points");
    }
    const Vec3 n_raw = a.cross(b);
    const double n_norm = n_raw.norm();
    const double sin_theta = n_norm / (la * lb);
    if (!std::isfinite(sin_theta) || sin_theta < 1e-5) {
        throw std::invalid_argument("circular arc points are collinear or near-collinear");
    }
    const Vec3 n_hat = n_raw / n_norm;
    const Vec3 e1 = a / la;
    const Vec3 e2 = n_hat.cross(e1);

    const double xb = la;
    const double xc = b.dot(e1);
    const double yc = b.dot(e2);  // strictly > 0 because n_hat aligns with a x b
    if (yc <= 1e-12) {
        throw std::invalid_argument("circular arc points are degenerate");
    }

    const double xo = 0.5 * xb;
    const double yo = (xc * xc + yc * yc - 2.0 * xc * xo) / (2.0 * yc);
    const double radius = std::hypot(xo, yo);
    const double max_chord = std::max({la, lb, lc});
    if (!std::isfinite(radius) || radius <= 1e-9 || radius > 1e5 * max_chord) {
        throw std::invalid_argument("circular arc has degenerate or ill-conditioned radius");
    }

    const Vec3 center = p0 + xo * e1 + yo * e2;
    const Vec3 u = (p0 - center) / radius;
    const Vec3 v = n_hat.cross(u);

    auto wrap_pos = [](double ang) {
        if (ang < 0.0) ang += 2.0 * M_PI;
        return ang;
    };
    const double ang_via = wrap_pos(std::atan2((p_via - center).dot(v), (p_via - center).dot(u)));
    const double ang_end = wrap_pos(std::atan2((p1 - center).dot(v), (p1 - center).dot(u)));
    if (!(ang_via > 1e-9 && ang_end > ang_via + 1e-9 && ang_end < 2.0 * M_PI - 1e-9)) {
        throw std::invalid_argument("circular arc directed ordering p0 -> via -> end is degenerate");
    }

    CartesianPrimitive prim;
    prim.kind = CartesianPrimitive::Kind::CircularArc;
    prim.p0 = p0;
    prim.p1 = p1;
    prim.arc_center = center;
    prim.arc_u = u;
    prim.arc_v = v;
    prim.arc_radius = radius;
    prim.arc_angle0 = 0.0;
    prim.arc_angle_span = ang_end;
    prim.has_via_tau = true;
    prim.arc_via_tau = ang_via / ang_end;
    prim.arc_length = radius * ang_end;
    return prim;
}

// Compute TCP world pose and optional analytic Jacobian [3/6 x J] for link + tcp_offset.
inline void eval_tcp_pose_and_jacobian(const KinematicsModel& kin,
                                       int link_id,
                                       const Vec3& tcp_pos_local,
                                       const Mat3& tcp_rot_local,
                                       const Vec3* body_pos,
                                       const Mat3* body_rot,
                                       const Vec3* j_origin,
                                       const Vec3* j_axis,
                                       Vec3& out_pos,
                                       Mat3& out_rot,
                                       Eigen::MatrixXd* out_jac_pos = nullptr,
                                       Eigen::MatrixXd* out_jac_ang = nullptr) {
    out_pos.noalias() = body_pos[link_id] + body_rot[link_id] * tcp_pos_local;
    out_rot.noalias() = body_rot[link_id] * tcp_rot_local;
    if (out_jac_pos || out_jac_ang) {
        const int J = kin.num_joints();
        const uint8_t* anc = kin.body_ancestor().data() + link_id * J;
        const auto& is_pris = kin.joint_prismatic();
        if (out_jac_pos) out_jac_pos->setZero(3, J);
        if (out_jac_ang) out_jac_ang->setZero(3, J);
        for (int j = 0; j < J; ++j) {
            if (!anc[j]) continue;
            if (is_pris[static_cast<size_t>(j)]) {
                if (out_jac_pos) out_jac_pos->col(j) = j_axis[j];
            } else {
                if (out_jac_pos) out_jac_pos->col(j) = j_axis[j].cross(out_pos - j_origin[j]);
                if (out_jac_ang) out_jac_ang->col(j) = j_axis[j];
            }
        }
    }
}

// Unwrap revolute joint angles by +/- 2*pi multiples toward q_prev while staying inside [lo, hi].
inline void unwrap_revolute_near(Eigen::VectorXd& q,
                                 const Eigen::VectorXd& q_prev,
                                 const Eigen::VectorXd& lo,
                                 const Eigen::VectorXd& hi,
                                 const std::vector<uint8_t>& is_pris,
                                 const std::vector<uint8_t>& active_mask) {
    const int J = static_cast<int>(q.size());
    constexpr double two_pi = 2.0 * M_PI;
    for (int j = 0; j < J; ++j) {
        if (!active_mask[static_cast<size_t>(j)] || is_pris[static_cast<size_t>(j)]) continue;
        const double diff = q_prev(j) - q(j);
        const double k = std::round(diff / two_pi);
        if (k != 0.0) {
            const double cand = q(j) + k * two_pi;
            if (cand >= lo(j) - 1e-12 && cand <= hi(j) + 1e-12) {
                q(j) = std::clamp(cand, lo(j), hi(j));
            }
        }
    }
}

struct JumpCheck {
    bool ok = true;
    double max_ratio = 0.0;
    int worst_joint = -1;
    double worst_delta = 0.0;
    double worst_limit = 0.0;
};

inline JumpCheck check_joint_jump(const Eigen::VectorXd& qa,
                                  const Eigen::VectorXd& qb,
                                  const std::vector<uint8_t>& is_pris,
                                  double rev_thresh,
                                  double pris_thresh) {
    JumpCheck out;
    const int J = static_cast<int>(qa.size());
    for (int j = 0; j < J; ++j) {
        const double lim = is_pris[static_cast<size_t>(j)] ? pris_thresh : rev_thresh;
        if (!std::isfinite(lim)) continue;
        const double dq = std::abs(qb(j) - qa(j));
        const double ratio = dq / lim;
        if (ratio > out.max_ratio) {
            out.max_ratio = ratio;
            out.worst_joint = j;
            out.worst_delta = dq;
            out.worst_limit = lim;
        }
        if (dq > lim + 1e-12) out.ok = false;
    }
    return out;
}

enum class StepIKStatus {
    Ok = 0,
    PoseFailed = 1,
    JumpExceeded = 2,
    Collision = 3,
};

struct StepIKResult {
    StepIKStatus status = StepIKStatus::PoseFailed;
    Eigen::VectorXd q;
    double jump_ratio = 0.0;
    double pos_err = 0.0;
    double rot_err = 0.0;
    std::string reason;
};

// Continuity-preserving projected Levenberg-Marquardt IK for a single Cartesian TCP target.
// Always seeded from q_prev with mild continuity regularization and continuous revolute unwrapping.
inline StepIKResult solve_cartesian_step_ik(
    const WorldSnapshot& snap,
    int link_id,
    const Vec3& tcp_pos_local,
    const Mat3& tcp_rot_local,
    const Vec3& target_pos,
    const Mat3& target_rot,
    bool has_rot,
    const Eigen::VectorXd& q_prev,
    const std::vector<uint8_t>& active_mask,
    const CartesianPlanConfigNative& cfg,
    const Eigen::VectorXd& margins,
    double pos_tol,
    double rot_tol) {
    StepIKResult out;
    const int J = snap.kin.num_joints();
    const int B = snap.kin.num_bodies();
    const int N_shapes = static_cast<int>(snap.shapes.size());
    const int P = static_cast<int>(snap.pairs.size());
    const auto& lo = snap.kin.lower();
    const auto& hi = snap.kin.upper();
    const auto& is_pris = snap.kin.joint_prismatic();

    std::vector<Vec3> bpos(static_cast<size_t>(B)), jorig(static_cast<size_t>(J)), jaxis(static_cast<size_t>(J));
    std::vector<Mat3> brot(static_cast<size_t>(B));
    std::vector<double> frame(static_cast<size_t>(N_shapes * 12));
    std::vector<int64_t> cand_rows, cand_pairs;
    std::vector<double> dist_data, col_grad;

    auto eval_pose_err = [&](const Eigen::VectorXd& qc, double& pe, double& re) -> bool {
        snap.kin.fk_single(qc.data(), bpos.data(), brot.data());
        Vec3 tp;
        Mat3 tr;
        eval_tcp_pose_and_jacobian(snap.kin, link_id, tcp_pos_local, tcp_rot_local,
                                   bpos.data(), brot.data(), nullptr, nullptr, tp, tr);
        pe = (tp - target_pos).norm();
        re = has_rot ? rot_geodesic_angle(target_rot, tr) : 0.0;
        return (pe <= pos_tol) && (!has_rot || re <= rot_tol);
    };

    auto eval_collision = [&](const Eigen::VectorXd& qc, std::string& col_reason) -> bool {
        if (!cfg.avoid_collisions || P == 0) return true;
        snap.compute_frames(qc.data(), 1, frame.data());
        snap.screen_aabbs(frame.data(), 1, margins.data(), false, 1, cand_rows, cand_pairs);
        if (!cand_rows.empty()) {
            auto hit = snap.first_violation_indexed(frame.data(), 1, cand_rows.data(), cand_pairs.data(),
                                                    static_cast<int64_t>(cand_rows.size()), margins.data(), false);
            if (hit.has_value()) {
                col_reason = "Cartesian IK collision on pair " + snap.pair_display_name(hit->second);
                return false;
            }
        }
        if (cfg.verify_original && snap.has_mesh) {
            snap.screen_aabbs(frame.data(), 1, margins.data(), true, 1, cand_rows, cand_pairs);
            if (!cand_rows.empty()) {
                auto hit = snap.first_violation_indexed(frame.data(), 1, cand_rows.data(), cand_pairs.data(),
                                                        static_cast<int64_t>(cand_rows.size()), margins.data(), true);
                if (hit.has_value()) {
                    col_reason = "Cartesian IK original mesh collision on pair " + snap.pair_display_name(hit->second);
                    return false;
                }
            }
        }
        return true;
    };

    auto run_lm = [&](bool with_collision) -> Eigen::VectorXd {
        Eigen::VectorXd q = q_prev;
        for (int j = 0; j < J; ++j) q(j) = std::clamp(q(j), lo(j), hi(j));

        const double w_pos = 1.0 / std::max(pos_tol, 1e-6);
        const double w_rot = has_rot ? (1.0 / std::max(rot_tol, 1e-6)) : 0.0;
        // Very mild continuity pull so redundant DOFs stay close to q_prev without sacrificing pose accuracy.
        const double w_stay = with_collision ? 0.5 : 1e-2;
        const int outer_rounds = (with_collision && P > 0) ? 4 : 1;
        Eigen::VectorXd dual = Eigen::VectorXd::Zero(P);
        double penalty = 50.0;
        std::vector<double> thr(static_cast<size_t>(P));

        Eigen::MatrixXd Jp(3, J), Jw(3, J);
        Eigen::MatrixXd H(J, J);
        Eigen::VectorXd g(J);
        Eigen::VectorXd ineq(P);

        auto eval_merit = [&](const Eigen::VectorXd& qc, Eigen::MatrixXd* H_out, Eigen::VectorXd* g_out,
                              Eigen::VectorXd* ineq_out = nullptr) -> double {
            const bool need_jac = (H_out != nullptr);
            snap.kin.fk_single(qc.data(), bpos.data(), brot.data(),
                               need_jac ? jorig.data() : nullptr, need_jac ? jaxis.data() : nullptr);
            Vec3 tp;
            Mat3 tr;
            eval_tcp_pose_and_jacobian(snap.kin, link_id, tcp_pos_local, tcp_rot_local,
                                       bpos.data(), brot.data(),
                                       need_jac ? jorig.data() : nullptr,
                                       need_jac ? jaxis.data() : nullptr,
                                       tp, tr,
                                       need_jac ? &Jp : nullptr,
                                       (need_jac && has_rot) ? &Jw : nullptr);
            if (need_jac) {
                H_out->setZero(J, J);
                g_out->setZero(J);
            }
            double cost = 0.0;
            // Continuity regularization toward q_prev
            for (int j = 0; j < J; ++j) {
                if (!active_mask[static_cast<size_t>(j)]) continue;
                const double r = w_stay * (qc(j) - q_prev(j));
                cost += r * r;
                if (need_jac) {
                    (*H_out)(j, j) += w_stay * w_stay;
                    (*g_out)(j) += w_stay * r;
                }
            }
            // Position residual
            const Vec3 rp = w_pos * (tp - target_pos);
            cost += rp.squaredNorm();
            if (need_jac) {
                Jp *= w_pos;
                H_out->noalias() += Jp.transpose() * Jp;
                g_out->noalias() += Jp.transpose() * rp;
            }
            // Orientation residual
            if (has_rot) {
                const Vec3 phi = rotation_log(target_rot.transpose() * tr);
                const Vec3 rr = w_rot * phi;
                cost += rr.squaredNorm();
                if (need_jac) {
                    const Mat3 J_rot_map = w_rot * so3_left_jacobian_inv(phi) * target_rot.transpose();
                    Eigen::MatrixXd Jr = J_rot_map * Jw;
                    H_out->noalias() += Jr.transpose() * Jr;
                    g_out->noalias() += Jr.transpose() * rr;
                }
            }
            // Optional collision avoidance (augmented Lagrangian)
            if (with_collision && P > 0) {
                for (int i = 0; i < N_shapes; ++i) {
                    const auto& ws = snap.shapes[static_cast<size_t>(i)];
                    Eigen::Map<Frame> dst(frame.data() + i * 12);
                    if (ws.body_id >= 0) {
                        dst.leftCols<3>().noalias() = brot[static_cast<size_t>(ws.body_id)] * ws.local_frame.leftCols<3>();
                        dst.col(3).noalias() = brot[static_cast<size_t>(ws.body_id)] * ws.local_frame.col(3) +
                                               bpos[static_cast<size_t>(ws.body_id)];
                    } else {
                        dst = ws.local_frame;
                    }
                }
                if (ineq_out) ineq_out->setConstant(P, -1.0);
                for (int p = 0; p < P; ++p) thr[static_cast<size_t>(p)] = margins(p) + dual(p) / penalty + 0.01;
                snap.screen_aabbs(frame.data(), 1, thr.data(), false, 1, cand_rows, cand_pairs);
                const int64_t C = static_cast<int64_t>(cand_rows.size());
                if (C > 0) {
                    dist_data.resize(static_cast<size_t>(C * 10));
                    for (int64_t k = 0; k < C; ++k) {
                        const auto& pr = snap.pairs[static_cast<size_t>(cand_pairs[static_cast<size_t>(k)])];
                        eval_coal_distance(*snap.shapes[static_cast<size_t>(pr.shape_a)].shape,
                                           *snap.shapes[static_cast<size_t>(pr.shape_b)].shape,
                                           frame.data() + pr.shape_a * 12, frame.data() + pr.shape_b * 12,
                                           false, std::numeric_limits<double>::infinity(),
                                           dist_data.data() + k * 10);
                    }
                    std::vector<int64_t> act_rows, act_pairs;
                    std::vector<double> act_data, act_res;
                    const double sqrt_pen = std::sqrt(penalty);
                    for (int64_t k = 0; k < C; ++k) {
                        const int64_t pid = cand_pairs[static_cast<size_t>(k)];
                        const double viol = margins(pid) - dist_data[static_cast<size_t>(k * 10)];
                        if (ineq_out) (*ineq_out)(pid) = viol;
                        const double aug = viol + dual(pid) / penalty;
                        if (aug > 0.0) {
                            const double sr = sqrt_pen * aug;
                            cost += sr * sr;
                            if (need_jac) {
                                act_rows.push_back(0);
                                act_pairs.push_back(pid);
                                act_res.push_back(sr);
                                act_data.insert(act_data.end(), dist_data.data() + k * 10,
                                                dist_data.data() + (k + 1) * 10);
                            }
                        }
                    }
                    if (need_jac && !act_rows.empty()) {
                        const int64_t K = static_cast<int64_t>(act_rows.size());
                        col_grad.resize(static_cast<size_t>(K * J));
                        snap.compute_analytic_gradients(frame.data(),
                                                        reinterpret_cast<const double*>(jorig.data()),
                                                        reinterpret_cast<const double*>(jaxis.data()),
                                                        act_rows.data(), act_pairs.data(), act_data.data(),
                                                        K, col_grad.data());
                        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> Jc(
                            col_grad.data(), K, J);
                        Eigen::Map<Eigen::VectorXd> rc(act_res.data(), K);
                        H_out->noalias() += penalty * (Jc.transpose() * Jc);
                        g_out->noalias() += sqrt_pen * (Jc.transpose() * rc);
                    }
                }
            }
            return cost;
        };

        for (int outer = 0; outer < outer_rounds; ++outer) {
            double damping = std::max(cfg.ik_damping, 1e-8);
            for (int it = 0; it < cfg.ik_iterations; ++it) {
                double pe = 0.0, re = 0.0;
                if (!with_collision && eval_pose_err(q, pe, re)) {
                    break;
                }
                const double cost = eval_merit(q, &H, &g);
                std::vector<int> free_idx;
                free_idx.reserve(static_cast<size_t>(J));
                double max_g = 0.0;
                for (int j = 0; j < J; ++j) {
                    if (!active_mask[static_cast<size_t>(j)]) continue;
                    const bool at_bound = (q(j) <= lo(j) && g(j) > 0.0) || (q(j) >= hi(j) && g(j) < 0.0);
                    if (!at_bound) {
                        free_idx.push_back(j);
                        max_g = std::max(max_g, std::abs(g(j)));
                    }
                }
                if (free_idx.empty() || max_g < 1e-9) break;

                const int F = static_cast<int>(free_idx.size());
                Eigen::MatrixXd Hf(F, F);
                Eigen::VectorXd gf(F);
                for (int a = 0; a < F; ++a) {
                    gf(a) = g(free_idx[static_cast<size_t>(a)]);
                    for (int b = 0; b < F; ++b) {
                        Hf(a, b) = H(free_idx[static_cast<size_t>(a)], free_idx[static_cast<size_t>(b)]);
                    }
                }

                bool accepted = false;
                for (int ls = 0; ls < 8; ++ls) {
                    Eigen::MatrixXd A = Hf;
                    for (int a = 0; a < F; ++a) A(a, a) += damping * (1.0 + Hf(a, a));
                    Eigen::LDLT<Eigen::MatrixXd> ldlt(A);
                    if (ldlt.info() != Eigen::Success) {
                        damping *= 10.0;
                        continue;
                    }
                    const Eigen::VectorXd step_f = ldlt.solve(-gf);
                    if (!step_f.allFinite()) {
                        damping *= 10.0;
                        continue;
                    }
                    Eigen::VectorXd q_try = q;
                    for (int a = 0; a < F; ++a) {
                        const int j = free_idx[static_cast<size_t>(a)];
                        q_try(j) = std::clamp(q(j) + step_f(a), lo(j), hi(j));
                    }
                    const double try_cost = eval_merit(q_try, nullptr, nullptr);
                    if (try_cost < cost - 1e-12) {
                        q = std::move(q_try);
                        damping = std::max(1e-8, damping * 0.4);
                        accepted = true;
                        break;
                    }
                    damping = std::min(1e6, damping * 4.0);
                }
                if (!accepted) break;
            }
            if (with_collision && P > 0) {
                double pe = 0.0, re = 0.0;
                std::string col_err;
                if (eval_pose_err(q, pe, re) && eval_collision(q, col_err)) break;
                eval_merit(q, nullptr, nullptr, &ineq);
                for (int p = 0; p < P; ++p) {
                    dual(p) = std::max(0.0, dual(p) + penalty * ineq(p));
                }
                penalty = std::min(1e5, penalty * 5.0);
            }
        }
        unwrap_revolute_near(q, q_prev, lo, hi, is_pris, active_mask);
        return q;
    };

    auto evaluate_candidate = [&](const Eigen::VectorXd& q_cand) -> StepIKResult {
        StepIKResult r;
        r.q = q_cand;
        const bool pose_ok = eval_pose_err(q_cand, r.pos_err, r.rot_err);
        if (!pose_ok) {
            r.status = StepIKStatus::PoseFailed;
            char buf[160];
            std::snprintf(buf, sizeof(buf),
                          "Cartesian IK failed to reach pose (pos_err=%.6f m, rot_err=%.6f rad)",
                          r.pos_err, r.rot_err);
            r.reason = buf;
            return r;
        }
        const auto jc = check_joint_jump(q_prev, q_cand, is_pris,
                                         cfg.jump_threshold_revolute, cfg.jump_threshold_prismatic);
        r.jump_ratio = jc.max_ratio;
        if (!jc.ok) {
            r.status = StepIKStatus::JumpExceeded;
            char buf[180];
            std::snprintf(buf, sizeof(buf),
                          "Cartesian joint jump exceeded on joint %d (delta=%.4f > limit=%.4f)",
                          jc.worst_joint, jc.worst_delta, jc.worst_limit);
            r.reason = buf;
            return r;
        }
        std::string col_err;
        if (!eval_collision(q_cand, col_err)) {
            r.status = StepIKStatus::Collision;
            r.reason = std::move(col_err);
            return r;
        }
        r.status = StepIKStatus::Ok;
        return r;
    };

    // Fast path: check if q_prev already satisfies pose + collision.
    {
        double pe = 0.0, re = 0.0;
        std::string col_err;
        if (eval_pose_err(q_prev, pe, re) && eval_collision(q_prev, col_err)) {
            out.status = StepIKStatus::Ok;
            out.q = q_prev;
            out.pos_err = pe;
            out.rot_err = re;
            return out;
        }
    }

    // 1. Continuity-preserving damped LM (minimal joint drift from q_prev).
    const Eigen::VectorXd q_cont = run_lm(false);
    out = evaluate_candidate(q_cont);
    if (out.status == StepIKStatus::Ok) return out;

    // 2. If pose & jump succeeded but collided, retry with collision-aware damped LM (for redundant nullspace).
    if (out.status == StepIKStatus::Collision && cfg.avoid_collisions && P > 0) {
        const Eigen::VectorXd q_col = run_lm(true);
        StepIKResult col_res = evaluate_candidate(q_col);
        if (col_res.status == StepIKStatus::Ok) return col_res;
    }
    return out;
}

struct TCPDeviationCheckResult {
    bool ok = true;
    double max_pos_err = 0.0;
    double max_rot_err = 0.0;
    std::vector<int> bad_segments;
    std::string reason;
};

// Sound Lipschitz + Conservative Advancement continuous TCP deviation verification
// along the ACTUAL played cubic (order 3) or quintic (order 5) Hermite spline.
inline TCPDeviationCheckResult verify_spline_tcp_deviation(
    const WorldSnapshot& snap,
    int link_id,
    const Vec3& tcp_pos_local,
    const Mat3& tcp_rot_local,
    const TrajectoryNative& traj,
    const std::vector<CartesianPrimitive>& prims,
    const std::vector<int>& seg_prim_idx,
    const std::vector<double>& seg_tau0,
    const std::vector<double>& seg_tau1,
    double max_pos_error,
    double max_rot_error) {
    TCPDeviationCheckResult res;
    const int T = traj.num_waypoints();
    const int J = traj.num_joints();
    const int B = snap.kin.num_bodies();
    const int n_segs = T - 1;
    if (n_segs <= 0) return res;

    const bool is_quintic = (traj.spline_order == 5);
    const auto& is_pris = snap.kin.joint_prismatic();
    const uint8_t* anc = snap.kin.body_ancestor().data() + link_id * J;

    // Exact/certified max |q_j| over trajectory for downstream prismatic joints.
    std::vector<double> q_abs_max(static_cast<size_t>(J), 0.0);
    for (int t = 0; t < T; ++t) {
        for (int j = 0; j < J; ++j) {
            q_abs_max[static_cast<size_t>(j)] = std::max(q_abs_max[static_cast<size_t>(j)], std::abs(traj.positions(t, j)));
        }
    }
    for (int seg = 0; seg < n_segs; ++seg) {
        const double h = traj.times(seg + 1) - traj.times(seg);
        if (h <= 0.0) continue;
        for (int j = 0; j < J; ++j) {
            if (!is_pris[static_cast<size_t>(j)]) continue;
            const double sm = is_quintic
                ? quintic_hermite_scalar_abs_max(traj.positions(seg, j), traj.velocities(seg, j),
                                                  traj.accelerations(seg, j), traj.positions(seg + 1, j),
                                                  traj.velocities(seg + 1, j), traj.accelerations(seg + 1, j), h)
                : hermite_scalar_abs_max(traj.positions(seg, j), traj.positions(seg + 1, j),
                                         traj.velocities(seg, j), traj.velocities(seg + 1, j), h);
            q_abs_max[static_cast<size_t>(j)] = std::max(q_abs_max[static_cast<size_t>(j)], sm);
        }
    }

    // Exact rotation-invariant TCP position & orientation Lipschitz weights.
    std::vector<double> L_pos(static_cast<size_t>(J), 0.0);
    snap.kin.body_lipschitz_weights(link_id, tcp_pos_local.norm(), q_abs_max.data(), L_pos.data());
    std::vector<double> L_rot(static_cast<size_t>(J), 0.0);
    for (int j = 0; j < J; ++j) {
        if (anc[j] && !is_pris[static_cast<size_t>(j)]) L_rot[static_cast<size_t>(j)] = 1.0;
    }

    std::vector<Vec3> bpos(static_cast<size_t>(B));
    std::vector<Mat3> brot(static_cast<size_t>(B));
    std::vector<double> q_eval(static_cast<size_t>(J));

    auto eval_q = [&](int seg, double u, double* out_q) {
        const double h = traj.times(seg + 1) - traj.times(seg);
        if (h <= 0.0) {
            for (int j = 0; j < J; ++j) {
                out_q[j] = (1.0 - u) * traj.positions(seg, j) + u * traj.positions(seg + 1, j);
            }
            return;
        }
        if (is_quintic) {
            for (int j = 0; j < J; ++j) {
                out_q[j] = QuinticPoly::from_endpoints(
                    traj.positions(seg, j), traj.velocities(seg, j), traj.accelerations(seg, j),
                    traj.positions(seg + 1, j), traj.velocities(seg + 1, j), traj.accelerations(seg + 1, j),
                    h).eval_pos(u);
            }
        } else {
            const double u2 = u * u;
            const double u3 = u2 * u;
            const double h00 = 2.0 * u3 - 3.0 * u2 + 1.0;
            const double h10 = (u3 - 2.0 * u2 + u) * h;
            const double h01 = -2.0 * u3 + 3.0 * u2;
            const double h11 = (u3 - u2) * h;
            for (int j = 0; j < J; ++j) {
                out_q[j] = h00 * traj.positions(seg, j) + h10 * traj.velocities(seg, j) +
                           h01 * traj.positions(seg + 1, j) + h11 * traj.velocities(seg + 1, j);
            }
        }
    };

    auto subinterval_disp = [&](int seg, double u0, double u1, double& d_pos, double& d_rot) {
        const double h = traj.times(seg + 1) - traj.times(seg);
        d_pos = 0.0;
        d_rot = 0.0;
        for (int j = 0; j < J; ++j) {
            const double lp = L_pos[static_cast<size_t>(j)];
            const double lr = L_rot[static_cast<size_t>(j)];
            if (lp == 0.0 && lr == 0.0) continue;
            double tv = 0.0;
            if (h <= 0.0) {
                tv = std::abs(traj.positions(seg + 1, j) - traj.positions(seg, j)) * (u1 - u0);
            } else if (is_quintic) {
                tv = quintic_hermite_scalar_total_variation_bound(
                    traj.positions(seg, j), traj.velocities(seg, j), traj.accelerations(seg, j),
                    traj.positions(seg + 1, j), traj.velocities(seg + 1, j), traj.accelerations(seg + 1, j),
                    h, u0, u1);
            } else {
                tv = hermite_scalar_total_variation(
                    traj.positions(seg + 1, j) - traj.positions(seg, j),
                    traj.velocities(seg, j), traj.velocities(seg + 1, j), h, u0, u1);
            }
            d_pos += lp * tv;
            d_rot += lr * tv;
        }
    };

    struct SubInterval {
        double u0, u1;
        double phi0, phi1;
        int depth;
    };
    std::vector<SubInterval> stack;
    stack.reserve(64);

    for (int seg = 0; seg < n_segs; ++seg) {
        const auto& prim = prims[static_cast<size_t>(seg_prim_idx[static_cast<size_t>(seg)])];
        const double tau0 = seg_tau0[static_cast<size_t>(seg)];
        const double tau1 = seg_tau1[static_cast<size_t>(seg)];

        auto eval_phi = [&](double u, double& pe, double& re) -> double {
            eval_q(seg, u, q_eval.data());
            snap.kin.fk_single(q_eval.data(), bpos.data(), brot.data());
            Vec3 tp;
            Mat3 tr;
            eval_tcp_pose_and_jacobian(snap.kin, link_id, tcp_pos_local, tcp_rot_local,
                                       bpos.data(), brot.data(), nullptr, nullptr, tp, tr);
            const double phi = prim.eval_coupled_error(tp, tr, tau0, tau1, max_pos_error, max_rot_error, pe, re);
            res.max_pos_err = std::max(res.max_pos_err, pe);
            res.max_rot_err = std::max(res.max_rot_err, re);
            return phi;
        };

        // Verify exact directed segment endpoints u=0 -> tau0 and u=1 -> tau1 so a stationary
        // or reversed curve can never pass by projecting onto the opposite endpoint.
        bool seg_bad = false;
        double worst_seg_pe = 0.0, worst_seg_re = 0.0;
        {
            const double ep_u[2] = {0.0, 1.0};
            const double ep_tau[2] = {tau0, tau1};
            for (int e = 0; e < 2; ++e) {
                eval_q(seg, ep_u[e], q_eval.data());
                snap.kin.fk_single(q_eval.data(), bpos.data(), brot.data());
                Vec3 tp;
                Mat3 tr;
                eval_tcp_pose_and_jacobian(snap.kin, link_id, tcp_pos_local, tcp_rot_local,
                                           bpos.data(), brot.data(), nullptr, nullptr, tp, tr);
                const double pe = (tp - prim.eval_pos(ep_tau[e])).norm();
                const double re = prim.has_rot ? rot_geodesic_angle(prim.eval_rot(ep_tau[e]), tr) : 0.0;
                const double phi_ep = std::max(pe / max_pos_error, prim.has_rot ? (re / max_rot_error) : 0.0);
                res.max_pos_err = std::max(res.max_pos_err, pe);
                res.max_rot_err = std::max(res.max_rot_err, re);
                worst_seg_pe = std::max(worst_seg_pe, pe);
                worst_seg_re = std::max(worst_seg_re, re);
                if (!std::isfinite(phi_ep) || phi_ep > 1.0 + 1e-9) {
                    seg_bad = true;
                }
            }
        }

        // Initial 4-substep grid catches obvious mid-segment bow-out immediately.
        constexpr int init_n = 4;
        double u_pts[init_n + 1];
        double phi_pts[init_n + 1];
        for (int i = 0; i <= init_n; ++i) {
            u_pts[i] = static_cast<double>(i) / static_cast<double>(init_n);
            double pe = 0.0, re = 0.0;
            phi_pts[i] = eval_phi(u_pts[i], pe, re);
            worst_seg_pe = std::max(worst_seg_pe, pe);
            worst_seg_re = std::max(worst_seg_re, re);
            if (!std::isfinite(phi_pts[i]) || phi_pts[i] > 1.0 + 1e-9) {
                seg_bad = true;
            }
        }

        if (!seg_bad) {
            stack.clear();
            for (int i = 0; i < init_n; ++i) {
                stack.push_back({u_pts[i], u_pts[i + 1], phi_pts[i], phi_pts[i + 1], 0});
            }
            while (!stack.empty()) {
                const SubInterval cur = stack.back();
                stack.pop_back();
                double d_pos = 0.0, d_rot = 0.0;
                subinterval_disp(seg, cur.u0, cur.u1, d_pos, d_rot);
                const double d_phi = std::max(d_pos / max_pos_error,
                                              prim.has_rot ? (d_rot / max_rot_error) : 0.0);
                // Certified upper bound on coupled normalized error over [cur.u0, cur.u1].
                const double phi_ub = 0.5 * (cur.phi0 + cur.phi1 + d_phi);
                if (phi_ub <= 1.0 + 1e-9) {
                    continue;  // Mathematically certified safe across entire subinterval!
                }
                if (d_phi <= 5e-4 || cur.depth >= 18) {
                    // Tightly localized within 0.05% of tolerance: reject if conservatively above 1.0 + 5e-4.
                    if (phi_ub > 1.0 + 5e-4) {
                        seg_bad = true;
                        worst_seg_pe = std::max(worst_seg_pe, phi_ub * max_pos_error);
                        break;
                    }
                    continue;
                }
                const double um = 0.5 * (cur.u0 + cur.u1);
                double pe_m = 0.0, re_m = 0.0;
                const double phi_m = eval_phi(um, pe_m, re_m);
                worst_seg_pe = std::max(worst_seg_pe, pe_m);
                worst_seg_re = std::max(worst_seg_re, re_m);
                if (!std::isfinite(phi_m) || phi_m > 1.0 + 1e-9) {
                    seg_bad = true;
                    break;
                }
                stack.push_back({cur.u0, um, cur.phi0, phi_m, cur.depth + 1});
                stack.push_back({um, cur.u1, phi_m, cur.phi1, cur.depth + 1});
            }
        }

        if (seg_bad) {
            res.ok = false;
            res.bad_segments.push_back(seg);
            if (res.reason.empty()) {
                char buf[200];
                std::snprintf(buf, sizeof(buf),
                              "retimed spline TCP deviation exceeded on segment %d (pos_err=%.6f m, rot_err=%.6f rad)",
                              seg, worst_seg_pe, worst_seg_re);
                res.reason = buf;
            }
        }
    }
    return res;
}

struct CartesianKnot {
    Eigen::VectorXd q;
    int prim_idx = 0;
    double tau = 0.0;
    double progress = 0.0;
    bool is_stop = false;
    int depth = 0;
    int completed_waypoints = 0;
};

struct PrefixVerifyResult {
    bool ok = false;
    TrajectoryNative trajectory;
    double max_pos_err = 0.0;
    double max_rot_err = 0.0;
    double retime_ms = 0.0;
    double validate_ms = 0.0;
    int earliest_bad_seg = -1;
    std::string reason;
};

// Retime and fully verify a Cartesian knot sequence against:
// 1. Continuous joint pos / vel / acc / jerk bounds
// 2. Discrete convex & original mesh collision + continuous CCD (strict_ccd pass-through)
// 3. Certified continuous TCP position & orientation deviation along the actual played spline
// Recovers via knot-velocity damping (keeping sharp-corner stop knots locked at 0)
// and mid-segment IK knot refinement when nonlinear FK curvature bows off the Cartesian path.
inline PrefixVerifyResult finalize_and_verify_cartesian_prefix(
    const WorldSnapshot& snap,
    int link_id,
    const Vec3& tcp_pos_local,
    const Mat3& tcp_rot_local,
    const std::vector<uint8_t>& active_mask,
    const CartesianPlanConfigNative& cfg,
    const Eigen::VectorXd& margins,
    const std::vector<CartesianPrimitive>& prims,
    std::vector<CartesianKnot> knots) {
    using Clock = std::chrono::steady_clock;
    PrefixVerifyResult out;
    const int J = snap.kin.num_joints();
    const int P = static_cast<int>(snap.pairs.size());
    const int N_shapes = static_cast<int>(snap.shapes.size());
    const Eigen::VectorXd vmax = snap.kin.velocity_limit() * cfg.velocity_scale;
    const Eigen::VectorXd amax = snap.kin.acceleration_limit() * cfg.acceleration_scale;
    const Eigen::VectorXd jmax = snap.kin.jerk_limit() * cfg.jerk_scale;
    const Eigen::VectorXd& lo = snap.kin.lower();
    const Eigen::VectorXd& hi = snap.kin.upper();
    const int substeps = std::max(2 * cfg.collision_substeps, 4);

    if (knots.size() < 2) {
        out.reason = "prefix has fewer than 2 knots";
        return out;
    }

    const double ik_pos_tol = 0.25 * cfg.max_pos_error;
    const double ik_rot_tol = 0.25 * cfg.max_rot_error;

    std::vector<double> samples, frames;
    std::vector<int64_t> screen_rows, screen_pairs, viol_rows, viol_pairs;

    for (int refine_round = 0; refine_round <= cfg.max_subdivision_depth + 2; ++refine_round) {
        const int K = static_cast<int>(knots.size());
        RowMatrixXd positions(K, J);
        Eigen::VectorXd knot_scales(K);
        std::vector<int> seg_prim_idx(static_cast<size_t>(K - 1));
        std::vector<double> seg_tau0(static_cast<size_t>(K - 1));
        std::vector<double> seg_tau1(static_cast<size_t>(K - 1));
        for (int k = 0; k < K; ++k) {
            positions.row(k) = knots[static_cast<size_t>(k)].q.transpose();
            knot_scales(k) = knots[static_cast<size_t>(k)].is_stop ? 0.0 : 1.0;
        }
        if (cfg.stop_at_endpoints) {
            knot_scales(0) = 0.0;
            knot_scales(K - 1) = 0.0;
        }
        for (int k = 0; k < K - 1; ++k) {
            const int pid = knots[static_cast<size_t>(k + 1)].prim_idx;
            seg_prim_idx[static_cast<size_t>(k)] = pid;
            seg_tau0[static_cast<size_t>(k)] = (knots[static_cast<size_t>(k)].prim_idx == pid)
                ? knots[static_cast<size_t>(k)].tau : 0.0;
            seg_tau1[static_cast<size_t>(k)] = knots[static_cast<size_t>(k + 1)].tau;
        }

        std::set<int> tcp_only_bad_segs;
        std::string last_error;
        int earliest_bad = K - 1;

        for (int attempt = 0; attempt < 8; ++attempt) {
            const auto t_ret0 = Clock::now();
            TrajectoryNative traj = retime_trajectory(
                positions, vmax, amax, cfg.stop_at_endpoints, false, 0.0,
                &knot_scales, &jmax, cfg.spline_order);
            out.retime_ms += std::chrono::duration<double, std::milli>(Clock::now() - t_ret0).count();

            const auto t_val0 = Clock::now();
            std::set<int> bad_segs;
            tcp_only_bad_segs.clear();
            last_error.clear();

            // 1. Continuous joint pos / vel / acc / jerk bounds.
            const auto bchk = check_trajectory_bounds(
                traj, vmax, amax, true, true, 1e-5, &lo, &hi, &jmax, cfg.spline_order == 5);
            if (!bchk.ok) {
                last_error = bchk.reason;
                for (const int s : bchk.bad_segments) bad_segs.insert(s);
            }

            // 2. Discrete + continuous CCD collision verification.
            if (bad_segs.empty() && cfg.avoid_collisions && P > 0) {
                traj.sample_segments(substeps, samples);
                const int64_t S = static_cast<int64_t>((K - 1) * substeps + 1);
                frames.resize(static_cast<size_t>(S * N_shapes * 12));
                snap.compute_frames(samples.data(), S, frames.data());

                const int n_passes = (cfg.verify_original && snap.has_mesh) ? 2 : 1;
                for (int pass = 0; pass < n_passes; ++pass) {
                    const bool verify_orig = (pass == 1);
                    const char* kind = verify_orig ? "mesh" : "convex";
                    snap.screen_aabbs(frames.data(), S, margins.data(), verify_orig, substeps, screen_rows, screen_pairs);
                    if (!screen_rows.empty()) {
                        snap.violating_pairs_indexed(frames.data(), S, screen_rows.data(), screen_pairs.data(),
                                                     static_cast<int64_t>(screen_rows.size()), margins.data(),
                                                     verify_orig, viol_rows, viol_pairs);
                        if (!viol_rows.empty()) {
                            const int seg0 = std::min(static_cast<int>(viol_rows[0] / substeps), K - 2);
                            last_error = std::string("Cartesian spline ") + kind + " clearance failed on segment " +
                                         std::to_string(seg0) + ", pair " + snap.pair_display_name(viol_pairs[0]);
                            for (const int64_t r : viol_rows) {
                                bad_segs.insert(std::min(static_cast<int>(r / substeps), K - 2));
                            }
                            break;
                        }
                    }
                    if (!verify_orig) {
                        auto ccd_hit = snap.first_violation_continuous(
                            traj.positions.data(), K, substeps, frames.data(), margins.data(),
                            false, 1e-3, traj.times.data(), traj.velocities.data(),
                            cfg.strict_ccd, traj.accelerations.data(), traj.spline_order);
                        if (ccd_hit.has_value()) {
                            last_error = std::string("Cartesian spline continuous ") + kind +
                                         " clearance failed on segment " + std::to_string(ccd_hit->seg) +
                                         ", pair " + snap.pair_display_name(ccd_hit->pair_id);
                            bad_segs.insert(ccd_hit->seg);
                            break;
                        }
                    }
                }
            }

            // 3. Certified continuous TCP position & orientation deviation on the actual played spline.
            TCPDeviationCheckResult tcp_chk;
            if (bad_segs.empty()) {
                tcp_chk = verify_spline_tcp_deviation(
                    snap, link_id, tcp_pos_local, tcp_rot_local, traj,
                    prims, seg_prim_idx, seg_tau0, seg_tau1,
                    cfg.max_pos_error, cfg.max_rot_error);
                if (!tcp_chk.ok) {
                    last_error = tcp_chk.reason;
                    for (const int s : tcp_chk.bad_segments) {
                        bad_segs.insert(s);
                        tcp_only_bad_segs.insert(s);
                    }
                }
            }
            out.validate_ms += std::chrono::duration<double, std::milli>(Clock::now() - t_val0).count();

            if (bad_segs.empty()) {
                out.ok = true;
                out.trajectory = std::move(traj);
                out.max_pos_err = tcp_chk.max_pos_err;
                out.max_rot_err = tcp_chk.max_rot_err;
                return out;
            }

            earliest_bad = *bad_segs.begin();
            bool damped = false;
            for (const int seg : bad_segs) {
                for (int knot : {seg, seg + 1}) {
                    if (knot_scales(knot) > 0.0) {
                        knot_scales(knot) = (attempt >= 2 || knot_scales(knot) <= 0.25) ? 0.0 : (knot_scales(knot) * 0.5);
                        damped = true;
                    }
                }
            }
            if (!damped) {
                break;
            }
        }

        // If damping exhausted and the failure is TCP bow-out on segments that can still be subdivided,
        // insert intermediate IK knots on offending segments to tighten the geometric curve.
        if (!tcp_only_bad_segs.empty() && knots.size() < 4096) {
            std::vector<CartesianKnot> refined;
            refined.reserve(knots.size() + tcp_only_bad_segs.size());
            bool inserted_any = false;
            for (int k = 0; k < K - 1; ++k) {
                refined.push_back(knots[static_cast<size_t>(k)]);
                if (tcp_only_bad_segs.count(k) > 0) {
                    const int seg_depth = std::max(knots[static_cast<size_t>(k)].depth,
                                                   knots[static_cast<size_t>(k + 1)].depth);
                    if (seg_depth < cfg.max_subdivision_depth + 2) {
                        const int pid = seg_prim_idx[static_cast<size_t>(k)];
                        const auto& prim = prims[static_cast<size_t>(pid)];
                        const double tm = 0.5 * (seg_tau0[static_cast<size_t>(k)] + seg_tau1[static_cast<size_t>(k)]);
                        const Eigen::VectorXd q_seed = 0.5 * (knots[static_cast<size_t>(k)].q +
                                                              knots[static_cast<size_t>(k + 1)].q);
                        const auto ik_m = solve_cartesian_step_ik(
                            snap, link_id, tcp_pos_local, tcp_rot_local,
                            prim.eval_pos(tm), prim.eval_rot(tm), prim.has_rot,
                            q_seed, active_mask, cfg, margins, ik_pos_tol, ik_rot_tol);
                        if (ik_m.status == StepIKStatus::Ok) {
                            const auto& is_pris = snap.kin.joint_prismatic();
                            const auto j1 = check_joint_jump(knots[static_cast<size_t>(k)].q, ik_m.q, is_pris,
                                                             cfg.jump_threshold_revolute, cfg.jump_threshold_prismatic);
                            const auto j2 = check_joint_jump(ik_m.q, knots[static_cast<size_t>(k + 1)].q, is_pris,
                                                             cfg.jump_threshold_revolute, cfg.jump_threshold_prismatic);
                            if (j1.ok && j2.ok) {
                                CartesianKnot mid;
                                mid.q = ik_m.q;
                                mid.prim_idx = pid;
                                mid.tau = tm;
                                mid.progress = prim.progress_start + tm * (prim.progress_end - prim.progress_start);
                                mid.is_stop = false;
                                mid.depth = seg_depth + 1;
                                mid.completed_waypoints = knots[static_cast<size_t>(k)].completed_waypoints;
                                refined.push_back(std::move(mid));
                                inserted_any = true;
                            }
                        }
                    }
                }
            }
            refined.push_back(knots.back());
            if (inserted_any) {
                knots = std::move(refined);
                continue;
            }
        }

        out.earliest_bad_seg = earliest_bad;
        out.reason = last_error.empty() ? "Cartesian trajectory verification failed" : last_error;
        return out;
    }

    out.earliest_bad_seg = 0;
    out.reason = "Cartesian trajectory refinement limit reached";
    return out;
}

}  // namespace cartesian_detail

// Plan a verified Cartesian TCP path on an immutable WorldSnapshot.
inline CartesianPlanResultNative plan_cartesian_path_on_snapshot(
    const WorldSnapshot& snap,
    const CartesianPlanRequestNative& req) {
    using Clock = std::chrono::steady_clock;
    const auto t_start = Clock::now();
    CartesianPlanResultNative res;
    res.world_version = snap.version;

    const auto& cfg = req.config;
    const int J = snap.kin.num_joints();
    const int B = snap.kin.num_bodies();
    if (req.link_id < 0 || req.link_id >= B) {
        throw std::invalid_argument("invalid link_id for Cartesian path planning");
    }
    if (req.start_q.size() != J || !req.start_q.allFinite()) {
        throw std::invalid_argument("start_q must be finite with length num_joints");
    }
    if (req.waypoints.empty()) {
        throw std::invalid_argument("provide at least one Cartesian waypoint");
    }
    if (!req.tcp_offset_position.allFinite()) {
        throw std::invalid_argument("tcp_offset_position must be finite");
    }
    const Mat3 tcp_rot_local = cartesian_detail::validate_and_orthonormalize_rot(
        req.tcp_offset_rotation, "tcp_offset_rotation");

    if (!std::isfinite(cfg.max_step) || cfg.max_step <= 0.0)
        throw std::invalid_argument("max_step must be finite and positive");
    if (!std::isfinite(cfg.max_rot_step) || cfg.max_rot_step <= 0.0)
        throw std::invalid_argument("max_rot_step must be finite and positive");
    if (!std::isfinite(cfg.max_pos_error) || cfg.max_pos_error <= 0.0)
        throw std::invalid_argument("max_pos_error must be finite and positive");
    if (!std::isfinite(cfg.max_rot_error) || cfg.max_rot_error <= 0.0)
        throw std::invalid_argument("max_rot_error must be finite and positive");
    if (std::isnan(cfg.jump_threshold_revolute) || cfg.jump_threshold_revolute <= 0.0)
        throw std::invalid_argument("jump_threshold_revolute must be positive (or +inf)");
    if (std::isnan(cfg.jump_threshold_prismatic) || cfg.jump_threshold_prismatic <= 0.0)
        throw std::invalid_argument("jump_threshold_prismatic must be positive (or +inf)");
    if (cfg.max_subdivision_depth < 0)
        throw std::invalid_argument("max_subdivision_depth must be >= 0");
    if (cfg.ik_iterations < 1)
        throw std::invalid_argument("ik_iterations must be >= 1");
    if (!std::isfinite(cfg.ik_damping) || cfg.ik_damping < 0.0)
        throw std::invalid_argument("ik_damping must be finite and nonnegative");
    if (!std::isfinite(cfg.margin) || cfg.margin < 0.0)
        throw std::invalid_argument("margin must be finite and nonnegative");
    if (cfg.has_world_margin && (!std::isfinite(cfg.world_margin) || cfg.world_margin < 0.0))
        throw std::invalid_argument("world_margin must be finite and nonnegative");
    if (cfg.collision_substeps < 1)
        throw std::invalid_argument("collision_substeps must be >= 1");
    if (cfg.spline_order != 3 && cfg.spline_order != 5)
        throw std::invalid_argument("spline_order must be 3 (cubic) or 5 (quintic)");
    if (!std::isfinite(cfg.velocity_scale) || cfg.velocity_scale <= 0.0)
        throw std::invalid_argument("velocity_scale must be finite and positive");
    if (!std::isfinite(cfg.acceleration_scale) || cfg.acceleration_scale <= 0.0)
        throw std::invalid_argument("acceleration_scale must be finite and positive");
    if (!std::isfinite(cfg.jerk_scale) || cfg.jerk_scale <= 0.0)
        throw std::invalid_argument("jerk_scale must be finite and positive");

    std::vector<uint8_t> mask(static_cast<size_t>(J), 1);
    if (!req.active_mask.empty()) {
        if (static_cast<int>(req.active_mask.size()) != J) {
            throw std::invalid_argument("active_mask length must match num_joints");
        }
        mask = req.active_mask;
    }

    // 1. Validate start configuration (strict joint limits matching WorldSnapshot::is_config_valid).
    const Eigen::VectorXd& lo = snap.kin.lower();
    const Eigen::VectorXd& hi = snap.kin.upper();
    for (int j = 0; j < J; ++j) {
        if (req.start_q(j) < lo(j) || req.start_q(j) > hi(j)) {
            res.reason = "start configuration violates joint limits";
            res.total_ms = std::chrono::duration<double, std::milli>(Clock::now() - t_start).count();
            return res;
        }
    }
    const Eigen::VectorXd margins = snap.pairs.empty()
        ? Eigen::VectorXd()
        : snap.pair_margins(cfg.margin, cfg.effective_world_margin());

    // Compute initial TCP pose at start_q.
    std::vector<Vec3> bpos(static_cast<size_t>(B));
    std::vector<Mat3> brot(static_cast<size_t>(B));
    snap.kin.fk_single(req.start_q.data(), bpos.data(), brot.data());
    Vec3 start_tcp_pos;
    Mat3 start_tcp_rot;
    cartesian_detail::eval_tcp_pose_and_jacobian(
        snap.kin, req.link_id, req.tcp_offset_position, tcp_rot_local,
        bpos.data(), brot.data(), nullptr, nullptr, start_tcp_pos, start_tcp_rot);

    // 2. Validate ALL waypoints and construct reference primitives (including circular arcs & blends).
    // Note: validate geometry BEFORE returning on start collision so invalid waypoints always raise.
    const int M = static_cast<int>(req.waypoints.size());
    std::vector<cartesian_detail::CartesianPrimitive> raw_segs;
    raw_segs.reserve(static_cast<size_t>(M));
    Vec3 prev_pos = start_tcp_pos;
    Mat3 prev_rot = start_tcp_rot;
    bool prev_has_rot = true;  // start_tcp_rot is always known!

    for (int i = 0; i < M; ++i) {
        const auto& wp = req.waypoints[static_cast<size_t>(i)];
        if (!wp.position.allFinite()) {
            throw std::invalid_argument("Cartesian waypoint position must be finite");
        }
        if (!std::isfinite(wp.blend_radius) || wp.blend_radius < 0.0) {
            throw std::invalid_argument("Cartesian waypoint blend_radius must be finite and nonnegative");
        }
        cartesian_detail::CartesianPrimitive seg;
        if (wp.motion_type == CartesianMotionType::Circular) {
            if (!wp.has_via_position) {
                throw std::invalid_argument("circular Cartesian waypoint requires via_position");
            }
            seg = cartesian_detail::make_circular_arc_primitive(prev_pos, wp.via_position, wp.position);
        } else if (wp.motion_type == CartesianMotionType::Linear) {
            seg.kind = cartesian_detail::CartesianPrimitive::Kind::Linear;
            seg.p0 = prev_pos;
            seg.p1 = wp.position;
            seg.arc_length = (wp.position - prev_pos).norm();
        } else {
            throw std::invalid_argument("unknown Cartesian motion_type");
        }

        const bool constrain_rot = wp.has_rotation || cfg.hold_orientation;
        seg.has_rot = constrain_rot;
        if (constrain_rot) {
            const Mat3 target_r = wp.has_rotation
                ? cartesian_detail::validate_and_orthonormalize_rot(wp.rotation, "waypoint rotation")
                : prev_rot;
            seg.dynamic_r0 = !prev_has_rot;
            seg.set_rotations(prev_rot, target_r);
            prev_rot = target_r;
            prev_has_rot = true;
        } else {
            seg.r0 = prev_rot;
            seg.r1 = prev_rot;
            prev_has_rot = false;
        }
        seg.orig_waypoint_idx = i;
        raw_segs.push_back(std::move(seg));
        prev_pos = wp.position;
    }

    // Now check start collision (after input validation).
    if (cfg.avoid_collisions && !snap.pairs.empty() &&
        !snap.is_config_valid(req.start_q.data(), margins, cfg.verify_original)) {
        res.reason = "start configuration is in collision";
        res.total_ms = std::chrono::duration<double, std::milli>(Clock::now() - t_start).count();
        return res;
    }

    // 3. Apply explicit corner handling: sharp corner stop (blend_radius == 0) vs G^1 cubic Bezier blend.
    std::vector<double> trim_start(static_cast<size_t>(M), 0.0);
    std::vector<double> trim_end(static_cast<size_t>(M), 0.0);
    for (int i = 0; i + 1 < M; ++i) {
        const double r_req = req.waypoints[static_cast<size_t>(i)].blend_radius;
        if (r_req <= 1e-9) continue;
        const auto& seg_in = raw_segs[static_cast<size_t>(i)];
        const auto& seg_out = raw_segs[static_cast<size_t>(i + 1)];
        const double l_in = seg_in.arc_length;
        const double l_out = seg_out.arc_length;
        if (l_in <= 1e-9 || l_out <= 1e-9) continue;
        // Clamp trims so adjacent blends never overlap and circular arc trims never consume via_position.
        double max_in = 0.49 * l_in;
        if (seg_in.has_via_tau) {
            max_in = std::min(max_in, 0.49 * (1.0 - seg_in.arc_via_tau) * l_in);
        }
        double max_out = 0.49 * l_out;
        if (seg_out.has_via_tau) {
            max_out = std::min(max_out, 0.49 * seg_out.arc_via_tau * l_out);
        }
        const double d = std::min({r_req, max_in, max_out});
        if (d > 1e-9) {
            trim_end[static_cast<size_t>(i)] = d;
            trim_start[static_cast<size_t>(i + 1)] = d;
        }
    }

    std::vector<cartesian_detail::CartesianPrimitive> prims;
    prims.reserve(static_cast<size_t>(2 * M));
    std::vector<int> wp_last_prim(static_cast<size_t>(M), 0);

    auto make_trimmed = [](const cartesian_detail::CartesianPrimitive& src, double t_lo, double t_hi) {
        cartesian_detail::CartesianPrimitive p = src;
        if (src.kind == cartesian_detail::CartesianPrimitive::Kind::Linear) {
            p.p0 = src.eval_pos(t_lo);
            p.p1 = src.eval_pos(t_hi);
            p.arc_length = (p.p1 - p.p0).norm();
        } else if (src.kind == cartesian_detail::CartesianPrimitive::Kind::CircularArc) {
            p.arc_angle0 = src.arc_angle0 + t_lo * src.arc_angle_span;
            p.arc_angle_span = (t_hi - t_lo) * src.arc_angle_span;
            p.p0 = p.eval_pos(0.0);
            p.p1 = p.eval_pos(1.0);
            p.arc_length = p.arc_radius * p.arc_angle_span;
            if (src.has_via_tau && t_hi > t_lo + 1e-12) {
                const double vt = (src.arc_via_tau - t_lo) / (t_hi - t_lo);
                p.has_via_tau = (vt > 1e-6 && vt < 1.0 - 1e-6);
                p.arc_via_tau = p.has_via_tau ? vt : 0.5;
            } else {
                p.has_via_tau = false;
            }
        }
        if (src.has_rot) {
            if (src.dynamic_r0) {
                // Start rotation is only known dynamically upon entering this segment (after an unconstrained segment).
                p.r0 = src.r0;
                p.r1 = src.r1;
                p.dynamic_r0 = true;
                p.dynamic_r1_frac = (1.0 - t_lo > 1e-12) ? ((t_hi - t_lo) / (1.0 - t_lo)) : 1.0;
            } else {
                const Mat3 r_lo = src.eval_rot(t_lo);
                const Mat3 r_hi = src.eval_rot(t_hi);
                p.set_rotations(r_lo, r_hi);
                p.dynamic_r0 = false;
                p.dynamic_r1_frac = 1.0;
            }
        }
        return p;
    };

    for (int i = 0; i < M; ++i) {
        const auto& src = raw_segs[static_cast<size_t>(i)];
        const double l_src = src.arc_length;
        const double t_lo = (l_src > 1e-12) ? (trim_start[static_cast<size_t>(i)] / l_src) : 0.0;
        const double t_hi = (l_src > 1e-12) ? (1.0 - trim_end[static_cast<size_t>(i)] / l_src) : 1.0;
        auto main_p = make_trimmed(src, t_lo, t_hi);
        const bool has_blend = (i + 1 < M) && (trim_end[static_cast<size_t>(i)] > 1e-9);
        // Sharp corner: stop at end of main_p. Last waypoint: honor stop_at_endpoints.
        main_p.stop_at_end = has_blend ? false : ((i + 1 < M) ? true : cfg.stop_at_endpoints);
        prims.push_back(std::move(main_p));

        if (has_blend) {
            const double d = trim_end[static_cast<size_t>(i)];
            const auto& next_src = raw_segs[static_cast<size_t>(i + 1)];
            const double next_t_lo = trim_start[static_cast<size_t>(i + 1)] / next_src.arc_length;
            const Vec3 A = src.eval_pos(t_hi);
            const Vec3 t_in = src.eval_tangent(t_hi);
            const Vec3 B_pt = next_src.eval_pos(next_t_lo);
            const Vec3 t_out = next_src.eval_tangent(next_t_lo);

            cartesian_detail::CartesianPrimitive bl;
            bl.kind = cartesian_detail::CartesianPrimitive::Kind::BlendBezier;
            bl.bez[0] = A;
            bl.bez[1] = A + (2.0 / 3.0) * d * t_in;
            bl.bez[2] = B_pt - (2.0 / 3.0) * d * t_out;
            bl.bez[3] = B_pt;
            bl.p0 = A;
            bl.p1 = B_pt;
            bl.arc_length = bl.compute_bezier_length();
            // Orientation transition across blend:
            // - If both incoming and outgoing constrain rotation, blend smoothly from src(t_hi) to next_src(next_t_lo).
            // - If incoming constrains rotation and outgoing releases orientation to free, finish the specified
            //   rotation all the way to src.r1 by the blend exit B_pt before releasing orientation on next_src!
            bl.has_rot = src.has_rot;
            if (bl.has_rot) {
                const Mat3 r_target = next_src.has_rot ? next_src.eval_rot(next_t_lo) : src.r1;
                bl.set_rotations(src.eval_rot(t_hi), r_target);
                bl.dynamic_r0 = src.dynamic_r0;
                bl.dynamic_r1_frac = 1.0;
            }
            bl.stop_at_end = false;
            bl.orig_waypoint_idx = i;
            bl.is_blend = true;
            prims.push_back(std::move(bl));
        }
        wp_last_prim[static_cast<size_t>(i)] = static_cast<int>(prims.size()) - 1;
    }

    // Assign monotone progress weights across primitives, reserving positive provisional weight
    // for unresolved dynamic pure-rotation segments until their actual start rotation is known.
    auto recompute_progress = [&]() -> double {
        double acc = 0.0;
        for (auto& pr : prims) {
            double w = 0.0;
            if (pr.arc_length > 1e-9) {
                w = pr.arc_length;
            } else if (pr.has_rot) {
                w = pr.dynamic_r0 ? std::max(0.1 * pr.rot_angle, 0.1 * M_PI_2) : (0.1 * pr.rot_angle);
            }
            pr.progress_start = acc;
            acc += w;
            pr.progress_end = acc;
        }
        return acc;
    };
    double total_progress = recompute_progress();

    // Degenerate zero-motion path (all waypoints identical to start pose): return verified stationary trajectory.
    if (total_progress <= 1e-12) {
        RowMatrixXd pos(2, J);
        pos.row(0) = req.start_q.transpose();
        pos.row(1) = req.start_q.transpose();
        const Eigen::VectorXd vmax = snap.kin.velocity_limit() * cfg.velocity_scale;
        const Eigen::VectorXd amax = snap.kin.acceleration_limit() * cfg.acceleration_scale;
        const Eigen::VectorXd jmax = snap.kin.jerk_limit() * cfg.jerk_scale;
        res.trajectory = retime_trajectory(pos, vmax, amax, cfg.stop_at_endpoints, false, 0.0,
                                           nullptr, &jmax, cfg.spline_order);
        res.has_trajectory = true;
        res.success = true;
        res.fraction = 1.0;
        res.completed_waypoints = M;
        res.max_pos_error = 0.0;
        res.max_rot_error = 0.0;
        res.total_ms = std::chrono::duration<double, std::milli>(Clock::now() - t_start).count();
        return res;
    }

    // 4. Sequential continuity-preserving damped IK with adaptive subdivision.
    const auto t_ik0 = Clock::now();
    const double ik_pos_tol = 0.25 * cfg.max_pos_error;
    const double ik_rot_tol = 0.25 * cfg.max_rot_error;

    std::vector<cartesian_detail::CartesianKnot> knots;
    knots.reserve(128);
    {
        cartesian_detail::CartesianKnot k0;
        k0.q = req.start_q;
        k0.prim_idx = 0;
        k0.tau = 0.0;
        k0.progress = 0.0;
        k0.is_stop = cfg.stop_at_endpoints;
        knots.push_back(std::move(k0));
    }

    std::string ik_stop_reason;
    bool ik_stopped = false;
    bool has_ik_fail_interval = false;
    bool ik_fail_is_collision = false;
    int ik_fail_prim = -1;
    double ik_fail_ta = 0.0;
    double ik_fail_tb = 0.0;
    std::vector<Vec3> mid_bpos(static_cast<size_t>(B));
    std::vector<Mat3> mid_brot(static_cast<size_t>(B));

    auto check_chord_mid = [&](const cartesian_detail::CartesianPrimitive& prim,
                               double tau_a, double tau_b,
                               const Eigen::VectorXd& qa, const Eigen::VectorXd& qb,
                               std::string& mid_err,
                               bool* out_is_collision = nullptr) -> bool {
        if (out_is_collision) *out_is_collision = false;
        const Eigen::VectorXd qm = 0.5 * (qa + qb);
        const auto& l = snap.kin.lower();
        const auto& u = snap.kin.upper();
        for (int j = 0; j < J; ++j) {
            if (qm(j) < l(j) - 1e-9 || qm(j) > u(j) + 1e-9) {
                mid_err = "Cartesian chord midpoint violates joint limits";
                return false;
            }
        }
        snap.kin.fk_single(qm.data(), mid_bpos.data(), mid_brot.data());
        Vec3 mp;
        Mat3 mr;
        cartesian_detail::eval_tcp_pose_and_jacobian(
            snap.kin, req.link_id, req.tcp_offset_position, tcp_rot_local,
            mid_bpos.data(), mid_brot.data(), nullptr, nullptr, mp, mr);
        double pe = 0.0, re = 0.0;
        const double phi = prim.eval_coupled_error(mp, mr, tau_a, tau_b,
                                                   cfg.max_pos_error, cfg.max_rot_error, pe, re);
        if (phi > 0.75) {
            mid_err = "Cartesian chord midpoint TCP deviation exceeded";
            return false;
        }
        if (cfg.avoid_collisions && !snap.pairs.empty() &&
            !snap.is_config_valid(qm.data(), margins, cfg.verify_original)) {
            mid_err = "Cartesian chord midpoint collision";
            if (out_is_collision) *out_is_collision = true;
            return false;
        }
        return true;
    };

    for (int p_idx = 0; p_idx < static_cast<int>(prims.size()); ++p_idx) {
        auto& prim = prims[static_cast<size_t>(p_idx)];
        if (prim.has_rot && prim.dynamic_r0) {
            snap.kin.fk_single(knots.back().q.data(), mid_bpos.data(), mid_brot.data());
            Vec3 cur_p;
            Mat3 cur_r;
            cartesian_detail::eval_tcp_pose_and_jacobian(
                snap.kin, req.link_id, req.tcp_offset_position, tcp_rot_local,
                mid_bpos.data(), mid_brot.data(), nullptr, nullptr, cur_p, cur_r);
            const Vec3 full_phi = rotation_log(cur_r.transpose() * prim.r1);
            const Mat3 target_r1 = cur_r * cartesian_detail::rodrigues_exp(prim.dynamic_r1_frac * full_phi);
            prim.set_rotations(cur_r, target_r1);
            prim.dynamic_r0 = false;
            // Recompute progress weights for this and remaining primitives now that true rot_angle is known.
            total_progress = recompute_progress();
        }
        if (prim.arc_length <= 1e-12 && (!prim.has_rot || prim.rot_angle <= 1e-12)) {
            // Zero-length primitive: propagate stop flag & completed waypoint count.
            if (prim.stop_at_end) knots.back().is_stop = true;
            if (p_idx == wp_last_prim[static_cast<size_t>(prim.orig_waypoint_idx)]) {
                knots.back().completed_waypoints = std::max(knots.back().completed_waypoints, prim.orig_waypoint_idx + 1);
            }
            continue;
        }

        // Build ordered nominal parameter breakpoints in [0, 1].
        // - Circular arcs: include arc_via_tau as mandatory milestone, and subdivide by angle (<= pi/4) & sagitta.
        // - Bezier blends: split at tau=0.5 (apex/reversal turning point) via de Casteljau, then recursively
        //   subdivide sub-curves using exact control-polygon arc-length upper bound (<= max_step) and
        //   convex-hull control-point-to-chord deviation upper bound (<= 0.5 * max_pos_error).
        std::vector<double> tau_breaks;
        auto point_to_chord_dist = [](const Vec3& p, const Vec3& a, const Vec3& b) -> double {
            const Vec3 d = b - a;
            const double d2 = d.squaredNorm();
            if (d2 <= 1e-24) return (p - a).norm();
            const double u = std::clamp((p - a).dot(d) / d2, 0.0, 1.0);
            return (p - (a + u * d)).norm();
        };
        auto split_cubic_bezier = [](const Vec3 c[4], Vec3 left[4], Vec3 right[4]) {
            const Vec3 q0 = 0.5 * (c[0] + c[1]);
            const Vec3 q1 = 0.5 * (c[1] + c[2]);
            const Vec3 q2 = 0.5 * (c[2] + c[3]);
            const Vec3 r0 = 0.5 * (q0 + q1);
            const Vec3 r1 = 0.5 * (q1 + q2);
            const Vec3 m  = 0.5 * (r0 + r1);
            left[0]  = c[0]; left[1]  = q0; left[2]  = r0; left[3]  = m;
            right[0] = m;    right[1] = r1; right[2] = q2; right[3] = c[3];
        };
        std::function<void(const Vec3[4], double, double, int)> append_bezier_de_casteljau =
            [&](const Vec3 c[4], double t0, double t1, int depth) {
                const double poly_len = (c[1] - c[0]).norm() + (c[2] - c[1]).norm() + (c[3] - c[2]).norm();
                const double chord_tol = 0.5 * cfg.max_pos_error;
                const double max_dev = std::max(point_to_chord_dist(c[1], c[0], c[3]),
                                                point_to_chord_dist(c[2], c[0], c[3]));
                const double span = t1 - t0;
                const bool need_len = (poly_len > cfg.max_step);
                const bool need_rot = (prim.has_rot && prim.rot_angle * span > cfg.max_rot_step);
                if ((max_dev > chord_tol || need_len || need_rot) && depth < 12) {
                    Vec3 left[4], right[4];
                    split_cubic_bezier(c, left, right);
                    const double tm = 0.5 * (t0 + t1);
                    append_bezier_de_casteljau(left, t0, tm, depth + 1);
                    append_bezier_de_casteljau(right, tm, t1, depth + 1);
                    return;
                }
                tau_breaks.push_back(t1);
            };
        auto append_subspan = [&](double t0, double t1) {
            const double span = t1 - t0;
            if (span <= 1e-12) return;
            const double sub_len = prim.arc_length * span;
            const double sub_rot = prim.has_rot ? (prim.rot_angle * span) : 0.0;
            int n_sub = 1;
            if (sub_len > 1e-12) {
                n_sub = std::max(n_sub, static_cast<int>(std::ceil(sub_len / cfg.max_step)));
            }
            if (sub_rot > 1e-12) {
                n_sub = std::max(n_sub, static_cast<int>(std::ceil(sub_rot / cfg.max_rot_step)));
            }
            if (prim.kind == cartesian_detail::CartesianPrimitive::Kind::CircularArc) {
                const double sub_ang = prim.arc_angle_span * span;
                n_sub = std::max(n_sub, static_cast<int>(std::ceil(sub_ang / (M_PI / 4.0))));
                if (prim.arc_radius > 0.5 * cfg.max_pos_error) {
                    const double cos_half = std::clamp(1.0 - (0.5 * cfg.max_pos_error) / prim.arc_radius, -1.0, 1.0);
                    const double max_dtheta = std::max(2.0 * std::acos(cos_half), 1e-3);
                    const int n_sag = std::min(512, static_cast<int>(std::ceil(sub_ang / max_dtheta)));
                    n_sub = std::max(n_sub, n_sag);
                }
            }
            for (int k = 1; k <= n_sub; ++k) {
                tau_breaks.push_back(t0 + span * (static_cast<double>(k) / static_cast<double>(n_sub)));
            }
        };
        tau_breaks.push_back(0.0);
        if (prim.kind == cartesian_detail::CartesianPrimitive::Kind::CircularArc &&
            prim.has_via_tau && prim.arc_via_tau > 1e-6 && prim.arc_via_tau < 1.0 - 1e-6) {
            append_subspan(0.0, prim.arc_via_tau);
            append_subspan(prim.arc_via_tau, 1.0);
        } else if (prim.kind == cartesian_detail::CartesianPrimitive::Kind::BlendBezier) {
            // Mandatory midpoint apex milestone via de Casteljau prevents skipping hairpin/reversal blends
            // and bounds every sub-curve's exact arc length <= max_step via control-polygon length!
            Vec3 left[4], right[4];
            split_cubic_bezier(prim.bez, left, right);
            append_bezier_de_casteljau(left, 0.0, 0.5, 0);
            append_bezier_de_casteljau(right, 0.5, 1.0, 0);
        } else {
            append_subspan(0.0, 1.0);
        }

        struct Task { double ta, tb; int depth; };
        std::vector<Task> tasks;
        const int n_steps = static_cast<int>(tau_breaks.size()) - 1;
        tasks.reserve(static_cast<size_t>(n_steps + cfg.max_subdivision_depth + 4));
        for (int s = n_steps - 1; s >= 0; --s) {
            tasks.push_back({tau_breaks[static_cast<size_t>(s)], tau_breaks[static_cast<size_t>(s + 1)], 0});
        }

        while (!tasks.empty()) {
            const Task cur = tasks.back();
            tasks.pop_back();
            const Eigen::VectorXd& q_prev = knots.back().q;
            const auto ik = cartesian_detail::solve_cartesian_step_ik(
                snap, req.link_id, req.tcp_offset_position, tcp_rot_local,
                prim.eval_pos(cur.tb), prim.eval_rot(cur.tb), prim.has_rot,
                q_prev, mask, cfg, margins, ik_pos_tol, ik_rot_tol);

            std::string mid_err;
            bool mid_is_col = false;
            bool mid_ok = true;
            if (ik.status == cartesian_detail::StepIKStatus::Ok) {
                mid_ok = check_chord_mid(prim, cur.ta, cur.tb, q_prev, ik.q, mid_err, &mid_is_col);
            }

            const bool approaching_jump = (ik.status == cartesian_detail::StepIKStatus::Ok) && (ik.jump_ratio > 0.75);
            const bool needs_subdiv = (ik.status != cartesian_detail::StepIKStatus::Ok) || !mid_ok || approaching_jump;

            if (needs_subdiv && cur.depth < cfg.max_subdivision_depth) {
                const double tm = 0.5 * (cur.ta + cur.tb);
                // Push right half first so left half runs next (LIFO stack -> strict left-to-right progress!).
                tasks.push_back({tm, cur.tb, cur.depth + 1});
                tasks.push_back({cur.ta, tm, cur.depth + 1});
                continue;
            }

            if (ik.status != cartesian_detail::StepIKStatus::Ok || !mid_ok) {
                ik_stopped = true;
                ik_stop_reason = (ik.status != cartesian_detail::StepIKStatus::Ok) ? ik.reason : mid_err;
                has_ik_fail_interval = true;
                ik_fail_is_collision = (ik.status == cartesian_detail::StepIKStatus::Collision) ||
                                       (ik.status == cartesian_detail::StepIKStatus::Ok && mid_is_col);
                ik_fail_prim = p_idx;
                ik_fail_ta = cur.ta;
                ik_fail_tb = cur.tb;
                break;
            }

            cartesian_detail::CartesianKnot kn;
            kn.q = ik.q;
            kn.prim_idx = p_idx;
            kn.tau = cur.tb;
            kn.progress = prim.progress_start + cur.tb * (prim.progress_end - prim.progress_start);
            kn.is_stop = (cur.tb >= 1.0 - 1e-12) && prim.stop_at_end;
            kn.depth = cur.depth;
            kn.completed_waypoints = knots.back().completed_waypoints;
            if (cur.tb >= 1.0 - 1e-12 && p_idx == wp_last_prim[static_cast<size_t>(prim.orig_waypoint_idx)]) {
                kn.completed_waypoints = std::max(kn.completed_waypoints, prim.orig_waypoint_idx + 1);
            }
            knots.push_back(std::move(kn));
        }
        if (ik_stopped) break;
    }
    res.ik_ms = std::chrono::duration<double, std::milli>(Clock::now() - t_ik0).count();

    // 5. Retime & strictly verify the solved prefix (bounds + continuous CCD + continuous TCP deviation).
    // Supports sub-segment bisection for BOTH verification failures AND collision-stopped IK intervals
    // (including when the very first interval fails before accepting knot 1).
    auto try_knots = [&](const std::vector<cartesian_detail::CartesianKnot>& sub) -> cartesian_detail::PrefixVerifyResult {
        return cartesian_detail::finalize_and_verify_cartesian_prefix(
            snap, req.link_id, req.tcp_offset_position, tcp_rot_local,
            mask, cfg, margins, prims, sub);
    };
    auto try_prefix = [&](int K_len) -> cartesian_detail::PrefixVerifyResult {
        std::vector<cartesian_detail::CartesianKnot> sub(knots.begin(), knots.begin() + K_len);
        return try_knots(sub);
    };

    if (knots.size() < 2 && !ik_stopped && knots.back().completed_waypoints == M) {
        RowMatrixXd pos(2, J);
        pos.row(0) = req.start_q.transpose();
        pos.row(1) = req.start_q.transpose();
        const Eigen::VectorXd vmax = snap.kin.velocity_limit() * cfg.velocity_scale;
        const Eigen::VectorXd amax = snap.kin.acceleration_limit() * cfg.acceleration_scale;
        const Eigen::VectorXd jmax = snap.kin.jerk_limit() * cfg.jerk_scale;
        res.trajectory = retime_trajectory(pos, vmax, amax, cfg.stop_at_endpoints, false, 0.0,
                                           nullptr, &jmax, cfg.spline_order);
        res.has_trajectory = true;
        res.success = true;
        res.fraction = 1.0;
        res.completed_waypoints = M;
        res.max_pos_error = 0.0;
        res.max_rot_error = 0.0;
        res.total_ms = std::chrono::duration<double, std::milli>(Clock::now() - t_start).count();
        return res;
    }

    int K_full = static_cast<int>(knots.size());
    cartesian_detail::PrefixVerifyResult ver;
    int verified_len = 0;
    bool truncated_by_verify = false;
    std::string verify_fail_reason;
    cartesian_detail::CartesianKnot last_verified_knot = knots.front();

    auto bisect_tail_interval = [&](int pid, double lo_t, double hi_t,
                                    std::vector<cartesian_detail::CartesianKnot> valid_prefix,
                                    bool update_verify_reason) {
        const auto& prim = prims[static_cast<size_t>(pid)];
        for (int bit = 0; bit < 10; ++bit) {
            if (hi_t - lo_t <= 1e-6) break;
            const double tm = 0.5 * (lo_t + hi_t);
            const Eigen::VectorXd& q_prev = valid_prefix.back().q;
            const auto ik_m = cartesian_detail::solve_cartesian_step_ik(
                snap, req.link_id, req.tcp_offset_position, tcp_rot_local,
                prim.eval_pos(tm), prim.eval_rot(tm), prim.has_rot,
                q_prev, mask, cfg, margins, ik_pos_tol, ik_rot_tol);
            std::string mid_err;
            const bool ok_step = (ik_m.status == cartesian_detail::StepIKStatus::Ok) &&
                                 check_chord_mid(prim, lo_t, tm, q_prev, ik_m.q, mid_err);
            if (!ok_step) {
                hi_t = tm;
                continue;
            }
            cartesian_detail::CartesianKnot kn_m;
            kn_m.q = ik_m.q;
            kn_m.prim_idx = pid;
            kn_m.tau = tm;
            kn_m.progress = prim.progress_start + tm * (prim.progress_end - prim.progress_start);
            kn_m.is_stop = false;
            kn_m.depth = valid_prefix.back().depth + 1;
            kn_m.completed_waypoints = valid_prefix.back().completed_waypoints;

            auto cand = valid_prefix;
            cand.push_back(kn_m);
            auto vm = try_knots(cand);
            res.retime_ms += vm.retime_ms;
            res.validate_ms += vm.validate_ms;
            if (vm.ok) {
                lo_t = tm;
                valid_prefix = std::move(cand);
                ver = std::move(vm);
                verified_len = static_cast<int>(valid_prefix.size());
                last_verified_knot = valid_prefix.back();
            } else {
                if (update_verify_reason && verify_fail_reason.empty()) verify_fail_reason = vm.reason;
                hi_t = tm;
            }
        }
    };

    if (K_full >= 2) {
        ver = try_prefix(K_full);
        res.retime_ms += ver.retime_ms;
        res.validate_ms += ver.validate_ms;
        if (ver.ok) {
            verified_len = K_full;
            last_verified_knot = knots.back();
        } else {
            truncated_by_verify = true;
            verify_fail_reason = ver.reason;
            // Binary search longest verifiable knot prefix in [2, min(K_full - 1, earliest_bad_seg + 1)].
            int hi_k = std::min(K_full - 1, std::max(1, ver.earliest_bad_seg + 1));
            if (hi_k >= 2) {
                auto v_hi = try_prefix(hi_k);
                res.retime_ms += v_hi.retime_ms;
                res.validate_ms += v_hi.validate_ms;
                if (v_hi.ok) {
                    ver = std::move(v_hi);
                    verified_len = hi_k;
                    last_verified_knot = knots[static_cast<size_t>(hi_k - 1)];
                } else {
                    verify_fail_reason = v_hi.reason;
                    int lo_k = 2;
                    int best_k = 0;
                    cartesian_detail::PrefixVerifyResult best_ver;
                    int r_k = std::min(hi_k - 1, std::max(1, v_hi.earliest_bad_seg + 1));
                    while (lo_k <= r_k) {
                        const int mid_k = lo_k + (r_k - lo_k) / 2;
                        auto vm = try_prefix(mid_k);
                        res.retime_ms += vm.retime_ms;
                        res.validate_ms += vm.validate_ms;
                        if (vm.ok) {
                            best_k = mid_k;
                            best_ver = std::move(vm);
                            lo_k = mid_k + 1;
                        } else {
                            verify_fail_reason = vm.reason;
                            r_k = std::min(mid_k - 1, std::max(1, vm.earliest_bad_seg + 1));
                        }
                    }
                    if (best_k >= 2) {
                        ver = std::move(best_ver);
                        verified_len = best_k;
                        last_verified_knot = knots[static_cast<size_t>(best_k - 1)];
                    }
                }
            }

            // Sub-segment bisection inside the first failing IK segment (including segment 0!).
            const int bad_seg = (verified_len >= 2) ? (verified_len - 1) : 0;
            if (bad_seg >= 0 && bad_seg + 1 < K_full) {
                const int pid = knots[static_cast<size_t>(bad_seg + 1)].prim_idx;
                const double lo_t = (knots[static_cast<size_t>(bad_seg)].prim_idx == pid)
                    ? knots[static_cast<size_t>(bad_seg)].tau : 0.0;
                const double hi_t = knots[static_cast<size_t>(bad_seg + 1)].tau;
                std::vector<cartesian_detail::CartesianKnot> valid_prefix(
                    knots.begin(), knots.begin() + (bad_seg + 1));
                bisect_tail_interval(pid, lo_t, hi_t, std::move(valid_prefix), true);
            }
        }
    }

    // If IK stopped due to collision after exhausting subdivision (including on the very first interval
    // where K_full == 1!) and verification did not already truncate earlier, bisect inside the failed
    // IK interval to recover the longest verified executable prefix.
    if (!truncated_by_verify && has_ik_fail_interval && ik_fail_is_collision) {
        bisect_tail_interval(ik_fail_prim, ik_fail_ta, ik_fail_tb, knots, false);
    }

    if (verified_len < 2 || !ver.ok) {
        res.fraction = 0.0;
        // Preserve zero-length waypoints already satisfied at start_q (e.g., wp[0] == start_pose).
        res.completed_waypoints = knots.front().completed_waypoints;
        if (truncated_by_verify && !verify_fail_reason.empty()) {
            res.reason = verify_fail_reason;
        } else if (!ik_stop_reason.empty()) {
            res.reason = ik_stop_reason;
        } else if (!verify_fail_reason.empty()) {
            res.reason = verify_fail_reason;
        } else {
            res.reason = "Cartesian planning made zero progress";
        }
        res.total_ms = std::chrono::duration<double, std::milli>(Clock::now() - t_start).count();
        return res;
    }

    // 6. Compute monotone fraction & completed_waypoints strictly from the verified executable prefix.
    const double achieved_progress = last_verified_knot.progress;
    const bool full_verified = (!ik_stopped) && (!truncated_by_verify) &&
                               (last_verified_knot.completed_waypoints == M) &&
                               (achieved_progress >= total_progress - 1e-9 * std::max(1.0, total_progress));
    res.fraction = full_verified
        ? 1.0
        : std::clamp((total_progress > 0.0) ? (achieved_progress / total_progress) : 0.0,
                     0.0, std::nextafter(1.0, 0.0));
    res.completed_waypoints = full_verified ? M : last_verified_knot.completed_waypoints;
    res.has_trajectory = true;
    res.trajectory = std::move(ver.trajectory);
    res.max_pos_error = ver.max_pos_err;
    res.max_rot_error = ver.max_rot_err;
    res.success = full_verified;
    if (!full_verified) {
        if (truncated_by_verify && !verify_fail_reason.empty()) {
            res.reason = verify_fail_reason;
        } else if (!ik_stop_reason.empty()) {
            res.reason = ik_stop_reason;
        } else if (!verify_fail_reason.empty()) {
            res.reason = verify_fail_reason;
        } else {
            res.reason = "Cartesian path incomplete";
        }
    }
    res.total_ms = std::chrono::duration<double, std::milli>(Clock::now() - t_start).count();
    return res;
}

}  // namespace gobot::planning
