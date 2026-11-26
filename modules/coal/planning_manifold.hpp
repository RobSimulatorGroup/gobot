#pragma once

#include "planning_retime.hpp"
#include "planning_world.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <queue>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace gobot::planning {

struct ManifoldPointEval {
    bool ok = true;
    bool antipodal = false;
    double phi = 0.0;        // normalized violation: <= 1.0 iff satisfied
    double pos_err = 0.0;
    double rot_err = 0.0;
    double pos_inv_tol = 0.0;
    double rot_inv_tol = 0.0;
    double rot_angle = 0.0;  // true SO(3) geodesic error angle (for Jacobian scaling near pi)
    bool needs_jinv_scale = false;
};

struct ManifoldSplineCheckResult {
    bool ok = true;
    double max_pos_err = 0.0;
    double max_rot_err = 0.0;
    std::vector<int> bad_segments;
    std::string reason;
};

namespace manifold_detail {

inline void validate_manifold_specs(const WorldSnapshot& snap,
                                    const std::vector<PoseConstraintSpec>& constraints) {
    const int B = snap.kin.num_bodies();
    for (const auto& c : constraints) {
        if (c.link_id < 0 || c.link_id >= B) {
            throw std::invalid_argument("manifold constraint has invalid link_id");
        }
        if (!c.has_position && !c.has_rotation) {
            throw std::invalid_argument("manifold constraint requires position, rotation, or both");
        }
        if (c.has_position) {
            if (!c.position.allFinite()) {
                throw std::invalid_argument("manifold constraint position must be finite");
            }
            for (int a = 0; a < 3; ++a) {
                if (std::isnan(c.pos_tol(a)) || c.pos_tol(a) <= 0.0) {
                    throw std::invalid_argument(
                        "manifold constraint position tolerance must be strictly positive (> 0 or +inf; zero tolerance is not supported)");
                }
            }
        }
        if (c.has_rotation) {
            if (!c.rotation.allFinite()) {
                throw std::invalid_argument("manifold constraint rotation must be finite");
            }
            for (int a = 0; a < 3; ++a) {
                if (std::isnan(c.rot_tol(a)) || c.rot_tol(a) <= 0.0) {
                    throw std::invalid_argument(
                        "manifold constraint rotation tolerance must be strictly positive (> 0 or +inf; zero tolerance is not supported)");
                }
            }
        }
    }
}

// Evaluate exact normalized manifold violation for a single PoseConstraintSpec at precomputed FK poses.
// Explicitly detects anti-parallel pi degeneracy where eval_pose_constraint's residual vanishes.
inline ManifoldPointEval eval_constraint_norm(const PoseConstraintSpec& c,
                                              const Vec3* body_pos,
                                              const Mat3* body_rot,
                                              double tol_scale = 1.0) {
    ManifoldPointEval out;
    const Vec3& p = body_pos[c.link_id];
    const Mat3& R = body_rot[c.link_id];

    auto scaled_tol = [tol_scale](double raw_tol) -> double {
        if (tol_scale >= 1.0) return raw_tol * tol_scale;
        // Never silently relax the user's tolerance when tightening for internal projection.
        return std::min(raw_tol, std::max(1e-12, raw_tol * tol_scale));
    };

    if (c.has_position) {
        const Vec3 dp = p - c.position;
        const bool iso = std::isfinite(c.pos_tol(0)) && c.pos_tol(0) == c.pos_tol(1) && c.pos_tol(1) == c.pos_tol(2);
        if (iso) {
            const double tol = scaled_tol(c.pos_tol(0));
            const double e = dp.norm();
            out.pos_err = e;
            out.pos_inv_tol = 1.0 / tol;
            out.phi = std::max(out.phi, e / tol);
            if (e > tol) out.ok = false;
        } else {
            double max_e = 0.0;
            double max_inv = 0.0;
            for (int a = 0; a < 3; ++a) {
                if (!std::isfinite(c.pos_tol(a))) continue;
                const double tol = scaled_tol(c.pos_tol(a));
                const double ea = std::abs(dp(a));
                max_e = std::max(max_e, ea);
                max_inv = std::max(max_inv, 1.0 / tol);
                out.phi = std::max(out.phi, ea / tol);
                if (ea > tol) out.ok = false;
            }
            out.pos_err = max_e;
            out.pos_inv_tol = max_inv;
        }
    }

    if (c.has_rotation) {
        int free_axis = -1;
        int free_count = 0;
        for (int a = 0; a < 3; ++a) {
            if (!std::isfinite(c.rot_tol(a))) {
                free_axis = a;
                ++free_count;
            }
        }
        if (free_count == 0) {
            const Mat3 R_rel = c.rotation.transpose() * R;
            const Vec3 phi = rotation_log(R_rel);
            const double ang = phi.norm();
            out.rot_angle = ang;
            const bool iso = (c.rot_tol(0) == c.rot_tol(1)) && (c.rot_tol(1) == c.rot_tol(2));
            if (iso) {
                const double tol = scaled_tol(c.rot_tol(0));
                out.rot_err = ang;
                out.rot_inv_tol = 1.0 / tol;
                out.phi = std::max(out.phi, ang / tol);
                if (ang > tol) out.ok = false;
            } else {
                out.needs_jinv_scale = true;
                double max_e = 0.0;
                double max_inv = 0.0;
                for (int a = 0; a < 3; ++a) {
                    const double tol = scaled_tol(c.rot_tol(a));
                    const double ea = std::abs(phi(a));
                    max_e = std::max(max_e, ea);
                    max_inv = std::max(max_inv, 1.0 / tol);
                    out.phi = std::max(out.phi, ea / tol);
                    if (ea > tol) out.ok = false;
                }
                out.rot_err = max_e;
                out.rot_inv_tol = max_inv;
            }
        } else if (free_count == 1) {
            // Axis alignment: true geodesic angle between link free_axis and target free_axis is strictly 1-Lipschitz on SO(3).
            const Vec3 u = R.col(free_axis);
            const Vec3 u_in_t = c.rotation.transpose() * u;
            const int a0 = (free_axis + 1) % 3;
            const int a1 = (free_axis + 2) % 3;
            const double sin_ang = std::hypot(u_in_t(a0), u_in_t(a1));
            const double ang = std::atan2(sin_ang, u_in_t(free_axis));
            const double tol = scaled_tol(std::min(c.rot_tol(a0), c.rot_tol(a1)));
            out.rot_angle = ang;
            out.rot_err = ang;
            out.rot_inv_tol = 1.0 / tol;
            out.phi = std::max(out.phi, ang / tol);
            if (ang > tol) out.ok = false;
            // Anti-parallel pi degeneracy: residual in eval_pose_constraint vanishes as u_in_t(free_axis) -> -1!
            out.antipodal = ang > std::min(c.rot_tol(a0), c.rot_tol(a1)) &&
                            u_in_t(free_axis) < 0.0 && sin_ang <= 1e-8;
        } else if (free_count == 2) {
            const Mat3 R_rel = c.rotation.transpose() * R;
            const Vec3 phi = rotation_log(R_rel);
            const double ang = phi.norm();
            out.rot_angle = ang;
            out.needs_jinv_scale = true;
            double max_e = 0.0;
            double max_inv = 0.0;
            for (int a = 0; a < 3; ++a) {
                if (!std::isfinite(c.rot_tol(a))) continue;
                const double tol = scaled_tol(c.rot_tol(a));
                const double ea = std::abs(phi(a));
                max_e = std::max(max_e, ea);
                max_inv = std::max(max_inv, 1.0 / tol);
                out.phi = std::max(out.phi, ea / tol);
                if (ea > tol) out.ok = false;
            }
            out.rot_err = max_e;
            out.rot_inv_tol = max_inv;
        }
    }
    return out;
}

inline bool is_on_manifold(const WorldSnapshot& snap,
                           const double* q,
                           const std::vector<PoseConstraintSpec>& constraints,
                           double tol_scale = 1.0,
                           bool* out_antipodal = nullptr) {
    if (out_antipodal) *out_antipodal = false;
    if (constraints.empty()) return true;
    const int B = snap.kin.num_bodies();
    std::vector<Vec3> bpos(static_cast<size_t>(B));
    std::vector<Mat3> brot(static_cast<size_t>(B));
    snap.kin.fk_single(q, bpos.data(), brot.data());
    for (const auto& c : constraints) {
        const auto ev = eval_constraint_norm(c, bpos.data(), brot.data(), tol_scale);
        if (ev.antipodal && out_antipodal) *out_antipodal = true;
        if (!ev.ok || ev.antipodal) return false;
    }
    return true;
}

// Compute stacked constraint residual & Jacobian (handling anti-parallel pi singularity cleanly).
inline bool eval_stacked_manifold(const WorldSnapshot& snap,
                                  const Eigen::VectorXd& q,
                                  const std::vector<PoseConstraintSpec>& constraints,
                                  double tol_scale,
                                  std::vector<Vec3>& bpos,
                                  std::vector<Mat3>& brot,
                                  std::vector<Vec3>& jorig,
                                  std::vector<Vec3>& jaxis,
                                  Eigen::VectorXd& out_r,
                                  Eigen::MatrixXd* out_J,
                                  bool& out_antipodal) {
    const int J = snap.kin.num_joints();
    const bool need_jac = (out_J != nullptr);
    snap.kin.fk_single(q.data(), bpos.data(), brot.data(),
                       need_jac ? jorig.data() : nullptr,
                       need_jac ? jaxis.data() : nullptr);

    int total_dim = 0;
    for (const auto& c : constraints) total_dim += c.residual_dim();
    out_r.resize(total_dim);
    if (need_jac) out_J->setZero(total_dim, J);

    bool all_ok = true;
    out_antipodal = false;
    int offset = 0;
    Eigen::VectorXd rc;
    Eigen::MatrixXd Jc;
    for (const auto& c : constraints) {
        const auto ev = eval_constraint_norm(c, bpos.data(), brot.data(), tol_scale);
        if (ev.antipodal) out_antipodal = true;
        if (!ev.ok || ev.antipodal) all_ok = false;

        snap.kin.eval_pose_constraint(c, bpos.data(), brot.data(),
                                      need_jac ? jorig.data() : nullptr,
                                      need_jac ? jaxis.data() : nullptr,
                                      rc, need_jac ? &Jc : nullptr);
        // Scale position and rotation rows consistently by 1/tolerance so LM converges uniformly.
        int pos_rows = 0;
        if (c.has_position) {
            for (int a = 0; a < 3; ++a) if (std::isfinite(c.pos_tol(a))) ++pos_rows;
        }
        const double wp = (ev.pos_inv_tol > 0.0) ? std::min(ev.pos_inv_tol, 1e4) : 10.0;
        const double wr = (ev.rot_inv_tol > 0.0) ? std::min(ev.rot_inv_tol, 1e4) : 10.0;
        for (int i = 0; i < rc.size(); ++i) {
            const double w = ((i < pos_rows) ? wp : wr) * c.weight;
            out_r(offset + i) = rc(i) * w;
            if (need_jac) out_J->row(offset + i) = Jc.row(i) * w;
        }
        offset += static_cast<int>(rc.size());
    }
    return all_ok;
}

// Project a joint increment dq onto the manifold tangent space at q_cur:
// dq_T = dq - J^T (J J^T + eps I)^{-1} (J dq) on active joints.
inline Eigen::VectorXd project_tangent_step(const WorldSnapshot& snap,
                                            const std::vector<PoseConstraintSpec>& constraints,
                                            const std::vector<uint8_t>& active_mask,
                                            const Eigen::VectorXd& q_cur,
                                            const Eigen::VectorXd& dq) {
    if (constraints.empty()) return dq;
    const int J = snap.kin.num_joints();
    const int B = snap.kin.num_bodies();
    std::vector<Vec3> bpos(static_cast<size_t>(B)), jorig(static_cast<size_t>(J)), jaxis(static_cast<size_t>(J));
    std::vector<Mat3> brot(static_cast<size_t>(B));
    Eigen::VectorXd r;
    Eigen::MatrixXd J_full;
    bool antipodal = false;
    eval_stacked_manifold(snap, q_cur, constraints, 1.0, bpos, brot, jorig, jaxis, r, &J_full, antipodal);
    if (antipodal || J_full.rows() == 0) return dq;

    // Mask out inactive columns.
    for (int j = 0; j < J; ++j) {
        if (!active_mask[static_cast<size_t>(j)]) J_full.col(j).setZero();
    }
    const int M = static_cast<int>(J_full.rows());
    Eigen::MatrixXd JJt = J_full * J_full.transpose();
    JJt.diagonal().array() += 1e-6;
    Eigen::LDLT<Eigen::MatrixXd> ldlt(JJt);
    if (ldlt.info() != Eigen::Success) return dq;
    Eigen::VectorXd dq_act = dq;
    for (int j = 0; j < J; ++j) {
        if (!active_mask[static_cast<size_t>(j)]) dq_act(j) = 0.0;
    }
    const Eigen::VectorXd corr = J_full.transpose() * ldlt.solve(J_full * dq_act);
    if (!corr.allFinite()) return dq_act;
    return dq_act - corr;
}

// Sound projected Levenberg-Marquardt manifold projector.
// Always checks true manifold `ok` (NOT residual norm!) and explicitly rejects anti-parallel pi degeneracy.
inline bool project_onto_manifold(const WorldSnapshot& snap,
                                  const std::vector<PoseConstraintSpec>& constraints,
                                  const std::vector<uint8_t>& active_mask,
                                  const Eigen::VectorXd& q_in,
                                  Eigen::VectorXd& q_out,
                                  double max_disp = std::numeric_limits<double>::infinity(),
                                  double target_tol_scale = 0.35,
                                  int max_iters = 25) {
    const int J = snap.kin.num_joints();
    const int B = snap.kin.num_bodies();
    const auto& lo = snap.kin.lower();
    const auto& hi = snap.kin.upper();

    q_out = q_in;
    for (int j = 0; j < J; ++j) q_out(j) = std::clamp(q_out(j), lo(j), hi(j));
    if (constraints.empty()) return true;

    std::vector<Vec3> bpos(static_cast<size_t>(B)), jorig(static_cast<size_t>(J)), jaxis(static_cast<size_t>(J));
    std::vector<Mat3> brot(static_cast<size_t>(B));
    Eigen::VectorXd r;
    Eigen::MatrixXd Jm;
    bool antipodal = false;

    if (eval_stacked_manifold(snap, q_out, constraints, target_tol_scale, bpos, brot, jorig, jaxis, r, nullptr, antipodal)) {
        return true;
    }
    if (antipodal) {
        // Reject anti-parallel pi singularity cleanly so sampler/extension resamples rather than stalling at residual=0.
        return false;
    }

    double damping = 1e-3;
    const double w_stay = 1e-3;
    for (int it = 0; it < max_iters; ++it) {
        const bool tight_ok = eval_stacked_manifold(
            snap, q_out, constraints, target_tol_scale, bpos, brot, jorig, jaxis, r, &Jm, antipodal);
        if (antipodal) return false;
        if (tight_ok) return (q_out - q_in).norm() <= max_disp;

        double cost = r.squaredNorm();
        Eigen::MatrixXd H = Jm.transpose() * Jm;
        Eigen::VectorXd g = Jm.transpose() * r;
        for (int j = 0; j < J; ++j) {
            if (!active_mask[static_cast<size_t>(j)]) continue;
            const double dj = w_stay * (q_out(j) - q_in(j));
            cost += dj * dj;
            H(j, j) += w_stay * w_stay;
            g(j) += w_stay * dj;
        }

        std::vector<int> free_idx;
        free_idx.reserve(static_cast<size_t>(J));
        double max_g = 0.0;
        for (int j = 0; j < J; ++j) {
            if (!active_mask[static_cast<size_t>(j)]) continue;
            const bool at_bound = (q_out(j) <= lo(j) && g(j) > 0.0) || (q_out(j) >= hi(j) && g(j) < 0.0);
            if (!at_bound) {
                free_idx.push_back(j);
                max_g = std::max(max_g, std::abs(g(j)));
            }
        }
        if (free_idx.empty() || max_g < 1e-10) break;

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
            Eigen::VectorXd q_try = q_out;
            for (int a = 0; a < F; ++a) {
                const int j = free_idx[static_cast<size_t>(a)];
                q_try(j) = std::clamp(q_out(j) + step_f(a), lo(j), hi(j));
            }
            if ((q_try - q_in).norm() > max_disp * 1.5) {
                damping *= 10.0;
                continue;
            }
            bool try_anti = false;
            Eigen::VectorXd r_try;
            eval_stacked_manifold(snap, q_try, constraints, target_tol_scale, bpos, brot, jorig, jaxis, r_try, nullptr, try_anti);
            if (try_anti) {
                damping *= 10.0;
                continue;
            }
            double try_cost = r_try.squaredNorm();
            for (int j = 0; j < J; ++j) {
                if (!active_mask[static_cast<size_t>(j)]) continue;
                const double dj = w_stay * (q_try(j) - q_in(j));
                try_cost += dj * dj;
            }
            if (try_cost < cost - 1e-12) {
                q_out = std::move(q_try);
                damping = std::max(1e-8, damping * 0.4);
                accepted = true;
                break;
            }
            damping = std::min(1e6, damping * 4.0);
        }
        if (!accepted) break;
    }

    // Final ground-truth manifold verification at true tolerance (1.0x), strictly rejecting antipodal pi.
    bool final_anti = false;
    const bool final_ok = is_on_manifold(snap, q_out.data(), constraints, 1.0, &final_anti);
    return final_ok && !final_anti && ((q_out - q_in).norm() <= max_disp);
}

// Verify a short linear segment [qa, qb] is BOTH collision-free AND stays on the manifold at every check step.
inline bool is_constrained_segment_valid(const WorldSnapshot& snap,
                                         const Eigen::VectorXd& qa,
                                         const Eigen::VectorXd& qb,
                                         const std::vector<PoseConstraintSpec>& constraints,
                                         const Eigen::VectorXd& margins,
                                         double max_step,
                                         bool verify_original) {
    if (!snap.is_segment_valid(qa, qb, margins, max_step, verify_original)) return false;
    if (constraints.empty()) return true;
    if (!is_on_manifold(snap, qb.data(), constraints, 1.0)) return false;
    const double dist = (qb - qa).norm();
    const int steps = std::max(2, static_cast<int>(std::ceil(dist / std::max(1e-3, max_step))));
    Eigen::VectorXd q(qa.size());
    for (int i = 1; i < steps; ++i) {
        const double alpha = static_cast<double>(i) / static_cast<double>(steps);
        q.noalias() = (1.0 - alpha) * qa + alpha * qb;
        if (!is_on_manifold(snap, q.data(), constraints, 1.0)) return false;
    }
    return true;
}

// Unified constrained geodesic trace from q_start toward q_goal (used by Extend, Connect, and Shortcutting).
// Stores EVERY intermediate projected state in out_chain. Returns:
//   0 = Trapped (made zero progress)
//   1 = Advanced (added >= 1 projected state, did not reach q_goal)
//   2 = Reached (reached q_goal with verified final constrained segment)
inline int trace_constrained_segment(const WorldSnapshot& snap,
                                     const Eigen::VectorXd& q_start,
                                     const Eigen::VectorXd& q_goal,
                                     const std::vector<PoseConstraintSpec>& constraints,
                                     const std::vector<uint8_t>& active_mask,
                                     const Eigen::VectorXd& margins,
                                     double step_size,
                                     double check_res,
                                     bool verify_original,
                                     int max_steps,
                                     std::vector<Eigen::VectorXd>& out_chain) {
    out_chain.clear();
    const double proj_step = std::min(step_size, 0.15);
    const double connect_eps = std::max(1e-3, 0.5 * proj_step);
    Eigen::VectorXd q_cur = q_start;

    for (int s = 0; s < max_steps; ++s) {
        const Eigen::VectorXd diff = q_goal - q_cur;
        const double d = diff.norm();
        if (d <= 1e-9) return 2;
        if (d <= proj_step &&
            is_constrained_segment_valid(snap, q_cur, q_goal, constraints, margins, check_res, verify_original)) {
            out_chain.push_back(q_goal);
            return 2;
        }

        const double h = std::min(proj_step, d);
        Eigen::VectorXd raw_step = (h / d) * diff;
        // Project step onto manifold tangent space when constrained (accelerates convergence along curved manifolds).
        Eigen::VectorXd tang_step = project_tangent_step(snap, constraints, active_mask, q_cur, raw_step);
        const double tn = tang_step.norm();
        Eigen::VectorXd q_cand = (tn > 1e-6) ? Eigen::VectorXd(q_cur + (h / tn) * tang_step)
                                             : Eigen::VectorXd(q_cur + raw_step);

        Eigen::VectorXd q_proj;
        if (!project_onto_manifold(snap, constraints, active_mask, q_cand, q_proj, 2.0 * h, 0.35, 20)) {
            // Fallback: try direct ambient step projection without tangent projection.
            if (!project_onto_manifold(snap, constraints, active_mask, q_cur + raw_step, q_proj, 2.0 * h, 0.35, 20)) {
                break;
            }
        }
        const double moved = (q_proj - q_cur).norm();
        if (moved < 1e-5 || moved > 2.2 * h) break;
        const double d_after = (q_goal - q_proj).norm();
        if (d_after >= d - 1e-5 && d > connect_eps) break;

        if (!is_constrained_segment_valid(snap, q_cur, q_proj, constraints, margins, check_res, verify_original)) {
            break;
        }
        out_chain.push_back(q_proj);
        q_cur = std::move(q_proj);

        if (d_after <= connect_eps &&
            is_constrained_segment_valid(snap, q_cur, q_goal, constraints, margins, check_res, verify_original)) {
            if (d_after > 1e-9) out_chain.push_back(q_goal);
            return 2;
        }
    }
    return out_chain.empty() ? 0 : 1;
}

// Unified constrained resampling onto `output_waypoints`: linearly interpolates along the verified
// projected polyline, projects interior samples onto the manifold, and verifies EVERY resampled segment
// for BOTH collision clearance and manifold satisfaction! Returns true iff all segments are valid.
inline bool constrained_resample_path(
    const WorldSnapshot& snap,
    const std::vector<Eigen::VectorXd>& path,
    int count,
    const std::vector<PoseConstraintSpec>& constraints,
    const std::vector<uint8_t>& active_mask,
    const Eigen::VectorXd& margins,
    double check_res,
    bool verify_original,
    std::vector<Eigen::VectorXd>& out_resampled) {
    if (count < 2 || path.size() < 2) {
        out_resampled = path;
        return out_resampled.size() >= 2;
    }
    out_resampled = WorldSnapshot::resample_path(path, count);
    if (constraints.empty()) {
        for (size_t k = 0; k + 1 < out_resampled.size(); ++k) {
            if (!snap.is_segment_valid(out_resampled[k], out_resampled[k + 1], margins, check_res, verify_original)) {
                return false;
            }
        }
        return true;
    }
    for (int i = 1; i + 1 < static_cast<int>(out_resampled.size()); ++i) {
        if (!is_on_manifold(snap, out_resampled[static_cast<size_t>(i)].data(), constraints, 0.5)) {
            Eigen::VectorXd q_proj;
            if (project_onto_manifold(snap, constraints, active_mask, out_resampled[static_cast<size_t>(i)],
                                      q_proj, 0.35, 0.35, 20) &&
                snap.is_config_valid(q_proj.data(), margins, verify_original)) {
                out_resampled[static_cast<size_t>(i)] = std::move(q_proj);
            }
        }
    }
    for (size_t k = 0; k + 1 < out_resampled.size(); ++k) {
        if (!is_constrained_segment_valid(
                snap, out_resampled[k], out_resampled[k + 1], constraints, margins, check_res, verify_original)) {
            return false;
        }
    }
    return true;
}

}  // namespace manifold_detail

// Certified continuous manifold verification along an ACTUAL played cubic (order 3) or quintic (order 5)
// Hermite spline (or linear polyline when `is_polyline == true`).
// Uses kinematic-chain Lipschitz bounds + exact/certified Hermite total variation + adaptive bisection.
inline ManifoldSplineCheckResult verify_spline_manifold_constraints(
    const WorldSnapshot& snap,
    const TrajectoryNative& traj,
    const std::vector<PoseConstraintSpec>& constraints,
    bool is_polyline = false) {
    ManifoldSplineCheckResult res;
    if (constraints.empty()) return res;
    manifold_detail::validate_manifold_specs(snap, constraints);

    const int T = traj.num_waypoints();
    const int J = traj.num_joints();
    const int B = snap.kin.num_bodies();
    const int n_segs = T - 1;
    if (n_segs <= 0) return res;

    const bool is_quintic = (!is_polyline) && (traj.spline_order == 5);
    const auto& is_pris = snap.kin.joint_prismatic();

    // Exact/certified max |q_j| over trajectory for downstream prismatic joints.
    std::vector<double> q_abs_max(static_cast<size_t>(J), 0.0);
    for (int t = 0; t < T; ++t) {
        for (int j = 0; j < J; ++j) {
            q_abs_max[static_cast<size_t>(j)] = std::max(q_abs_max[static_cast<size_t>(j)], std::abs(traj.positions(t, j)));
        }
    }
    if (!is_polyline) {
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
    }

    // Precompute exact Lipschitz weights per constraint.
    const int C = static_cast<int>(constraints.size());
    std::vector<std::vector<double>> L_pos(static_cast<size_t>(C), std::vector<double>(static_cast<size_t>(J), 0.0));
    std::vector<std::vector<double>> L_rot(static_cast<size_t>(C), std::vector<double>(static_cast<size_t>(J), 0.0));
    for (int ci = 0; ci < C; ++ci) {
        const auto& c = constraints[static_cast<size_t>(ci)];
        if (c.has_position) {
            snap.kin.body_lipschitz_weights(c.link_id, 0.0, q_abs_max.data(), L_pos[static_cast<size_t>(ci)].data());
        }
        if (c.has_rotation) {
            const uint8_t* anc = snap.kin.body_ancestor().data() + c.link_id * J;
            for (int j = 0; j < J; ++j) {
                if (anc[j] && !is_pris[static_cast<size_t>(j)]) L_rot[static_cast<size_t>(ci)][static_cast<size_t>(j)] = 1.0;
            }
        }
    }

    std::vector<Vec3> bpos(static_cast<size_t>(B));
    std::vector<Mat3> brot(static_cast<size_t>(B));
    std::vector<double> q_eval(static_cast<size_t>(J));

    auto eval_q = [&](int seg, double u, double* out_q) {
        const double h = is_polyline ? 0.0 : (traj.times(seg + 1) - traj.times(seg));
        if (is_polyline || h <= 0.0) {
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

    auto subinterval_tv = [&](int seg, double u0, double u1, std::vector<double>& tv_out) {
        const double h = is_polyline ? 0.0 : (traj.times(seg + 1) - traj.times(seg));
        for (int j = 0; j < J; ++j) {
            if (is_polyline || h <= 0.0) {
                tv_out[static_cast<size_t>(j)] = std::abs(traj.positions(seg + 1, j) - traj.positions(seg, j)) * (u1 - u0);
            } else if (is_quintic) {
                tv_out[static_cast<size_t>(j)] = quintic_hermite_scalar_total_variation_bound(
                    traj.positions(seg, j), traj.velocities(seg, j), traj.accelerations(seg, j),
                    traj.positions(seg + 1, j), traj.velocities(seg + 1, j), traj.accelerations(seg + 1, j),
                    h, u0, u1);
            } else {
                tv_out[static_cast<size_t>(j)] = hermite_scalar_total_variation(
                    traj.positions(seg + 1, j) - traj.positions(seg, j),
                    traj.velocities(seg, j), traj.velocities(seg + 1, j), h, u0, u1);
            }
        }
    };

    struct SubInterval {
        double u0, u1;
        double psi0, psi1;
        double ang0, ang1;
        double psi_ub;
        int depth;
        bool operator<(const SubInterval& o) const {
            return psi_ub < o.psi_ub;  // max-heap by certified upper bound
        }
    };
    std::vector<double> tv_buf(static_cast<size_t>(J));
    std::vector<Vec3> origins(J), axes(J);
    std::vector<bool> rotating_body(B), fixed_axis(J);

    for (int seg = 0; seg < n_segs; ++seg) {
        bool seg_bad = false;
        bool uncertified = false;
        double worst_pe = 0.0, worst_re = 0.0;
        // A slide's world axis stays fixed if all upstream revolute joints are
        // stationary on this segment, including between equal endpoint positions.
        snap.kin.fk_single(traj.positions.row(seg).data(), bpos.data(), brot.data(), origins.data(), axes.data());
        for (int b = 0; b < B; ++b) {
            const int parent = snap.kin.parent()[b];
            const int j = snap.kin.joint_idx()[b];
            rotating_body[b] = parent >= 0 && rotating_body[parent];
            if (j < 0) continue;
            if (is_pris[j]) {
                fixed_axis[j] = !rotating_body[b];
            } else if (traj.positions(seg, j) != traj.positions(seg + 1, j) ||
                       (!is_polyline && (traj.velocities(seg, j) != 0.0 || traj.velocities(seg + 1, j) != 0.0)) ||
                       (is_quintic && (traj.accelerations(seg, j) != 0.0 || traj.accelerations(seg + 1, j) != 0.0))) {
                rotating_body[b] = true;
            }
        }

        for (int ci = 0; ci < C && !seg_bad; ++ci) {
            const auto& c = constraints[static_cast<size_t>(ci)];
            auto pos_weights = L_pos[ci];
            if (c.has_position && c.pos_tol.minCoeff() != c.pos_tol.maxCoeff()) {
                // d_pos is normalized below by the smallest tolerance. A fixed
                // prismatic axis contributes only its component on each limited axis.
                const Vec3 scale = Vec3::Constant(c.pos_tol.minCoeff()).cwiseQuotient(c.pos_tol);
                for (int j = 0; j < J; ++j) {
                    if (is_pris[j] && fixed_axis[j] && pos_weights[j] > 0.0) {
                        pos_weights[j] = axes[j].cwiseAbs().cwiseProduct(scale).maxCoeff();
                    }
                }
            }
            auto eval_at_u = [&](double u, double& out_ang, bool& out_anti, bool& out_jinv,
                                 double& out_p_inv, double& out_r_inv) -> double {
                eval_q(seg, u, q_eval.data());
                snap.kin.fk_single(q_eval.data(), bpos.data(), brot.data());
                const auto ev = manifold_detail::eval_constraint_norm(c, bpos.data(), brot.data(), 1.0);
                res.max_pos_err = std::max(res.max_pos_err, ev.pos_err);
                res.max_rot_err = std::max(res.max_rot_err, ev.rot_err);
                worst_pe = std::max(worst_pe, ev.pos_err);
                worst_re = std::max(worst_re, ev.rot_err);
                out_ang = ev.rot_angle;
                out_anti = ev.antipodal;
                out_jinv = ev.needs_jinv_scale;
                out_p_inv = ev.pos_inv_tol;
                out_r_inv = ev.rot_inv_tol;
                return ev.antipodal ? std::numeric_limits<double>::infinity() : ev.phi;
            };

            constexpr double kStrictEps = 1e-9;
            constexpr int init_n = 4;
            double u_pts[init_n + 1];
            double psi_pts[init_n + 1];
            double ang_pts[init_n + 1];
            bool needs_jinv = false;
            double p_inv = 0.0, r_inv = 0.0;
            for (int i = 0; i <= init_n; ++i) {
                u_pts[i] = static_cast<double>(i) / static_cast<double>(init_n);
                bool anti = false, ji = false;
                double pi_v = 0.0, ri_v = 0.0;
                psi_pts[i] = eval_at_u(u_pts[i], ang_pts[i], anti, ji, pi_v, ri_v);
                needs_jinv = needs_jinv || ji;
                p_inv = std::max(p_inv, pi_v);
                r_inv = std::max(r_inv, ri_v);
                if (anti || !std::isfinite(psi_pts[i]) || psi_pts[i] > 1.0 + kStrictEps) {
                    seg_bad = true;
                    break;
                }
            }
            if (seg_bad) break;

            auto compute_psi_ub = [&](double u0, double u1, double psi0, double psi1,
                                      double ang0, double ang1, double& out_d_psi) -> double {
                subinterval_tv(seg, u0, u1, tv_buf);
                double d_pos = 0.0, d_rot = 0.0;
                for (int j = 0; j < J; ++j) {
                    d_pos += pos_weights[j] * tv_buf[static_cast<size_t>(j)];
                    d_rot += L_rot[static_cast<size_t>(ci)][static_cast<size_t>(j)] * tv_buf[static_cast<size_t>(j)];
                }
                double rot_factor = 1.0;
                if (needs_jinv && d_rot > 0.0) {
                    // For |phi_a(R)| = (theta / sin(theta/2)) * |q_a|, the exact global Lipschitz bound
                    // w.r.t. geodesic SO(3) distance is 1 + pi/2 everywhere on SO(3) including theta = pi!
                    // For smaller angles away from pi, (theta/2)/sin(theta/2) is tighter.
                    constexpr double kGlobalRotCompLip = 1.0 + 0.5 * M_PI;
                    const double ang_ub = std::min(M_PI, std::max(ang0, ang1) + d_rot);
                    if (ang_ub >= M_PI - 0.1) {
                        rot_factor = kGlobalRotCompLip;
                    } else if (ang_ub > 1e-4) {
                        rot_factor = std::min(kGlobalRotCompLip, (0.5 * ang_ub) / std::sin(0.5 * ang_ub));
                    }
                }
                out_d_psi = std::max(d_pos * p_inv, d_rot * rot_factor * r_inv);
                return 0.5 * (psi0 + psi1 + out_d_psi);
            };

            std::priority_queue<SubInterval> pq;
            for (int i = 0; i < init_n; ++i) {
                double d_psi = 0.0;
                const double ub = compute_psi_ub(u_pts[i], u_pts[i + 1], psi_pts[i], psi_pts[i + 1],
                                                 ang_pts[i], ang_pts[i + 1], d_psi);
                if (!std::isfinite(ub) || ub > 1.0 + kStrictEps) {
                    pq.push({u_pts[i], u_pts[i + 1], psi_pts[i], psi_pts[i + 1], ang_pts[i], ang_pts[i + 1], ub, 0});
                }
            }

            // Best-first refinement of the highest certified upper bound: never passes any spline
            // whose certified upper bound exceeds 1.0 + 1e-9 (strict tolerance, no 5e-4 slack).
            constexpr int kMaxSubdivEvals = 4096;
            constexpr int kMaxDepth = 24;
            int evals = 0;
            while (!pq.empty()) {
                const SubInterval cur = pq.top();
                pq.pop();
                if (!std::isfinite(cur.psi_ub) || cur.depth >= kMaxDepth || evals >= kMaxSubdivEvals) {
                    seg_bad = true;
                    uncertified = true;
                    break;
                }
                const double um = 0.5 * (cur.u0 + cur.u1);
                double ang_m = 0.0, pi_v = 0.0, ri_v = 0.0;
                bool anti_m = false, ji_m = false;
                const double psi_m = eval_at_u(um, ang_m, anti_m, ji_m, pi_v, ri_v);
                ++evals;
                if (anti_m || !std::isfinite(psi_m) || psi_m > 1.0 + kStrictEps) {
                    seg_bad = true;
                    break;
                }
                double d_l = 0.0, d_r = 0.0;
                const double ub_l = compute_psi_ub(cur.u0, um, cur.psi0, psi_m, cur.ang0, ang_m, d_l);
                const double ub_r = compute_psi_ub(um, cur.u1, psi_m, cur.psi1, ang_m, cur.ang1, d_r);
                if (!std::isfinite(ub_l) || ub_l > 1.0 + kStrictEps) {
                    pq.push({cur.u0, um, cur.psi0, psi_m, cur.ang0, ang_m, ub_l, cur.depth + 1});
                }
                if (!std::isfinite(ub_r) || ub_r > 1.0 + kStrictEps) {
                    pq.push({um, cur.u1, psi_m, cur.psi1, ang_m, cur.ang1, ub_r, cur.depth + 1});
                }
            }
        }

        if (seg_bad) {
            res.ok = false;
            res.bad_segments.push_back(seg);
            if (res.reason.empty()) {
                char buf[200];
                std::snprintf(buf, sizeof(buf),
                              "continuous manifold constraint %s on segment %d (pos_err=%.6f m, rot_err=%.6f rad)",
                              uncertified ? "could not be certified" : "violated", seg, worst_pe, worst_re);
                res.reason = buf;
            }
        }
    }
    return res;
}

// Manifold-Projected Bidirectional RRT (CBiRRT2) on an immutable WorldSnapshot.
// Stores every intermediate projected state on tree edges and uses unified constrained
// extend / connect / shortcutting / resampling.
inline RRTResultNative plan_constrained_birrt(
    const WorldSnapshot& snap,
    const Eigen::VectorXd& start,
    const Eigen::VectorXd& goal,
    const std::vector<PoseConstraintSpec>& constraints,
    const std::vector<uint8_t>& active_mask,
    double margin,
    double world_margin,
    double step_size,
    int max_iterations,
    double timeout_s,
    uint64_t rng_seed,
    bool verify_original,
    int shortcut_iters,
    int output_waypoints) {
    if (constraints.empty()) {
        return snap.plan_birrt(start, goal, active_mask, margin, world_margin,
                               step_size, max_iterations, timeout_s, rng_seed,
                               verify_original, shortcut_iters, output_waypoints);
    }
    manifold_detail::validate_manifold_specs(snap, constraints);

    const auto t0 = std::chrono::steady_clock::now();
    RRTResultNative out;
    const int J = snap.kin.num_joints();
    const Eigen::VectorXd margins = snap.pair_margins(margin, world_margin);
    const Eigen::VectorXd& lo = snap.kin.lower();
    const Eigen::VectorXd& hi = snap.kin.upper();
    const double check_res = std::min(0.05, step_size * 0.25);

    if (!snap.is_config_valid(start.data(), margins, verify_original) ||
        !snap.is_config_valid(goal.data(), margins, verify_original) ||
        !manifold_detail::is_on_manifold(snap, start.data(), constraints, 1.0) ||
        !manifold_detail::is_on_manifold(snap, goal.data(), constraints, 1.0)) {
        return out;
    }

    // Fast path: direct constrained geodesic trace from start to goal.
    {
        std::vector<Eigen::VectorXd> direct_chain;
        const int st = manifold_detail::trace_constrained_segment(
            snap, start, goal, constraints, active_mask, margins,
            step_size, check_res, verify_original, 64, direct_chain);
        if (st == 2) {
            std::vector<Eigen::VectorXd> full;
            full.reserve(direct_chain.size() + 2);
            full.push_back(start);
            if (direct_chain.empty()) {
                // Identical start == goal: emit at least 2 waypoints {start, goal} so downstream never reads OOB.
                full.push_back(goal);
            } else {
                for (auto& q : direct_chain) full.push_back(std::move(q));
            }
            out.dense_waypoints = full;
            if (output_waypoints >= 2) {
                std::vector<Eigen::VectorXd> resampled;
                const bool res_ok = manifold_detail::constrained_resample_path(
                    snap, full, output_waypoints, constraints, active_mask, margins, check_res, verify_original, resampled);
                out.waypoints = std::move(resampled);
                out.success = res_ok;
            } else {
                out.waypoints = std::move(full);
                out.success = true;
            }
            out.elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            return out;
        }
    }

    struct Node { Eigen::VectorXd q; int parent; };
    std::vector<Node> tree_a{{start, -1}}, tree_b{{goal, -1}};
    tree_a.reserve(2048);
    tree_b.reserve(2048);
    std::mt19937_64 rng(rng_seed);
    std::uniform_real_distribution<double> u01(0.0, 1.0);

    Eigen::VectorXd s_lo(J), s_hi(J);
    for (int j = 0; j < J; ++j) {
        if (std::isfinite(lo(j)) && std::isfinite(hi(j))) {
            s_lo(j) = lo(j);
            s_hi(j) = hi(j);
        } else {
            const double c = 0.5 * (start(j) + goal(j));
            const double span = std::max(M_PI, std::abs(goal(j) - start(j)));
            s_lo(j) = std::max(lo(j), c - span);
            s_hi(j) = std::min(hi(j), c + span);
        }
    }

    auto nearest = [](const std::vector<Node>& tree, const Eigen::VectorXd& target) -> int {
        int best = 0;
        double best_d2 = (tree[0].q - target).squaredNorm();
        for (int i = 1; i < static_cast<int>(tree.size()); ++i) {
            const double d2 = (tree[static_cast<size_t>(i)].q - target).squaredNorm();
            if (d2 < best_d2) {
                best_d2 = d2;
                best = i;
            }
        }
        return best;
    };

    std::vector<Eigen::VectorXd> step_chain;
    step_chain.reserve(64);

    // Constrained extend: takes up to `max_ext_steps` projected sub-steps toward target,
    // storing EVERY intermediate projected state directly on the tree!
    auto extend_tree = [&](std::vector<Node>& tree, const Eigen::VectorXd& target, int max_ext_steps) -> int {
        const int near_id = nearest(tree, target);
        const Eigen::VectorXd q_near = tree[static_cast<size_t>(near_id)].q;
        const int st = manifold_detail::trace_constrained_segment(
            snap, q_near, target, constraints, active_mask, margins,
            step_size, check_res, verify_original, max_ext_steps, step_chain);
        if (st == 0) return 0;
        int parent = near_id;
        for (auto& q_node : step_chain) {
            tree.push_back({std::move(q_node), parent});
            parent = static_cast<int>(tree.size()) - 1;
        }
        return st;
    };

    auto path_length = [](const std::vector<Eigen::VectorXd>& p, int i0, int i1) {
        double len = 0.0;
        for (int k = i0; k < i1; ++k) {
            len += (p[static_cast<size_t>(k + 1)] - p[static_cast<size_t>(k)]).norm();
        }
        return len;
    };

    bool swapped = false;
    Eigen::VectorXd sample(J);
    for (int it = 0; it < max_iterations; ++it) {
        if ((it & 7) == 0 && timeout_s > 0.0) {
            const double elapsed_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            if (elapsed_s >= timeout_s) break;
        }
        out.iterations = it + 1;
        auto& ta = swapped ? tree_b : tree_a;
        auto& tb = swapped ? tree_a : tree_b;

        for (int j = 0; j < J; ++j) {
            sample(j) = active_mask[static_cast<size_t>(j)]
                ? (s_lo(j) + u01(rng) * (s_hi(j) - s_lo(j)))
                : start(j);
        }
        // 30% of samples are projected onto the manifold (rejecting anti-parallel pi singularities)
        // to guide tree growth cleanly along lower-dimensional constraint manifolds.
        if (u01(rng) < 0.30) {
            Eigen::VectorXd q_proj;
            if (!manifold_detail::project_onto_manifold(
                    snap, constraints, active_mask, sample, q_proj,
                    std::numeric_limits<double>::infinity(), 0.5, 20)) {
                swapped = !swapped;
                continue;
            }
            sample = std::move(q_proj);
        }

        if (extend_tree(ta, sample, 4) != 0) {
            const Eigen::VectorXd target_q = ta.back().q;
            if (extend_tree(tb, target_q, 48) == 2) {
                std::vector<Eigen::VectorXd> pa, pb;
                for (int cur = static_cast<int>(tree_a.size()) - 1; cur >= 0; cur = tree_a[static_cast<size_t>(cur)].parent)
                    pa.push_back(tree_a[static_cast<size_t>(cur)].q);
                std::reverse(pa.begin(), pa.end());
                for (int cur = static_cast<int>(tree_b.size()) - 1; cur >= 0; cur = tree_b[static_cast<size_t>(cur)].parent)
                    pb.push_back(tree_b[static_cast<size_t>(cur)].q);
                if (!pb.empty() && (pa.back() - pb.front()).norm() < 1e-9) {
                    pa.insert(pa.end(), pb.begin() + 1, pb.end());
                } else {
                    pa.insert(pa.end(), pb.begin(), pb.end());
                }

                // Unified constrained shortcutting: replace pa[i..j] ONLY with verified shorter constrained trace!
                std::vector<Eigen::VectorXd> sc_chain;
                for (int sc = 0; sc < shortcut_iters && pa.size() > 2; ++sc) {
                    const int n = static_cast<int>(pa.size());
                    const int i = static_cast<int>(rng() % static_cast<uint64_t>(n - 2));
                    const int j = i + 2 + static_cast<int>(rng() % static_cast<uint64_t>(n - i - 2));
                    const double old_len = path_length(pa, i, j);
                    const int st = manifold_detail::trace_constrained_segment(
                        snap, pa[static_cast<size_t>(i)], pa[static_cast<size_t>(j)],
                        constraints, active_mask, margins,
                        step_size, check_res, verify_original, 32, sc_chain);
                    if (st == 2 && !sc_chain.empty()) {
                        double new_len = (sc_chain.front() - pa[static_cast<size_t>(i)]).norm();
                        for (size_t k = 0; k + 1 < sc_chain.size(); ++k) {
                            new_len += (sc_chain[k + 1] - sc_chain[k]).norm();
                        }
                        if (new_len < old_len - 1e-4) {
                            // sc_chain ends at pa[j]; replace (i+1 .. j) with sc_chain.
                            std::vector<Eigen::VectorXd> next_pa;
                            next_pa.reserve(pa.size() - static_cast<size_t>(j - i) + sc_chain.size());
                            next_pa.insert(next_pa.end(), pa.begin(), pa.begin() + i + 1);
                            for (auto& qn : sc_chain) next_pa.push_back(std::move(qn));
                            next_pa.insert(next_pa.end(), pa.begin() + j + 1, pa.end());
                            pa = std::move(next_pa);
                        }
                    }
                }

                if (pa.size() < 2) pa.push_back(goal);
                out.dense_waypoints = pa;
                if (output_waypoints >= 2) {
                    std::vector<Eigen::VectorXd> resampled;
                    const bool res_ok = manifold_detail::constrained_resample_path(
                        snap, pa, output_waypoints, constraints, active_mask, margins, check_res, verify_original, resampled);
                    out.waypoints = std::move(resampled);
                    out.success = res_ok;
                } else {
                    out.waypoints = std::move(pa);
                    out.success = true;
                }
                out.elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                return out;
            }
        }
        swapped = !swapped;
    }

    out.elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return out;
}

}  // namespace gobot::planning
