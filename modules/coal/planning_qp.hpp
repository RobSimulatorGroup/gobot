#pragma once

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace gobot::planning {

struct L1BoxQPResult {
    Eigen::VectorXd x;
    bool solved = false;
    int outer_iters = 0;
    int newton_iters = 0;
    double prim_res = 0.0;
    double dual_res = 0.0;
};

// Reusable workspace for solving the exact TrajOpt L1-hinge box-constrained QP:
//   min_{l <= x <= u}  0.5 * x^T H x + g^T x + penalty * sum_i max(0, v_i + a_i^T x)
// Eliminates all m slack variables analytically via Proximal Augmented Lagrangian +
// Semismooth Newton (ProxAL-SSN), solving only n_free x n_free SPD Cholesky systems.
class L1BoxQPSolver {
public:
    void setup(const Eigen::MatrixXd& H,
               const Eigen::VectorXd& g,
               const Eigen::MatrixXd& A,
               const Eigen::VectorXd& v,
               double penalty) {
        n_full_ = static_cast<int>(g.size());
        m_full_ = static_cast<int>(v.size());
        H_full_ = H;
        g_full_ = g;
        A_full_ = A;
        v_full_ = v;
        penalty_ = penalty;
        has_warm_ = false;
        warm_x_.setZero(n_full_);
        warm_lambda_.setZero(m_full_);
        warm_nu_.setZero(n_full_);
    }

    L1BoxQPResult solve(const Eigen::VectorXd& lower,
                        const Eigen::VectorXd& upper,
                        double eps_abs = 1e-5,
                        double eps_rel = 1e-5,
                        int max_outer = 28,
                        int max_newton = 25) {
        L1BoxQPResult out;
        if (n_full_ == 0) {
            out.x.resize(0);
            out.solved = true;
            return out;
        }
        if (lower.size() != n_full_ || upper.size() != n_full_) {
            return out;
        }
        for (int j = 0; j < n_full_; ++j) {
            if (!std::isfinite(lower(j)) || !std::isfinite(upper(j)) || lower(j) > upper(j) + 1e-12) {
                return out;
            }
        }

        // 1. Identify pinned (l_j == u_j) vs free variables.
        std::vector<int> free_idx;
        free_idx.reserve(static_cast<size_t>(n_full_));
        Eigen::VectorXd x_pin = Eigen::VectorXd::Zero(n_full_);
        bool any_pinned = false;
        bool any_nonzero_pin = false;
        for (int j = 0; j < n_full_; ++j) {
            if (upper(j) - lower(j) <= 1e-14) {
                const double c = 0.5 * (lower(j) + upper(j));
                x_pin(j) = c;
                any_pinned = true;
                if (std::abs(c) > 0.0) any_nonzero_pin = true;
            } else {
                free_idx.push_back(j);
            }
        }
        const int n = static_cast<int>(free_idx.size());
        if (n == 0) {
            out.x = x_pin;
            out.solved = true;
            return out;
        }

        // Extract reduced problem over free variables.
        Eigen::MatrixXd H(n, n);
        Eigen::VectorXd g(n), lo(n), hi(n);
        Eigen::VectorXd g_shift = any_nonzero_pin ? (g_full_ + H_full_ * x_pin) : g_full_;
        if (!any_pinned) {
            H = H_full_;
            g = g_shift;
            lo = lower;
            hi = upper;
        } else {
            for (int c = 0; c < n; ++c) {
                const int cj = free_idx[static_cast<size_t>(c)];
                g(c) = g_shift(cj);
                lo(c) = lower(cj);
                hi(c) = upper(cj);
                for (int r = 0; r < n; ++r) {
                    H(r, c) = H_full_(free_idx[static_cast<size_t>(r)], cj);
                }
            }
        }

        // 2. Filter strictly inactive inequality rows over the trust-region box [lo, hi]
        //    and normalize active rows to unit L2 norm for optimal conditioning.
        std::vector<int> kept_rows;
        kept_rows.reserve(static_cast<size_t>(m_full_));
        std::vector<double> row_norms;
        row_norms.reserve(static_cast<size_t>(m_full_));
        Eigen::VectorXd v_shift = (m_full_ > 0 && any_nonzero_pin) ? (v_full_ + A_full_ * x_pin) : v_full_;

        for (int i = 0; i < m_full_; ++i) {
            double worst = v_shift(i);
            double norm2 = 0.0;
            for (int c = 0; c < n; ++c) {
                const double a = A_full_(i, free_idx[static_cast<size_t>(c)]);
                worst += (a > 0.0) ? (a * hi(c)) : (a * lo(c));
                norm2 += a * a;
            }
            if (worst > 0.0 && norm2 > 1e-24) {
                kept_rows.push_back(i);
                row_norms.push_back(std::sqrt(norm2));
            }
        }
        const int m = static_cast<int>(kept_rows.size());

        // Scale objective so penalty_eff <= 10.0, keeping Hessian & duals well-conditioned.
        const double h_diag_mean = std::max(1e-6, H.diagonal().mean());
        const double obj_scale = std::max({1.0, penalty_ / 10.0, h_diag_mean / 100.0});
        const double inv_scale = 1.0 / obj_scale;
        H *= inv_scale;
        g *= inv_scale;
        const double pen_s = penalty_ * inv_scale;

        Eigen::MatrixXd A(m, n);
        Eigen::VectorXd v(m), mu(m), inv_norm(m);
        for (int r = 0; r < m; ++r) {
            const int orig = kept_rows[static_cast<size_t>(r)];
            const double nr = row_norms[static_cast<size_t>(r)];
            const double inv_n = 1.0 / nr;
            inv_norm(r) = inv_n;
            v(r) = v_shift(orig) * inv_n;
            mu(r) = pen_s * nr;
            for (int c = 0; c < n; ++c) {
                A(r, c) = A_full_(orig, free_idx[static_cast<size_t>(c)]) * inv_n;
            }
        }

        // Fast path when m == 0 and unconstrained minimizer is already inside [lo, hi].
        if (m == 0) {
            Eigen::MatrixXd H_reg = H;
            H_reg.diagonal().array() += 1e-12;
            Eigen::LLT<Eigen::MatrixXd> llt(H_reg);
            if (llt.info() == Eigen::Success) {
                Eigen::VectorXd x0 = llt.solve(-g);
                if (x0.allFinite() && (x0.array() >= lo.array() - 1e-12).all() &&
                    (x0.array() <= hi.array() + 1e-12).all()) {
                    x0 = x0.cwiseMax(lo).cwiseMin(hi);
                    out.x = x_pin;
                    for (int c = 0; c < n; ++c) out.x(free_idx[static_cast<size_t>(c)]) = x0(c);
                    out.solved = true;
                    out.newton_iters = 1;
                    warm_x_ = out.x;
                    warm_nu_.setZero(n_full_);
                    if (m_full_ > 0) warm_lambda_.setZero(m_full_);
                    has_warm_ = true;
                    return out;
                }
            }
        }

        // 3. Initialize primal & dual iterates (warm-started across trust-region retries).
        Eigen::VectorXd x(n), nu(n), lam(m);
        if (has_warm_) {
            for (int c = 0; c < n; ++c) {
                const int cj = free_idx[static_cast<size_t>(c)];
                x(c) = std::clamp(warm_x_(cj), lo(c), hi(c));
                nu(c) = warm_nu_(cj) * inv_scale;
            }
            for (int r = 0; r < m; ++r) {
                const int orig = kept_rows[static_cast<size_t>(r)];
                lam(r) = std::clamp(warm_lambda_(orig) * row_norms[static_cast<size_t>(r)] * inv_scale, 0.0, mu(r));
            }
        } else {
            x.setZero(n);
            for (int c = 0; c < n; ++c) x(c) = std::clamp(0.0, lo(c), hi(c));
            nu.setZero(n);
            lam.setZero(m);
        }

        const double h_scale = std::max(1e-3, H.diagonal().mean());
        double rho_A = std::clamp(4.0 * h_scale, 1.0, 1e3);
        double rho_B = std::clamp(20.0 * h_scale, 10.0, 1e4);
        const double eps_prox = 1e-7 * std::max(1.0, h_scale);
        const double g_inf = std::max(1.0, g.lpNorm<Eigen::Infinity>());

        // Preallocated work buffers (zero allocations inside inner Newton loop).
        Eigen::VectorXd x_k = x;
        Eigen::VectorXd r_vec(m), w_vec(m), z_vec(n);
        Eigen::VectorXd Hx(n), grad(n), d(n), x_try(n), r_try(m), Ad(m), Hd(n);
        Eigen::MatrixXd M(n, n);
        Eigen::MatrixXd A_act;
        std::vector<int> act_rows;
        act_rows.reserve(static_cast<size_t>(m));

        auto eval_phi = [&](const Eigen::VectorXd& xv,
                            const Eigen::VectorXd& Hxv,
                            const Eigen::VectorXd& rv) -> double {
            double val = 0.5 * xv.dot(Hxv) + g.dot(xv) + 0.5 * eps_prox * (xv - x_k).squaredNorm();
            const double inv_2rhoA = 0.5 / rho_A;
            for (int i = 0; i < m; ++i) {
                const double y = lam(i) + rho_A * rv(i);
                if (y <= 0.0) {
                    // Constant term omitted; only differences in Phi matter for line search.
                } else if (y < mu(i)) {
                    val += y * y * inv_2rhoA;
                } else {
                    val += mu(i) * rv(i) + (2.0 * mu(i) * lam(i) - mu(i) * mu(i)) * inv_2rhoA;
                }
            }
            const double inv_2rhoB = 0.5 / rho_B;
            for (int j = 0; j < n; ++j) {
                const double zu = nu(j) + rho_B * (xv(j) - hi(j));
                if (zu > 0.0) {
                    val += zu * zu * inv_2rhoB;
                } else {
                    const double zl = nu(j) + rho_B * (xv(j) - lo(j));
                    if (zl < 0.0) val += zl * zl * inv_2rhoB;
                }
            }
            return val;
        };

        auto eval_grad_and_multipliers = [&](const Eigen::VectorXd& xv,
                                             const Eigen::VectorXd& Hxv,
                                             const Eigen::VectorXd& rv) {
            for (int i = 0; i < m; ++i) {
                w_vec(i) = std::clamp(lam(i) + rho_A * rv(i), 0.0, mu(i));
            }
            for (int j = 0; j < n; ++j) {
                const double zu = nu(j) + rho_B * (xv(j) - hi(j));
                if (zu > 0.0) {
                    z_vec(j) = zu;
                } else {
                    const double zl = nu(j) + rho_B * (xv(j) - lo(j));
                    z_vec(j) = (zl < 0.0) ? zl : 0.0;
                }
            }
            grad.noalias() = Hxv + g + eps_prox * (xv - x_k) + z_vec;
            if (m > 0) grad.noalias() += A.transpose() * w_vec;
        };

        double prev_box_viol = std::numeric_limits<double>::infinity();
        double prev_hinge_viol = std::numeric_limits<double>::infinity();
        int total_newton = 0;

        for (int outer = 0; outer < max_outer; ++outer) {
            out.outer_iters = outer + 1;
            x_k = x;
            Hx.noalias() = H * x;
            if (m > 0) r_vec.noalias() = A * x + v;

            const double inner_tol = std::max(1e-9, std::min(1e-4, 0.25 * eps_abs)) * g_inf;

            for (int ssn = 0; ssn < max_newton; ++ssn) {
                eval_grad_and_multipliers(x, Hx, r_vec);
                const double g_norm = grad.lpNorm<Eigen::Infinity>();
                if (g_norm <= inner_tol) break;

                ++total_newton;

                // Assemble generalized Hessian M = H + eps*I + rho_A * A_act^T * A_act + rho_B * D_box.
                M = H;
                for (int j = 0; j < n; ++j) {
                    double diag_add = eps_prox;
                    const double zu = nu(j) + rho_B * (x(j) - hi(j));
                    const double zl = nu(j) + rho_B * (x(j) - lo(j));
                    if (zu > 0.0 || zl < 0.0) diag_add += rho_B;
                    M(j, j) += diag_add;
                }
                act_rows.clear();
                for (int i = 0; i < m; ++i) {
                    const double y = lam(i) + rho_A * r_vec(i);
                    if (y > 0.0 && y < mu(i)) act_rows.push_back(i);
                }
                if (!act_rows.empty()) {
                    const int ka = static_cast<int>(act_rows.size());
                    A_act.resize(ka, n);
                    for (int k = 0; k < ka; ++k) A_act.row(k) = A.row(act_rows[static_cast<size_t>(k)]);
                    M.selfadjointView<Eigen::Lower>().rankUpdate(A_act.transpose(), rho_A);
                }

                Eigen::LLT<Eigen::MatrixXd> llt(M);
                if (llt.info() != Eigen::Success) {
                    // Add small Levenberg-Marquardt damping if near-singular.
                    const double lm = 1e-6 * std::max(1.0, M.diagonal().maxCoeff());
                    M.diagonal().array() += lm;
                    llt.compute(M);
                    if (llt.info() != Eigen::Success) break;
                }
                d = llt.solve(-grad);
                if (!d.allFinite()) break;

                const double dir_deriv = grad.dot(d);
                if (dir_deriv >= 0.0) break;

                // Precompute directional products so Armijo line search costs O(n + m) per trial.
                Hd.noalias() = H * d;
                if (m > 0) Ad.noalias() = A * d;
                const double phi0 = eval_phi(x, Hx, r_vec);
                double alpha = 1.0;
                bool stepped = false;
                for (int ls = 0; ls < 16; ++ls) {
                    x_try.noalias() = x + alpha * d;
                    const Eigen::VectorXd Hx_try = Hx + alpha * Hd;
                    if (m > 0) r_try.noalias() = r_vec + alpha * Ad;
                    const double phi_try = eval_phi(x_try, Hx_try, r_try);
                    if (phi_try <= phi0 + 1e-4 * alpha * dir_deriv + 1e-14 * std::abs(phi0)) {
                        x = x_try;
                        Hx = Hx_try;
                        if (m > 0) r_vec = r_try;
                        stepped = true;
                        break;
                    }
                    alpha *= 0.5;
                }
                if (!stepped) break;
                if (alpha * d.lpNorm<Eigen::Infinity>() <= 1e-12) break;
            }

            // Update dual multipliers from the exact Moreau/box projections.
            eval_grad_and_multipliers(x, Hx, r_vec);
            lam = w_vec;
            nu = z_vec;

            // Evaluate exact primal, complementarity, and dual stationarity residuals.
            double box_viol = 0.0;
            for (int j = 0; j < n; ++j) {
                box_viol = std::max(box_viol, std::max(x(j) - hi(j), lo(j) - x(j)));
                if (nu(j) > 1e-9) box_viol = std::max(box_viol, std::abs(x(j) - hi(j)));
                else if (nu(j) < -1e-9) box_viol = std::max(box_viol, std::abs(x(j) - lo(j)));
            }
            double hinge_viol = 0.0;
            for (int i = 0; i < m; ++i) {
                // Convert back to original unnormalized residual units for accurate stopping.
                const double r_orig = r_vec(i) * row_norms[static_cast<size_t>(i)];
                if (lam(i) <= 1e-12 * mu(i)) {
                    hinge_viol = std::max(hinge_viol, std::max(0.0, r_orig));
                } else if (lam(i) >= (1.0 - 1e-12) * mu(i)) {
                    hinge_viol = std::max(hinge_viol, std::max(0.0, -r_orig));
                } else {
                    hinge_viol = std::max(hinge_viol, std::abs(r_orig));
                }
            }
            // Stationarity residual without the proximal term.
            Eigen::VectorXd stat = Hx + g + nu;
            if (m > 0) stat.noalias() += A.transpose() * lam;
            const double dual_res = stat.lpNorm<Eigen::Infinity>();
            const double prim_res = std::max(box_viol, hinge_viol);

            const double x_inf = std::max(1.0, x.lpNorm<Eigen::Infinity>());
            const double prim_tol = eps_abs + eps_rel * x_inf;
            const double dual_tol = eps_abs + eps_rel * g_inf;

            out.prim_res = prim_res;
            out.dual_res = dual_res;
            out.newton_iters = total_newton;

            if (prim_res <= prim_tol && dual_res <= dual_tol) {
                out.solved = true;
                break;
            }

            if (box_viol > 0.25 * prev_box_viol && box_viol > prim_tol) {
                rho_B = std::min(1e9, rho_B * 10.0);
            }
            if (hinge_viol > 0.25 * prev_hinge_viol && hinge_viol > prim_tol) {
                rho_A = std::min(1e9, rho_A * 10.0);
            }
            prev_box_viol = box_viol;
            prev_hinge_viol = hinge_viol;
        }

        // Accept slightly inaccurate solutions that are finite and well within SQP tolerance.
        if (!out.solved && x.allFinite() && out.prim_res <= 1e-3 && out.dual_res <= 1e-2 * g_inf) {
            out.solved = true;
        }
        if (!out.solved || !x.allFinite()) {
            out.solved = false;
            return out;
        }

        // Clamp strictly to box bounds and reconstruct full vector + save warm start.
        x = x.cwiseMax(lo).cwiseMin(hi);
        out.x = x_pin;
        warm_nu_.setZero(n_full_);
        for (int c = 0; c < n; ++c) {
            const int cj = free_idx[static_cast<size_t>(c)];
            out.x(cj) = x(c);
            warm_nu_(cj) = nu(c) * obj_scale;
        }
        warm_x_ = out.x;
        if (m_full_ > 0) {
            warm_lambda_.setZero(m_full_);
            for (int r = 0; r < m; ++r) {
                const int orig = kept_rows[static_cast<size_t>(r)];
                warm_lambda_(orig) = lam(r) * inv_norm(r) * obj_scale;
            }
        }
        has_warm_ = true;
        return out;
    }

private:
    int n_full_ = 0;
    int m_full_ = 0;
    Eigen::MatrixXd H_full_;
    Eigen::VectorXd g_full_;
    Eigen::MatrixXd A_full_;
    Eigen::VectorXd v_full_;
    double penalty_ = 10.0;

    bool has_warm_ = false;
    Eigen::VectorXd warm_x_;
    Eigen::VectorXd warm_lambda_;
    Eigen::VectorXd warm_nu_;
};

inline L1BoxQPResult solve_l1_box_qp(const Eigen::MatrixXd& H,
                                     const Eigen::VectorXd& g,
                                     const Eigen::MatrixXd& A,
                                     const Eigen::VectorXd& v,
                                     const Eigen::VectorXd& lower,
                                     const Eigen::VectorXd& upper,
                                     double penalty,
                                     double eps_abs = 1e-5,
                                     double eps_rel = 1e-5) {
    L1BoxQPSolver solver;
    solver.setup(H, g, A, v, penalty);
    return solver.solve(lower, upper, eps_abs, eps_rel);
}

}  // namespace gobot::planning
