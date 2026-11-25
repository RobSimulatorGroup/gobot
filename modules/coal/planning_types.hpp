#pragma once

#include <coal/BV/OBBRSS.h>
#include <coal/BVH/BVH_model.h>
#include <coal/collision.h>
#include <coal/distance.h>
#include <coal/shape/convex.h>
#include <coal/shape/geometric_shapes.h>

#include <Eigen/Core>
#include <Eigen/Dense>

#ifdef GOBOT_PLANNING_HAS_TBB
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/task_arena.h>
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

namespace gobot::planning {

using Geometry = std::shared_ptr<coal::CollisionGeometry>;
using Vec3 = Eigen::Vector3d;
using Mat3 = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>;
using Frame = Eigen::Matrix<double, 3, 4, Eigen::RowMajor>;

class ThreadPool {
public:
    static ThreadPool& instance() {
        static ThreadPool pool;
        return pool;
    }

    unsigned int max_threads() const {
#ifdef GOBOT_PLANNING_HAS_TBB
        return static_cast<unsigned int>(std::max(1, tbb::this_task_arena::max_concurrency()));
#else
        return num_threads_;
#endif
    }

    void parallel_for(int64_t count, int64_t min_chunk,
                      const std::function<void(int64_t, int64_t)>& fn) {
        if (count <= 0) return;
        const int64_t grain = std::max<int64_t>(1, min_chunk);
#ifdef GOBOT_PLANNING_HAS_TBB
        if (count < grain * 2) {
            fn(0, count);
            return;
        }
        tbb::parallel_for(
            tbb::blocked_range<int64_t>(0, count, static_cast<size_t>(grain)),
            [&](const tbb::blocked_range<int64_t>& r) {
                fn(r.begin(), r.end());
            });
        return;
#else
        if (num_threads_ <= 1 || count < grain * 2) {
            fn(0, count);
            return;
        }
        std::unique_lock<std::mutex> dispatch_lock(dispatch_mutex_, std::try_to_lock);
        if (!dispatch_lock.owns_lock()) {
            fn(0, count);
            return;
        }
        const int64_t max_chunks = std::min<int64_t>(num_threads_, (count + grain - 1) / grain);
        if (max_chunks <= 1) {
            fn(0, count);
            return;
        }
        const int64_t chunk = (count + max_chunks - 1) / max_chunks;
        std::vector<std::exception_ptr> errors(static_cast<size_t>(max_chunks), nullptr);
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            active_fn_ = &fn;
            active_errors_ = &errors;
            total_count_ = count;
            chunk_size_ = chunk;
            active_chunks_ = max_chunks;
            pending_workers_ = max_chunks - 1;
            ++generation_;
        }
        cv_start_.notify_all();

        try {
            fn(0, std::min(chunk, count));
        } catch (...) {
            errors[0] = std::current_exception();
        }

        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            cv_done_.wait(lock, [this] { return pending_workers_ == 0; });
            active_fn_ = nullptr;
            active_errors_ = nullptr;
        }
        for (const auto& err : errors) {
            if (err) std::rethrow_exception(err);
        }
#endif
    }

private:
    ThreadPool() {
#ifndef GOBOT_PLANNING_HAS_TBB
        const unsigned int hw = std::thread::hardware_concurrency();
        num_threads_ = std::min(16u, std::max(1u, hw));
        workers_.reserve(num_threads_ > 1 ? num_threads_ - 1 : 0);
        for (unsigned int id = 1; id < num_threads_; ++id) {
            workers_.emplace_back([this, id] { worker_loop(static_cast<int64_t>(id)); });
        }
#endif
    }

    ~ThreadPool() {
#ifndef GOBOT_PLANNING_HAS_TBB
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            stop_ = true;
            ++generation_;
        }
        cv_start_.notify_all();
        for (auto& worker : workers_) {
            if (worker.joinable()) worker.join();
        }
#endif
    }

    void worker_loop(int64_t id) {
        uint64_t seen = 0;
        while (true) {
            const std::function<void(int64_t, int64_t)>* fn = nullptr;
            std::vector<std::exception_ptr>* errors = nullptr;
            int64_t begin = 0, end = 0;
            bool run = false;
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                cv_start_.wait(lock, [this, seen] { return generation_ != seen; });
                seen = generation_;
                if (stop_) return;
                if (id < active_chunks_) {
                    fn = active_fn_;
                    errors = active_errors_;
                    begin = id * chunk_size_;
                    end = std::min(begin + chunk_size_, total_count_);
                    run = true;
                }
            }
            if (run) {
                try {
                    if (begin < end) (*fn)(begin, end);
                } catch (...) {
                    (*errors)[static_cast<size_t>(id)] = std::current_exception();
                }
                std::lock_guard<std::mutex> lock(queue_mutex_);
                if (--pending_workers_ == 0) cv_done_.notify_one();
            }
        }
    }

    unsigned int num_threads_ = 1;
    std::vector<std::thread> workers_;
    std::mutex dispatch_mutex_;
    std::mutex queue_mutex_;
    std::condition_variable cv_start_;
    std::condition_variable cv_done_;
    const std::function<void(int64_t, int64_t)>* active_fn_ = nullptr;
    std::vector<std::exception_ptr>* active_errors_ = nullptr;
    int64_t total_count_ = 0;
    int64_t chunk_size_ = 0;
    int64_t active_chunks_ = 0;
    int64_t pending_workers_ = 0;
    uint64_t generation_ = 0;
    bool stop_ = false;
};

inline Mat3 skew_matrix(const Vec3& v) {
    Mat3 s;
    s << 0.0, -v(2), v(1),
         v(2), 0.0, -v(0),
         -v(1), v(0), 0.0;
    return s;
}

inline Vec3 rotation_log(const Mat3& m) {
    // Shortest rotation vector, well-conditioned near pi.
    const double w = 1.0 + m(0, 0) + m(1, 1) + m(2, 2);
    const double x = 1.0 + m(0, 0) - m(1, 1) - m(2, 2);
    const double y = 1.0 - m(0, 0) + m(1, 1) - m(2, 2);
    const double z = 1.0 - m(0, 0) - m(1, 1) + m(2, 2);
    double q[4];
    if (w >= x && w >= y && w >= z) {
        q[0] = w; q[1] = m(2, 1) - m(1, 2); q[2] = m(0, 2) - m(2, 0); q[3] = m(1, 0) - m(0, 1);
    } else if (x >= y && x >= z) {
        q[0] = m(2, 1) - m(1, 2); q[1] = x; q[2] = m(0, 1) + m(1, 0); q[3] = m(0, 2) + m(2, 0);
    } else if (y >= z) {
        q[0] = m(0, 2) - m(2, 0); q[1] = m(0, 1) + m(1, 0); q[2] = y; q[3] = m(1, 2) + m(2, 1);
    } else {
        q[0] = m(1, 0) - m(0, 1); q[1] = m(0, 2) + m(2, 0); q[2] = m(1, 2) + m(2, 1); q[3] = z;
    }
    if (q[0] < 0.0) {
        for (int i = 0; i < 4; ++i) q[i] = -q[i];
    }
    const Vec3 vec(q[1], q[2], q[3]);
    const double norm = vec.norm();
    constexpr double eps = std::numeric_limits<double>::epsilon();
    const double scale = (norm > eps)
        ? (2.0 * std::atan2(norm, q[0]) / norm)
        : (2.0 / std::max(q[0], eps));
    return vec * scale;
}

inline Mat3 so3_left_jacobian_inv(const Vec3& phi) {
    const double theta = phi.norm();
    const Mat3 S = skew_matrix(phi);
    const Mat3 S2 = S * S;
    if (theta < 1e-5) {
        return Mat3::Identity() - 0.5 * S + (1.0 / 12.0) * S2;
    }
    const double half = 0.5 * theta;
    const double cot_half = std::cos(half) / std::sin(half);
    const double coeff = (1.0 - half * cot_half) / (theta * theta);
    return Mat3::Identity() - 0.5 * S + coeff * S2;
}

inline coal::Transform3s transform_from(const double* data) {
    Eigen::Map<const Frame> frame(data);
    if (!frame.allFinite()) throw std::invalid_argument("nonfinite shape transform");
    return coal::Transform3s(frame.leftCols<3>(), frame.col(3));
}

inline coal::AABB world_aabb(const Geometry& geometry, const coal::Transform3s& pose) {
    const auto& local = geometry->aabb_local;
    const coal::Vec3s center = pose.transform(local.center());
    const coal::Vec3s half = pose.getRotation().cwiseAbs() * ((local.max_ - local.min_) * 0.5);
    return coal::AABB(center - half, center + half);
}

inline void transform_aabb(const double* frame, const Vec3& lc, const Vec3& lh,
                           double* out_lo, double* out_hi) {
    const double eps16 = 16.0 * std::numeric_limits<double>::epsilon();
    const double cx = lc(0), cy = lc(1), cz = lc(2);
    const double hx = lh(0), hy = lh(1), hz = lh(2);
    const double acx = std::abs(cx), acy = std::abs(cy), acz = std::abs(cz);
    for (int a = 0; a < 3; ++a) {
        const double r0 = frame[a * 4 + 0], r1 = frame[a * 4 + 1], r2 = frame[a * 4 + 2], t = frame[a * 4 + 3];
        const double ar0 = std::abs(r0), ar1 = std::abs(r1), ar2 = std::abs(r2);
        const double center = r0 * cx + r1 * cy + r2 * cz + t;
        const double extent0 = ar0 * hx + ar1 * hy + ar2 * hz;
        const double scale = std::abs(t) + (ar0 * acx + ar1 * acy + ar2 * acz) + extent0;
        const double extent = extent0 + eps16 * (1.0 + scale);
        out_lo[a] = center - extent;
        out_hi[a] = center + extent;
    }
}

struct CollisionShape {
    std::string label;
    int kind = 0;  // 1=box, 2=sphere, 3=cylinder, 4=capsule, 5=mesh, 6=plane
    Geometry convex;
    Geometry original;
    Vec3 convex_center = Vec3::Zero();
    Vec3 convex_half = Vec3::Zero();
    Vec3 orig_center = Vec3::Zero();
    Vec3 orig_half = Vec3::Zero();

    // Primitive metadata for optional GPU backends (e.g. MuJoCo Warp).
    Vec3 box_half_size = Vec3::Zero();
    double radius = 0.0;
    double height = 0.0;
    std::vector<Vec3> hull_points;

    void finalize_bounds() {
        if (!original) original = convex;
        convex->computeLocalAABB();
        original->computeLocalAABB();
        if (kind != 6) {
            const auto& cb = convex->aabb_local;
            convex_center = (cb.min_ + cb.max_) * 0.5;
            convex_half = (cb.max_ - cb.min_) * 0.5;
            const auto& ob = original->aabb_local;
            orig_center = (ob.min_ + ob.max_) * 0.5;
            orig_half = (ob.max_ - ob.min_) * 0.5;
        }
    }
};

struct WorldShape {
    std::string name;
    std::string link_key;  // canonical "robot::link" or "::obstacle"
    int body_id = -1;      // >= 0 if mobile, -1 if static
    Frame local_frame = Frame::Zero();  // link-local if mobile, world if static
    int layer = 1;
    int mask = 1;
    bool is_attached = false;
    std::unordered_set<std::string> attached_touch_links;
    std::shared_ptr<CollisionShape> shape;
    // Cached static world AABBs (only used when body_id < 0).
    Vec3 fixed_lo_convex = Vec3::Zero();
    Vec3 fixed_hi_convex = Vec3::Zero();
    Vec3 fixed_lo_orig = Vec3::Zero();
    Vec3 fixed_hi_orig = Vec3::Zero();

    void update_fixed_bounds() {
        if (body_id >= 0 || shape->kind == 6) return;
        transform_aabb(local_frame.data(), shape->convex_center, shape->convex_half,
                       fixed_lo_convex.data(), fixed_hi_convex.data());
        transform_aabb(local_frame.data(), shape->orig_center, shape->orig_half,
                       fixed_lo_orig.data(), fixed_hi_orig.data());
    }
};

struct PoseConstraintSpec {
    int link_id = -1;
    bool has_position = false;
    bool has_rotation = false;
    Vec3 position = Vec3::Zero();
    Mat3 rotation = Mat3::Identity();
    Vec3 pos_tol = Vec3::Constant(1e-4);
    Vec3 rot_tol = Vec3::Constant(1e-3);
    double weight = 1.0;

    // Count of scalar residual components for Jacobian/SQP assembly.
    // Free axes (tol == inf) are omitted.
    int residual_dim() const {
        int d = 0;
        if (has_position) {
            for (int a = 0; a < 3; ++a) if (std::isfinite(pos_tol(a))) ++d;
        }
        if (has_rotation) {
            int free_count = 0;
            for (int a = 0; a < 3; ++a) if (!std::isfinite(rot_tol(a))) ++free_count;
            if (free_count == 0) d += 3;
            else if (free_count == 1) d += 2;  // singularity-free axis alignment
            else if (free_count == 2) d += 1;  // single constrained rotation component
        }
        return d;
    }
};

// Multi-branch fixed-or-mutable-base kinematic tree.
class KinematicsModel {
public:
    struct Branch {
        std::string robot_name;
        int root_body = 0;
        Vec3 root_pos = Vec3::Zero();
        Mat3 root_rot = Mat3::Identity();
    };

    int num_bodies() const { return static_cast<int>(parent_.size()); }
    int num_joints() const { return static_cast<int>(lower_.size()); }

    void initialize(
        std::vector<int> parent,
        std::vector<int> kinds,
        std::vector<int> joint_idx,
        std::vector<Mat3> a_rot,
        std::vector<Vec3> a_pos,
        std::vector<Mat3> b_rot,
        std::vector<Vec3> b_pos,
        std::vector<Vec3> axes,
        std::vector<Branch> branches,
        std::vector<std::string> body_names,
        std::vector<std::string> joint_names,
        Eigen::VectorXd lower,
        Eigen::VectorXd upper,
        Eigen::VectorXd velocity_limit,
        Eigen::VectorXd acceleration_limit,
        Eigen::VectorXd initial_q) {
        parent_ = std::move(parent);
        kinds_ = std::move(kinds);
        joint_idx_ = std::move(joint_idx);
        a_rot_ = std::move(a_rot);
        a_pos_ = std::move(a_pos);
        b_rot_ = std::move(b_rot);
        b_pos_ = std::move(b_pos);
        axes_ = std::move(axes);
        branches_ = std::move(branches);
        body_names_ = std::move(body_names);
        joint_names_ = std::move(joint_names);
        lower_ = std::move(lower);
        upper_ = std::move(upper);
        velocity_limit_ = std::move(velocity_limit);
        acceleration_limit_ = std::move(acceleration_limit);
        initial_q_ = std::move(initial_q);

        const int B = num_bodies();
        const int J = num_joints();
        skew_.resize(static_cast<size_t>(B));
        skew2_.resize(static_cast<size_t>(B));
        for (int i = 0; i < B; ++i) {
            skew_[i] = skew_matrix(axes_[i]);
            skew2_[i] = skew_[i] * skew_[i];
        }
        body_branch_.assign(static_cast<size_t>(B), -1);
        for (size_t br = 0; br < branches_.size(); ++br) {
            body_branch_[static_cast<size_t>(branches_[br].root_body)] = static_cast<int>(br);
        }
        body_ancestor_.assign(static_cast<size_t>(B * J), 0);
        joint_prismatic_.assign(static_cast<size_t>(J), 0);
        body_joint_base_reach_.assign(static_cast<size_t>(B * J), 0.0);
        body_joint_pris_desc_.assign(static_cast<size_t>(B * J), {});
        for (int b = 0; b < B; ++b) {
            if (joint_idx_[b] >= 0 && kinds_[b] == 3) {
                joint_prismatic_[static_cast<size_t>(joint_idx_[b])] = 1;
            }
            // Walk up from body b to root, accumulating exact kinematic chain reach bounds.
            double acc_reach = 0.0;
            std::vector<int> pris_below;
            int cur = b;
            while (cur >= 0) {
                const int j = joint_idx_[cur];
                if (j >= 0) {
                    const size_t slot = static_cast<size_t>(b * J + j);
                    body_ancestor_[slot] = 1;
                    // From joint j's origin to body cur's origin is at most ||b_pos_[cur]||;
                    // downstream links add acc_reach + sum(|q_p|) for downstream prismatic joints.
                    body_joint_base_reach_[slot] = acc_reach + b_pos_[static_cast<size_t>(cur)].norm();
                    body_joint_pris_desc_[slot] = pris_below;
                }
                if (parent_[cur] >= 0) {
                    acc_reach += a_pos_[static_cast<size_t>(cur)].norm() + b_pos_[static_cast<size_t>(cur)].norm();
                    if (j >= 0 && kinds_[cur] == 3) {
                        pris_below.push_back(j);
                    }
                }
                cur = parent_[cur];
            }
        }
    }

    // Exact rotation-invariant Lipschitz upper bound R_max(shape, j) on point displacement
    // per unit joint motion dq_j for any point in a shape attached to `body_id`.
    // `shape_body_radius` must be >= max_{x in shape} ||x_in_body_frame||_2.
    // `pris_abs_max` (optional length J) bounds |q_p| for any downstream prismatic joints.
    void body_lipschitz_weights(int body_id, double shape_body_radius,
                                const double* pris_abs_max, double* out_weights) const {
        const int J = num_joints();
        if (body_id < 0) {
            std::fill(out_weights, out_weights + J, 0.0);
            return;
        }
        const uint8_t* anc = body_ancestor_.data() + body_id * J;
        for (int j = 0; j < J; ++j) {
            if (!anc[j]) {
                out_weights[j] = 0.0;
                continue;
            }
            if (joint_prismatic_[static_cast<size_t>(j)]) {
                out_weights[j] = 1.0;
                continue;
            }
            const size_t slot = static_cast<size_t>(body_id * J + j);
            double r = body_joint_base_reach_[slot] + shape_body_radius;
            for (const int pj : body_joint_pris_desc_[slot]) {
                if (pris_abs_max != nullptr) {
                    r += pris_abs_max[pj];
                } else {
                    const double bound = std::max(std::abs(lower_(pj)), std::abs(upper_(pj)));
                    r += std::isfinite(bound) ? bound : 1e6;
                }
            }
            out_weights[j] = r;
        }
    }

    void set_branch_pose(int branch_idx, const Vec3& pos, const Mat3& rot) {
        if (branch_idx < 0 || branch_idx >= static_cast<int>(branches_.size()))
            throw std::out_of_range("invalid robot branch index");
        branches_[static_cast<size_t>(branch_idx)].root_pos = pos;
        branches_[static_cast<size_t>(branch_idx)].root_rot = rot;
    }

    // Single-configuration FK into caller-provided buffers (zero heap alloc on hot path).
    void fk_single(const double* q, Vec3* pos, Mat3* rot,
                   Vec3* j_origin = nullptr, Vec3* j_axis = nullptr) const {
        const int B = num_bodies();
        const Mat3 eye3 = Mat3::Identity();
        for (int i = 0; i < B; ++i) {
            const int p = parent_[i];
            if (p < 0) {
                const auto& br = branches_[static_cast<size_t>(body_branch_[i])];
                pos[i] = br.root_pos;
                rot[i] = br.root_rot;
                continue;
            }
            const int j = joint_idx_[i];
            const int kind = kinds_[i];
            const Mat3 anchor_r = rot[p] * a_rot_[i];
            Vec3 anchor_p = pos[p] + rot[p] * a_pos_[i];
            Mat3 moving_r = anchor_r;
            if (kind == 1 || kind == 2) {
                const double angle = q[j];
                moving_r.noalias() = anchor_r * (eye3 + std::sin(angle) * skew_[i] + (1.0 - std::cos(angle)) * skew2_[i]);
            }
            Vec3 world_axis;
            if (kind == 3 || (j_origin != nullptr && j >= 0)) {
                world_axis.noalias() = anchor_r * axes_[i];
            }
            if (j_origin != nullptr && j >= 0) {
                j_origin[j] = anchor_p;
                j_axis[j] = world_axis;
            }
            if (kind == 3) {
                anchor_p.noalias() += world_axis * q[j];
            }
            pos[i].noalias() = anchor_p + moving_r * b_pos_[i];
            rot[i].noalias() = moving_r * b_rot_[i];
        }
    }

    // Evaluate PoseConstraint residual and analytic Jacobian for a single configuration.
    // Returns true if residual satisfies constraint tolerances.
    bool eval_pose_constraint(const PoseConstraintSpec& c,
                              const Vec3* body_pos, const Mat3* body_rot,
                              const Vec3* j_origin, const Vec3* j_axis,
                              Eigen::VectorXd& residual,
                              Eigen::MatrixXd* jacobian = nullptr) const {
        const int J = num_joints();
        const int dim = c.residual_dim();
        residual.resize(dim);
        if (jacobian) jacobian->setZero(dim, J);
        const Vec3& p = body_pos[c.link_id];
        const Mat3& R = body_rot[c.link_id];
        const uint8_t* anc = body_ancestor_.data() + c.link_id * J;

        int row = 0;
        bool ok = true;

        if (c.has_position) {
            const Vec3 dp = p - c.position;
            // Check tolerance: isotropic L2 if all 3 equal and finite, else per-axis box.
            const bool iso = std::isfinite(c.pos_tol(0)) && c.pos_tol(0) == c.pos_tol(1) && c.pos_tol(1) == c.pos_tol(2);
            if (iso) {
                if (dp.norm() > c.pos_tol(0)) ok = false;
            }
            for (int a = 0; a < 3; ++a) {
                if (!std::isfinite(c.pos_tol(a))) continue;
                if (!iso && std::abs(dp(a)) > c.pos_tol(a)) ok = false;
                residual(row) = dp(a);
                if (jacobian) {
                    for (int j = 0; j < J; ++j) {
                        if (!anc[j]) continue;
                        if (joint_prismatic_[j]) {
                            (*jacobian)(row, j) = j_axis[j](a);
                        } else {
                            const Vec3 v = j_axis[j].cross(p - j_origin[j]);
                            (*jacobian)(row, j) = v(a);
                        }
                    }
                }
                ++row;
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
                const bool iso = c.rot_tol(0) == c.rot_tol(1) && c.rot_tol(1) == c.rot_tol(2);
                if (iso) {
                    if (phi.norm() > c.rot_tol(0)) ok = false;
                } else {
                    for (int a = 0; a < 3; ++a)
                        if (std::abs(phi(a)) > c.rot_tol(a)) ok = false;
                }
                residual.segment<3>(row) = phi;
                if (jacobian) {
                    const Mat3 J_inv_Rt = so3_left_jacobian_inv(phi) * c.rotation.transpose();
                    for (int j = 0; j < J; ++j) {
                        if (!anc[j] || joint_prismatic_[j]) continue;
                        jacobian->block<3, 1>(row, j) = J_inv_Rt * j_axis[j];
                    }
                }
                row += 3;
            } else if (free_count == 1) {
                // Singularity-free axis alignment: align link's free_axis with target's free_axis.
                const Vec3 u = R.col(free_axis);
                const Vec3 u_in_t = c.rotation.transpose() * u;
                const int a0 = (free_axis + 1) % 3;
                const int a1 = (free_axis + 2) % 3;
                // Angle between axes = atan2(hypot(u_in_t[a0], u_in_t[a1]), u_in_t[free_axis]).
                const double sin_ang = std::hypot(u_in_t(a0), u_in_t(a1));
                const double ang = std::atan2(sin_ang, u_in_t(free_axis));
                const double tol = std::min(c.rot_tol(a0), c.rot_tol(a1));
                if (ang > tol) ok = false;
                // Residual is projection onto the two orthogonal target axes (scaled to radians when nearly anti-parallel).
                double scale = 1.0;
                if (u_in_t(free_axis) < 0.0 && sin_ang > 1e-8) {
                    scale = ang / sin_ang;
                }
                residual(row + 0) = scale * u_in_t(a0);
                residual(row + 1) = scale * u_in_t(a1);
                if (jacobian) {
                    // d(R_t^T u)/dq_j = R_t^T (a_j x u)
                    for (int j = 0; j < J; ++j) {
                        if (!anc[j] || joint_prismatic_[j]) continue;
                        const Vec3 du_t = c.rotation.transpose() * j_axis[j].cross(u);
                        (*jacobian)(row + 0, j) = du_t(a0);
                        (*jacobian)(row + 1, j) = du_t(a1);
                    }
                }
                row += 2;
            } else if (free_count == 2) {
                // Single constrained rotation component via rotation_log.
                const Mat3 R_rel = c.rotation.transpose() * R;
                const Vec3 phi = rotation_log(R_rel);
                const Mat3 J_inv_Rt = jacobian
                    ? Mat3(so3_left_jacobian_inv(phi) * c.rotation.transpose())
                    : Mat3(Mat3::Identity());
                for (int a = 0; a < 3; ++a) {
                    if (!std::isfinite(c.rot_tol(a))) continue;
                    if (std::abs(phi(a)) > c.rot_tol(a)) ok = false;
                    residual(row) = phi(a);
                    if (jacobian) {
                        for (int j = 0; j < J; ++j) {
                            if (!anc[j] || joint_prismatic_[j]) continue;
                            (*jacobian)(row, j) = J_inv_Rt.row(a).dot(j_axis[j]);
                        }
                    }
                    ++row;
                }
            }
        }
        return ok;
    }

    const std::vector<int>& parent() const { return parent_; }
    const std::vector<int>& kinds() const { return kinds_; }
    const std::vector<int>& joint_idx() const { return joint_idx_; }
    const std::vector<std::string>& body_names() const { return body_names_; }
    const std::vector<std::string>& joint_names() const { return joint_names_; }
    const std::vector<Branch>& branches() const { return branches_; }
    const Eigen::VectorXd& lower() const { return lower_; }
    const Eigen::VectorXd& upper() const { return upper_; }
    const Eigen::VectorXd& velocity_limit() const { return velocity_limit_; }
    const Eigen::VectorXd& acceleration_limit() const { return acceleration_limit_; }
    const Eigen::VectorXd& initial_q() const { return initial_q_; }
    const std::vector<uint8_t>& body_ancestor() const { return body_ancestor_; }
    const std::vector<uint8_t>& joint_prismatic() const { return joint_prismatic_; }

private:
    std::vector<int> parent_;
    std::vector<int> kinds_;
    std::vector<int> joint_idx_;
    std::vector<Mat3> a_rot_, b_rot_;
    std::vector<Vec3> a_pos_, b_pos_, axes_;
    std::vector<Mat3> skew_, skew2_;
    std::vector<Branch> branches_;
    std::vector<int> body_branch_;
    std::vector<std::string> body_names_;
    std::vector<std::string> joint_names_;
    Eigen::VectorXd lower_, upper_, velocity_limit_, acceleration_limit_, initial_q_;
    std::vector<uint8_t> body_ancestor_;
    std::vector<uint8_t> joint_prismatic_;
    std::vector<double> body_joint_base_reach_;
    std::vector<std::vector<int>> body_joint_pris_desc_;
};

// Exact total variation int_{u0}^{u1} |dp/du| du of a scalar cubic Hermite segment
// with displacement d = p1 - p0, endpoint velocities v0, v1, duration h, over [u0, u1] in [0, 1].
inline double hermite_scalar_total_variation(double d, double v0, double v1, double h,
                                             double u0, double u1) {
    if (u1 <= u0) return 0.0;
    auto eval = [&](double u) {
        const double u2 = u * u;
        const double u3 = u2 * u;
        return (3.0 * u2 - 2.0 * u3) * d + (u3 - 2.0 * u2 + u) * (h * v0) + (u3 - u2) * (h * v1);
    };
    const double inv_h = 1.0 / std::max(h, 1e-12);
    const double six_d_h = 6.0 * d * inv_h;
    const double c1 = six_d_h - 4.0 * v0 - 2.0 * v1;
    const double c2 = -six_d_h + 3.0 * (v0 + v1);

    // Roots of derivative v(u) = c2 * u^2 + c1 * u + v0 = 0 strictly inside (u0, u1).
    double pts[4];
    int n_pts = 0;
    pts[n_pts++] = u0;
    if (std::abs(c2) > 1e-12) {
        const double disc = c1 * c1 - 4.0 * c2 * v0;
        if (disc > 0.0) {
            const double sq = std::sqrt(disc);
            double r1 = (-c1 - sq) / (2.0 * c2);
            double r2 = (-c1 + sq) / (2.0 * c2);
            if (r1 > r2) std::swap(r1, r2);
            if (r1 > u0 && r1 < u1) pts[n_pts++] = r1;
            if (r2 > u0 && r2 < u1 && std::abs(r2 - r1) > 1e-12) pts[n_pts++] = r2;
        }
    } else if (std::abs(c1) > 1e-12) {
        const double r = -v0 / c1;
        if (r > u0 && r < u1) pts[n_pts++] = r;
    }
    pts[n_pts++] = u1;

    double tv = 0.0;
    double prev = eval(pts[0]);
    for (int i = 1; i < n_pts; ++i) {
        const double cur = eval(pts[i]);
        tv += std::abs(cur - prev);
        prev = cur;
    }
    return tv;
}

}  // namespace gobot::planning
