#pragma once

#include "planning_types.hpp"

#include <chrono>
#include <optional>
#include <random>
#include <unordered_map>
#include <unordered_set>

namespace gobot::planning {

struct ActivePair {
    int shape_a = 0;
    int shape_b = 0;
    bool is_self = false;
    bool is_plane = false;
};

struct CollisionEvalResult {
    std::vector<int64_t> rows;
    std::vector<int64_t> pair_ids;
    std::vector<double> residuals;
    std::vector<double> gradients;   // [K, J] when linearize == true
    std::vector<double> witnesses;   // [K, 6] world (p_a, p_b) for diagnostics/debug draw
    std::vector<double> frames;      // [S, N_shapes, 12]
    std::vector<uint8_t> seg_ok;     // [S / group_size]
};

struct IKProblem {
    std::vector<PoseConstraintSpec> targets;
    Eigen::VectorXd seed;
};

struct IKResultNative {
    Eigen::VectorXd q;
    bool success = false;
    std::string reason;
};

struct RRTResultNative {
    bool success = false;
    std::vector<Eigen::VectorXd> waypoints;
    int iterations = 0;
    double elapsed_ms = 0.0;
};

inline std::string canonical_pair_key(const std::string& a, const std::string& b) {
    return (a < b) ? (a + "\x1f" + b) : (b + "\x1f" + a);
}

inline void eval_coal_distance(const CollisionShape& a, const CollisionShape& b,
                               const double* fa, const double* fb,
                               bool verify, double clearance, double* out) {
    const auto ta = transform_from(fa);
    const auto tb = transform_from(fb);
    const auto& ga = verify ? a.original : a.convex;
    const auto& gb = verify ? b.original : b.convex;
    if (verify && std::isfinite(clearance) &&
        ga->getNodeType() != coal::GEOM_HALFSPACE && gb->getNodeType() != coal::GEOM_HALFSPACE) {
        const double gap = world_aabb(ga, ta).distance(world_aabb(gb, tb));
        if (gap > clearance) {
            out[0] = gap;
            return;
        }
    }
    if (verify && (ga != a.convex || gb != b.convex)) {
        coal::CollisionResult hit;
        coal::collide(ga.get(), ta, gb.get(), tb, coal::CollisionRequest(), hit);
        if (hit.isCollision()) {
            out[0] = -std::numeric_limits<double>::infinity();
            return;
        }
    }
    coal::DistanceResult result;
    coal::distance(ga.get(), ta, gb.get(), tb, coal::DistanceRequest(), result);
    out[0] = result.min_distance;
    if (!std::isfinite(out[0]) || out[0] == std::numeric_limits<double>::max())
        throw std::runtime_error("invalid distance");
    if (verify) return;
    if (!result.nearest_points[0].allFinite() || !result.nearest_points[1].allFinite() ||
        !result.normal.allFinite() || std::abs(result.normal.norm() - 1.0) > 1e-5)
        throw std::runtime_error("invalid witnesses or separation normal");
    Eigen::Map<coal::Vec3s> local_a(out + 1), local_b(out + 4), normal(out + 7);
    local_a = ta.inverseTransform(result.nearest_points[0]);
    local_b = tb.inverseTransform(result.nearest_points[1]);
    normal = -result.normal;
}

inline bool eval_coal_clearance(const CollisionShape& a, const CollisionShape& b,
                                const double* fa, const double* fb,
                                bool verify, double margin) {
    const auto ta = transform_from(fa);
    const auto tb = transform_from(fb);
    const auto& ga = verify ? a.original : a.convex;
    const auto& gb = verify ? b.original : b.convex;
    const double threshold = std::max(0.0, margin - 1e-6);

    if (ga->getNodeType() != coal::GEOM_HALFSPACE && gb->getNodeType() != coal::GEOM_HALFSPACE) {
        if (world_aabb(ga, ta).distance(world_aabb(gb, tb)) > threshold) {
            return true;
        }
    }
    coal::CollisionRequest req(coal::NO_REQUEST, 1);
    req.security_margin = threshold;
    req.distance_upper_bound = threshold;
    coal::CollisionResult hit;
    coal::collide(ga.get(), ta, gb.get(), tb, req, hit);
    if (!hit.isCollision()) {
        return true;
    }
    if (verify && (ga != a.convex || gb != b.convex)) {
        coal::CollisionRequest zero_req(coal::NO_REQUEST, 1);
        coal::CollisionResult zero_hit;
        coal::collide(ga.get(), ta, gb.get(), tb, zero_req, zero_hit);
        if (zero_hit.isCollision()) {
            return false;
        }
    }
    coal::DistanceResult dist;
    coal::distance(ga.get(), ta, gb.get(), tb, coal::DistanceRequest(), dist);
    if (!std::isfinite(dist.min_distance) || dist.min_distance == std::numeric_limits<double>::max())
        throw std::runtime_error("invalid distance");
    return dist.min_distance >= threshold;
}

class PlanningWorld {
public:
    KinematicsModel kin;

    uint64_t version() const { return version_; }
    uint64_t topology_version() const { return topology_version_; }

    void set_link_index(std::unordered_map<std::string, int> map) {
        link_index_ = std::move(map);
    }

    int link_id(const std::string& link_key) const {
        auto it = link_index_.find(link_key);
        if (it == link_index_.end()) throw std::invalid_argument("unknown link: " + link_key);
        return it->second;
    }

    void add_shape(WorldShape ws) {
        if (shape_index_.count(ws.name))
            throw std::invalid_argument("duplicate collision shape name: " + ws.name);
        ws.update_fixed_bounds();
        const int idx = static_cast<int>(shapes_.size());
        shape_index_[ws.name] = idx;
        shapes_.push_back(std::move(ws));
        pairs_dirty_ = true;
        ++version_;
        ++topology_version_;
    }

    void update_obstacle_pose(const std::string& name, const Frame& world_frame) {
        auto it = shape_index_.find(name);
        if (it == shape_index_.end()) throw std::invalid_argument("unknown obstacle: " + name);
        auto& ws = shapes_[static_cast<size_t>(it->second)];
        if (ws.body_id >= 0) throw std::invalid_argument("cannot set world pose of attached/robot shape: " + name);
        ws.local_frame = world_frame;
        ws.update_fixed_bounds();
        ++version_;
    }

    void remove_obstacle(const std::string& name) {
        auto it = shape_index_.find(name);
        if (it == shape_index_.end()) throw std::invalid_argument("unknown obstacle: " + name);
        const int idx = it->second;
        if (shapes_[static_cast<size_t>(idx)].body_id >= 0)
            throw std::invalid_argument("cannot remove robot/attached shape directly; detach first: " + name);
        shapes_.erase(shapes_.begin() + idx);
        shape_index_.clear();
        for (size_t i = 0; i < shapes_.size(); ++i) shape_index_[shapes_[i].name] = static_cast<int>(i);
        pairs_dirty_ = true;
        ++version_;
        ++topology_version_;
    }

    void attach_object(const std::string& name, const std::string& link_key,
                       const Frame& local_frame, const std::vector<std::string>& touch_links) {
        auto it = shape_index_.find(name);
        if (it == shape_index_.end()) throw std::invalid_argument("unknown object to attach: " + name);
        const int bid = link_id(link_key);
        auto& ws = shapes_[static_cast<size_t>(it->second)];
        ws.body_id = bid;
        ws.link_key = link_key;
        ws.local_frame = local_frame;
        // Allow collision between the attached object and explicitly specified touch links.
        for (const auto& tl : touch_links) {
            disabled_pairs_.insert(canonical_pair_key(link_key, tl));
        }
        pairs_dirty_ = true;
        ++version_;
        ++topology_version_;
    }

    void detach_object(const std::string& name, const Eigen::VectorXd& q_current) {
        auto it = shape_index_.find(name);
        if (it == shape_index_.end()) throw std::invalid_argument("unknown object to detach: " + name);
        auto& ws = shapes_[static_cast<size_t>(it->second)];
        if (ws.body_id < 0) throw std::invalid_argument("object is not attached: " + name);
        const int B = kin.num_bodies();
        std::vector<Vec3> bpos(static_cast<size_t>(B));
        std::vector<Mat3> brot(static_cast<size_t>(B));
        kin.fk_single(q_current.data(), bpos.data(), brot.data());
        const Mat3& R = brot[static_cast<size_t>(ws.body_id)];
        const Vec3& p = bpos[static_cast<size_t>(ws.body_id)];
        Frame world_f;
        world_f.leftCols<3>() = R * ws.local_frame.leftCols<3>();
        world_f.col(3) = R * ws.local_frame.col(3) + p;
        ws.body_id = -1;
        ws.link_key = "::" + ws.name;
        ws.local_frame = world_f;
        ws.update_fixed_bounds();
        pairs_dirty_ = true;
        ++version_;
        ++topology_version_;
    }

    void set_collision_enabled(const std::string& link_a, const std::string& link_b, bool enabled) {
        const auto key = canonical_pair_key(link_a, link_b);
        if (enabled) disabled_pairs_.erase(key);
        else disabled_pairs_.insert(key);
        pairs_dirty_ = true;
        ++version_;
        ++topology_version_;
    }

    void set_robot_base_pose(int branch_idx, const Vec3& pos, const Mat3& rot) {
        kin.set_branch_pose(branch_idx, pos, rot);
        ++version_;
    }

    void ensure_pairs() const {
        if (!pairs_dirty_) return;
        std::lock_guard<std::mutex> lock(cache_mutex_);
        if (!pairs_dirty_) return;
        pairs_.clear();
        mobile_shapes_.clear();
        shape_move_slot_.assign(shapes_.size(), -1);
        std::vector<int> fixed_shapes;
        for (int i = 0; i < static_cast<int>(shapes_.size()); ++i) {
            if (shapes_[static_cast<size_t>(i)].body_id >= 0) {
                shape_move_slot_[static_cast<size_t>(i)] = static_cast<int>(mobile_shapes_.size());
                mobile_shapes_.push_back(i);
            } else {
                fixed_shapes.push_back(i);
            }
        }
        auto enabled = [&](int i, int j) {
            const auto& a = shapes_[static_cast<size_t>(i)];
            const auto& b = shapes_[static_cast<size_t>(j)];
            if (a.link_key == b.link_key) return false;
            if (disabled_pairs_.count(canonical_pair_key(a.link_key, b.link_key))) return false;
            return ((a.layer & b.mask) != 0) || ((b.layer & a.mask) != 0);
        };
        for (size_t a = 0; a < mobile_shapes_.size(); ++a) {
            for (size_t b = a + 1; b < mobile_shapes_.size(); ++b) {
                const int ia = mobile_shapes_[a], ib = mobile_shapes_[b];
                if (enabled(ia, ib)) {
                    const bool pl = (shapes_[ia].shape->kind == 6 || shapes_[ib].shape->kind == 6);
                    pairs_.push_back({ia, ib, true, pl});
                }
            }
        }
        self_pair_count_ = static_cast<int>(pairs_.size());
        for (const int ia : mobile_shapes_) {
            for (const int ib : fixed_shapes) {
                if (enabled(ia, ib)) {
                    const bool pl = (shapes_[ia].shape->kind == 6 || shapes_[ib].shape->kind == 6);
                    pairs_.push_back({ia, ib, false, pl});
                }
            }
        }
        has_mesh_ = false;
        for (const auto& pr : pairs_) {
            if (shapes_[pr.shape_a].shape->kind == 5 || shapes_[pr.shape_b].shape->kind == 5) {
                has_mesh_ = true;
                break;
            }
        }
        pairs_dirty_ = false;
    }

    const std::vector<WorldShape>& shapes() const { return shapes_; }
    const std::vector<ActivePair>& active_pairs() const { ensure_pairs(); return pairs_; }
    int self_pair_count() const { ensure_pairs(); return self_pair_count_; }
    bool has_mesh() const { ensure_pairs(); return has_mesh_; }

    Eigen::VectorXd pair_margins(double margin, double world_margin) const {
        ensure_pairs();
        Eigen::VectorXd m(static_cast<Eigen::Index>(pairs_.size()));
        for (size_t i = 0; i < pairs_.size(); ++i) {
            m(static_cast<Eigen::Index>(i)) = pairs_[i].is_self ? margin : world_margin;
        }
        return m;
    }

    // Compute world frames [S, N_shapes, 12] and optional joint origins/axes [S, J, 3].
    void compute_frames(const double* q_batch, int64_t S,
                        double* out_frames,
                        double* out_jorig = nullptr,
                        double* out_jaxis = nullptr) const {
        ensure_pairs();
        const int B = kin.num_bodies();
        const int J = kin.num_joints();
        const int N = static_cast<int>(shapes_.size());
        const bool need_joints = (out_jorig != nullptr && out_jaxis != nullptr);

        auto work = [&](int64_t s_begin, int64_t s_end) {
            std::vector<Vec3> bpos(static_cast<size_t>(B));
            std::vector<Mat3> brot(static_cast<size_t>(B));
            std::vector<Vec3> jorig(need_joints ? static_cast<size_t>(J) : 0);
            std::vector<Vec3> jaxis(need_joints ? static_cast<size_t>(J) : 0);
            for (int64_t s = s_begin; s < s_end; ++s) {
                kin.fk_single(q_batch + s * J, bpos.data(), brot.data(),
                              need_joints ? jorig.data() : nullptr,
                              need_joints ? jaxis.data() : nullptr);
                if (need_joints) {
                    for (int j = 0; j < J; ++j) {
                        Eigen::Map<Vec3>(out_jorig + (s * J + j) * 3) = jorig[static_cast<size_t>(j)];
                        Eigen::Map<Vec3>(out_jaxis + (s * J + j) * 3) = jaxis[static_cast<size_t>(j)];
                    }
                }
                double* sf = out_frames + s * N * 12;
                for (int i = 0; i < N; ++i) {
                    const auto& ws = shapes_[static_cast<size_t>(i)];
                    Eigen::Map<Frame> dst(sf + i * 12);
                    if (ws.body_id >= 0) {
                        const Mat3& R = brot[static_cast<size_t>(ws.body_id)];
                        const Vec3& p = bpos[static_cast<size_t>(ws.body_id)];
                        dst.leftCols<3>().noalias() = R * ws.local_frame.leftCols<3>();
                        dst.col(3).noalias() = R * ws.local_frame.col(3) + p;
                    } else {
                        dst = ws.local_frame;
                    }
                }
            }
        };
        ThreadPool::instance().parallel_for(S, 32, work);
    }

    // Fast C++ AABB screening (segment union + per-sample).
    void screen_aabbs(const double* frames, int64_t S, const double* thresholds,
                      bool verify_original, int64_t group_size,
                      std::vector<int64_t>& out_rows, std::vector<int64_t>& out_pairs) const {
        ensure_pairs();
        out_rows.clear();
        out_pairs.clear();
        const int64_t N = static_cast<int64_t>(shapes_.size());
        const int64_t P = static_cast<int64_t>(pairs_.size());
        const int64_t M = static_cast<int64_t>(mobile_shapes_.size());
        if (S == 0 || P == 0) return;

        std::vector<double> mlo(static_cast<size_t>(S * M * 3));
        std::vector<double> mhi(static_cast<size_t>(S * M * 3));

        auto compute_moving = [&](int64_t s_begin, int64_t s_end) {
            for (int64_t s = s_begin; s < s_end; ++s) {
                const double* sf = frames + s * N * 12;
                double* lo_s = mlo.data() + s * M * 3;
                double* hi_s = mhi.data() + s * M * 3;
                for (int64_t m = 0; m < M; ++m) {
                    const int sid = mobile_shapes_[static_cast<size_t>(m)];
                    const auto& sh = *shapes_[static_cast<size_t>(sid)].shape;
                    if (sh.kind == 6) continue;
                    const Vec3& lc = verify_original ? sh.orig_center : sh.convex_center;
                    const Vec3& lh = verify_original ? sh.orig_half : sh.convex_half;
                    transform_aabb(sf + sid * 12, lc, lh, lo_s + m * 3, hi_s + m * 3);
                }
            }
        };
        ThreadPool::instance().parallel_for(S, 32, compute_moving);

        auto overlaps = [](const double* alo, const double* ahi,
                           const double* blo, const double* bhi, double thr) -> bool {
            for (int a = 0; a < 3; ++a) {
                const double g0 = alo[a] - bhi[a];
                const double g1 = blo[a] - ahi[a];
                if ((g0 > g1 ? g0 : g1) > thr) return false;
            }
            return true;
        };

        auto get_b = [&](int64_t s, int sid, const double*& lo, const double*& hi) {
            const int slot = shape_move_slot_[static_cast<size_t>(sid)];
            if (slot >= 0) {
                const size_t off = static_cast<size_t>((s * M + slot) * 3);
                lo = mlo.data() + off;
                hi = mhi.data() + off;
            } else {
                const auto& ws = shapes_[static_cast<size_t>(sid)];
                lo = (verify_original ? ws.fixed_lo_orig : ws.fixed_lo_convex).data();
                hi = (verify_original ? ws.fixed_hi_orig : ws.fixed_hi_convex).data();
            }
        };

        if (group_size <= 1 || S < 2) {
            for (int64_t s = 0; s < S; ++s) {
                for (int64_t p = 0; p < P; ++p) {
                    const auto& pr = pairs_[static_cast<size_t>(p)];
                    if (pr.is_plane) {
                        out_rows.push_back(s);
                        out_pairs.push_back(p);
                        continue;
                    }
                    const double *alo, *ahi, *blo, *bhi;
                    get_b(s, pr.shape_a, alo, ahi);
                    get_b(s, pr.shape_b, blo, bhi);
                    if (overlaps(alo, ahi, blo, bhi, thresholds[p])) {
                        out_rows.push_back(s);
                        out_pairs.push_back(p);
                    }
                }
            }
            return;
        }

        std::vector<int64_t> starts;
        for (int64_t s = 0; s < S - 1; s += group_size) starts.push_back(s);
        const int64_t G = static_cast<int64_t>(starts.size());
        std::vector<double> glo(static_cast<size_t>(M * 3)), ghi(static_cast<size_t>(M * 3));
        std::vector<int64_t> selected;
        selected.reserve(static_cast<size_t>(P));

        for (int64_t g = 0; g < G; ++g) {
            const int64_t start = starts[static_cast<size_t>(g)];
            const int64_t stop = (g + 1 < G) ? starts[static_cast<size_t>(g + 1)] : S;
            const double* flo = mlo.data() + start * M * 3;
            const double* fhi = mhi.data() + start * M * 3;
            std::copy(flo, flo + M * 3, glo.begin());
            std::copy(fhi, fhi + M * 3, ghi.begin());
            for (int64_t s = start + 1; s < stop; ++s) {
                const double* slo = mlo.data() + s * M * 3;
                const double* shi = mhi.data() + s * M * 3;
                for (int64_t k = 0; k < M * 3; ++k) {
                    if (slo[k] < glo[static_cast<size_t>(k)]) glo[static_cast<size_t>(k)] = slo[k];
                    if (shi[k] > ghi[static_cast<size_t>(k)]) ghi[static_cast<size_t>(k)] = shi[k];
                }
            }
            selected.clear();
            for (int64_t p = 0; p < P; ++p) {
                const auto& pr = pairs_[static_cast<size_t>(p)];
                if (pr.is_plane) {
                    selected.push_back(p);
                    continue;
                }
                const int sa = shape_move_slot_[static_cast<size_t>(pr.shape_a)];
                const int sb = shape_move_slot_[static_cast<size_t>(pr.shape_b)];
                const double* alo = sa >= 0 ? glo.data() + sa * 3
                                            : (verify_original ? shapes_[pr.shape_a].fixed_lo_orig : shapes_[pr.shape_a].fixed_lo_convex).data();
                const double* ahi = sa >= 0 ? ghi.data() + sa * 3
                                            : (verify_original ? shapes_[pr.shape_a].fixed_hi_orig : shapes_[pr.shape_a].fixed_hi_convex).data();
                const double* blo = sb >= 0 ? glo.data() + sb * 3
                                            : (verify_original ? shapes_[pr.shape_b].fixed_lo_orig : shapes_[pr.shape_b].fixed_lo_convex).data();
                const double* bhi = sb >= 0 ? ghi.data() + sb * 3
                                            : (verify_original ? shapes_[pr.shape_b].fixed_hi_orig : shapes_[pr.shape_b].fixed_hi_convex).data();
                if (overlaps(alo, ahi, blo, bhi, thresholds[p])) selected.push_back(p);
            }
            if (selected.empty()) continue;
            for (int64_t s = start; s < stop; ++s) {
                for (const int64_t p : selected) {
                    const auto& pr = pairs_[static_cast<size_t>(p)];
                    if (pr.is_plane) {
                        out_rows.push_back(s);
                        out_pairs.push_back(p);
                        continue;
                    }
                    const double *alo, *ahi, *blo, *bhi;
                    get_b(s, pr.shape_a, alo, ahi);
                    get_b(s, pr.shape_b, blo, bhi);
                    if (overlaps(alo, ahi, blo, bhi, thresholds[p])) {
                        out_rows.push_back(s);
                        out_pairs.push_back(p);
                    }
                }
            }
        }
    }

    // Parallel indexed distance query.
    void query_distances_indexed(const double* frames, int64_t S,
                                 const int64_t* rows, const int64_t* pair_ids, int64_t K,
                                 bool verify, double* out) const {
        ensure_pairs();
        const int64_t N = static_cast<int64_t>(shapes_.size());
        const int64_t width = verify ? 1 : 10;
        auto work = [&](int64_t begin, int64_t end) {
            for (int64_t k = begin; k < end; ++k) {
                const int64_t s = rows[k];
                const int64_t pid = pair_ids[k];
                const auto& pr = pairs_[static_cast<size_t>(pid)];
                const double* fa = frames + (s * N + pr.shape_a) * 12;
                const double* fb = frames + (s * N + pr.shape_b) * 12;
                eval_coal_distance(*shapes_[pr.shape_a].shape, *shapes_[pr.shape_b].shape,
                                   fa, fb, verify, std::numeric_limits<double>::infinity(),
                                   out + k * width);
            }
        };
        ThreadPool::instance().parallel_for(K, 64, work);
    }

    // Early-exit clearance check on indexed candidates.
    std::optional<std::pair<int64_t, int64_t>> first_violation_indexed(
        const double* frames, int64_t S,
        const int64_t* rows, const int64_t* pair_ids, int64_t K,
        const double* margins, bool verify) const {
        ensure_pairs();
        if (K <= 0) return std::nullopt;
        const int64_t N = static_cast<int64_t>(shapes_.size());
        std::atomic<int64_t> earliest(K);
        auto work = [&](int64_t begin, int64_t end) {
            for (int64_t k = begin; k < end; ++k) {
                if (k >= earliest.load(std::memory_order_relaxed)) break;
                const int64_t s = rows[k];
                const int64_t pid = pair_ids[k];
                const auto& pr = pairs_[static_cast<size_t>(pid)];
                const double* fa = frames + (s * N + pr.shape_a) * 12;
                const double* fb = frames + (s * N + pr.shape_b) * 12;
                if (!eval_coal_clearance(*shapes_[pr.shape_a].shape, *shapes_[pr.shape_b].shape,
                                         fa, fb, verify, margins[pid])) {
                    int64_t prev = earliest.load(std::memory_order_relaxed);
                    while (k < prev && !earliest.compare_exchange_weak(prev, k, std::memory_order_relaxed)) {}
                    break;
                }
            }
        };
        ThreadPool::instance().parallel_for(K, 128, work);
        const int64_t hit = earliest.load();
        if (hit >= K) return std::nullopt;
        return std::make_pair(rows[hit], pair_ids[hit]);
    }

    // Analytic geometric collision gradient for residual = margin - distance.
    void compute_analytic_gradients(const double* frames, const double* jorig, const double* jaxis,
                                    const int64_t* rows, const int64_t* pair_ids, const double* data,
                                    int64_t K, double* out_grad, double* out_witnesses = nullptr) const {
        ensure_pairs();
        const int64_t N = static_cast<int64_t>(shapes_.size());
        const int J = kin.num_joints();
        const auto& anc = kin.body_ancestor();
        const auto& pris = kin.joint_prismatic();

        auto work = [&](int64_t begin, int64_t end) {
            for (int64_t k = begin; k < end; ++k) {
                const int64_t s = rows[k];
                const int64_t pid = pair_ids[k];
                const auto& pr = pairs_[static_cast<size_t>(pid)];
                const double* d = data + k * 10;
                const Vec3 la(d[1], d[2], d[3]);
                const Vec3 lb(d[4], d[5], d[6]);
                const Vec3 n(d[7], d[8], d[9]);

                Eigen::Map<const Frame> fa(frames + (s * N + pr.shape_a) * 12);
                const Vec3 wa = fa.leftCols<3>() * la + fa.col(3);
                Eigen::Map<const Frame> fb(frames + (s * N + pr.shape_b) * 12);
                const Vec3 wb = fb.leftCols<3>() * lb + fb.col(3);
                if (out_witnesses) {
                    Eigen::Map<Vec3>(out_witnesses + k * 6 + 0) = wa;
                    Eigen::Map<Vec3>(out_witnesses + k * 6 + 3) = wb;
                }

                const int ba = shapes_[static_cast<size_t>(pr.shape_a)].body_id;
                const int bb = shapes_[static_cast<size_t>(pr.shape_b)].body_id;
                double* g = out_grad + k * J;
                for (int j = 0; j < J; ++j) {
                    double val = 0.0;
                    const bool ma = (ba >= 0) && (anc[static_cast<size_t>(ba * J + j)] != 0);
                    const bool mb = (bb >= 0) && (anc[static_cast<size_t>(bb * J + j)] != 0);
                    if (ma || mb) {
                        const Eigen::Map<const Vec3> oj(jorig + (s * J + j) * 3);
                        const Eigen::Map<const Vec3> aj(jaxis + (s * J + j) * 3);
                        if (pris[static_cast<size_t>(j)]) {
                            const double dot = aj.dot(n);
                            if (ma) val -= dot;
                            if (mb) val += dot;
                        } else {
                            if (ma) val -= aj.cross(wa - oj).dot(n);
                            if (mb) val += aj.cross(wb - oj).dot(n);
                        }
                    }
                    g[j] = val;
                }
            }
        };
        ThreadPool::instance().parallel_for(K, 128, work);
    }

    // All-in-one C++ SQP collision row evaluation (FK + AABB + Coal + analytic Jacobian).
    CollisionEvalResult eval_collision_rows(const double* q_batch, int64_t S,
                                            const double* opt_margins, const double* check_margins,
                                            double buffer, bool linearize, int64_t group_size,
                                            bool segment_minimum) const {
        ensure_pairs();
        CollisionEvalResult res;
        const int64_t N = static_cast<int64_t>(shapes_.size());
        const int64_t P = static_cast<int64_t>(pairs_.size());
        const int J = kin.num_joints();
        const int64_t n_segs = (group_size > 0) ? (S / group_size) : S;
        res.seg_ok.assign(static_cast<size_t>(n_segs), 1);
        res.frames.resize(static_cast<size_t>(S * N * 12));
        std::vector<double> jorig, jaxis;
        if (linearize) {
            jorig.resize(static_cast<size_t>(S * J * 3));
            jaxis.resize(static_cast<size_t>(S * J * 3));
        }
        compute_frames(q_batch, S, res.frames.data(),
                       linearize ? jorig.data() : nullptr,
                       linearize ? jaxis.data() : nullptr);
        if (P == 0) return res;

        std::vector<double> thresholds(static_cast<size_t>(P));
        for (int64_t p = 0; p < P; ++p) thresholds[static_cast<size_t>(p)] = opt_margins[p] + buffer;

        std::vector<int64_t> c_rows, c_pairs;
        screen_aabbs(res.frames.data(), S, thresholds.data(), false, group_size, c_rows, c_pairs);
        const int64_t C = static_cast<int64_t>(c_rows.size());
        if (C == 0) return res;

        std::vector<double> data(static_cast<size_t>(C * 10));
        query_distances_indexed(res.frames.data(), S, c_rows.data(), c_pairs.data(), C, false, data.data());

        // Check exact uninflated clearance for seg_ok and collect rows within buffer.
        std::vector<int64_t> near;
        near.reserve(static_cast<size_t>(C));
        for (int64_t k = 0; k < C; ++k) {
            const int64_t pid = c_pairs[static_cast<size_t>(k)];
            const double dist = data[static_cast<size_t>(k * 10)];
            const double req_min = std::max(0.0, check_margins[pid] - 1e-6);
            if (!std::isfinite(dist) || dist < req_min) {
                if (group_size > 0) {
                    res.seg_ok[static_cast<size_t>(c_rows[static_cast<size_t>(k)] / group_size)] = 0;
                }
            }
            const double r = opt_margins[pid] - dist;
            if (r > -buffer) near.push_back(k);
        }

        if (segment_minimum && !near.empty() && group_size > 0) {
            // Keep worst sample per (segment, pair), breaking ties by earliest sample.
            struct Key { int64_t grp; double neg_res; int64_t idx; };
            std::vector<Key> items;
            items.reserve(near.size());
            for (const int64_t k : near) {
                const int64_t seg = c_rows[static_cast<size_t>(k)] / group_size;
                const int64_t pid = c_pairs[static_cast<size_t>(k)];
                const double r = opt_margins[pid] - data[static_cast<size_t>(k * 10)];
                items.push_back({seg * P + pid, -r, k});
            }
            std::stable_sort(items.begin(), items.end(), [](const Key& a, const Key& b) {
                if (a.grp != b.grp) return a.grp < b.grp;
                return a.neg_res < b.neg_res;
            });
            near.clear();
            int64_t last_grp = -1;
            for (const auto& it : items) {
                if (it.grp != last_grp) {
                    near.push_back(it.idx);
                    last_grp = it.grp;
                }
            }
            std::sort(near.begin(), near.end());
        }

        const int64_t K = static_cast<int64_t>(near.size());
        res.rows.resize(static_cast<size_t>(K));
        res.pair_ids.resize(static_cast<size_t>(K));
        res.residuals.resize(static_cast<size_t>(K));
        std::vector<double> near_data;
        if (linearize && K > 0) near_data.resize(static_cast<size_t>(K * 10));
        for (int64_t i = 0; i < K; ++i) {
            const int64_t k = near[static_cast<size_t>(i)];
            res.rows[static_cast<size_t>(i)] = c_rows[static_cast<size_t>(k)];
            res.pair_ids[static_cast<size_t>(i)] = c_pairs[static_cast<size_t>(k)];
            res.residuals[static_cast<size_t>(i)] = opt_margins[c_pairs[static_cast<size_t>(k)]] - data[static_cast<size_t>(k * 10)];
            if (linearize) {
                std::copy(data.data() + k * 10, data.data() + (k + 1) * 10, near_data.data() + i * 10);
            }
        }
        if (linearize && K > 0) {
            res.gradients.resize(static_cast<size_t>(K * J));
            res.witnesses.resize(static_cast<size_t>(K * 6));
            compute_analytic_gradients(res.frames.data(), jorig.data(), jaxis.data(),
                                       res.rows.data(), res.pair_ids.data(), near_data.data(),
                                       K, res.gradients.data(), res.witnesses.data());
        }
        return res;
    }

    // Fast single-configuration validity check (bounds + convex + optional original mesh).
    bool is_config_valid(const double* q, const Eigen::VectorXd& margins, bool verify_original) const {
        ensure_pairs();
        const int J = kin.num_joints();
        const auto& lo = kin.lower();
        const auto& hi = kin.upper();
        for (int j = 0; j < J; ++j) {
            if (!std::isfinite(q[j]) || q[j] < lo(j) || q[j] > hi(j)) return false;
        }
        if (pairs_.empty()) return true;
        const int N = static_cast<int>(shapes_.size());
        std::vector<double> frame(static_cast<size_t>(N * 12));
        compute_frames(q, 1, frame.data());
        std::vector<int64_t> rows, pids;
        screen_aabbs(frame.data(), 1, margins.data(), false, 1, rows, pids);
        if (!rows.empty()) {
            if (first_violation_indexed(frame.data(), 1, rows.data(), pids.data(),
                                        static_cast<int64_t>(rows.size()), margins.data(), false).has_value())
                return false;
        }
        if (verify_original && has_mesh_) {
            screen_aabbs(frame.data(), 1, margins.data(), true, 1, rows, pids);
            if (!rows.empty()) {
                if (first_violation_indexed(frame.data(), 1, rows.data(), pids.data(),
                                            static_cast<int64_t>(rows.size()), margins.data(), true).has_value())
                    return false;
            }
        }
        return true;
    }

    // Bisection-order segment validity check for fast BiRRT edge rejection.
    bool is_segment_valid(const Eigen::VectorXd& qa, const Eigen::VectorXd& qb,
                          const Eigen::VectorXd& margins, double max_step, bool verify_original) const {
        const double dist = (qb - qa).norm();
        const int steps = std::max(1, static_cast<int>(std::ceil(dist / max_step)));
        if (steps == 1) return is_config_valid(qb.data(), margins, verify_original);
        // Check endpoints first, then bisection order (midpoint, quarters, eighths...).
        if (!is_config_valid(qb.data(), margins, verify_original)) return false;
        std::vector<std::pair<int, int>> stack;
        stack.reserve(static_cast<size_t>(steps));
        stack.emplace_back(0, steps);
        Eigen::VectorXd q(qa.size());
        while (!stack.empty()) {
            auto [l, r] = stack.back();
            stack.pop_back();
            if (r - l <= 1) continue;
            const int mid = (l + r) / 2;
            const double t = static_cast<double>(mid) / static_cast<double>(steps);
            q.noalias() = (1.0 - t) * qa + t * qb;
            if (!is_config_valid(q.data(), margins, verify_original)) return false;
            stack.emplace_back(l, mid);
            stack.emplace_back(mid, r);
        }
        return true;
    }

    // Batched parallel C++ IK solver (projected LM + augmented Lagrangian for clearance).
    std::vector<IKResultNative> solve_ik_batch(const std::vector<IKProblem>& problems,
                                               const std::vector<uint8_t>& active_mask,
                                               double margin, double world_margin,
                                               int iterations, bool verify_original) const {
        ensure_pairs();
        const int B_prob = static_cast<int>(problems.size());
        std::vector<IKResultNative> results(static_cast<size_t>(B_prob));
        if (B_prob == 0) return results;

        const int J = kin.num_joints();
        const int B_links = kin.num_bodies();
        const int N_shapes = static_cast<int>(shapes_.size());
        const int P = static_cast<int>(pairs_.size());
        const Eigen::VectorXd margins = pair_margins(margin, world_margin);
        const Eigen::VectorXd& lo = kin.lower();
        const Eigen::VectorXd& hi = kin.upper();

        auto solve_one = [&](int idx) {
            const auto& prob = problems[static_cast<size_t>(idx)];
            auto& out = results[static_cast<size_t>(idx)];
            if (prob.targets.empty()) {
                out.reason = "provide at least one pose constraint";
                return;
            }
            Eigen::VectorXd q = prob.seed.size() == J ? prob.seed : kin.initial_q();
            for (int j = 0; j < J; ++j) q(j) = std::clamp(q(j), lo(j), hi(j));
            const Eigen::VectorXd q0 = q;

            std::vector<Vec3> bpos(static_cast<size_t>(B_links)), jorig(static_cast<size_t>(J)), jaxis(static_cast<size_t>(J));
            std::vector<Mat3> brot(static_cast<size_t>(B_links));
            std::vector<double> frame(static_cast<size_t>(N_shapes * 12));
            std::vector<int64_t> cand_rows, cand_pairs;
            std::vector<double> dist_data;
            std::vector<double> col_grad;

            auto eval_fk_frames = [&](const Eigen::VectorXd& qc, bool need_j) {
                kin.fk_single(qc.data(), bpos.data(), brot.data(),
                              need_j ? jorig.data() : nullptr, need_j ? jaxis.data() : nullptr);
                for (int i = 0; i < N_shapes; ++i) {
                    const auto& ws = shapes_[static_cast<size_t>(i)];
                    Eigen::Map<Frame> dst(frame.data() + i * 12);
                    if (ws.body_id >= 0) {
                        dst.leftCols<3>().noalias() = brot[ws.body_id] * ws.local_frame.leftCols<3>();
                        dst.col(3).noalias() = brot[ws.body_id] * ws.local_frame.col(3) + bpos[ws.body_id];
                    } else {
                        dst = ws.local_frame;
                    }
                }
            };

            auto check_pose_and_collision = [&](const Eigen::VectorXd& qc, std::string& fail_reason) -> bool {
                eval_fk_frames(qc, false);
                Eigen::VectorXd r_tmp;
                for (const auto& t : prob.targets) {
                    if (!kin.eval_pose_constraint(t, bpos.data(), brot.data(), nullptr, nullptr, r_tmp, nullptr)) {
                        fail_reason = "IK pose tolerance not satisfied";
                        return false;
                    }
                }
                if (P > 0) {
                    screen_aabbs(frame.data(), 1, margins.data(), false, 1, cand_rows, cand_pairs);
                    if (!cand_rows.empty()) {
                        auto hit = first_violation_indexed(frame.data(), 1, cand_rows.data(), cand_pairs.data(),
                                                           static_cast<int64_t>(cand_rows.size()), margins.data(), false);
                        if (hit.has_value()) {
                            const auto& pr = pairs_[static_cast<size_t>(hit->second)];
                            fail_reason = "IK collision clearance failed for " + shapes_[pr.shape_a].name + " / " + shapes_[pr.shape_b].name;
                            return false;
                        }
                    }
                    if (verify_original && has_mesh_) {
                        screen_aabbs(frame.data(), 1, margins.data(), true, 1, cand_rows, cand_pairs);
                        if (!cand_rows.empty()) {
                            auto hit = first_violation_indexed(frame.data(), 1, cand_rows.data(), cand_pairs.data(),
                                                               static_cast<int64_t>(cand_rows.size()), margins.data(), true);
                            if (hit.has_value()) {
                                const auto& pr = pairs_[static_cast<size_t>(hit->second)];
                                fail_reason = "IK original mesh clearance failed for " + shapes_[pr.shape_a].name + " / " + shapes_[pr.shape_b].name;
                                return false;
                            }
                        }
                    }
                }
                return true;
            };

            std::string last_reason;
            if (check_pose_and_collision(q, last_reason)) {
                out.q = q;
                out.success = true;
                return;
            }

            Eigen::VectorXd dual = Eigen::VectorXd::Zero(P);
            double penalty = 10.0;
            const int outer_rounds = (P > 0) ? 5 : 1;
            std::vector<double> thr(static_cast<size_t>(P));
            Eigen::VectorXd best_q = q;
            double best_obj = std::numeric_limits<double>::infinity();

            auto eval_merit = [&](const Eigen::VectorXd& qc, Eigen::MatrixXd* H_out, Eigen::VectorXd* g_out,
                                  Eigen::VectorXd* ineq_out = nullptr) -> double {
                const bool need_jac = (H_out != nullptr);
                eval_fk_frames(qc, need_jac);
                if (need_jac) {
                    H_out->setZero(J, J);
                    g_out->setZero(J);
                }
                double cost = 0.0;
                // Small regularization toward seed q0.
                const double w_reg = 1e-3;
                for (int j = 0; j < J; ++j) {
                    if (!active_mask[static_cast<size_t>(j)]) continue;
                    const double r = w_reg * (qc(j) - q0(j));
                    cost += r * r;
                    if (need_jac) {
                        (*H_out)(j, j) += w_reg * w_reg;
                        (*g_out)(j) += w_reg * r;
                    }
                }
                Eigen::VectorXd r_t;
                Eigen::MatrixXd J_t;
                for (const auto& t : prob.targets) {
                    kin.eval_pose_constraint(t, bpos.data(), brot.data(),
                                             need_jac ? jorig.data() : nullptr,
                                             need_jac ? jaxis.data() : nullptr,
                                             r_t, need_jac ? &J_t : nullptr);
                    const double wp = 10.0 * t.weight;
                    const double wr = 1.0 * t.weight;
                    int pos_rows = 0;
                    if (t.has_position) {
                        for (int a = 0; a < 3; ++a) if (std::isfinite(t.pos_tol(a))) ++pos_rows;
                    }
                    for (int row = 0; row < r_t.size(); ++row) {
                        const double w = (row < pos_rows) ? wp : wr;
                        r_t(row) *= w;
                        if (need_jac) J_t.row(row) *= w;
                    }
                    cost += r_t.squaredNorm();
                    if (need_jac) {
                        H_out->noalias() += J_t.transpose() * J_t;
                        g_out->noalias() += J_t.transpose() * r_t;
                    }
                }
                if (P > 0) {
                    if (ineq_out) ineq_out->setConstant(P, -1.0);
                    for (int p = 0; p < P; ++p) thr[static_cast<size_t>(p)] = margins(p) + dual(p) / penalty + 0.01;
                    screen_aabbs(frame.data(), 1, thr.data(), false, 1, cand_rows, cand_pairs);
                    const int64_t C = static_cast<int64_t>(cand_rows.size());
                    if (C > 0) {
                        dist_data.resize(static_cast<size_t>(C * 10));
                        for (int64_t k = 0; k < C; ++k) {
                            const auto& pr = pairs_[static_cast<size_t>(cand_pairs[static_cast<size_t>(k)])];
                            eval_coal_distance(*shapes_[pr.shape_a].shape, *shapes_[pr.shape_b].shape,
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
                                    act_data.insert(act_data.end(), dist_data.data() + k * 10, dist_data.data() + (k + 1) * 10);
                                }
                            }
                        }
                        if (need_jac && !act_rows.empty()) {
                            const int64_t K = static_cast<int64_t>(act_rows.size());
                            col_grad.resize(static_cast<size_t>(K * J));
                            compute_analytic_gradients(frame.data(),
                                                       reinterpret_cast<const double*>(jorig.data()),
                                                       reinterpret_cast<const double*>(jaxis.data()),
                                                       act_rows.data(), act_pairs.data(), act_data.data(),
                                                       K, col_grad.data());
                            Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> Jc(col_grad.data(), K, J);
                            Eigen::Map<Eigen::VectorXd> rc(act_res.data(), K);
                            H_out->noalias() += penalty * (Jc.transpose() * Jc);
                            g_out->noalias() += sqrt_pen * (Jc.transpose() * rc);
                        }
                    }
                }
                return cost;
            };

            Eigen::MatrixXd H(J, J);
            Eigen::VectorXd g(J), ineq(P);
            for (int outer = 0; outer < outer_rounds; ++outer) {
                double damping = 1e-3;
                double cost = eval_merit(q, nullptr, nullptr);
                for (int it = 0; it < iterations; ++it) {
                    cost = eval_merit(q, &H, &g);
                    // Project out inactive joints and active box constraints.
                    std::vector<uint8_t> free(static_cast<size_t>(J), 0);
                    double max_g = 0.0;
                    for (int j = 0; j < J; ++j) {
                        if (!active_mask[static_cast<size_t>(j)]) continue;
                        const bool at_bound = (q(j) <= lo(j) && g(j) > 0.0) || (q(j) >= hi(j) && g(j) < 0.0);
                        if (!at_bound) {
                            free[static_cast<size_t>(j)] = 1;
                            max_g = std::max(max_g, std::abs(g(j)));
                        }
                    }
                    if (max_g <= 1e-10) break;
                    for (int i = 0; i < J; ++i) {
                        if (!free[static_cast<size_t>(i)]) {
                            H.row(i).setZero();
                            H.col(i).setZero();
                            H(i, i) = 1.0;
                            g(i) = 0.0;
                        }
                    }
                    bool accepted = false;
                    for (int damp_it = 0; damp_it < 8; ++damp_it) {
                        Eigen::MatrixXd sys = H;
                        for (int j = 0; j < J; ++j) {
                            if (free[static_cast<size_t>(j)]) {
                                sys(j, j) += damping * std::max(1.0, H(j, j));
                            }
                        }
                        Eigen::LDLT<Eigen::MatrixXd> ldlt(sys);
                        if (ldlt.info() != Eigen::Success) {
                            damping *= 10.0;
                            continue;
                        }
                        Eigen::VectorXd step = ldlt.solve(-g);
                        Eigen::VectorXd trial = q + step;
                        for (int j = 0; j < J; ++j) {
                            trial(j) = active_mask[static_cast<size_t>(j)] ? std::clamp(trial(j), lo(j), hi(j)) : q(j);
                        }
                        const double trial_cost = eval_merit(trial, nullptr, nullptr);
                        if (trial_cost < cost) {
                            q = trial;
                            cost = trial_cost;
                            damping = std::max(1e-7, damping / 3.0);
                            accepted = true;
                            break;
                        }
                        damping *= 10.0;
                    }
                    if (!accepted) break;
                    if (check_pose_and_collision(q, last_reason)) {
                        const double obj = (q - q0).squaredNorm();
                        if (obj < best_obj) {
                            best_obj = obj;
                            best_q = q;
                        }
                        out.q = best_q;
                        out.success = true;
                        return;
                    }
                }
                if (check_pose_and_collision(q, last_reason)) {
                    out.q = q;
                    out.success = true;
                    return;
                }
                if (P > 0) {
                    eval_merit(q, nullptr, nullptr, &ineq);
                    for (int p = 0; p < P; ++p) {
                        dual(p) = std::max(0.0, dual(p) + penalty * ineq(p));
                    }
                    penalty *= 10.0;
                }
            }
            out.success = false;
            out.reason = last_reason.empty() ? "IK found no feasible configuration" : last_reason;
        };

        ThreadPool::instance().parallel_for(B_prob, 1, [&](int64_t b, int64_t e) {
            for (int64_t i = b; i < e; ++i) solve_one(static_cast<int>(i));
        });
        return results;
    }

    // C++ BiRRT-Connect global motion planner with shortcutting and uniform resampling.
    RRTResultNative plan_birrt(const Eigen::VectorXd& start, const Eigen::VectorXd& goal,
                               const std::vector<uint8_t>& active_mask,
                               double margin, double world_margin,
                               double step_size, int max_iterations, double timeout_s,
                               uint64_t rng_seed, bool verify_original,
                               int shortcut_iters, int output_waypoints) const {
        ensure_pairs();
        const auto t0 = std::chrono::steady_clock::now();
        RRTResultNative out;
        const int J = kin.num_joints();
        const Eigen::VectorXd margins = pair_margins(margin, world_margin);
        const Eigen::VectorXd& lo = kin.lower();
        const Eigen::VectorXd& hi = kin.upper();
        const double check_res = std::min(0.05, step_size * 0.25);

        if (!is_config_valid(start.data(), margins, verify_original) ||
            !is_config_valid(goal.data(), margins, verify_original)) {
            return out;
        }
        // Quick direct connection check.
        if (is_segment_valid(start, goal, margins, check_res, verify_original)) {
            out.success = true;
            out.waypoints = resample_path({start, goal}, output_waypoints);
            out.elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            return out;
        }

        struct Node { Eigen::VectorXd q; int parent; };
        std::vector<Node> tree_a{{start, -1}}, tree_b{{goal, -1}};
        tree_a.reserve(1024);
        tree_b.reserve(1024);
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

        auto extend = [&](std::vector<Node>& tree, const Eigen::VectorXd& target) -> int {
            // Returns: 0 = trapped, 1 = advanced, 2 = reached.
            const int near_id = nearest(tree, target);
            const Eigen::VectorXd& q_near = tree[static_cast<size_t>(near_id)].q;
            Eigen::VectorXd diff = target - q_near;
            const double d = diff.norm();
            if (d < 1e-9) return 2;
            const bool reached = (d <= step_size);
            Eigen::VectorXd q_new = reached ? target : (q_near + (step_size / d) * diff);
            if (!is_segment_valid(q_near, q_new, margins, check_res, verify_original)) return 0;
            tree.push_back({std::move(q_new), near_id});
            return reached ? 2 : 1;
        };

        auto connect = [&](std::vector<Node>& tree, const Eigen::VectorXd& target) -> int {
            int status = 1;
            while (status == 1) {
                status = extend(tree, target);
            }
            return status;
        };

        bool swapped = false;
        Eigen::VectorXd sample(J);
        for (int it = 0; it < max_iterations; ++it) {
            if ((it & 15) == 0 && timeout_s > 0.0) {
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
            if (extend(ta, sample) != 0) {
                if (connect(tb, ta.back().q) == 2) {
                    // Extract path start -> goal.
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
                    // Shortcutting passes.
                    for (int sc = 0; sc < shortcut_iters && pa.size() > 2; ++sc) {
                        const int n = static_cast<int>(pa.size());
                        int i = static_cast<int>(rng() % static_cast<uint64_t>(n - 2));
                        int j = i + 2 + static_cast<int>(rng() % static_cast<uint64_t>(n - i - 2));
                        if (is_segment_valid(pa[static_cast<size_t>(i)], pa[static_cast<size_t>(j)],
                                             margins, check_res, verify_original)) {
                            pa.erase(pa.begin() + i + 1, pa.begin() + j);
                        }
                    }
                    out.success = true;
                    out.waypoints = resample_path(pa, output_waypoints);
                    out.elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                    return out;
                }
            }
            swapped = !swapped;
        }
        out.elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        return out;
    }

    static std::vector<Eigen::VectorXd> resample_path(const std::vector<Eigen::VectorXd>& path, int count) {
        if (count < 2 || path.size() < 2) return path;
        std::vector<double> cum(path.size(), 0.0);
        for (size_t i = 1; i < path.size(); ++i) {
            cum[i] = cum[i - 1] + (path[i] - path[i - 1]).norm();
        }
        const double total = cum.back();
        std::vector<Eigen::VectorXd> out;
        out.reserve(static_cast<size_t>(count));
        out.push_back(path.front());
        if (total < 1e-12) {
            for (int i = 1; i < count; ++i) out.push_back(path.back());
            return out;
        }
        size_t seg = 0;
        for (int i = 1; i < count - 1; ++i) {
            const double s = total * static_cast<double>(i) / static_cast<double>(count - 1);
            while (seg + 1 < path.size() - 1 && cum[seg + 1] < s) ++seg;
            const double seg_len = cum[seg + 1] - cum[seg];
            const double alpha = (seg_len > 1e-12) ? ((s - cum[seg]) / seg_len) : 0.0;
            out.push_back((1.0 - alpha) * path[seg] + alpha * path[seg + 1]);
        }
        out.push_back(path.back());
        return out;
    }

private:
    std::vector<WorldShape> shapes_;
    std::unordered_map<std::string, int> shape_index_;
    std::unordered_map<std::string, int> link_index_;
    std::unordered_set<std::string> disabled_pairs_;

    mutable std::mutex cache_mutex_;
    mutable bool pairs_dirty_ = true;
    mutable std::vector<ActivePair> pairs_;
    mutable std::vector<int> mobile_shapes_;
    mutable std::vector<int> shape_move_slot_;
    mutable int self_pair_count_ = 0;
    mutable bool has_mesh_ = false;
    uint64_t version_ = 0;
    uint64_t topology_version_ = 0;
};

}  // namespace gobot::planning
