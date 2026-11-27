#pragma once

#include "planning_esdf.hpp"
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
    bool cancelled = false;
    std::vector<Eigen::VectorXd> waypoints;
    std::vector<Eigen::VectorXd> dense_waypoints;  // un-downsampled verified chain (for constrained BiRRT)
    int iterations = 0;
    double elapsed_ms = 0.0;
};

struct ContinuousViolation {
    int seg = 0;
    double u = 0.0;
    int64_t pair_id = 0;
    double distance = 0.0;
};

struct GpuSceneExport {
    uint64_t version = 0;
    uint64_t topology_version = 0;
    int num_bodies = 0;
    int num_joints = 0;
    int num_branches = 0;

    // Kinematics (flat C-order arrays ready for float32/int32 GPU upload).
    std::vector<int32_t> parent;           // [B]
    std::vector<int32_t> kinds;            // [B]
    std::vector<int32_t> joint_idx;        // [B]
    std::vector<int32_t> body_branch;      // [B]
    std::vector<float> a_rot;              // [B * 9]
    std::vector<float> a_pos;              // [B * 3]
    std::vector<float> b_rot;              // [B * 9]
    std::vector<float> b_pos;              // [B * 3]
    std::vector<float> axes;               // [B * 3]
    std::vector<float> root_pos;           // [N_br * 3]
    std::vector<float> root_rot;           // [N_br * 9]
    std::vector<uint8_t> body_ancestor;    // [B * J]
    std::vector<uint8_t> joint_prismatic;  // [J]
    std::vector<float> lower;              // [J]
    std::vector<float> upper;              // [J]
    std::vector<float> velocity_limit;     // [J]
    std::vector<float> acceleration_limit; // [J]

    // Link spheres (all mobile link/payload spheres in body_id frame).
    int num_spheres = 0;
    std::vector<int32_t> sphere_body;      // [NS]
    std::vector<float> sphere_center;      // [NS * 3]
    std::vector<float> sphere_radius;      // [NS]
    std::vector<uint8_t> sphere_esdf_active;  // [NS]

    // Self-collision sphere pairs filtered by ACM, layer/mask, and attached_touch_links.
    std::vector<int32_t> self_sphere_pairs;   // [N_self * 2]

    // Static obstacles (exported individually; static meshes use tight PCA-OBB box proxies).
    int num_static = 0;
    std::vector<int32_t> static_kind;         // [N_stat] 1=box/OBB, 2=sphere, 3=cyl, 4=cap, 6=plane
    std::vector<float> static_pos;            // [N_stat * 3]
    std::vector<float> static_rot;            // [N_stat * 9]
    std::vector<float> static_size;           // [N_stat * 3]
    std::vector<int32_t> static_sphere_pairs; // [N_sp_stat * 2] (sphere_idx, static_idx)

    // 3D ESDF snapshot (shared immutable float32 grid).
    bool has_esdf = false;
    int esdf_nx = 0;
    int esdf_ny = 0;
    int esdf_nz = 0;
    std::array<float, 3> esdf_origin{0.0f, 0.0f, 0.0f};
    float esdf_resolution = 0.05f;
    int32_t esdf_from_occupancy = 0;
    float esdf_inflation_radius = 0.0f;
    float esdf_max_distance = 1e30f;
    float esdf_lipschitz_bound = 1.0f;
    std::shared_ptr<const VoxelESDF> esdf_grid;
};

struct SyncBaseUpdate {
    int branch_idx = 0;
    Vec3 pos = Vec3::Zero();
    Mat3 rot = Mat3::Identity();
};

struct SyncPoseUpdate {
    std::string name;
    Frame world_frame = Frame::Zero();
};

struct SyncShapeReplacement {
    std::string name;
    std::string link_key;
    Frame world_frame = Frame::Zero();
    int layer = 1;
    int mask = 1;
    std::shared_ptr<CollisionShape> shape;  // nullptr => reuse existing CollisionShape/BVH
};

struct SyncBatch {
    std::vector<SyncBaseUpdate> base_updates;
    std::vector<SyncPoseUpdate> pose_updates;
    std::vector<std::string> remove_names;
    std::vector<SyncShapeReplacement> replace_shapes;
    std::vector<WorldShape> add_shapes;
};

struct SyncResultNative {
    uint64_t version = 0;
    uint64_t topology_version = 0;
    bool changed = false;
    int poses_updated = 0;
    int shapes_added = 0;
    int shapes_removed = 0;
    int shapes_replaced = 0;
    int robot_bases_updated = 0;
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

// Immutable Copy-on-Write snapshot of a PlanningWorld.
// All query and planning methods operate lock-free on a const WorldSnapshot.
class WorldSnapshot {
public:
    KinematicsModel kin;
    std::unordered_map<std::string, int> link_index;
    std::vector<WorldShape> shapes;
    std::unordered_map<std::string, int> shape_index;
    std::unordered_set<std::string> auto_disabled_pairs;
    std::unordered_map<std::string, bool> user_acm_overrides;

    std::vector<ActivePair> pairs;
    std::vector<int> mobile_shapes;
    std::vector<int> shape_move_slot;
    std::vector<double> mobile_body_radius_convex;
    std::vector<double> mobile_body_radius_orig;
    std::shared_ptr<const VoxelESDF> esdf;
    std::unordered_map<std::string, CustomLinkSpheres> custom_link_spheres;
    std::vector<LinkSphere> link_spheres;
    std::vector<LinkSphere> esdf_spheres;
    int self_pair_count = 0;
    bool has_mesh = false;
    uint64_t version = 0;
    uint64_t topology_version = 0;

    bool has_esdf() const {
        return esdf && esdf->valid() && !esdf->is_empty && !esdf_spheres.empty();
    }

    bool has_collision_checks() const {
        return !pairs.empty() || has_esdf();
    }

    double resolve_esdf_margin(const double* margins, double explicit_margin) const {
        if (explicit_margin >= 0.0) return explicit_margin;
        if (margins != nullptr) {
            if (static_cast<int>(pairs.size()) > self_pair_count) return margins[self_pair_count];
            if (!pairs.empty()) return margins[0];
        }
        return 0.0;
    }

    double resolve_esdf_margin(const Eigen::VectorXd& margins, double explicit_margin) const {
        if (explicit_margin >= 0.0) return explicit_margin;
        if (margins.size() > self_pair_count) return margins(self_pair_count);
        if (margins.size() > 0) return margins(0);
        return 0.0;
    }

    bool is_pair_disabled_by_acm(const std::string& link_a, const std::string& link_b) const {
        const auto key = canonical_pair_key(link_a, link_b);
        auto it = user_acm_overrides.find(key);
        if (it != user_acm_overrides.end()) return !it->second;
        return auto_disabled_pairs.count(key) > 0;
    }

    std::vector<std::string> attached_shape_names() const {
        std::vector<std::string> out;
        for (const auto& ws : shapes) {
            if (ws.is_attached) out.push_back(ws.name);
        }
        return out;
    }

    std::vector<std::string> shape_names() const {
        std::vector<std::string> out;
        out.reserve(shapes.size());
        for (const auto& ws : shapes) out.push_back(ws.name);
        return out;
    }

    int link_id(const std::string& link_key) const {
        auto it = link_index.find(link_key);
        if (it == link_index.end()) throw std::invalid_argument("unknown link: " + link_key);
        return it->second;
    }

    std::string pair_display_name(int64_t pid) const {
        if (pid < 0) {
            const size_t si = static_cast<size_t>(-(pid + 1));
            if (si < esdf_spheres.size()) {
                return "('" + esdf_spheres[si].shape_name + "', 'esdf')";
            }
            return "('robot_sphere', 'esdf')";
        }
        const auto& pr = pairs[static_cast<size_t>(pid)];
        return "('" + shapes[static_cast<size_t>(pr.shape_a)].name + "', '" +
               shapes[static_cast<size_t>(pr.shape_b)].name + "')";
    }

    void rebuild_esdf_spheres() {
        esdf_spheres.clear();
        if (!esdf || !esdf->valid() || esdf->is_empty) return;
        auto enabled = [&](const std::string& link, int layer, int mask) {
            return ((layer & esdf->collision_mask) || (mask & esdf->collision_layer))
                   && !is_pair_disabled_by_acm(link, "::esdf");
        };
        for (const int sid : mobile_shapes) {
            const auto& ws = shapes[static_cast<size_t>(sid)];
            if (ws.shape->kind == 6 && !custom_link_spheres.count(ws.link_key)
                    && enabled(ws.link_key, ws.layer, ws.mask)) {
                throw std::invalid_argument("ESDF cannot bound mobile halfspace: " + ws.name);
            }
        }
        esdf_spheres.reserve(link_spheres.size());
        for (const auto& sp : link_spheres) {
            if (sp.body_id < 0) continue;
            if (enabled(sp.link_key, sp.layer, sp.mask)) {
                esdf_spheres.push_back(sp);
            }
        }
    }

    void rebuild_link_spheres() {
        link_spheres.clear();
        const int B = kin.num_bodies();
        std::vector<int> body_ref_shape(static_cast<size_t>(std::max(0, B)), -1);
        for (const int sid : mobile_shapes) {
            const auto& ws = shapes[static_cast<size_t>(sid)];
            if (ws.body_id >= 0 && ws.body_id < B && body_ref_shape[static_cast<size_t>(ws.body_id)] < 0) {
                body_ref_shape[static_cast<size_t>(ws.body_id)] = sid;
            }
        }

        // Auto-generate conservative spheres for every mobile shape unless its link_key has a custom override.
        for (const int sid : mobile_shapes) {
            const auto& ws = shapes[static_cast<size_t>(sid)];
            if (ws.body_id < 0 || !ws.shape || ws.shape->kind == 6) continue;
            if (custom_link_spheres.count(ws.link_key) > 0) continue;
            const auto local_sphs = esdf_detail::generate_shape_sphere_cover(*ws.shape);
            for (const auto& [sc, sr] : local_sphs) {
                LinkSphere sp;
                sp.body_id = ws.body_id;
                sp.shape_id = sid;
                sp.ref_shape_id = sid;
                sp.link_key = ws.link_key;
                sp.shape_name = ws.name;
                sp.ref_shape_center = sc;
                sp.local_center = ws.local_frame.leftCols<3>() * sc + ws.local_frame.col(3);
                sp.radius = sr;
                sp.body_reach_radius = sp.local_center.norm();
                sp.layer = ws.layer;
                sp.mask = ws.mask;
                link_spheres.push_back(std::move(sp));
            }
        }

        // Append custom user-specified link spheres.
        for (const auto& [lk, custom] : custom_link_spheres) {
            int bid = -1;
            int payload_sid = -1;
            auto lit = link_index.find(lk);
            if (lit != link_index.end()) {
                bid = lit->second;
            } else {
                for (const int sid : mobile_shapes) {
                    if (shapes[static_cast<size_t>(sid)].link_key == lk) {
                        bid = shapes[static_cast<size_t>(sid)].body_id;
                        payload_sid = sid;
                        break;
                    }
                }
            }
            if (bid < 0 || bid >= B) continue;
            const int ref_sid = payload_sid >= 0 ? payload_sid : body_ref_shape[static_cast<size_t>(bid)];
            for (size_t k = 0; k < custom.spheres.size(); ++k) {
                const Vec3& lc = custom.spheres[k].first;
                const double lr = custom.spheres[k].second;
                LinkSphere sp;
                sp.body_id = bid;
                sp.shape_id = -1;
                sp.ref_shape_id = ref_sid;
                sp.link_key = lk;
                sp.shape_name = lk + "/sphere#" + std::to_string(k);
                sp.local_center = lc;
                if (payload_sid >= 0) {
                    const auto& f = shapes[static_cast<size_t>(payload_sid)].local_frame;
                    sp.ref_shape_center = lc;
                    sp.local_center = f.leftCols<3>() * lc + f.col(3);
                } else if (ref_sid >= 0) {
                    const auto& rws = shapes[static_cast<size_t>(ref_sid)];
                    sp.ref_shape_center = rws.local_frame.leftCols<3>().transpose() * (lc - rws.local_frame.col(3));
                }
                sp.radius = lr;
                sp.body_reach_radius = sp.local_center.norm();
                sp.layer = custom.layer;
                sp.mask = custom.mask;
                link_spheres.push_back(std::move(sp));
            }
        }
        rebuild_esdf_spheres();
    }

    void rebuild_pairs() {
        pairs.clear();
        mobile_shapes.clear();
        mobile_body_radius_convex.clear();
        mobile_body_radius_orig.clear();
        shape_move_slot.assign(shapes.size(), -1);
        std::vector<int> fixed_shapes;
        for (int i = 0; i < static_cast<int>(shapes.size()); ++i) {
            const auto& ws = shapes[static_cast<size_t>(i)];
            if (ws.body_id >= 0) {
                shape_move_slot[static_cast<size_t>(i)] = static_cast<int>(mobile_shapes.size());
                mobile_shapes.push_back(i);
                if (ws.shape->kind == 6) {
                    mobile_body_radius_convex.push_back(0.0);
                    mobile_body_radius_orig.push_back(0.0);
                } else {
                    const Vec3 c_cvx = ws.local_frame.leftCols<3>() * ws.shape->convex_center + ws.local_frame.col(3);
                    const Vec3 c_org = ws.local_frame.leftCols<3>() * ws.shape->orig_center + ws.local_frame.col(3);
                    mobile_body_radius_convex.push_back(c_cvx.norm() + ws.shape->convex_half.norm());
                    mobile_body_radius_orig.push_back(c_org.norm() + ws.shape->orig_half.norm());
                }
            } else {
                fixed_shapes.push_back(i);
            }
        }
        auto enabled = [&](int i, int j) {
            const auto& a = shapes[static_cast<size_t>(i)];
            const auto& b = shapes[static_cast<size_t>(j)];
            if (a.link_key == b.link_key) return false;
            if (a.is_attached && a.attached_touch_links.count(b.link_key)) return false;
            if (b.is_attached && b.attached_touch_links.count(a.link_key)) return false;
            if (is_pair_disabled_by_acm(a.link_key, b.link_key)) return false;
            return ((a.layer & b.mask) != 0) || ((b.layer & a.mask) != 0);
        };
        for (size_t a = 0; a < mobile_shapes.size(); ++a) {
            for (size_t b = a + 1; b < mobile_shapes.size(); ++b) {
                const int ia = mobile_shapes[a], ib = mobile_shapes[b];
                if (enabled(ia, ib)) {
                    const bool pl = (shapes[static_cast<size_t>(ia)].shape->kind == 6 ||
                                     shapes[static_cast<size_t>(ib)].shape->kind == 6);
                    pairs.push_back({ia, ib, true, pl});
                }
            }
        }
        self_pair_count = static_cast<int>(pairs.size());
        for (const int ia : mobile_shapes) {
            for (const int ib : fixed_shapes) {
                if (enabled(ia, ib)) {
                    const bool pl = (shapes[static_cast<size_t>(ia)].shape->kind == 6 ||
                                     shapes[static_cast<size_t>(ib)].shape->kind == 6);
                    pairs.push_back({ia, ib, false, pl});
                }
            }
        }
        has_mesh = false;
        for (const auto& pr : pairs) {
            if (shapes[static_cast<size_t>(pr.shape_a)].shape->kind == 5 ||
                shapes[static_cast<size_t>(pr.shape_b)].shape->kind == 5) {
                has_mesh = true;
                break;
            }
        }
        rebuild_link_spheres();
    }

    Eigen::VectorXd pair_margins(double margin, double world_margin) const {
        Eigen::VectorXd m(static_cast<Eigen::Index>(pairs.size()));
        for (size_t i = 0; i < pairs.size(); ++i) {
            m(static_cast<Eigen::Index>(i)) = pairs[i].is_self ? margin : world_margin;
        }
        return m;
    }

    // Compute world frames [S, N_shapes, 12] and optional joint origins/axes [S, J, 3].
    void compute_frames(const double* q_batch, int64_t S,
                        double* out_frames,
                        double* out_jorig = nullptr,
                        double* out_jaxis = nullptr) const {
        const int B = kin.num_bodies();
        const int J = kin.num_joints();
        const int N = static_cast<int>(shapes.size());
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
                if (N == 0) continue;
                double* sf = out_frames + s * N * 12;
                for (int i = 0; i < N; ++i) {
                    const auto& ws = shapes[static_cast<size_t>(i)];
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
        out_rows.clear();
        out_pairs.clear();
        const int64_t N = static_cast<int64_t>(shapes.size());
        const int64_t P = static_cast<int64_t>(pairs.size());
        const int64_t M = static_cast<int64_t>(mobile_shapes.size());
        if (S == 0 || P == 0) return;

        std::vector<double> mlo(static_cast<size_t>(S * M * 3));
        std::vector<double> mhi(static_cast<size_t>(S * M * 3));

        auto compute_moving = [&](int64_t s_begin, int64_t s_end) {
            for (int64_t s = s_begin; s < s_end; ++s) {
                const double* sf = frames + s * N * 12;
                double* lo_s = mlo.data() + s * M * 3;
                double* hi_s = mhi.data() + s * M * 3;
                for (int64_t m = 0; m < M; ++m) {
                    const int sid = mobile_shapes[static_cast<size_t>(m)];
                    const auto& sh = *shapes[static_cast<size_t>(sid)].shape;
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
            const int slot = shape_move_slot[static_cast<size_t>(sid)];
            if (slot >= 0) {
                const size_t off = static_cast<size_t>((s * M + slot) * 3);
                lo = mlo.data() + off;
                hi = mhi.data() + off;
            } else {
                const auto& ws = shapes[static_cast<size_t>(sid)];
                lo = (verify_original ? ws.fixed_lo_orig : ws.fixed_lo_convex).data();
                hi = (verify_original ? ws.fixed_hi_orig : ws.fixed_hi_convex).data();
            }
        };

        if (group_size <= 1 || S < 2) {
            for (int64_t s = 0; s < S; ++s) {
                for (int64_t p = 0; p < P; ++p) {
                    const auto& pr = pairs[static_cast<size_t>(p)];
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
                const auto& pr = pairs[static_cast<size_t>(p)];
                if (pr.is_plane) {
                    selected.push_back(p);
                    continue;
                }
                const int sa = shape_move_slot[static_cast<size_t>(pr.shape_a)];
                const int sb = shape_move_slot[static_cast<size_t>(pr.shape_b)];
                const double* alo = sa >= 0 ? glo.data() + sa * 3
                                            : (verify_original ? shapes[static_cast<size_t>(pr.shape_a)].fixed_lo_orig : shapes[static_cast<size_t>(pr.shape_a)].fixed_lo_convex).data();
                const double* ahi = sa >= 0 ? ghi.data() + sa * 3
                                            : (verify_original ? shapes[static_cast<size_t>(pr.shape_a)].fixed_hi_orig : shapes[static_cast<size_t>(pr.shape_a)].fixed_hi_convex).data();
                const double* blo = sb >= 0 ? glo.data() + sb * 3
                                            : (verify_original ? shapes[static_cast<size_t>(pr.shape_b)].fixed_lo_orig : shapes[static_cast<size_t>(pr.shape_b)].fixed_lo_convex).data();
                const double* bhi = sb >= 0 ? ghi.data() + sb * 3
                                            : (verify_original ? shapes[static_cast<size_t>(pr.shape_b)].fixed_hi_orig : shapes[static_cast<size_t>(pr.shape_b)].fixed_hi_convex).data();
                if (overlaps(alo, ahi, blo, bhi, thresholds[p])) selected.push_back(p);
            }
            if (selected.empty()) continue;
            for (int64_t s = start; s < stop; ++s) {
                for (const int64_t p : selected) {
                    const auto& pr = pairs[static_cast<size_t>(p)];
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
        (void)S;
        const int64_t N = static_cast<int64_t>(shapes.size());
        const int64_t width = verify ? 1 : 10;
        auto work = [&](int64_t begin, int64_t end) {
            for (int64_t k = begin; k < end; ++k) {
                const int64_t s = rows[k];
                const int64_t pid = pair_ids[k];
                const auto& pr = pairs[static_cast<size_t>(pid)];
                const double* fa = frames + (s * N + pr.shape_a) * 12;
                const double* fb = frames + (s * N + pr.shape_b) * 12;
                eval_coal_distance(*shapes[static_cast<size_t>(pr.shape_a)].shape,
                                   *shapes[static_cast<size_t>(pr.shape_b)].shape,
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
        (void)S;
        if (K <= 0) return std::nullopt;
        const int64_t N = static_cast<int64_t>(shapes.size());
        std::atomic<int64_t> earliest(K);
        auto work = [&](int64_t begin, int64_t end) {
            for (int64_t k = begin; k < end; ++k) {
                if (k >= earliest.load(std::memory_order_relaxed)) break;
                const int64_t s = rows[k];
                const int64_t pid = pair_ids[k];
                const auto& pr = pairs[static_cast<size_t>(pid)];
                const double* fa = frames + (s * N + pr.shape_a) * 12;
                const double* fb = frames + (s * N + pr.shape_b) * 12;
                if (!eval_coal_clearance(*shapes[static_cast<size_t>(pr.shape_a)].shape,
                                         *shapes[static_cast<size_t>(pr.shape_b)].shape,
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

    // Parallel early-exit clearance check returning all violating (sample_row, pair_id) indices.
    void violating_pairs_indexed(
        const double* frames, int64_t S,
        const int64_t* rows, const int64_t* pair_ids, int64_t K,
        const double* margins, bool verify,
        std::vector<int64_t>& out_rows, std::vector<int64_t>& out_pairs) const {
        (void)S;
        out_rows.clear();
        out_pairs.clear();
        if (K <= 0) return;
        const int64_t N = static_cast<int64_t>(shapes.size());
        std::vector<uint8_t> bad(static_cast<size_t>(K), 0);
        auto work = [&](int64_t begin, int64_t end) {
            for (int64_t k = begin; k < end; ++k) {
                const int64_t s = rows[k];
                const int64_t pid = pair_ids[k];
                const auto& pr = pairs[static_cast<size_t>(pid)];
                const double* fa = frames + (s * N + pr.shape_a) * 12;
                const double* fb = frames + (s * N + pr.shape_b) * 12;
                if (!eval_coal_clearance(*shapes[static_cast<size_t>(pr.shape_a)].shape,
                                         *shapes[static_cast<size_t>(pr.shape_b)].shape,
                                         fa, fb, verify, margins[pid])) {
                    bad[static_cast<size_t>(k)] = 1;
                }
            }
        };
        ThreadPool::instance().parallel_for(K, 128, work);
        for (int64_t k = 0; k < K; ++k) {
            if (bad[static_cast<size_t>(k)]) {
                out_rows.push_back(rows[k]);
                out_pairs.push_back(pair_ids[k]);
            }
        }
    }

    // Compute world-space centers [S * esdf_spheres.size()] for active ESDF spheres.
    // Reuses `frames` with zero extra FK whenever all spheres reference a mobile WorldShape;
    // falls back to lightweight body FK for any custom sphere on a shapeless link.
    void compute_esdf_sphere_centers(const double* q_samples, const double* frames, int64_t S,
                                     std::vector<Vec3>& out_centers) const {
        const size_t NS = esdf_spheres.size();
        out_centers.resize(static_cast<size_t>(S) * NS);
        if (S <= 0 || NS == 0) return;
        const int64_t N_shapes = static_cast<int64_t>(shapes.size());
        bool all_have_ref = (frames != nullptr);
        if (all_have_ref) {
            for (const auto& sp : esdf_spheres) {
                if (sp.ref_shape_id < 0) { all_have_ref = false; break; }
            }
        }
        if (all_have_ref) {
            ThreadPool::instance().parallel_for(S, 32, [&](int64_t sb, int64_t se) {
                for (int64_t s = sb; s < se; ++s) {
                    const double* sf = frames + s * N_shapes * 12;
                    Vec3* dst = out_centers.data() + static_cast<size_t>(s) * NS;
                    for (size_t i = 0; i < NS; ++i) {
                        const auto& sp = esdf_spheres[i];
                        Eigen::Map<const Frame> f(sf + sp.ref_shape_id * 12);
                        dst[i].noalias() = f.leftCols<3>() * sp.ref_shape_center + f.col(3);
                    }
                }
            });
            return;
        }
        if (q_samples == nullptr) {
            throw std::invalid_argument("q_samples required when custom ESDF spheres lack a reference shape");
        }
        const int B = kin.num_bodies();
        const int J = kin.num_joints();
        ThreadPool::instance().parallel_for(S, 32, [&](int64_t sb, int64_t se) {
            std::vector<Vec3> bpos(static_cast<size_t>(B));
            std::vector<Mat3> brot(static_cast<size_t>(B));
            for (int64_t s = sb; s < se; ++s) {
                kin.fk_single(q_samples + s * J, bpos.data(), brot.data());
                Vec3* dst = out_centers.data() + static_cast<size_t>(s) * NS;
                for (size_t i = 0; i < NS; ++i) {
                    const auto& sp = esdf_spheres[i];
                    dst[i].noalias() = brot[static_cast<size_t>(sp.body_id)] * sp.local_center +
                                       bpos[static_cast<size_t>(sp.body_id)];
                }
            }
        });
    }

    // Earliest discrete ESDF sphere violation on S samples. Returns {sample_row, -(sphere_idx + 1)}.
    std::optional<std::pair<int64_t, int64_t>> first_esdf_violation_samples(
        const double* q_samples, const double* frames, int64_t S, double esdf_margin) const {
        if (S <= 0 || !has_esdf()) return std::nullopt;
        const size_t NS = esdf_spheres.size();
        const double req = std::max(0.0, esdf_margin - 1e-6);
        std::vector<Vec3> centers;
        compute_esdf_sphere_centers(q_samples, frames, S, centers);
        std::atomic<int64_t> earliest(S);
        std::mutex mtx;
        std::optional<std::pair<int64_t, int64_t>> hit;
        ThreadPool::instance().parallel_for(S, 32, [&](int64_t sb, int64_t se) {
            for (int64_t s = sb; s < se; ++s) {
                if (s >= earliest.load(std::memory_order_relaxed)) break;
                const Vec3* row = centers.data() + static_cast<size_t>(s) * NS;
                for (size_t i = 0; i < NS; ++i) {
                    const double d = esdf->collision_distance(row[i]) - esdf_spheres[i].radius;
                    if (!std::isfinite(d) || d < req) {
                        std::lock_guard<std::mutex> lk(mtx);
                        if (s < earliest.load(std::memory_order_relaxed)) {
                            earliest.store(s, std::memory_order_relaxed);
                            hit = std::make_pair(s, -static_cast<int64_t>(i + 1));
                        }
                        break;
                    }
                }
            }
        });
        return hit;
    }

    // All discrete ESDF sphere violations on S samples.
    void violating_esdf_samples(
        const double* q_samples, const double* frames, int64_t S, double esdf_margin,
        std::vector<int64_t>& out_rows, std::vector<int64_t>& out_pids) const {
        out_rows.clear();
        out_pids.clear();
        if (S <= 0 || !has_esdf()) return;
        const size_t NS = esdf_spheres.size();
        const double req = std::max(0.0, esdf_margin - 1e-6);
        std::vector<Vec3> centers;
        compute_esdf_sphere_centers(q_samples, frames, S, centers);
        std::vector<int64_t> first_bad_sphere(static_cast<size_t>(S), -1);
        ThreadPool::instance().parallel_for(S, 32, [&](int64_t sb, int64_t se) {
            for (int64_t s = sb; s < se; ++s) {
                const Vec3* row = centers.data() + static_cast<size_t>(s) * NS;
                for (size_t i = 0; i < NS; ++i) {
                    const double d = esdf->collision_distance(row[i]) - esdf_spheres[i].radius;
                    if (!std::isfinite(d) || d < req) {
                        first_bad_sphere[static_cast<size_t>(s)] = static_cast<int64_t>(i);
                        break;
                    }
                }
            }
        });
        for (int64_t s = 0; s < S; ++s) {
            if (first_bad_sphere[static_cast<size_t>(s)] >= 0) {
                out_rows.push_back(s);
                out_pids.push_back(-(first_bad_sphere[static_cast<size_t>(s)] + 1));
            }
        }
    }

    // Signed sphere-to-ESDF clearances [S, esdf_spheres.size()] for a batch of configurations.
    void query_esdf_sphere_distances(const double* q_batch, int64_t S, double* out_dists) const {
        const size_t NS = esdf_spheres.size();
        if (S <= 0 || NS == 0) return;
        if (!has_esdf()) {
            std::fill(out_dists, out_dists + static_cast<size_t>(S) * NS,
                      std::numeric_limits<double>::infinity());
            return;
        }
        std::vector<Vec3> centers;
        compute_esdf_sphere_centers(q_batch, nullptr, S, centers);
        ThreadPool::instance().parallel_for(S, 32, [&](int64_t sb, int64_t se) {
            for (int64_t s = sb; s < se; ++s) {
                const Vec3* row = centers.data() + static_cast<size_t>(s) * NS;
                double* dst = out_dists + static_cast<size_t>(s) * NS;
                for (size_t i = 0; i < NS; ++i) {
                    dst[i] = esdf->collision_distance(row[i]) - esdf_spheres[i].radius;
                }
            }
        });
    }

    // Sound Conservative Advancement continuous clearance check using exact kinematic-chain
    // Lipschitz upper bounds (Schwarzer-Saha-Latombe) on point arc-length displacement.
    // Supports BOTH linear joint interpolation (velocities == nullptr) and cubic/quintic Hermite
    // curves, BOTH convex and original meshes, AND trilinear 3D ESDF sphere-cover obstacles.
    // When strict == false (default), any undetected inter-sample dip below `margin` is bounded
    // strictly by `< 0.5 * disp_tol`. When strict == true, any interval that cannot be proven
    // safe within `disp_tol` is conservatively reported as a violation.
    std::optional<ContinuousViolation> first_violation_continuous(
        const double* q_waypoints, int T, int substeps,
        const double* frames, const double* margins,
        bool verify_original = false, double disp_tol = 5e-4,
        const double* times = nullptr, const double* velocities = nullptr,
        bool strict = false,
        const double* accelerations = nullptr, int spline_order = 3,
        double esdf_margin = -1.0,
        const std::atomic<bool>* cancel_flag = nullptr,
        bool* was_cancelled = nullptr) const {
        if (was_cancelled != nullptr) *was_cancelled = false;
        if (cancel_flag != nullptr && cancel_flag->load(std::memory_order_relaxed)) {
            if (was_cancelled != nullptr) *was_cancelled = true;
            return std::nullopt;
        }
        if (T < 2) {
            throw std::invalid_argument("q_waypoints must have at least 2 waypoints");
        }
        if (substeps < 1) {
            throw std::invalid_argument("substeps must be >= 1");
        }
        if (!(disp_tol > 0.0 && std::isfinite(disp_tol)))
            throw std::invalid_argument("disp_tol must be finite and positive");
        if (spline_order != 3 && spline_order != 5) {
            throw std::invalid_argument("spline_order must be 3 (cubic) or 5 (quintic)");
        }
        if ((times != nullptr) != (velocities != nullptr)) {
            throw std::invalid_argument("times and velocities must both be provided or both omitted");
        }
        if (spline_order == 5 && (times == nullptr || velocities == nullptr || accelerations == nullptr)) {
            throw std::invalid_argument("quintic continuous collision check requires times, velocities, and accelerations");
        }
        if (times != nullptr) {
            for (int t = 0; t < T; ++t) {
                if (!std::isfinite(times[t])) {
                    throw std::invalid_argument("times must be finite");
                }
            }
            for (int seg = 0; seg < T - 1; ++seg) {
                if (times[seg + 1] < times[seg]) {
                    throw std::invalid_argument("times must be non-decreasing");
                }
            }
        }
        const int64_t P = static_cast<int64_t>(pairs.size());
        const int64_t M = static_cast<int64_t>(mobile_shapes.size());
        const int64_t N = static_cast<int64_t>(shapes.size());
        const bool check_esdf = has_esdf();
        const int64_t NS = check_esdf ? static_cast<int64_t>(esdf_spheres.size()) : 0;
        const int J = kin.num_joints();
        const int B_links = kin.num_bodies();
        const int n_segs = T - 1;
        const int64_t S = static_cast<int64_t>(n_segs) * substeps + 1;
        if (n_segs <= 0 || ((P == 0 || M == 0) && NS == 0)) return std::nullopt;
        const double eff_esdf_margin = check_esdf ? resolve_esdf_margin(margins, esdf_margin) : 0.0;
        const double esdf_req = std::max(0.0, eff_esdf_margin - 1e-6);
        const double esdf_lip = check_esdf && esdf->from_occupancy ? esdf->lipschitz_bound : 1.0;
        const bool is_quintic = (spline_order == 5 && times != nullptr);
        const bool is_hermite = (spline_order == 3 && times != nullptr);

        // Compute exact/certified conservative max |q_j(u)| over the trajectory for any downstream prismatic joints
        // (exact analytical extremum for cubic Hermite, certified Bernstein branch-and-bound for quintic Hermite,
        // both sound even without finite joint limits).
        std::vector<double> q_abs_max(static_cast<size_t>(J), 0.0);
        for (int t = 0; t < T; ++t) {
            for (int j = 0; j < J; ++j) {
                q_abs_max[static_cast<size_t>(j)] = std::max(q_abs_max[static_cast<size_t>(j)],
                                                             std::abs(q_waypoints[t * J + j]));
            }
        }
        if (is_quintic || is_hermite) {
            const auto& is_pris = kin.joint_prismatic();
            for (int seg = 0; seg < n_segs; ++seg) {
                const double h = times[seg + 1] - times[seg];
                if (h <= 0.0) continue;
                const double* qa = q_waypoints + seg * J;
                const double* qb = q_waypoints + (seg + 1) * J;
                const double* va = velocities + seg * J;
                const double* vb = velocities + (seg + 1) * J;
                const double* aa = is_quintic ? (accelerations + seg * J) : nullptr;
                const double* ab = is_quintic ? (accelerations + (seg + 1) * J) : nullptr;
                for (int j = 0; j < J; ++j) {
                    if (is_pris[static_cast<size_t>(j)]) {
                        const double seg_max = is_quintic
                            ? quintic_hermite_scalar_abs_max(qa[j], va[j], aa[j], qb[j], vb[j], ab[j], h)
                            : hermite_scalar_abs_max(qa[j], qb[j], va[j], vb[j], h);
                        q_abs_max[static_cast<size_t>(j)] = std::max(q_abs_max[static_cast<size_t>(j)], seg_max);
                    }
                }
            }
        }

        // Precompute exact kinematic-chain Lipschitz weights [M, J] for every mobile shape.
        std::vector<double> lip_weights(static_cast<size_t>(M * J), 0.0);
        const auto& rads = verify_original ? mobile_body_radius_orig : mobile_body_radius_convex;
        for (int64_t m = 0; m < M; ++m) {
            const int sid = mobile_shapes[static_cast<size_t>(m)];
            kin.body_lipschitz_weights(shapes[static_cast<size_t>(sid)].body_id,
                                       rads[static_cast<size_t>(m)],
                                       q_abs_max.data(),
                                       lip_weights.data() + m * J);
        }

        // Precompute exact kinematic-chain Lipschitz weights [NS, J] for ESDF link spheres.
        std::vector<double> esdf_lip_weights(static_cast<size_t>(NS * J), 0.0);
        for (int64_t si = 0; si < NS; ++si) {
            const auto& sp = esdf_spheres[static_cast<size_t>(si)];
            kin.body_lipschitz_weights(sp.body_id, sp.body_reach_radius,
                                       q_abs_max.data(),
                                       esdf_lip_weights.data() + si * J);
        }

        // Compute per-sample moving AABBs in parallel (only when Coal pairs exist).
        std::vector<double> mlo(P > 0 ? static_cast<size_t>(S * M * 3) : 0);
        std::vector<double> mhi(P > 0 ? static_cast<size_t>(S * M * 3) : 0);
        if (P > 0 && M > 0) {
            ThreadPool::instance().parallel_for(S, 32, [&](int64_t s_beg, int64_t s_end) {
                for (int64_t s = s_beg; s < s_end; ++s) {
                    const double* sf = frames + s * N * 12;
                    double* lo_s = mlo.data() + s * M * 3;
                    double* hi_s = mhi.data() + s * M * 3;
                    for (int64_t m = 0; m < M; ++m) {
                        const int sid = mobile_shapes[static_cast<size_t>(m)];
                        const auto& sh = *shapes[static_cast<size_t>(sid)].shape;
                        if (sh.kind == 6) continue;
                        const Vec3& lc = verify_original ? sh.orig_center : sh.convex_center;
                        const Vec3& lh = verify_original ? sh.orig_half : sh.convex_half;
                        transform_aabb(sf + sid * 12, lc, lh, lo_s + m * 3, hi_s + m * 3);
                    }
                }
            });
        }

        auto aabb_gap_sq = [](const double* alo, const double* ahi,
                              const double* blo, const double* bhi) -> double {
            double g2 = 0.0;
            for (int a = 0; a < 3; ++a) {
                double g = 0.0;
                if (alo[a] > bhi[a]) g = alo[a] - bhi[a];
                else if (blo[a] > ahi[a]) g = blo[a] - ahi[a];
                g2 += g * g;
            }
            return g2;
        };

        auto get_b = [&](int64_t s, int sid, const double*& lo, const double*& hi) {
            const int slot = shape_move_slot[static_cast<size_t>(sid)];
            if (slot >= 0) {
                const size_t off = static_cast<size_t>((s * M + slot) * 3);
                lo = mlo.data() + off;
                hi = mhi.data() + off;
            } else {
                const auto& ws = shapes[static_cast<size_t>(sid)];
                lo = (verify_original ? ws.fixed_lo_orig : ws.fixed_lo_convex).data();
                hi = (verify_original ? ws.fixed_hi_orig : ws.fixed_hi_convex).data();
            }
        };

        auto eval_q = [&](int seg, double u, double* q_out) {
            const double* qa = q_waypoints + seg * J;
            const double* qb = q_waypoints + (seg + 1) * J;
            if (is_quintic) {
                const double h = times[seg + 1] - times[seg];
                if (h <= 0.0) {
                    std::copy(qa, qa + J, q_out);
                    return;
                }
                const double* va = velocities + seg * J;
                const double* vb = velocities + (seg + 1) * J;
                const double* aa = accelerations + seg * J;
                const double* ab = accelerations + (seg + 1) * J;
                for (int j = 0; j < J; ++j) {
                    eval_quintic_hermite_scalar(qa[j], va[j], aa[j], qb[j], vb[j], ab[j], h, u, q_out + j);
                }
            } else if (is_hermite) {
                const double h = times[seg + 1] - times[seg];
                if (h <= 0.0) {
                    std::copy(qa, qa + J, q_out);
                    return;
                }
                const double* va = velocities + seg * J;
                const double* vb = velocities + (seg + 1) * J;
                for (int j = 0; j < J; ++j) {
                    eval_cubic_hermite_scalar(qa[j], va[j], qb[j], vb[j], h, u, q_out + j);
                }
            } else {
                const double w = 1.0 - u;
                for (int j = 0; j < J; ++j) q_out[j] = w * qa[j] + u * qb[j];
            }
        };

        // Exact/certified kinematic-chain displacement upper bound for a weight vector on [u0, u1] of `seg`.
        auto weights_disp_bound = [&](int seg, double u0, double u1,
                                      const double* wa, const double* wb) -> double {
            const double* qa = q_waypoints + seg * J;
            const double* qb = q_waypoints + (seg + 1) * J;
            double total = 0.0;
            if (is_quintic) {
                const double h = times[seg + 1] - times[seg];
                if (h <= 0.0) {
                    for (int j = 0; j < J; ++j) {
                        const double w = (wa ? wa[j] : 0.0) + (wb ? wb[j] : 0.0);
                        if (w > 0.0) total += w * std::abs(qb[j] - qa[j]);
                    }
                    return total;
                }
                const double* va = velocities + seg * J;
                const double* vb = velocities + (seg + 1) * J;
                const double* aa = accelerations + seg * J;
                const double* ab = accelerations + (seg + 1) * J;
                for (int j = 0; j < J; ++j) {
                    const double w = (wa ? wa[j] : 0.0) + (wb ? wb[j] : 0.0);
                    if (w > 0.0) {
                        total += w * quintic_hermite_scalar_total_variation_bound(
                            qa[j], va[j], aa[j], qb[j], vb[j], ab[j], h, u0, u1);
                    }
                }
            } else if (is_hermite) {
                const double h = times[seg + 1] - times[seg];
                if (h <= 0.0) {
                    for (int j = 0; j < J; ++j) {
                        const double w = (wa ? wa[j] : 0.0) + (wb ? wb[j] : 0.0);
                        if (w > 0.0) total += w * std::abs(qb[j] - qa[j]);
                    }
                    return total;
                }
                const double* va = velocities + seg * J;
                const double* vb = velocities + (seg + 1) * J;
                for (int j = 0; j < J; ++j) {
                    const double w = (wa ? wa[j] : 0.0) + (wb ? wb[j] : 0.0);
                    if (w > 0.0) {
                        total += w * hermite_scalar_total_variation(qb[j] - qa[j], va[j], vb[j], h, u0, u1);
                    }
                }
            } else {
                const double du = u1 - u0;
                for (int j = 0; j < J; ++j) {
                    const double w = (wa ? wa[j] : 0.0) + (wb ? wb[j] : 0.0);
                    if (w > 0.0) total += w * std::abs(qb[j] - qa[j]) * du;
                }
            }
            return total;
        };

        auto pair_disp_bound = [&](int seg, double u0, double u1, int sa, int sb) -> double {
            const double* wa = (sa >= 0) ? (lip_weights.data() + sa * J) : nullptr;
            const double* wb = (sb >= 0) ? (lip_weights.data() + sb * J) : nullptr;
            return weights_disp_bound(seg, u0, u1, wa, wb);
        };

        auto sphere_disp_bound = [&](int seg, double u0, double u1, int64_t si) -> double {
            return weights_disp_bound(seg, u0, u1, esdf_lip_weights.data() + si * J, nullptr);
        };

        const double inv_sub = 1.0 / static_cast<double>(substeps);

        // Precompute ESDF sphere world centers at all S samples when ESDF CCD is active.
        std::vector<Vec3> esdf_centers;
        if (check_esdf) {
            bool all_have_ref = (frames != nullptr);
            if (all_have_ref) {
                for (const auto& sp : esdf_spheres) {
                    if (sp.ref_shape_id < 0) { all_have_ref = false; break; }
                }
            }
            if (all_have_ref) {
                compute_esdf_sphere_centers(nullptr, frames, S, esdf_centers);
            } else {
                std::vector<double> q_samp(static_cast<size_t>(S * J));
                for (int seg = 0; seg < n_segs; ++seg) {
                    for (int k = 0; k < substeps; ++k) {
                        eval_q(seg, static_cast<double>(k) * inv_sub, q_samp.data() + (seg * substeps + k) * J);
                    }
                }
                std::copy(q_waypoints + (T - 1) * J, q_waypoints + T * J,
                          q_samp.data() + (S - 1) * J);
                compute_esdf_sphere_centers(q_samp.data(), nullptr, S, esdf_centers);
            }
        }

        std::atomic<int64_t> earliest_s(S);
        std::atomic<bool> cancelled{false};
        std::mutex hit_mutex;
        std::optional<ContinuousViolation> best_hit;

        auto is_cancelled = [&]() -> bool {
            if (cancel_flag != nullptr && cancel_flag->load(std::memory_order_relaxed)) {
                cancelled.store(true, std::memory_order_relaxed);
                return true;
            }
            return false;
        };

        auto check_range = [&](int64_t s_beg, int64_t s_end) {
            std::vector<Vec3> bpos(static_cast<size_t>(B_links));
            std::vector<Mat3> brot(static_cast<size_t>(B_links));
            std::vector<double> q_mid(static_cast<size_t>(J));
            double dist_buf[10];

            auto eval_pair_frame = [&](int sid, Frame& out_f) {
                const auto& ws = shapes[static_cast<size_t>(sid)];
                if (ws.body_id >= 0) {
                    out_f.leftCols<3>().noalias() = brot[static_cast<size_t>(ws.body_id)] * ws.local_frame.leftCols<3>();
                    out_f.col(3).noalias() = brot[static_cast<size_t>(ws.body_id)] * ws.local_frame.col(3) + bpos[static_cast<size_t>(ws.body_id)];
                } else {
                    out_f = ws.local_frame;
                }
            };

            struct Interval {
                double u0, u1;
                double d0, d1;
            };
            std::vector<Interval> stack;
            stack.reserve(32);

            for (int64_t s = s_beg; s < s_end; ++s) {
                if (is_cancelled() || s >= earliest_s.load(std::memory_order_relaxed)) break;
                const int seg = static_cast<int>(s / substeps);
                const int sub_k = static_cast<int>(s % substeps);
                const double u_start = static_cast<double>(sub_k) * inv_sub;
                const double u_end = static_cast<double>(sub_k + 1) * inv_sub;

                for (int64_t p = 0; p < P; ++p) {
                    if (is_cancelled() || s >= earliest_s.load(std::memory_order_relaxed)) break;
                    const auto& pr = pairs[static_cast<size_t>(p)];
                    const int sa = shape_move_slot[static_cast<size_t>(pr.shape_a)];
                    const int sb = shape_move_slot[static_cast<size_t>(pr.shape_b)];
                    const double delta_ab = pair_disp_bound(seg, u_start, u_end, sa, sb);
                    const double req = std::max(0.0, margins[p] - 1e-6);

                    if (!pr.is_plane) {
                        const double *alo0, *ahi0, *blo0, *bhi0;
                        const double *alo1, *ahi1, *blo1, *bhi1;
                        get_b(s, pr.shape_a, alo0, ahi0);
                        get_b(s, pr.shape_b, blo0, bhi0);
                        get_b(s + 1, pr.shape_a, alo1, ahi1);
                        get_b(s + 1, pr.shape_b, blo1, bhi1);
                        const double g0 = std::sqrt(aabb_gap_sq(alo0, ahi0, blo0, bhi0));
                        const double g1 = std::sqrt(aabb_gap_sq(alo1, ahi1, blo1, bhi1));
                        if ((g0 - req) + (g1 - req) > delta_ab) continue;
                    } else {
                        // Exact AABB-to-halfspace signed distance lower bound (prunes distant shapes from floor/wall planes).
                        const int plane_sid = (shapes[static_cast<size_t>(pr.shape_a)].shape->kind == 6) ? pr.shape_a : pr.shape_b;
                        const int other_sid = (plane_sid == pr.shape_a) ? pr.shape_b : pr.shape_a;
                        if (shapes[static_cast<size_t>(other_sid)].shape->kind != 6) {
                            auto plane_aabb_gap = [&](int64_t s_idx) -> double {
                                Eigen::Map<const Frame> pf(frames + (s_idx * N + plane_sid) * 12);
                                const Vec3 n = pf.leftCols<3>().col(2);
                                const Vec3 p0 = pf.col(3);
                                const double *olo, *ohi;
                                get_b(s_idx, other_sid, olo, ohi);
                                const Vec3 ctr(0.5 * (olo[0] + ohi[0]), 0.5 * (olo[1] + ohi[1]), 0.5 * (olo[2] + ohi[2]));
                                const Vec3 hlf(0.5 * (ohi[0] - olo[0]), 0.5 * (ohi[1] - olo[1]), 0.5 * (ohi[2] - olo[2]));
                                return n.dot(ctr - p0) - n.cwiseAbs().dot(hlf);
                            };
                            const double g0 = plane_aabb_gap(s);
                            const double g1 = plane_aabb_gap(s + 1);
                            if ((g0 - req) + (g1 - req) > delta_ab) continue;
                        }
                    }

                    const double* fa0_ptr = frames + (s * N + pr.shape_a) * 12;
                    const double* fb0_ptr = frames + (s * N + pr.shape_b) * 12;
                    const double* fa1_ptr = frames + ((s + 1) * N + pr.shape_a) * 12;
                    const double* fb1_ptr = frames + ((s + 1) * N + pr.shape_b) * 12;
                    const auto& sh_a = *shapes[static_cast<size_t>(pr.shape_a)].shape;
                    const auto& sh_b = *shapes[static_cast<size_t>(pr.shape_b)].shape;

                    // Since original triangle meshes are subsets of their convex hulls (d_orig >= d_convex),
                    // check convex distance first when verify_original is requested to prune safe mesh pairs fast.
                    if (verify_original && (sh_a.original != sh_a.convex || sh_b.original != sh_b.convex)) {
                        eval_coal_distance(sh_a, sh_b, fa0_ptr, fb0_ptr, false, req + delta_ab, dist_buf);
                        const double dc0 = dist_buf[0];
                        if (std::isfinite(dc0) && dc0 >= req) {
                            eval_coal_distance(sh_a, sh_b, fa1_ptr, fb1_ptr, false, req + delta_ab, dist_buf);
                            const double dc1 = dist_buf[0];
                            if (std::isfinite(dc1) && dc1 >= req && (dc0 - req) + (dc1 - req) > delta_ab) {
                                continue;
                            }
                        }
                    }

                    eval_coal_distance(sh_a, sh_b, fa0_ptr, fb0_ptr, verify_original, req + delta_ab, dist_buf);
                    const double d0 = dist_buf[0];
                    if (!std::isfinite(d0) || d0 < req) {
                        std::lock_guard<std::mutex> lk(hit_mutex);
                        if (s < earliest_s.load(std::memory_order_relaxed)) {
                            earliest_s.store(s, std::memory_order_relaxed);
                            best_hit = ContinuousViolation{seg, u_start, p, d0};
                        }
                        break;
                    }
                    eval_coal_distance(sh_a, sh_b, fa1_ptr, fb1_ptr, verify_original, req + delta_ab, dist_buf);
                    const double d1 = dist_buf[0];
                    if (!std::isfinite(d1) || d1 < req) {
                        std::lock_guard<std::mutex> lk(hit_mutex);
                        if (s < earliest_s.load(std::memory_order_relaxed)) {
                            earliest_s.store(s, std::memory_order_relaxed);
                            best_hit = ContinuousViolation{seg, u_end, p, d1};
                        }
                        break;
                    }
                    if ((d0 - req) + (d1 - req) > delta_ab) continue;
                    if (delta_ab <= disp_tol) {
                        if (strict) {
                            std::lock_guard<std::mutex> lk(hit_mutex);
                            if (s < earliest_s.load(std::memory_order_relaxed)) {
                                earliest_s.store(s, std::memory_order_relaxed);
                                best_hit = ContinuousViolation{seg, 0.5 * (u_start + u_end), p,
                                                               0.5 * (d0 + d1 - delta_ab)};
                            }
                            break;
                        }
                        continue;
                    }

                    // Recursive Conservative Advancement bisection along the exact curve.
                    stack.clear();
                    stack.push_back({u_start, u_end, d0, d1});
                    while (!stack.empty()) {
                        if (is_cancelled()) break;
                        const Interval cur = stack.back();
                        stack.pop_back();
                        const double um = 0.5 * (cur.u0 + cur.u1);
                        eval_q(seg, um, q_mid.data());
                        kin.fk_single(q_mid.data(), bpos.data(), brot.data());
                        Frame fam, fbm;
                        eval_pair_frame(pr.shape_a, fam);
                        eval_pair_frame(pr.shape_b, fbm);
                        eval_coal_distance(sh_a, sh_b, fam.data(), fbm.data(), verify_original,
                                           std::numeric_limits<double>::infinity(), dist_buf);
                        const double dm = dist_buf[0];
                        if (!std::isfinite(dm) || dm < req) {
                            std::lock_guard<std::mutex> lk(hit_mutex);
                            if (s < earliest_s.load(std::memory_order_relaxed)) {
                                earliest_s.store(s, std::memory_order_relaxed);
                                best_hit = ContinuousViolation{seg, um, p, dm};
                            }
                            stack.clear();
                            break;
                        }
                        const double dL = pair_disp_bound(seg, cur.u0, um, sa, sb);
                        if ((cur.d0 - req) + (dm - req) <= dL) {
                            if (dL > disp_tol) {
                                stack.push_back({cur.u0, um, cur.d0, dm});
                            } else if (strict) {
                                std::lock_guard<std::mutex> lk(hit_mutex);
                                if (s < earliest_s.load(std::memory_order_relaxed)) {
                                    earliest_s.store(s, std::memory_order_relaxed);
                                    best_hit = ContinuousViolation{seg, 0.5 * (cur.u0 + um), p,
                                                                   0.5 * (cur.d0 + dm - dL)};
                                }
                                stack.clear();
                                break;
                            }
                        }
                        const double dR = pair_disp_bound(seg, um, cur.u1, sa, sb);
                        if ((dm - req) + (cur.d1 - req) <= dR) {
                            if (dR > disp_tol) {
                                stack.push_back({um, cur.u1, dm, cur.d1});
                            } else if (strict) {
                                std::lock_guard<std::mutex> lk(hit_mutex);
                                if (s < earliest_s.load(std::memory_order_relaxed)) {
                                    earliest_s.store(s, std::memory_order_relaxed);
                                    best_hit = ContinuousViolation{seg, 0.5 * (um + cur.u1), p,
                                                                   0.5 * (dm + cur.d1 - dR)};
                                }
                                stack.clear();
                                break;
                            }
                        }
                    }
                }

                // Sound Conservative Advancement continuous check for ESDF link spheres.
                if (check_esdf && !is_cancelled()) {
                    for (int64_t si = 0; si < NS; ++si) {
                        if (is_cancelled() || s >= earliest_s.load(std::memory_order_relaxed)) break;
                        const auto& sp = esdf_spheres[static_cast<size_t>(si)];
                        const int64_t pid = -(si + 1);
                        const Vec3& p0 = esdf_centers[static_cast<size_t>(s * NS + si)];
                        const Vec3& p1 = esdf_centers[static_cast<size_t>((s + 1) * NS + si)];
                        const double d0 = esdf->collision_distance(p0) - sp.radius;
                        if (!std::isfinite(d0) || d0 < esdf_req) {
                            std::lock_guard<std::mutex> lk(hit_mutex);
                            if (s < earliest_s.load(std::memory_order_relaxed)) {
                                earliest_s.store(s, std::memory_order_relaxed);
                                best_hit = ContinuousViolation{seg, u_start, pid, d0};
                            }
                            break;
                        }
                        const double d1 = esdf->collision_distance(p1) - sp.radius;
                        if (!std::isfinite(d1) || d1 < esdf_req) {
                            std::lock_guard<std::mutex> lk(hit_mutex);
                            if (s < earliest_s.load(std::memory_order_relaxed)) {
                                earliest_s.store(s, std::memory_order_relaxed);
                                best_hit = ContinuousViolation{seg, u_end, pid, d1};
                            }
                            break;
                        }
                        const double delta_d = esdf_lip * sphere_disp_bound(seg, u_start, u_end, si);
                        if ((d0 - esdf_req) + (d1 - esdf_req) > delta_d) continue;
                        if (delta_d <= disp_tol) {
                            if (strict) {
                                std::lock_guard<std::mutex> lk(hit_mutex);
                                if (s < earliest_s.load(std::memory_order_relaxed)) {
                                    earliest_s.store(s, std::memory_order_relaxed);
                                    best_hit = ContinuousViolation{seg, 0.5 * (u_start + u_end), pid,
                                                                   0.5 * (d0 + d1 - delta_d)};
                                }
                                break;
                            }
                            continue;
                        }
                        stack.clear();
                        stack.push_back({u_start, u_end, d0, d1});
                        while (!stack.empty()) {
                            if (is_cancelled()) break;
                            const Interval cur = stack.back();
                            stack.pop_back();
                            const double um = 0.5 * (cur.u0 + cur.u1);
                            eval_q(seg, um, q_mid.data());
                            kin.fk_single(q_mid.data(), bpos.data(), brot.data());
                            const Vec3 pm = brot[static_cast<size_t>(sp.body_id)] * sp.local_center +
                                            bpos[static_cast<size_t>(sp.body_id)];
                            const double dm = esdf->collision_distance(pm) - sp.radius;
                            if (!std::isfinite(dm) || dm < esdf_req) {
                                std::lock_guard<std::mutex> lk(hit_mutex);
                                if (s < earliest_s.load(std::memory_order_relaxed)) {
                                    earliest_s.store(s, std::memory_order_relaxed);
                                    best_hit = ContinuousViolation{seg, um, pid, dm};
                                }
                                stack.clear();
                                break;
                            }
                            const double dL = esdf_lip * sphere_disp_bound(seg, cur.u0, um, si);
                            if ((cur.d0 - esdf_req) + (dm - esdf_req) <= dL) {
                                if (dL > disp_tol) {
                                    stack.push_back({cur.u0, um, cur.d0, dm});
                                } else if (strict) {
                                    std::lock_guard<std::mutex> lk(hit_mutex);
                                    if (s < earliest_s.load(std::memory_order_relaxed)) {
                                        earliest_s.store(s, std::memory_order_relaxed);
                                        best_hit = ContinuousViolation{seg, 0.5 * (cur.u0 + um), pid,
                                                                       0.5 * (cur.d0 + dm - dL)};
                                    }
                                    stack.clear();
                                    break;
                                }
                            }
                            const double dR = esdf_lip * sphere_disp_bound(seg, um, cur.u1, si);
                            if ((dm - esdf_req) + (cur.d1 - esdf_req) <= dR) {
                                if (dR > disp_tol) {
                                    stack.push_back({um, cur.u1, dm, cur.d1});
                                } else if (strict) {
                                    std::lock_guard<std::mutex> lk(hit_mutex);
                                    if (s < earliest_s.load(std::memory_order_relaxed)) {
                                        earliest_s.store(s, std::memory_order_relaxed);
                                        best_hit = ContinuousViolation{seg, 0.5 * (um + cur.u1), pid,
                                                                       0.5 * (dm + cur.d1 - dR)};
                                    }
                                    stack.clear();
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        };

        ThreadPool::instance().parallel_for(S - 1, 8, check_range);
        if (cancelled.load(std::memory_order_relaxed)) {
            if (was_cancelled != nullptr) *was_cancelled = true;
            return std::nullopt;
        }
        return best_hit;
    }

    // Analytic geometric collision gradient for residual = margin - distance.
    void compute_analytic_gradients(const double* frames, const double* jorig, const double* jaxis,
                                    const int64_t* rows, const int64_t* pair_ids, const double* data,
                                    int64_t K, double* out_grad, double* out_witnesses = nullptr) const {
        const int64_t N = static_cast<int64_t>(shapes.size());
        const int J = kin.num_joints();
        const auto& anc = kin.body_ancestor();
        const auto& pris = kin.joint_prismatic();

        auto work = [&](int64_t begin, int64_t end) {
            for (int64_t k = begin; k < end; ++k) {
                const int64_t s = rows[k];
                const int64_t pid = pair_ids[k];
                const auto& pr = pairs[static_cast<size_t>(pid)];
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

                const int ba = shapes[static_cast<size_t>(pr.shape_a)].body_id;
                const int bb = shapes[static_cast<size_t>(pr.shape_b)].body_id;
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

    // All-in-one C++ SQP collision row evaluation (FK + AABB + Coal + ESDF sphere-cover + analytic Jacobian).
    CollisionEvalResult eval_collision_rows(const double* q_batch, int64_t S,
                                            const double* opt_margins, const double* check_margins,
                                            double buffer, bool linearize, int64_t group_size,
                                            bool segment_minimum,
                                            double esdf_opt_margin = -1.0,
                                            double esdf_check_margin = -1.0) const {
        CollisionEvalResult res;
        const int64_t N = static_cast<int64_t>(shapes.size());
        const int64_t P = static_cast<int64_t>(pairs.size());
        const bool use_esdf = has_esdf();
        const int J = kin.num_joints();
        const int64_t n_segs = (group_size > 0 && S > 0) ? 1 + (S - 1) / group_size : S;
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
        if (P == 0 && !use_esdf) return res;

        if (P > 0) {
            std::vector<double> thresholds(static_cast<size_t>(P));
            for (int64_t p = 0; p < P; ++p) thresholds[static_cast<size_t>(p)] = opt_margins[p] + buffer;

            std::vector<int64_t> c_rows, c_pairs;
            screen_aabbs(res.frames.data(), S, thresholds.data(), false, group_size, c_rows, c_pairs);
            const int64_t C = static_cast<int64_t>(c_rows.size());
            if (C > 0) {
                std::vector<double> data(static_cast<size_t>(C * 10));
                query_distances_indexed(res.frames.data(), S, c_rows.data(), c_pairs.data(), C, false, data.data());

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
            }
        }

        // Fast O(1) per-sphere 3D ESDF lookup + analytic trilinear Jacobian.
        if (use_esdf && S > 0) {
            const double opt_m = resolve_esdf_margin(opt_margins, esdf_opt_margin);
            const double chk_m = resolve_esdf_margin(check_margins, esdf_check_margin);
            const double req_min = std::max(0.0, chk_m - 1e-6);
            const size_t NS = esdf_spheres.size();
            std::vector<Vec3> centers;
            compute_esdf_sphere_centers(q_batch, res.frames.data(), S, centers);

            struct EsdfHit {
                int64_t row;
                int64_t sphere_idx;
                double sdf_val;
                double residual;
            };
            std::vector<EsdfHit> hits;
            hits.reserve(static_cast<size_t>(std::min<int64_t>(S * static_cast<int64_t>(NS), 512)));
            for (int64_t s = 0; s < S; ++s) {
                const Vec3* crow = centers.data() + static_cast<size_t>(s) * NS;
                for (size_t si = 0; si < NS; ++si) {
                    const double sdf_val = esdf->collision_distance(crow[si]);
                    const double dist = sdf_val - esdf_spheres[si].radius;
                    if (!std::isfinite(dist) || dist < req_min) {
                        if (group_size > 0) {
                            res.seg_ok[static_cast<size_t>(s / group_size)] = 0;
                        }
                    }
                    const double r = opt_m - dist;
                    if (r > -buffer) {
                        hits.push_back({s, static_cast<int64_t>(si), sdf_val, r});
                    }
                }
            }

            if (segment_minimum && !hits.empty() && group_size > 0) {
                struct Key { int64_t grp; double neg_res; size_t idx; };
                std::vector<Key> items;
                items.reserve(hits.size());
                const int64_t NS_i64 = static_cast<int64_t>(NS);
                for (size_t k = 0; k < hits.size(); ++k) {
                    const int64_t seg = hits[k].row / group_size;
                    items.push_back({seg * NS_i64 + hits[k].sphere_idx, -hits[k].residual, k});
                }
                std::stable_sort(items.begin(), items.end(), [](const Key& a, const Key& b) {
                    if (a.grp != b.grp) return a.grp < b.grp;
                    return a.neg_res < b.neg_res;
                });
                std::vector<EsdfHit> reduced;
                reduced.reserve(items.size());
                int64_t last_grp = -1;
                for (const auto& it : items) {
                    if (it.grp != last_grp) {
                        reduced.push_back(hits[it.idx]);
                        last_grp = it.grp;
                    }
                }
                hits = std::move(reduced);
            }

            if (!hits.empty()) {
                const size_t prev_K = res.rows.size();
                const size_t add_K = hits.size();
                res.rows.resize(prev_K + add_K);
                res.pair_ids.resize(prev_K + add_K);
                res.residuals.resize(prev_K + add_K);
                if (linearize) {
                    res.gradients.resize((prev_K + add_K) * static_cast<size_t>(J), 0.0);
                    res.witnesses.resize((prev_K + add_K) * 6, 0.0);
                }
                const auto& anc = kin.body_ancestor();
                const auto& pris = kin.joint_prismatic();
                for (size_t i = 0; i < add_K; ++i) {
                    const auto& h = hits[i];
                    const size_t out_i = prev_K + i;
                    res.rows[out_i] = h.row;
                    res.pair_ids[out_i] = -(h.sphere_idx + 1);
                    res.residuals[out_i] = h.residual;
                    if (linearize) {
                        const auto& sp = esdf_spheres[static_cast<size_t>(h.sphere_idx)];
                        const Vec3& pw = centers[static_cast<size_t>(h.row) * NS + static_cast<size_t>(h.sphere_idx)];
                        Vec3 g_sdf = Vec3::Zero();
                        const double sv = esdf->collision_distance(pw, &g_sdf, true);
                        const double g_norm = g_sdf.norm();
                        const Vec3 n_hat = (g_norm > 1e-12) ? Vec3(g_sdf / g_norm) : Vec3(Vec3::UnitZ());
                        const Vec3 wa = pw - sp.radius * n_hat;
                        const Vec3 wb = pw - sv * n_hat;
                        Eigen::Map<Vec3>(res.witnesses.data() + out_i * 6 + 0) = wa;
                        Eigen::Map<Vec3>(res.witnesses.data() + out_i * 6 + 3) = wb;

                        double* g = res.gradients.data() + out_i * static_cast<size_t>(J);
                        const int ba = sp.body_id;
                        for (int j = 0; j < J; ++j) {
                            if (ba >= 0 && anc[static_cast<size_t>(ba * J + j)] != 0) {
                                const Eigen::Map<const Vec3> oj(jorig.data() + (h.row * J + j) * 3);
                                const Eigen::Map<const Vec3> aj(jaxis.data() + (h.row * J + j) * 3);
                                if (pris[static_cast<size_t>(j)]) {
                                    g[j] = -aj.dot(g_sdf);
                                } else {
                                    g[j] = -aj.cross(pw - oj).dot(g_sdf);
                                }
                            } else {
                                g[j] = 0.0;
                            }
                        }
                    }
                }
            }
        }
        return res;
    }

    bool is_config_valid(const double* q, const Eigen::VectorXd& margins, bool verify_original,
                         double esdf_margin = -1.0) const {
        const int J = kin.num_joints();
        const auto& lo = kin.lower();
        const auto& hi = kin.upper();
        for (int j = 0; j < J; ++j) {
            if (!std::isfinite(q[j]) || q[j] < lo(j) || q[j] > hi(j)) return false;
        }
        const bool check_pairs = !pairs.empty();
        const bool check_esdf = has_esdf();
        if (!check_pairs && !check_esdf) return true;

        if (check_pairs) {
            const int N = static_cast<int>(shapes.size());
            std::vector<double> frame(static_cast<size_t>(N * 12));
            compute_frames(q, 1, frame.data());
            std::vector<int64_t> rows, pids;
            screen_aabbs(frame.data(), 1, margins.data(), false, 1, rows, pids);
            if (!rows.empty()) {
                if (first_violation_indexed(frame.data(), 1, rows.data(), pids.data(),
                                            static_cast<int64_t>(rows.size()), margins.data(), false).has_value())
                    return false;
            }
            if (verify_original && has_mesh) {
                screen_aabbs(frame.data(), 1, margins.data(), true, 1, rows, pids);
                if (!rows.empty()) {
                    if (first_violation_indexed(frame.data(), 1, rows.data(), pids.data(),
                                                static_cast<int64_t>(rows.size()), margins.data(), true).has_value())
                        return false;
                }
            }
            if (check_esdf) {
                const double em = resolve_esdf_margin(margins, esdf_margin);
                if (first_esdf_violation_samples(q, frame.data(), 1, em).has_value()) return false;
            }
            return true;
        }

        // ESDF-only fast path (no shape frames needed).
        const int B = kin.num_bodies();
        std::vector<Vec3> bpos(static_cast<size_t>(B));
        std::vector<Mat3> brot(static_cast<size_t>(B));
        kin.fk_single(q, bpos.data(), brot.data());
        const double req = std::max(0.0, resolve_esdf_margin(margins, esdf_margin) - 1e-6);
        for (const auto& sp : esdf_spheres) {
            const Vec3 pw = brot[static_cast<size_t>(sp.body_id)] * sp.local_center +
                            bpos[static_cast<size_t>(sp.body_id)];
            const double d = esdf->collision_distance(pw) - sp.radius;
            if (!std::isfinite(d) || d < req) return false;
        }
        return true;
    }

    bool is_segment_valid(const Eigen::VectorXd& qa, const Eigen::VectorXd& qb,
                          const Eigen::VectorXd& margins, double max_step, bool verify_original,
                          double esdf_margin = -1.0, const std::atomic<bool>* cancel_flag = nullptr) const {
        if (cancel_flag && cancel_flag->load(std::memory_order_relaxed)) return false;
        const double dist = (qb - qa).norm();
        const int steps = std::max(1, static_cast<int>(std::ceil(dist / max_step)));
        if (steps == 1) return is_config_valid(qb.data(), margins, verify_original, esdf_margin);
        if (!is_config_valid(qb.data(), margins, verify_original, esdf_margin)) return false;
        std::vector<std::pair<int, int>> stack;
        stack.reserve(static_cast<size_t>(steps));
        stack.emplace_back(0, steps);
        Eigen::VectorXd q(qa.size());
        while (!stack.empty()) {
            if (cancel_flag && cancel_flag->load(std::memory_order_relaxed)) return false;
            auto [l, r] = stack.back();
            stack.pop_back();
            if (r - l <= 1) continue;
            const int mid = (l + r) / 2;
            const double t = static_cast<double>(mid) / static_cast<double>(steps);
            q.noalias() = (1.0 - t) * qa + t * qb;
            if (!is_config_valid(q.data(), margins, verify_original, esdf_margin)) return false;
            stack.emplace_back(l, mid);
            stack.emplace_back(mid, r);
        }
        return true;
    }

    // Batched parallel C++ IK solver (projected LM + augmented Lagrangian for clearance).
    std::vector<IKResultNative> solve_ik_batch(const std::vector<IKProblem>& problems,
                                               const std::vector<uint8_t>& active_mask,
                                               double margin, double world_margin,
                                               int iterations, bool verify_original,
                                               const std::atomic<bool>* cancel_flag = nullptr) const {
        const int B_prob = static_cast<int>(problems.size());
        std::vector<IKResultNative> results(static_cast<size_t>(B_prob));
        if (B_prob == 0) return results;
        auto is_cancelled = [&]() -> bool {
            return cancel_flag != nullptr && cancel_flag->load(std::memory_order_relaxed);
        };
        if (is_cancelled()) {
            for (auto& r : results) {
                r.success = false;
                r.reason = "planning cancelled";
            }
            return results;
        }

        const int J = kin.num_joints();
        const int B_links = kin.num_bodies();
        const int N_shapes = static_cast<int>(shapes.size());
        const int P = static_cast<int>(pairs.size());
        const bool use_esdf = has_esdf();
        const int NS_esdf = use_esdf ? static_cast<int>(esdf_spheres.size()) : 0;
        const int total_col = P + NS_esdf;
        const Eigen::VectorXd margins = pair_margins(margin, world_margin);
        const Eigen::VectorXd& lo = kin.lower();
        const Eigen::VectorXd& hi = kin.upper();
        const double esdf_req = std::max(0.0, world_margin - 1e-6);
        const auto& anc = kin.body_ancestor();
        const auto& pris = kin.joint_prismatic();

        auto solve_one = [&](int idx) {
            const auto& prob = problems[static_cast<size_t>(idx)];
            auto& out = results[static_cast<size_t>(idx)];
            if (is_cancelled()) {
                out.success = false;
                out.reason = "planning cancelled";
                return;
            }
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
                    const auto& ws = shapes[static_cast<size_t>(i)];
                    Eigen::Map<Frame> dst(frame.data() + i * 12);
                    if (ws.body_id >= 0) {
                        dst.leftCols<3>().noalias() = brot[static_cast<size_t>(ws.body_id)] * ws.local_frame.leftCols<3>();
                        dst.col(3).noalias() = brot[static_cast<size_t>(ws.body_id)] * ws.local_frame.col(3) + bpos[static_cast<size_t>(ws.body_id)];
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
                            fail_reason = "IK collision clearance failed for " + pair_display_name(hit->second);
                            return false;
                        }
                    }
                    if (verify_original && has_mesh) {
                        screen_aabbs(frame.data(), 1, margins.data(), true, 1, cand_rows, cand_pairs);
                        if (!cand_rows.empty()) {
                            auto hit = first_violation_indexed(frame.data(), 1, cand_rows.data(), cand_pairs.data(),
                                                               static_cast<int64_t>(cand_rows.size()), margins.data(), true);
                            if (hit.has_value()) {
                                fail_reason = "IK original mesh clearance failed for " + pair_display_name(hit->second);
                                return false;
                            }
                        }
                    }
                }
                if (use_esdf) {
                    for (int si = 0; si < NS_esdf; ++si) {
                        const auto& sp = esdf_spheres[static_cast<size_t>(si)];
                        const Vec3 pw = brot[static_cast<size_t>(sp.body_id)] * sp.local_center +
                                        bpos[static_cast<size_t>(sp.body_id)];
                        const double d = esdf->collision_distance(pw) - sp.radius;
                        if (!std::isfinite(d) || d < esdf_req) {
                            fail_reason = "IK ESDF clearance failed for " + pair_display_name(-static_cast<int64_t>(si + 1));
                            return false;
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

            Eigen::VectorXd dual = Eigen::VectorXd::Zero(total_col);
            double penalty = 10.0;
            const int outer_rounds = (total_col > 0) ? 5 : 1;
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
                if (ineq_out && total_col > 0) ineq_out->setConstant(total_col, -1.0);
                if (P > 0) {
                    for (int p = 0; p < P; ++p) thr[static_cast<size_t>(p)] = margins(p) + dual(p) / penalty + 0.01;
                    screen_aabbs(frame.data(), 1, thr.data(), false, 1, cand_rows, cand_pairs);
                    const int64_t C = static_cast<int64_t>(cand_rows.size());
                    if (C > 0) {
                        dist_data.resize(static_cast<size_t>(C * 10));
                        for (int64_t k = 0; k < C; ++k) {
                            const auto& pr = pairs[static_cast<size_t>(cand_pairs[static_cast<size_t>(k)])];
                            eval_coal_distance(*shapes[static_cast<size_t>(pr.shape_a)].shape,
                                               *shapes[static_cast<size_t>(pr.shape_b)].shape,
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
                if (use_esdf) {
                    const double sqrt_pen = std::sqrt(penalty);
                    Eigen::VectorXd g_row(J);
                    for (int si = 0; si < NS_esdf; ++si) {
                        const auto& sp = esdf_spheres[static_cast<size_t>(si)];
                        const Vec3 pw = brot[static_cast<size_t>(sp.body_id)] * sp.local_center +
                                        bpos[static_cast<size_t>(sp.body_id)];
                        Vec3 g_sdf = Vec3::Zero();
                        const double sv = esdf->collision_distance(pw, need_jac ? &g_sdf : nullptr, true);
                        const double viol = world_margin - (sv - sp.radius);
                        const int slot = P + si;
                        if (ineq_out) (*ineq_out)(slot) = viol;
                        const double aug = viol + dual(slot) / penalty;
                        if (aug > 0.0) {
                            const double sr = sqrt_pen * aug;
                            cost += sr * sr;
                            if (need_jac) {
                                g_row.setZero();
                                const int ba = sp.body_id;
                                for (int j = 0; j < J; ++j) {
                                    if (ba >= 0 && anc[static_cast<size_t>(ba * J + j)] != 0) {
                                        if (pris[static_cast<size_t>(j)]) {
                                            g_row(j) = -jaxis[static_cast<size_t>(j)].dot(g_sdf);
                                        } else {
                                            g_row(j) = -jaxis[static_cast<size_t>(j)].cross(pw - jorig[static_cast<size_t>(j)]).dot(g_sdf);
                                        }
                                    }
                                }
                                H_out->noalias() += penalty * (g_row * g_row.transpose());
                                g_out->noalias() += (sqrt_pen * sr) * g_row;
                            }
                        }
                    }
                }
                return cost;
            };

            Eigen::MatrixXd H(J, J);
            Eigen::VectorXd g(J), ineq(total_col);
            for (int outer = 0; outer < outer_rounds; ++outer) {
                if (is_cancelled()) {
                    out.success = false;
                    out.reason = "planning cancelled";
                    return;
                }
                double damping = 1e-3;
                double cost = eval_merit(q, nullptr, nullptr);
                for (int it = 0; it < iterations; ++it) {
                    if (is_cancelled()) {
                        out.success = false;
                        out.reason = "planning cancelled";
                        return;
                    }
                    cost = eval_merit(q, &H, &g);
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
                        if (is_cancelled()) {
                            out.success = false;
                            out.reason = "planning cancelled";
                            return;
                        }
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
                if (total_col > 0) {
                    eval_merit(q, nullptr, nullptr, &ineq);
                    for (int p = 0; p < total_col; ++p) {
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
                               int shortcut_iters, int output_waypoints,
                               const std::atomic<bool>* cancel_flag = nullptr) const {
        const auto t0 = std::chrono::steady_clock::now();
        RRTResultNative out;
        auto is_cancelled = [&]() -> bool {
            return cancel_flag != nullptr && cancel_flag->load(std::memory_order_relaxed);
        };
        if (is_cancelled()) {
            out.cancelled = true;
            out.elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            return out;
        }
        const int J = kin.num_joints();
        const Eigen::VectorXd margins = pair_margins(margin, world_margin);
        const Eigen::VectorXd& lo = kin.lower();
        const Eigen::VectorXd& hi = kin.upper();
        const double check_res = std::min(0.05, step_size * 0.25);

        if (!is_config_valid(start.data(), margins, verify_original, world_margin) ||
            !is_config_valid(goal.data(), margins, verify_original, world_margin)) {
            return out;
        }
        if (is_segment_valid(start, goal, margins, check_res, verify_original, world_margin, cancel_flag)) {
            out.waypoints = resample_path({start, goal}, output_waypoints, cancel_flag);
            out.cancelled = is_cancelled();
            out.success = !out.cancelled;
            if (out.cancelled) out.waypoints.clear();
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
            if (is_cancelled()) return 0;
            const int near_id = nearest(tree, target);
            const Eigen::VectorXd& q_near = tree[static_cast<size_t>(near_id)].q;
            Eigen::VectorXd diff = target - q_near;
            const double d = diff.norm();
            if (d < 1e-9) return 2;
            const bool reached = (d <= step_size);
            Eigen::VectorXd q_new = reached ? target : (q_near + (step_size / d) * diff);
            if (!is_segment_valid(q_near, q_new, margins, check_res, verify_original, world_margin, cancel_flag)) return 0;
            tree.push_back({std::move(q_new), near_id});
            return reached ? 2 : 1;
        };

        auto connect = [&](std::vector<Node>& tree, const Eigen::VectorXd& target) -> int {
            int status = 1;
            while (status == 1 && !is_cancelled()) status = extend(tree, target);
            return is_cancelled() ? 0 : status;
        };

        bool swapped = false;
        Eigen::VectorXd sample(J);
        for (int it = 0; it < max_iterations; ++it) {
            if (is_cancelled()) {
                out.cancelled = true;
                break;
            }
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
                    for (int sc = 0; sc < shortcut_iters && pa.size() > 2; ++sc) {
                        if (is_cancelled()) {
                            out.cancelled = true;
                            out.elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                            return out;
                        }
                        const int n = static_cast<int>(pa.size());
                        int i = static_cast<int>(rng() % static_cast<uint64_t>(n - 2));
                        int j = i + 2 + static_cast<int>(rng() % static_cast<uint64_t>(n - i - 2));
                        if (is_segment_valid(pa[static_cast<size_t>(i)], pa[static_cast<size_t>(j)],
                                             margins, check_res, verify_original, world_margin, cancel_flag)) {
                            pa.erase(pa.begin() + i + 1, pa.begin() + j);
                        }
                    }
                    if (is_cancelled()) {
                        out.cancelled = true;
                        out.elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                        return out;
                    }
                    out.waypoints = resample_path(pa, output_waypoints, cancel_flag);
                    out.cancelled = is_cancelled();
                    out.success = !out.cancelled;
                    if (out.cancelled) out.waypoints.clear();
                    out.elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                    return out;
                }
            }
            swapped = !swapped;
        }
        if (is_cancelled()) out.cancelled = true;
        out.elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        return out;
    }

    static std::vector<Eigen::VectorXd> resample_path(const std::vector<Eigen::VectorXd>& path, int count,
                                                      const std::atomic<bool>* cancel_flag = nullptr) {
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
            for (int i = 1; i < count; ++i) {
                if (cancel_flag && cancel_flag->load(std::memory_order_relaxed)) return {};
                out.push_back(path.back());
            }
            return out;
        }
        size_t seg = 0;
        for (int i = 1; i < count - 1; ++i) {
            if (cancel_flag && cancel_flag->load(std::memory_order_relaxed)) return {};
            const double s = total * static_cast<double>(i) / static_cast<double>(count - 1);
            while (seg + 1 < path.size() - 1 && cum[seg + 1] < s) ++seg;
            const double seg_len = cum[seg + 1] - cum[seg];
            const double alpha = (seg_len > 1e-12) ? ((s - cum[seg]) / seg_len) : 0.0;
            out.push_back((1.0 - alpha) * path[seg] + alpha * path[seg + 1]);
        }
        out.push_back(path.back());
        return out;
    }

    // Atomic collision validation (FK + AABB screen + discrete check + continuous CCD for convex
    // and optional original mesh) executed entirely on this immutable WorldSnapshot.
    struct ValidationResult {
        bool valid = true;
        std::string reason;
    };

    ValidationResult validate_trajectory_collision(
        const double* q_waypoints, int T,
        const double* samples, int64_t S, int substeps,
        double margin, double world_margin,
        bool verify_original = true, double disp_tol = 5e-4,
        const double* times = nullptr, const double* velocities = nullptr,
        bool strict_ccd = false,
        const double* accelerations = nullptr, int spline_order = 3) const {
        if (T < 2) {
            throw std::invalid_argument("q_waypoints must have at least 2 waypoints");
        }
        if (substeps < 1) {
            throw std::invalid_argument("substeps must be >= 1");
        }
        if (!(disp_tol > 0.0 && std::isfinite(disp_tol)))
            throw std::invalid_argument("disp_tol must be finite and positive");
        if (spline_order != 3 && spline_order != 5) {
            throw std::invalid_argument("spline_order must be 3 (cubic) or 5 (quintic)");
        }
        if ((times != nullptr) != (velocities != nullptr)) {
            throw std::invalid_argument("times and velocities must both be provided or both omitted");
        }
        if (spline_order == 5 && (times == nullptr || velocities == nullptr || accelerations == nullptr)) {
            throw std::invalid_argument("quintic continuous collision check requires times, velocities, and accelerations");
        }
        if (times != nullptr) {
            for (int t = 0; t < T; ++t) {
                if (!std::isfinite(times[t])) {
                    throw std::invalid_argument("times must be finite");
                }
            }
            for (int seg = 0; seg < T - 1; ++seg) {
                if (times[seg + 1] < times[seg]) {
                    throw std::invalid_argument("times must be non-decreasing");
                }
            }
        }
        const int64_t expected_S = static_cast<int64_t>(T - 1) * static_cast<int64_t>(substeps) + 1;
        if (S != expected_S) {
            throw std::invalid_argument("samples row count must equal (T - 1) * substeps + 1");
        }
        const bool check_pairs = !pairs.empty();
        const bool check_esdf = has_esdf();
        if (!check_pairs && !check_esdf) return {true, ""};
        const int64_t N = static_cast<int64_t>(shapes.size());
        const auto margins = pair_margins(margin, world_margin);
        std::vector<double> frames(static_cast<size_t>(S * N * 12));
        compute_frames(samples, S, frames.data());

        std::vector<int64_t> rows, pids;
        if (check_pairs) {
            screen_aabbs(frames.data(), S, margins.data(), false, substeps, rows, pids);
            if (!rows.empty()) {
                auto hit = first_violation_indexed(frames.data(), S, rows.data(), pids.data(),
                                                   static_cast<int64_t>(rows.size()), margins.data(), false);
                if (hit.has_value()) {
                    const int seg = std::min(static_cast<int>(hit->first / substeps), T - 2);
                    return {false, "convex clearance failed on segment " + std::to_string(seg)
                                   + ", pair " + pair_display_name(hit->second)};
                }
            }
        }
        if (check_esdf) {
            auto ehit = first_esdf_violation_samples(samples, frames.data(), S, world_margin);
            if (ehit.has_value()) {
                const int seg = std::min(static_cast<int>(ehit->first / substeps), T - 2);
                return {false, "ESDF clearance failed on segment " + std::to_string(seg)
                               + ", pair " + pair_display_name(ehit->second)};
            }
        }
        auto ccd = first_violation_continuous(
            q_waypoints, T, substeps, frames.data(), margins.data(),
            false, disp_tol, times, velocities, strict_ccd, accelerations, spline_order, world_margin);
        if (ccd.has_value()) {
            const char* kind = (ccd->pair_id < 0) ? "ESDF" : "convex";
            return {false, std::string("continuous ") + kind + " clearance failed on segment " + std::to_string(ccd->seg)
                           + ", pair " + pair_display_name(ccd->pair_id)};
        }
        if (check_pairs && verify_original && has_mesh) {
            rows.clear();
            pids.clear();
            screen_aabbs(frames.data(), S, margins.data(), true, substeps, rows, pids);
            if (!rows.empty()) {
                auto hit = first_violation_indexed(frames.data(), S, rows.data(), pids.data(),
                                                   static_cast<int64_t>(rows.size()), margins.data(), true);
                if (hit.has_value()) {
                    const int seg = std::min(static_cast<int>(hit->first / substeps), T - 2);
                    return {false, "mesh clearance failed on segment " + std::to_string(seg)
                                   + ", pair " + pair_display_name(hit->second)};
                }
            }
            auto ccd_m = first_violation_continuous(
                q_waypoints, T, substeps, frames.data(), margins.data(),
                true, disp_tol, times, velocities, strict_ccd, accelerations, spline_order, world_margin);
            if (ccd_m.has_value()) {
                return {false, "continuous mesh clearance failed on segment " + std::to_string(ccd_m->seg)
                               + ", pair " + pair_display_name(ccd_m->pair_id)};
            }
        }
        return {true, ""};
    }

    // Export immutable kinematics, link spheres, ACM-filtered self/static collision pairs,
    // individual static analytic primitives, and ESDF metadata for GPU-resident batch optimizers.
    GpuSceneExport export_gpu_scene() const {
        GpuSceneExport out;
        out.version = version;
        out.topology_version = topology_version;
        const int B = kin.num_bodies();
        const int J = kin.num_joints();
        const int N_br = static_cast<int>(kin.branches().size());
        out.num_bodies = B;
        out.num_joints = J;
        out.num_branches = N_br;

        out.parent.resize(static_cast<size_t>(B));
        out.kinds.resize(static_cast<size_t>(B));
        out.joint_idx.resize(static_cast<size_t>(B));
        out.body_branch.resize(static_cast<size_t>(B));
        out.a_rot.resize(static_cast<size_t>(B) * 9);
        out.a_pos.resize(static_cast<size_t>(B) * 3);
        out.b_rot.resize(static_cast<size_t>(B) * 9);
        out.b_pos.resize(static_cast<size_t>(B) * 3);
        out.axes.resize(static_cast<size_t>(B) * 3);

        for (int i = 0; i < B; ++i) {
            const size_t si = static_cast<size_t>(i);
            out.parent[si] = static_cast<int32_t>(kin.parent()[si]);
            out.kinds[si] = static_cast<int32_t>(kin.kinds()[si]);
            out.joint_idx[si] = static_cast<int32_t>(kin.joint_idx()[si]);
            out.body_branch[si] = static_cast<int32_t>(kin.body_branch()[si]);
            for (int r = 0; r < 3; ++r) {
                out.a_pos[si * 3 + static_cast<size_t>(r)] = static_cast<float>(kin.a_pos()[si](r));
                out.b_pos[si * 3 + static_cast<size_t>(r)] = static_cast<float>(kin.b_pos()[si](r));
                out.axes[si * 3 + static_cast<size_t>(r)] = static_cast<float>(kin.axes()[si](r));
                for (int c = 0; c < 3; ++c) {
                    out.a_rot[si * 9 + static_cast<size_t>(r * 3 + c)] = static_cast<float>(kin.a_rot()[si](r, c));
                    out.b_rot[si * 9 + static_cast<size_t>(r * 3 + c)] = static_cast<float>(kin.b_rot()[si](r, c));
                }
            }
        }

        out.root_pos.resize(static_cast<size_t>(N_br) * 3);
        out.root_rot.resize(static_cast<size_t>(N_br) * 9);
        for (int br = 0; br < N_br; ++br) {
            const size_t sb = static_cast<size_t>(br);
            const auto& branch = kin.branches()[sb];
            for (int r = 0; r < 3; ++r) {
                out.root_pos[sb * 3 + static_cast<size_t>(r)] = static_cast<float>(branch.root_pos(r));
                for (int c = 0; c < 3; ++c) {
                    out.root_rot[sb * 9 + static_cast<size_t>(r * 3 + c)] = static_cast<float>(branch.root_rot(r, c));
                }
            }
        }

        out.body_ancestor = kin.body_ancestor();
        out.joint_prismatic = kin.joint_prismatic();
        out.lower.resize(static_cast<size_t>(J));
        out.upper.resize(static_cast<size_t>(J));
        out.velocity_limit.resize(static_cast<size_t>(J));
        out.acceleration_limit.resize(static_cast<size_t>(J));
        for (int j = 0; j < J; ++j) {
            const size_t sj = static_cast<size_t>(j);
            out.lower[sj] = static_cast<float>(kin.lower()(j));
            out.upper[sj] = static_cast<float>(kin.upper()(j));
            out.velocity_limit[sj] = static_cast<float>(kin.velocity_limit()(j));
            out.acceleration_limit[sj] = static_cast<float>(kin.acceleration_limit()(j));
        }

        // Link spheres for GPU optimization:
        //  1) Conservative ESDF spheres (exact match to snap.esdf_spheres / world.esdf_distances).
        //  2) Tight non-bloated spheres for static-primitive & self-collision checks (mobile boxes use
        //     flush inset corner/edge/face spheres with 0.0 mm protrusion; fixed-base bodies with zero
        //     movable ancestors are omitted from static checks so bolted bases never corrupt ranking).
        const bool esdf_ok = has_esdf();
        const int J_kin = kin.num_joints();
        std::vector<uint8_t> body_movable(static_cast<size_t>(B), 0);
        for (int b = 0; b < B; ++b) {
            for (int j = 0; j < J_kin; ++j) {
                if (kin.body_ancestor()[static_cast<size_t>(b * J_kin + j)] != 0) {
                    body_movable[static_cast<size_t>(b)] = 1;
                    break;
                }
            }
        }

        auto source_shape_for = [&](const LinkSphere& sp) -> const WorldShape* {
            if (sp.shape_id >= 0 && sp.shape_id < static_cast<int>(shapes.size())) {
                return &shapes[static_cast<size_t>(sp.shape_id)];
            }
            if (sp.ref_shape_id >= 0 && sp.ref_shape_id < static_cast<int>(shapes.size()) &&
                shapes[static_cast<size_t>(sp.ref_shape_id)].link_key == sp.link_key) {
                return &shapes[static_cast<size_t>(sp.ref_shape_id)];
            }
            return nullptr;
        };

        struct GpuOptSphere {
            int body_id = -1;
            Vec3 local_center = Vec3::Zero();
            double radius = 0.0;
            int layer = 1;
            int mask = 1;
            std::string link_key;
            const WorldShape* src = nullptr;
            bool esdf_active = false;
            bool geom_active = false;
        };

        std::vector<GpuOptSphere> all_spheres;
        all_spheres.reserve(link_spheres.size() * 2);

        // Identify which mobile shapes are boxes (kind == 1) without custom_link_spheres so they get
        // exact flush surface spheres (0.0 mm protrusion) for static/self checks while keeping exact
        // conservative snap.link_spheres for ESDF parity.
        std::unordered_set<int> flush_box_shapes;
        for (const int sid : mobile_shapes) {
            const auto& ws = shapes[static_cast<size_t>(sid)];
            if (ws.body_id >= 0 && ws.shape && ws.shape->kind == 1 && custom_link_spheres.count(ws.link_key) == 0) {
                flush_box_shapes.insert(sid);
            }
        }

        for (const auto& sp : link_spheres) {
            const WorldShape* src = source_shape_for(sp);
            const bool is_esdf = esdf_ok && sp.body_id >= 0 &&
                (((sp.layer & esdf->collision_mask) != 0) || ((sp.mask & esdf->collision_layer) != 0)) &&
                !is_pair_disabled_by_acm(sp.link_key, "::esdf");
            const bool replaced_by_flush = (sp.shape_id >= 0 && flush_box_shapes.count(sp.shape_id) > 0);
            if (!is_esdf && replaced_by_flush) continue;
            GpuOptSphere gs;
            gs.body_id = sp.body_id;
            gs.local_center = sp.local_center;
            gs.radius = sp.radius;
            gs.layer = sp.layer;
            gs.mask = sp.mask;
            gs.link_key = sp.link_key;
            gs.src = src;
            gs.esdf_active = is_esdf;
            gs.geom_active = !replaced_by_flush;
            all_spheres.push_back(std::move(gs));
        }

        // Emit exact flush surface samples (0.0 mm protrusion AND 0.0 mm corner inset gap)
        // for mobile Box shapes: 8 exact corners (r=0), 12 exact edge midpoints (r=0),
        // and inscribed face-tangent spheres (r=min_h).
        for (const int sid : mobile_shapes) {
            if (flush_box_shapes.count(sid) == 0) continue;
            const auto& ws = shapes[static_cast<size_t>(sid)];
            const Vec3 h = ws.shape->box_half_size;
            const double min_h = std::min({h(0), h(1), h(2)});
            if (min_h <= 1e-6) continue;
            auto emit = [&](const Vec3& sc, double sr) {
                GpuOptSphere gs;
                gs.body_id = ws.body_id;
                gs.local_center = ws.local_frame.leftCols<3>() * sc + ws.local_frame.col(3);
                gs.radius = sr;
                gs.layer = ws.layer;
                gs.mask = ws.mask;
                gs.link_key = ws.link_key;
                gs.src = &ws;
                gs.esdf_active = false;
                gs.geom_active = true;
                all_spheres.push_back(std::move(gs));
            };
            // 8 exact corner points (r = 0 -> exact vertex-to-obstacle distance).
            for (int sx : {-1, 1}) {
                for (int sy : {-1, 1}) {
                    for (int sz : {-1, 1}) {
                        emit(Vec3(sx * h(0), sy * h(1), sz * h(2)), 0.0);
                    }
                }
            }
            // Edge midpoint points (r = 0) along edges longer than 8 cm.
            if (h(0) > 0.04) {
                for (int sy : {-1, 1}) for (int sz : {-1, 1}) emit(Vec3(0.0, sy * h(1), sz * h(2)), 0.0);
            }
            if (h(1) > 0.04) {
                for (int sx : {-1, 1}) for (int sz : {-1, 1}) emit(Vec3(sx * h(0), 0.0, sz * h(2)), 0.0);
            }
            if (h(2) > 0.04) {
                for (int sx : {-1, 1}) for (int sy : {-1, 1}) emit(Vec3(sx * h(0), sy * h(1), 0.0), 0.0);
            }
            // Face-tangent inscribed spheres strictly inside the box (0.0 mm protrusion).
            const double rf = min_h;
            bool emitted_inscribed = false;
            if (h(0) > rf + 1e-4) {
                emit(Vec3(h(0) - rf, 0.0, 0.0), rf);
                emit(Vec3(-(h(0) - rf), 0.0, 0.0), rf);
                emitted_inscribed = true;
            }
            if (h(1) > rf + 1e-4) {
                emit(Vec3(0.0, h(1) - rf, 0.0), rf);
                emit(Vec3(0.0, -(h(1) - rf), 0.0), rf);
                emitted_inscribed = true;
            }
            if (h(2) > rf + 1e-4) {
                emit(Vec3(0.0, 0.0, h(2) - rf), rf);
                emit(Vec3(0.0, 0.0, -(h(2) - rf)), rf);
                emitted_inscribed = true;
            }
            if (!emitted_inscribed) {
                emit(Vec3::Zero(), rf);
            }
        }

        const int NS = static_cast<int>(all_spheres.size());
        out.num_spheres = NS;
        out.sphere_body.resize(static_cast<size_t>(NS));
        out.sphere_center.resize(static_cast<size_t>(NS) * 3);
        out.sphere_radius.resize(static_cast<size_t>(NS));
        out.sphere_esdf_active.assign(static_cast<size_t>(NS), 0);

        for (int i = 0; i < NS; ++i) {
            const size_t si = static_cast<size_t>(i);
            const auto& gs = all_spheres[si];
            out.sphere_body[si] = static_cast<int32_t>(gs.body_id);
            out.sphere_center[si * 3 + 0] = static_cast<float>(gs.local_center(0));
            out.sphere_center[si * 3 + 1] = static_cast<float>(gs.local_center(1));
            out.sphere_center[si * 3 + 2] = static_cast<float>(gs.local_center(2));
            out.sphere_radius[si] = static_cast<float>(gs.radius);
            out.sphere_esdf_active[si] = gs.esdf_active ? 1 : 0;
        }

        // Self-collision sphere pairs filtered by ACM, touch_links, and layer/mask.
        for (int a = 0; a < NS; ++a) {
            const auto& sa = all_spheres[static_cast<size_t>(a)];
            if (!sa.geom_active || sa.body_id < 0) continue;
            for (int b = a + 1; b < NS; ++b) {
                const auto& sb = all_spheres[static_cast<size_t>(b)];
                if (!sb.geom_active || sb.body_id < 0 || sa.body_id == sb.body_id || sa.link_key == sb.link_key) continue;
                if (!body_movable[static_cast<size_t>(sa.body_id)] && !body_movable[static_cast<size_t>(sb.body_id)]) continue;
                if (sa.src != nullptr && sa.src->is_attached && sa.src->attached_touch_links.count(sb.link_key) > 0) continue;
                if (sb.src != nullptr && sb.src->is_attached && sb.src->attached_touch_links.count(sa.link_key) > 0) continue;
                if (is_pair_disabled_by_acm(sa.link_key, sb.link_key)) continue;
                if (((sa.layer & sb.mask) == 0) && ((sb.layer & sa.mask) == 0)) continue;
                out.self_sphere_pairs.push_back(static_cast<int32_t>(a));
                out.self_sphere_pairs.push_back(static_cast<int32_t>(b));
            }
        }

        // Export individual static shapes (never merging multi-shape concave assets into one OBB).
        std::vector<int> static_shape_ids;
        for (int i = 0; i < static_cast<int>(shapes.size()); ++i) {
            if (shapes[static_cast<size_t>(i)].body_id < 0 && shapes[static_cast<size_t>(i)].shape) {
                static_shape_ids.push_back(i);
            }
        }
        const int N_stat = static_cast<int>(static_shape_ids.size());
        out.num_static = N_stat;
        out.static_kind.resize(static_cast<size_t>(N_stat));
        out.static_pos.resize(static_cast<size_t>(N_stat) * 3);
        out.static_rot.resize(static_cast<size_t>(N_stat) * 9);
        out.static_size.resize(static_cast<size_t>(N_stat) * 3);

        for (int k = 0; k < N_stat; ++k) {
            const size_t sk = static_cast<size_t>(k);
            const auto& ws = shapes[static_cast<size_t>(static_shape_ids[sk])];
            const auto& sh = *ws.shape;
            Mat3 R = ws.local_frame.leftCols<3>();
            Vec3 p = ws.local_frame.col(3);
            Vec3 sz = Vec3::Zero();
            int32_t gkind = static_cast<int32_t>(sh.kind);

            if (sh.kind == 1) {
                sz = sh.box_half_size;
            } else if (sh.kind == 2) {
                sz = Vec3(sh.radius, 0.0, 0.0);
            } else if (sh.kind == 3 || sh.kind == 4) {
                sz = Vec3(sh.radius, 0.5 * sh.height, 0.0);
            } else if (sh.kind == 5) {
                // Static mesh -> tight PCA-OBB box proxy (exact Coal mesh verified in C++ finalize).
                gkind = 1;
                if (sh.hull_points.size() >= 3) {
                    Vec3 mean = Vec3::Zero();
                    for (const auto& pt : sh.hull_points) mean += pt;
                    mean /= static_cast<double>(sh.hull_points.size());
                    Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
                    for (const auto& pt : sh.hull_points) {
                        const Vec3 d = pt - mean;
                        cov.noalias() += d * d.transpose();
                    }
                    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(cov);
                    Mat3 basis = Mat3::Identity();
                    if (eig.info() == Eigen::Success) {
                        basis = eig.eigenvectors();
                        if (basis.determinant() < 0.0) basis.col(0) = -basis.col(0);
                    }
                    Vec3 lo = Vec3::Constant(std::numeric_limits<double>::infinity());
                    Vec3 hi = Vec3::Constant(-std::numeric_limits<double>::infinity());
                    for (const auto& pt : sh.hull_points) {
                        const Vec3 lp = basis.transpose() * (pt - mean);
                        lo = lo.cwiseMin(lp);
                        hi = hi.cwiseMax(lp);
                    }
                    const Vec3 obb_local_c = mean + basis * (0.5 * (lo + hi));
                    p = p + R * obb_local_c;
                    R = R * basis;
                    sz = 0.5 * (hi - lo);
                } else {
                    p = p + R * sh.convex_center;
                    sz = sh.convex_half;
                }
            } else if (sh.kind == 6) {
                sz = Vec3::Zero();
            }

            out.static_kind[sk] = gkind;
            for (int r = 0; r < 3; ++r) {
                out.static_pos[sk * 3 + static_cast<size_t>(r)] = static_cast<float>(p(r));
                out.static_size[sk * 3 + static_cast<size_t>(r)] = static_cast<float>(sz(r));
                for (int c = 0; c < 3; ++c) {
                    out.static_rot[sk * 9 + static_cast<size_t>(r * 3 + c)] = static_cast<float>(R(r, c));
                }
            }
        }

        // Active (sphere_idx, static_idx) pairs filtered by ACM, touch_links, layer/mask, and movability.
        for (int a = 0; a < NS; ++a) {
            const auto& sa = all_spheres[static_cast<size_t>(a)];
            if (!sa.geom_active || sa.body_id < 0 || !body_movable[static_cast<size_t>(sa.body_id)]) continue;
            for (int k = 0; k < N_stat; ++k) {
                const auto& wb = shapes[static_cast<size_t>(static_shape_ids[static_cast<size_t>(k)])];
                if (sa.link_key == wb.link_key) continue;
                if (sa.src != nullptr && sa.src->is_attached && sa.src->attached_touch_links.count(wb.link_key) > 0) continue;
                if (is_pair_disabled_by_acm(sa.link_key, wb.link_key)) continue;
                if (((sa.layer & wb.mask) == 0) && ((wb.layer & sa.mask) == 0)) continue;
                out.static_sphere_pairs.push_back(static_cast<int32_t>(a));
                out.static_sphere_pairs.push_back(static_cast<int32_t>(k));
            }
        }

        // ESDF metadata and shared grid reference.
        out.has_esdf = esdf_ok;
        if (esdf_ok) {
            out.esdf_nx = esdf->nx;
            out.esdf_ny = esdf->ny;
            out.esdf_nz = esdf->nz;
            out.esdf_origin = {static_cast<float>(esdf->origin(0)),
                               static_cast<float>(esdf->origin(1)),
                               static_cast<float>(esdf->origin(2))};
            out.esdf_resolution = static_cast<float>(esdf->resolution);
            out.esdf_from_occupancy = esdf->from_occupancy ? 1 : 0;
            out.esdf_inflation_radius = static_cast<float>(esdf->inflation_radius);
            out.esdf_max_distance = std::isfinite(esdf->max_distance)
                ? static_cast<float>(std::min(esdf->max_distance, 1e30))
                : 1e30f;
            out.esdf_lipschitz_bound = static_cast<float>(esdf->lipschitz_bound);
            out.esdf_grid = esdf;
        }
        return out;
    }
};

// Thread-safe mutable PlanningWorld backed by Copy-on-Write WorldSnapshot.
// Mutations clone the lightweight snapshot metadata (sharing immutable CollisionShape pointers),
// while concurrent planners/queries read an immutable snapshot lock-free.
class PlanningWorld {
public:
    PlanningWorld() : state_(std::make_shared<WorldSnapshot>()) {}

    std::shared_ptr<const WorldSnapshot> snapshot() const {
        std::lock_guard<std::mutex> lock(state_mutex_);
        return state_;
    }

    uint64_t version() const { return snapshot()->version; }
    uint64_t topology_version() const { return snapshot()->topology_version; }
    int num_bodies() const { return snapshot()->kin.num_bodies(); }
    int num_joints() const { return snapshot()->kin.num_joints(); }
    int num_shapes() const { return static_cast<int>(snapshot()->shapes.size()); }
    int num_pairs() const { return static_cast<int>(snapshot()->pairs.size()); }
    int self_pair_count() const { return snapshot()->self_pair_count; }
    bool has_mesh() const { return snapshot()->has_mesh; }
    bool has_esdf() const { return snapshot()->has_esdf(); }
    int num_link_spheres() const { return static_cast<int>(snapshot()->link_spheres.size()); }
    int num_esdf_spheres() const { return static_cast<int>(snapshot()->esdf_spheres.size()); }

    void init_kinematics(
        std::vector<int> parent,
        std::vector<int> kinds,
        std::vector<int> joint_idx,
        std::vector<Mat3> a_rot,
        std::vector<Vec3> a_pos,
        std::vector<Mat3> b_rot,
        std::vector<Vec3> b_pos,
        std::vector<Vec3> axes,
        std::vector<KinematicsModel::Branch> branches,
        std::vector<std::string> body_names,
        std::vector<std::string> joint_names,
        Eigen::VectorXd lower,
        Eigen::VectorXd upper,
        Eigen::VectorXd velocity_limit,
        Eigen::VectorXd acceleration_limit,
        Eigen::VectorXd initial_q,
        std::unordered_map<std::string, int> link_index) {
        std::lock_guard<std::mutex> lock(state_mutex_);
        auto next = std::make_shared<WorldSnapshot>(*state_);
        next->kin.initialize(std::move(parent), std::move(kinds), std::move(joint_idx),
                             std::move(a_rot), std::move(a_pos), std::move(b_rot), std::move(b_pos),
                             std::move(axes), std::move(branches), std::move(body_names), std::move(joint_names),
                             std::move(lower), std::move(upper), std::move(velocity_limit),
                             std::move(acceleration_limit), std::move(initial_q));
        next->link_index = std::move(link_index);
        next->rebuild_pairs();
        ++next->version;
        ++next->topology_version;
        state_ = std::move(next);
    }

    void add_shape(WorldShape ws) {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (state_->shape_index.count(ws.name))
            throw std::invalid_argument("duplicate collision shape name: " + ws.name);
        auto next = std::make_shared<WorldSnapshot>(*state_);
        ws.update_fixed_bounds();
        const int idx = static_cast<int>(next->shapes.size());
        next->shape_index[ws.name] = idx;
        next->shapes.push_back(std::move(ws));
        next->rebuild_pairs();
        ++next->version;
        ++next->topology_version;
        state_ = std::move(next);
    }

    void update_obstacle_pose(const std::string& name, const Frame& world_frame) {
        std::lock_guard<std::mutex> lock(state_mutex_);
        auto it = state_->shape_index.find(name);
        if (it == state_->shape_index.end()) throw std::invalid_argument("unknown obstacle: " + name);
        if (state_->shapes[static_cast<size_t>(it->second)].body_id >= 0)
            throw std::invalid_argument("cannot set world pose of attached/robot shape: " + name);
        auto next = std::make_shared<WorldSnapshot>(*state_);
        auto& ws = next->shapes[static_cast<size_t>(it->second)];
        ws.local_frame = world_frame;
        ws.update_fixed_bounds();
        ++next->version;
        state_ = std::move(next);
    }

    void remove_obstacle(const std::string& name) {
        std::lock_guard<std::mutex> lock(state_mutex_);
        auto it = state_->shape_index.find(name);
        if (it == state_->shape_index.end()) throw std::invalid_argument("unknown obstacle: " + name);
        const int idx = it->second;
        if (state_->shapes[static_cast<size_t>(idx)].body_id >= 0)
            throw std::invalid_argument("cannot remove robot/attached shape directly; detach first: " + name);
        auto next = std::make_shared<WorldSnapshot>(*state_);
        next->custom_link_spheres.erase("::" + name);
        next->shapes.erase(next->shapes.begin() + idx);
        next->shape_index.clear();
        for (size_t i = 0; i < next->shapes.size(); ++i) {
            next->shape_index[next->shapes[i].name] = static_cast<int>(i);
        }
        next->rebuild_pairs();
        ++next->version;
        ++next->topology_version;
        state_ = std::move(next);
    }

    void attach_object(const std::string& name, const std::string& link_key,
                       const Frame& local_frame, const std::vector<std::string>& touch_links) {
        std::lock_guard<std::mutex> lock(state_mutex_);
        auto it = state_->shape_index.find(name);
        if (it == state_->shape_index.end()) throw std::invalid_argument("unknown object to attach: " + name);
        const int bid = state_->link_id(link_key);
        auto next = std::make_shared<WorldSnapshot>(*state_);
        auto& ws = next->shapes[static_cast<size_t>(it->second)];
        if (!ws.is_attached) {
            ws.detached_link_key = ws.link_key;
        }
        ws.body_id = bid;
        ws.is_attached = true;
        ws.link_key = "::" + ws.name;
        ws.local_frame = local_frame;
        ws.attached_touch_links.clear();
        ws.attached_touch_links.insert(link_key);
        for (const auto& tl : touch_links) ws.attached_touch_links.insert(tl);
        next->rebuild_pairs();
        ++next->version;
        ++next->topology_version;
        state_ = std::move(next);
    }

    void detach_object(const std::string& name, const Eigen::VectorXd& q_current) {
        std::lock_guard<std::mutex> lock(state_mutex_);
        auto it = state_->shape_index.find(name);
        if (it == state_->shape_index.end()) throw std::invalid_argument("unknown object to detach: " + name);
        const auto& cur_ws = state_->shapes[static_cast<size_t>(it->second)];
        if (!cur_ws.is_attached || cur_ws.body_id < 0)
            throw std::invalid_argument("object is not attached: " + name);
        const int B = state_->kin.num_bodies();
        std::vector<Vec3> bpos(static_cast<size_t>(B));
        std::vector<Mat3> brot(static_cast<size_t>(B));
        state_->kin.fk_single(q_current.data(), bpos.data(), brot.data());
        const Mat3& R = brot[static_cast<size_t>(cur_ws.body_id)];
        const Vec3& p = bpos[static_cast<size_t>(cur_ws.body_id)];
        Frame world_f;
        world_f.leftCols<3>() = R * cur_ws.local_frame.leftCols<3>();
        world_f.col(3) = R * cur_ws.local_frame.col(3) + p;

        auto next = std::make_shared<WorldSnapshot>(*state_);
        auto& ws = next->shapes[static_cast<size_t>(it->second)];
        ws.body_id = -1;
        ws.is_attached = false;
        ws.attached_touch_links.clear();
        ws.link_key = ws.detached_link_key.empty() ? ("::" + ws.name) : ws.detached_link_key;
        ws.detached_link_key.clear();
        ws.local_frame = world_f;
        ws.update_fixed_bounds();
        next->rebuild_pairs();
        ++next->version;
        ++next->topology_version;
        state_ = std::move(next);
    }

    void set_auto_collision_disabled(const std::string& link_a, const std::string& link_b) {
        std::lock_guard<std::mutex> lock(state_mutex_);
        const auto key = canonical_pair_key(link_a, link_b);
        if (state_->auto_disabled_pairs.count(key) > 0) return;
        auto next = std::make_shared<WorldSnapshot>(*state_);
        next->auto_disabled_pairs.insert(key);
        next->rebuild_pairs();
        ++next->version;
        ++next->topology_version;
        state_ = std::move(next);
    }

    void set_collision_enabled(const std::string& link_a, const std::string& link_b, bool enabled) {
        std::lock_guard<std::mutex> lock(state_mutex_);
        auto next = std::make_shared<WorldSnapshot>(*state_);
        const auto key = canonical_pair_key(link_a, link_b);
        next->user_acm_overrides[key] = enabled;
        next->rebuild_pairs();
        ++next->version;
        ++next->topology_version;
        state_ = std::move(next);
    }

    void set_robot_base_pose(int branch_idx, const Vec3& pos, const Mat3& rot) {
        std::lock_guard<std::mutex> lock(state_mutex_);
        auto next = std::make_shared<WorldSnapshot>(*state_);
        next->kin.set_branch_pose(branch_idx, pos, rot);
        ++next->version;
        state_ = std::move(next);
    }

    void set_jerk_limits(const Eigen::VectorXd& jerk_limits) {
        std::lock_guard<std::mutex> lock(state_mutex_);
        auto next = std::make_shared<WorldSnapshot>(*state_);
        next->kin.set_jerk_limits(jerk_limits);
        ++next->version;
        state_ = std::move(next);
    }

    // --- Voxel / 3D ESDF & Link Sphere-Cover APIs ---

    void set_occupancy_grid(const uint8_t* occupied, int nx, int ny, int nz,
                            const Vec3& origin, double resolution,
                            double max_distance = std::numeric_limits<double>::infinity(),
                            double inflation_radius = 0.0,
                            int collision_layer = 1, int collision_mask = 1) {
        auto grid = VoxelESDF::from_occupancy_grid(
            occupied, nx, ny, nz, origin, resolution, max_distance, inflation_radius,
            collision_layer, collision_mask);
        std::lock_guard<std::mutex> lock(state_mutex_);
        auto next = std::make_shared<WorldSnapshot>(*state_);
        next->esdf = std::move(grid);
        next->rebuild_esdf_spheres();
        ++next->version;
        state_ = std::move(next);
    }

    void set_point_cloud(const double* points_xyz, int64_t num_points, double resolution,
                         const std::optional<Vec3>& origin = std::nullopt,
                         const std::optional<std::array<int, 3>>& shape = std::nullopt,
                         double padding = 0.2,
                         double point_radius = 0.0,
                         double max_distance = std::numeric_limits<double>::infinity(),
                         int collision_layer = 1, int collision_mask = 1) {
        auto grid = VoxelESDF::from_point_cloud(
            points_xyz, num_points, resolution, origin, shape, padding, point_radius,
            max_distance, collision_layer, collision_mask);
        std::lock_guard<std::mutex> lock(state_mutex_);
        auto next = std::make_shared<WorldSnapshot>(*state_);
        next->esdf = std::move(grid);
        next->rebuild_esdf_spheres();
        ++next->version;
        state_ = std::move(next);
    }

    void set_esdf(const double* values, int nx, int ny, int nz,
                  const Vec3& origin, double resolution,
                  int collision_layer = 1, int collision_mask = 1) {
        auto grid = VoxelESDF::from_esdf_array(
            values, nx, ny, nz, origin, resolution, collision_layer, collision_mask);
        std::lock_guard<std::mutex> lock(state_mutex_);
        auto next = std::make_shared<WorldSnapshot>(*state_);
        next->esdf = std::move(grid);
        next->rebuild_esdf_spheres();
        ++next->version;
        state_ = std::move(next);
    }

    void clear_esdf() {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (!state_->esdf) return;
        auto next = std::make_shared<WorldSnapshot>(*state_);
        next->esdf.reset();
        next->esdf_spheres.clear();
        ++next->version;
        state_ = std::move(next);
    }

    void set_link_spheres(const std::string& link_key,
                          const std::vector<std::pair<Vec3, double>>& spheres,
                          int collision_layer = 1, int collision_mask = 1) {
        for (const auto& [c, r] : spheres) {
            if (!c.allFinite() || !std::isfinite(r) || r < 0.0) {
                throw std::invalid_argument("link sphere center must be finite and radius must be non-negative");
            }
        }
        std::lock_guard<std::mutex> lock(state_mutex_);
        bool valid_link = state_->link_index.count(link_key) > 0;
        if (!valid_link) {
            for (const auto& ws : state_->shapes) {
                if (ws.body_id >= 0 && ws.link_key == link_key) {
                    valid_link = true;
                    break;
                }
            }
        }
        if (!valid_link) {
            throw std::invalid_argument("unknown link for custom spheres: " + link_key);
        }
        auto next = std::make_shared<WorldSnapshot>(*state_);
        next->custom_link_spheres[link_key] = CustomLinkSpheres{spheres, collision_layer, collision_mask};
        next->rebuild_link_spheres();
        ++next->version;
        state_ = std::move(next);
    }

    void clear_link_spheres(const std::string& link_key = "") {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (link_key.empty()) {
            if (state_->custom_link_spheres.empty()) return;
            auto next = std::make_shared<WorldSnapshot>(*state_);
            next->custom_link_spheres.clear();
            next->rebuild_link_spheres();
            ++next->version;
            state_ = std::move(next);
            return;
        }
        if (state_->custom_link_spheres.count(link_key) == 0) return;
        auto next = std::make_shared<WorldSnapshot>(*state_);
        next->custom_link_spheres.erase(link_key);
        next->rebuild_link_spheres();
        ++next->version;
        state_ = std::move(next);
    }

    std::vector<LinkSphere> link_spheres() const {
        return snapshot()->link_spheres;
    }

    std::vector<LinkSphere> esdf_spheres() const {
        return snapshot()->esdf_spheres;
    }

    void query_esdf(const double* points_xyz, int64_t N, double* out_dist, double* out_grad = nullptr) const {
        auto snap = snapshot();
        const auto& grid = snap->esdf;
        if (!grid || !grid->valid() || grid->is_empty) {
            std::fill(out_dist, out_dist + N, std::numeric_limits<double>::infinity());
            if (out_grad != nullptr) std::fill(out_grad, out_grad + N * 3, 0.0);
            return;
        }
        ThreadPool::instance().parallel_for(N, 256, [&](int64_t b, int64_t e) {
            for (int64_t i = b; i < e; ++i) {
                const Vec3 p(points_xyz[i * 3 + 0], points_xyz[i * 3 + 1], points_xyz[i * 3 + 2]);
                if (out_grad != nullptr) {
                    Vec3 g;
                    out_dist[i] = grid->query(p, &g, false);
                    out_grad[i * 3 + 0] = g(0);
                    out_grad[i * 3 + 1] = g(1);
                    out_grad[i * 3 + 2] = g(2);
                } else {
                    out_dist[i] = grid->query(p, nullptr, false);
                }
            }
        });
    }

    void query_esdf_sphere_distances(const double* q_batch, int64_t S, double* out_dists) const {
        snapshot()->query_esdf_sphere_distances(q_batch, S, out_dists);
    }

    // Atomic batch synchronization: validates everything on a single CoW copy, rebuilds pairs
    // at most once when topology/filter changes, and publishes one new snapshot (or zero if no-op).
    // Strong exception guarantee: any validation error leaves state_ completely untouched.
    SyncResultNative apply_sync_batch(const SyncBatch& batch) {
        std::lock_guard<std::mutex> lock(state_mutex_);
        SyncResultNative out;
        out.version = state_->version;
        out.topology_version = state_->topology_version;
        if (batch.base_updates.empty() && batch.pose_updates.empty() &&
            batch.remove_names.empty() && batch.replace_shapes.empty() &&
            batch.add_shapes.empty()) {
            return out;
        }

        auto next = std::make_shared<WorldSnapshot>(*state_);
        bool topology_changed = false;

        // 1. Robot base pose updates (validates branch index).
        for (const auto& bu : batch.base_updates) {
            next->kin.set_branch_pose(bu.branch_idx, bu.pos, bu.rot);
            ++out.robot_bases_updated;
        }

        // 2. Removals (must exist, must be static non-attached obstacle, no duplicates).
        if (!batch.remove_names.empty()) {
            std::vector<uint8_t> remove_mask(next->shapes.size(), 0);
            for (const auto& name : batch.remove_names) {
                auto it = next->shape_index.find(name);
                if (it == next->shape_index.end()) {
                    throw std::invalid_argument("sync remove: unknown obstacle: " + name);
                }
                const size_t idx = static_cast<size_t>(it->second);
                if (remove_mask[idx]) {
                    throw std::invalid_argument("sync remove: duplicate obstacle name: " + name);
                }
                const auto& ws = next->shapes[idx];
                if (ws.body_id >= 0 || ws.is_attached) {
                    throw std::invalid_argument("sync remove: cannot remove robot/attached shape: " + name);
                }
                next->custom_link_spheres.erase("::" + name);
                remove_mask[idx] = 1;
                ++out.shapes_removed;
            }
            std::vector<WorldShape> kept;
            kept.reserve(next->shapes.size() - static_cast<size_t>(out.shapes_removed));
            for (size_t i = 0; i < next->shapes.size(); ++i) {
                if (!remove_mask[i]) kept.push_back(std::move(next->shapes[i]));
            }
            next->shapes = std::move(kept);
            next->shape_index.clear();
            for (size_t i = 0; i < next->shapes.size(); ++i) {
                next->shape_index[next->shapes[i].name] = static_cast<int>(i);
            }
            topology_changed = true;
        }

        // 3. Replacements (geometry, filter, or link_key change on existing static obstacle).
        for (const auto& rep : batch.replace_shapes) {
            auto it = next->shape_index.find(rep.name);
            if (it == next->shape_index.end()) {
                throw std::invalid_argument("sync replace: unknown obstacle: " + rep.name);
            }
            auto& ws = next->shapes[static_cast<size_t>(it->second)];
            if (ws.body_id >= 0 || ws.is_attached) {
                throw std::invalid_argument("sync replace: cannot replace robot/attached shape: " + rep.name);
            }
            if (!rep.link_key.empty()) ws.link_key = rep.link_key;
            ws.local_frame = rep.world_frame;
            ws.layer = rep.layer;
            ws.mask = rep.mask;
            if (rep.shape) ws.shape = rep.shape;
            ws.update_fixed_bounds();
            ++out.shapes_replaced;
            topology_changed = true;
        }

        // 4. Additions (new static obstacles).
        for (const auto& add_ws : batch.add_shapes) {
            if (add_ws.body_id >= 0 || add_ws.is_attached) {
                throw std::invalid_argument("sync add: added obstacle must be static: " + add_ws.name);
            }
            if (!add_ws.shape) {
                throw std::invalid_argument("sync add: null collision shape: " + add_ws.name);
            }
            if (next->shape_index.count(add_ws.name)) {
                throw std::invalid_argument("sync add: duplicate collision shape name: " + add_ws.name);
            }
            WorldShape ws = add_ws;
            ws.update_fixed_bounds();
            const int idx = static_cast<int>(next->shapes.size());
            next->shape_index[ws.name] = idx;
            next->shapes.push_back(std::move(ws));
            ++out.shapes_added;
            topology_changed = true;
        }

        // 5. Pure pose updates (no geometry/BVH/pair rebuild).
        for (const auto& pu : batch.pose_updates) {
            auto it = next->shape_index.find(pu.name);
            if (it == next->shape_index.end()) {
                throw std::invalid_argument("sync pose: unknown obstacle: " + pu.name);
            }
            auto& ws = next->shapes[static_cast<size_t>(it->second)];
            if (ws.body_id >= 0 || ws.is_attached) {
                throw std::invalid_argument("sync pose: cannot set world pose of robot/attached shape: " + pu.name);
            }
            ws.local_frame = pu.world_frame;
            ws.update_fixed_bounds();
            ++out.poses_updated;
        }

        if (topology_changed) {
            next->rebuild_pairs();
            ++next->topology_version;
        }
        ++next->version;
        state_ = std::move(next);

        out.version = state_->version;
        out.topology_version = state_->topology_version;
        out.changed = true;
        return out;
    }

    std::vector<std::string> attached_shape_names() const {
        return snapshot()->attached_shape_names();
    }

    std::vector<std::string> shape_names() const {
        return snapshot()->shape_names();
    }

    // Convenience thread-safe pass-throughs operating on an atomic immutable snapshot.
    Eigen::VectorXd pair_margins(double margin, double world_margin) const {
        return snapshot()->pair_margins(margin, world_margin);
    }
    void compute_frames(const double* q_batch, int64_t S, double* out_frames,
                        double* out_jorig = nullptr, double* out_jaxis = nullptr) const {
        snapshot()->compute_frames(q_batch, S, out_frames, out_jorig, out_jaxis);
    }
    void screen_aabbs(const double* frames, int64_t S, const double* thresholds,
                      bool verify_original, int64_t group_size,
                      std::vector<int64_t>& out_rows, std::vector<int64_t>& out_pairs) const {
        snapshot()->screen_aabbs(frames, S, thresholds, verify_original, group_size, out_rows, out_pairs);
    }
    void query_distances_indexed(const double* frames, int64_t S,
                                 const int64_t* rows, const int64_t* pair_ids, int64_t K,
                                 bool verify, double* out) const {
        snapshot()->query_distances_indexed(frames, S, rows, pair_ids, K, verify, out);
    }
    std::optional<std::pair<int64_t, int64_t>> first_violation_indexed(
        const double* frames, int64_t S, const int64_t* rows, const int64_t* pair_ids,
        int64_t K, const double* margins, bool verify) const {
        return snapshot()->first_violation_indexed(frames, S, rows, pair_ids, K, margins, verify);
    }
    void violating_pairs_indexed(
        const double* frames, int64_t S, const int64_t* rows, const int64_t* pair_ids,
        int64_t K, const double* margins, bool verify,
        std::vector<int64_t>& out_rows, std::vector<int64_t>& out_pairs) const {
        snapshot()->violating_pairs_indexed(frames, S, rows, pair_ids, K, margins, verify, out_rows, out_pairs);
    }
    std::optional<ContinuousViolation> first_violation_continuous(
        const double* q_waypoints, int T, int substeps, const double* frames,
        const double* margins, bool verify_original = false, double disp_tol = 5e-4,
        const double* times = nullptr, const double* velocities = nullptr,
        bool strict = false, const double* accelerations = nullptr, int spline_order = 3,
        double esdf_margin = -1.0) const {
        return snapshot()->first_violation_continuous(
            q_waypoints, T, substeps, frames, margins, verify_original, disp_tol, times, velocities,
            strict, accelerations, spline_order, esdf_margin);
    }
    void compute_analytic_gradients(const double* frames, const double* jorig, const double* jaxis,
                                    const int64_t* rows, const int64_t* pair_ids, const double* data,
                                    int64_t K, double* out_grad, double* out_witnesses = nullptr) const {
        snapshot()->compute_analytic_gradients(frames, jorig, jaxis, rows, pair_ids, data, K, out_grad, out_witnesses);
    }
    CollisionEvalResult eval_collision_rows(const double* q_batch, int64_t S,
                                            const double* opt_margins, const double* check_margins,
                                            double buffer, bool linearize, int64_t group_size,
                                            bool segment_minimum,
                                            double esdf_opt_margin = -1.0,
                                            double esdf_check_margin = -1.0) const {
        return snapshot()->eval_collision_rows(q_batch, S, opt_margins, check_margins, buffer, linearize,
                                               group_size, segment_minimum, esdf_opt_margin, esdf_check_margin);
    }
    std::vector<IKResultNative> solve_ik_batch(const std::vector<IKProblem>& problems,
                                               const std::vector<uint8_t>& active_mask,
                                               double margin, double world_margin,
                                               int iterations, bool verify_original) const {
        return snapshot()->solve_ik_batch(problems, active_mask, margin, world_margin, iterations, verify_original);
    }
    RRTResultNative plan_birrt(const Eigen::VectorXd& start, const Eigen::VectorXd& goal,
                               const std::vector<uint8_t>& active_mask,
                               double margin, double world_margin,
                               double step_size, int max_iterations, double timeout_s,
                               uint64_t rng_seed, bool verify_original,
                               int shortcut_iters, int output_waypoints) const {
        return snapshot()->plan_birrt(start, goal, active_mask, margin, world_margin, step_size,
                                      max_iterations, timeout_s, rng_seed, verify_original,
                                      shortcut_iters, output_waypoints);
    }

private:
    mutable std::mutex state_mutex_;
    std::shared_ptr<const WorldSnapshot> state_;
};

}  // namespace gobot::planning
