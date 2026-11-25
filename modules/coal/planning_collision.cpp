#include "planning_planner.hpp"
#include "planning_qp.hpp"
#include "planning_retime.hpp"
#include "planning_sqp.hpp"
#include "planning_world.hpp"

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
namespace gp = gobot::planning;

namespace gobot::planning {
int gobot_planning_abi_version() noexcept;
bool gobot_planning_has_tbb() noexcept;
}  // namespace gobot::planning

namespace {
using Input = py::array_t<double, py::array::c_style | py::array::forcecast>;
using Indices = py::array_t<int64_t, py::array::c_style | py::array::forcecast>;
using MaskInput = py::array_t<bool, py::array::c_style | py::array::forcecast>;

std::shared_ptr<gp::CollisionShape> make_collision_shape(const std::string& name, const py::dict& spec) {
    auto sh = std::make_shared<gp::CollisionShape>();
    sh->label = name;
    sh->kind = spec["type"].cast<int>();
    switch (sh->kind) {
        case 1: {
            const auto sz = spec["size"].cast<coal::Vec3s>();
            sh->convex = std::make_shared<coal::Box>(sz);
            sh->box_half_size = sz * 0.5;
            break;
        }
        case 2: {
            sh->radius = spec["radius"].cast<double>();
            sh->convex = std::make_shared<coal::Sphere>(sh->radius);
            break;
        }
        case 3: {
            sh->radius = spec["radius"].cast<double>();
            sh->height = spec["height"].cast<double>();
            sh->convex = std::make_shared<coal::Cylinder>(sh->radius, sh->height);
            break;
        }
        case 4: {
            sh->radius = spec["radius"].cast<double>();
            sh->height = spec["height"].cast<double>();
            sh->convex = std::make_shared<coal::Capsule>(sh->radius, sh->height);
            break;
        }
        case 5: {
            const auto points = spec["vertices"].cast<std::vector<coal::Vec3s>>();
            const auto indices = spec["indices"].cast<std::vector<size_t>>();
            if (points.size() < 3 || indices.empty() || indices.size() % 3 != 0)
                throw std::invalid_argument(name + ": expected an indexed triangle mesh");
            std::vector<coal::Triangle> triangles;
            triangles.reserve(indices.size() / 3);
            for (size_t i = 0; i < indices.size(); i += 3) {
                for (size_t j = i; j < i + 3; ++j)
                    if (indices[j] >= points.size()) throw std::invalid_argument(name + ": invalid triangle index");
                triangles.emplace_back(indices[i], indices[i + 1], indices[i + 2]);
            }
            auto mesh = std::make_shared<coal::BVHModel<coal::OBBRSS>>();
            if (mesh->beginModel() != 0 || mesh->addSubModel(points, triangles) != 0 || mesh->endModel() != 0)
                throw std::invalid_argument(name + ": failed to build triangle BVH");
            sh->original = mesh;
            auto hull = std::make_shared<std::vector<coal::Vec3s>>(spec["hull_points"].cast<std::vector<coal::Vec3s>>());
            sh->hull_points = *hull;
            auto faces = std::make_shared<std::vector<coal::Triangle>>();
            for (const auto& face : spec["hull_faces"].cast<std::vector<std::array<size_t, 3>>>()) {
                if (face[0] >= hull->size() || face[1] >= hull->size() || face[2] >= hull->size())
                    throw std::invalid_argument(name + ": invalid convex hull face");
                faces->emplace_back(face[0], face[1], face[2]);
            }
            if (faces->empty()) throw std::invalid_argument(name + ": empty convex hull");
            sh->convex = std::make_shared<coal::Convex<coal::Triangle>>(hull, hull->size(), faces, faces->size());
            break;
        }
        case 6:
            sh->convex = std::make_shared<coal::Halfspace>(coal::Vec3s::UnitZ(), 0);
            break;
        default:
            throw std::invalid_argument(name + ": unsupported collision shape");
    }
    sh->finalize_bounds();
    return sh;
}

gp::Frame frame_from_array(const Input& arr) {
    if (arr.ndim() != 2 || arr.shape(0) != 3 || arr.shape(1) != 4)
        throw std::invalid_argument("expected 3x4 rigid frame");
    gp::Frame f;
    std::copy(arr.data(), arr.data() + 12, f.data());
    return f;
}

gp::SolverConfigNative parse_solver_config(const py::dict& d) {
    gp::SolverConfigNative cfg;
    cfg.waypoints = d["waypoints"].cast<int>();
    cfg.margin = d["margin"].cast<double>();
    if (d.contains("world_margin") && !d["world_margin"].is_none()) {
        cfg.has_world_margin = true;
        cfg.world_margin = d["world_margin"].cast<double>();
    }
    cfg.collision_substeps = d["collision_substeps"].cast<int>();
    if (d.contains("max_segment_length") && !d["max_segment_length"].is_none()) {
        cfg.max_segment_length = d["max_segment_length"].cast<double>();
    }
    cfg.iterations = d["iterations"].cast<int>();
    cfg.stop_at_endpoints = d["stop_at_endpoints"].cast<bool>();
    cfg.first_feasible = d["first_feasible"].cast<bool>();
    cfg.use_global_seeding = d["use_global_seeding"].cast<bool>();
    cfg.num_seeds = d["num_seeds"].cast<int>();
    cfg.rrt_timeout_s = d["rrt_timeout_s"].cast<double>();
    cfg.rrt_step_size = d["rrt_step_size"].cast<double>();
    cfg.rrt_max_iterations = d["rrt_max_iterations"].cast<int>();
    cfg.rrt_shortcut_iters = d["rrt_shortcut_iters"].cast<int>();
    cfg.velocity_scale = d["velocity_scale"].cast<double>();
    cfg.acceleration_scale = d["acceleration_scale"].cast<double>();
    if (d.contains("jerk_scale") && !d["jerk_scale"].is_none()) {
        cfg.jerk_scale = d["jerk_scale"].cast<double>();
    }
    if (d.contains("spline_order") && !d["spline_order"].is_none()) {
        cfg.spline_order = d["spline_order"].cast<int>();
    }
    if (d.contains("strict_ccd") && !d["strict_ccd"].is_none()) {
        cfg.strict_ccd = d["strict_ccd"].cast<bool>();
    }
    if (d.contains("dt") && !d["dt"].is_none()) {
        cfg.has_dt = true;
        cfg.dt = d["dt"].cast<double>();
    }
    cfg.retime = d["retime"].cast<bool>();
    return cfg;
}

gp::MotionPlanRequestNative parse_plan_request(const py::dict& d, int J) {
    gp::MotionPlanRequestNative req;
    req.config = parse_solver_config(d["config"].cast<py::dict>());
    req.start_is_pose = d["start_is_pose"].cast<bool>();
    if (req.start_is_pose) {
        req.start_poses = d["start_poses"].cast<std::vector<gp::PoseConstraintSpec>>();
    } else {
        req.start_q = d["start_q"].cast<Eigen::VectorXd>();
    }
    if (d.contains("start_seed") && !d["start_seed"].is_none()) {
        req.has_start_seed = true;
        req.start_seed = d["start_seed"].cast<Eigen::VectorXd>();
    }
    req.goal_is_pose = d["goal_is_pose"].cast<bool>();
    if (req.goal_is_pose) {
        req.goal_pose_sets = d["goal_pose_sets"].cast<std::vector<std::vector<gp::PoseConstraintSpec>>>();
    } else {
        req.goal_qs = d["goal_qs"].cast<std::vector<Eigen::VectorXd>>();
    }
    if (d.contains("goal_seed") && !d["goal_seed"].is_none()) {
        req.has_goal_seed = true;
        req.goal_seed = d["goal_seed"].cast<Eigen::VectorXd>();
    }
    if (d.contains("pose_constraints") && !d["pose_constraints"].is_none()) {
        const py::dict pc_dict = d["pose_constraints"].cast<py::dict>();
        for (const auto& kv : pc_dict) {
            auto vec = kv.second.cast<std::vector<gp::PoseConstraintSpec>>();
            if (!vec.empty()) req.pose_constraints.emplace_back(kv.first.cast<int>(), std::move(vec));
        }
    }
    if (d.contains("joint_targets") && !d["joint_targets"].is_none()) {
        const py::dict jt_dict = d["joint_targets"].cast<py::dict>();
        for (const auto& kv : jt_dict) {
            req.joint_targets.emplace_back(kv.first.cast<int>(), kv.second.cast<Eigen::VectorXd>());
        }
    }
    if (d.contains("seed") && !d["seed"].is_none()) {
        const Input s_arr = d["seed"].cast<Input>();
        if (s_arr.ndim() != 2 || s_arr.shape(1) != J)
            throw std::invalid_argument("seed must have shape [T, J]");
        req.has_seed = true;
        req.seed.resize(s_arr.shape(0), J);
        std::copy(s_arr.data(), s_arr.data() + s_arr.shape(0) * J, req.seed.data());
    }
    if (d.contains("active_mask") && !d["active_mask"].is_none()) {
        const MaskInput m_in = d["active_mask"].cast<MaskInput>();
        req.active_mask.assign(static_cast<size_t>(J), 1);
        if (m_in.size() == J) {
            for (int j = 0; j < J; ++j) req.active_mask[static_cast<size_t>(j)] = m_in.data()[j] ? 1 : 0;
        }
    }
    return req;
}

py::dict trajectory_to_dict(const gp::TrajectoryNative& traj) {
    const py::ssize_t T = traj.num_waypoints();
    const py::ssize_t J = traj.num_joints();
    py::array_t<double> times(T), pos({T, J}), vel({T, J}), acc({T, J});
    std::copy(traj.times.data(), traj.times.data() + T, times.mutable_data());
    std::copy(traj.positions.data(), traj.positions.data() + T * J, pos.mutable_data());
    std::copy(traj.velocities.data(), traj.velocities.data() + T * J, vel.mutable_data());
    std::copy(traj.accelerations.data(), traj.accelerations.data() + T * J, acc.mutable_data());
    py::dict d;
    d["times"] = std::move(times);
    d["positions"] = std::move(pos);
    d["velocities"] = std::move(vel);
    d["accelerations"] = std::move(acc);
    d["dt"] = traj.has_dt ? py::cast(traj.dt) : py::none();
    d["spline_order"] = traj.spline_order;
    return d;
}

py::dict plan_result_to_dict(const gp::PlanResultNative& r) {
    py::dict d;
    d["success"] = r.success;
    d["trajectory"] = r.success ? py::object(trajectory_to_dict(r.trajectory)) : py::none();
    if (r.has_initial_positions) {
        const py::ssize_t T = r.initial_positions.rows();
        const py::ssize_t J = r.initial_positions.cols();
        py::array_t<double> init_arr({T, J});
        std::copy(r.initial_positions.data(), r.initial_positions.data() + T * J, init_arr.mutable_data());
        d["initial_positions"] = std::move(init_arr);
    } else {
        d["initial_positions"] = py::none();
    }
    d["goal_index"] = r.goal_index;
    d["world_version"] = r.world_version;
    d["reason"] = r.success ? py::none() : py::object(py::str(r.reason));
    py::dict diag;
    diag["total_ms"] = r.diagnostics.total_ms;
    diag["ik_ms"] = r.diagnostics.ik_ms;
    diag["seed_ms"] = r.diagnostics.seed_ms;
    diag["sqp_ms"] = r.diagnostics.sqp_ms;
    diag["validate_ms"] = r.diagnostics.validate_ms;
    diag["retime_ms"] = r.diagnostics.retime_ms;
    diag["qp_calls"] = r.diagnostics.qp_calls;
    diag["sqp_iterations"] = r.diagnostics.sqp_iterations;
    diag["seed_strategy"] = r.diagnostics.seed_strategy;
    if (!r.diagnostics.witnesses.empty()) {
        const py::ssize_t K = static_cast<py::ssize_t>(r.diagnostics.witnesses.size() / 6);
        py::array_t<double> wit_arr({K, py::ssize_t(6)});
        std::copy(r.diagnostics.witnesses.begin(), r.diagnostics.witnesses.end(), wit_arr.mutable_data());
        diag["witnesses"] = std::move(wit_arr);
    } else {
        diag["witnesses"] = py::none();
    }
    d["diagnostics"] = std::move(diag);
    return d;
}

// Standalone forward_kinematics kept for gobot._kinematics.SceneKinematics CPU acceleration.
py::tuple forward_kinematics(const Indices& parent, const Indices& kinds, const Indices& joint_idx,
                             const Input& parent_joint, const Input& joint_child, const Input& axes,
                             const Indices& tracked_idx, const Input& root_pos, const Input& root_rot,
                             const Input& q, bool return_joints) {
    const py::ssize_t B = parent.size();
    const py::ssize_t T = tracked_idx.size();
    const py::ssize_t S = q.shape(0);
    const py::ssize_t J = q.shape(1);
    const bool batched_root = (root_pos.ndim() == 2);

    py::array_t<double> out_pos({S, T, py::ssize_t(3)});
    py::array_t<double> out_rot({S, T, py::ssize_t(3), py::ssize_t(3)});
    py::array_t<double> out_jpos = return_joints ? py::array_t<double>({S, J, py::ssize_t(3)}) : py::array_t<double>(0);
    py::array_t<double> out_jaxis = return_joints ? py::array_t<double>({S, J, py::ssize_t(3)}) : py::array_t<double>(0);

    const int64_t* par = parent.data();
    const int64_t* knd = kinds.data();
    const int64_t* jid = joint_idx.data();
    const int64_t* trk = tracked_idx.data();
    const double* pj_ptr = parent_joint.data();
    const double* jc_ptr = joint_child.data();
    const double* ax_ptr = axes.data();
    const double* rp_ptr = root_pos.data();
    const double* rr_ptr = root_rot.data();
    const double* q_ptr = q.data();

    double* opos = out_pos.mutable_data();
    double* orot = out_rot.mutable_data();
    double* ojpos = return_joints ? out_jpos.mutable_data() : nullptr;
    double* ojax = return_joints ? out_jaxis.mutable_data() : nullptr;

    {
        py::gil_scoped_release release;
        std::vector<gp::Mat3> a_rot(static_cast<size_t>(B)), b_rot(static_cast<size_t>(B));
        std::vector<gp::Vec3> a_pos(static_cast<size_t>(B)), b_pos(static_cast<size_t>(B)), ax(static_cast<size_t>(B));
        std::vector<gp::Mat3> skew(static_cast<size_t>(B)), skew2(static_cast<size_t>(B));
        for (py::ssize_t i = 0; i < B; ++i) {
            const double* a = pj_ptr + i * 16;
            const double* b = jc_ptr + i * 16;
            a_rot[i] << a[0], a[1], a[2], a[4], a[5], a[6], a[8], a[9], a[10];
            a_pos[i] << a[3], a[7], a[11];
            b_rot[i] << b[0], b[1], b[2], b[4], b[5], b[6], b[8], b[9], b[10];
            b_pos[i] << b[3], b[7], b[11];
            ax[i] << ax_ptr[i * 3 + 0], ax_ptr[i * 3 + 1], ax_ptr[i * 3 + 2];
            skew[i] = gp::skew_matrix(ax[i]);
            skew2[i] = skew[i] * skew[i];
        }
        const gp::Mat3 eye3 = gp::Mat3::Identity();

        auto work = [&](int64_t s_begin, int64_t s_end) {
            std::vector<gp::Vec3> pos(static_cast<size_t>(B));
            std::vector<gp::Mat3> rot(static_cast<size_t>(B));
            for (int64_t s = s_begin; s < s_end; ++s) {
                const double* rp = batched_root ? (rp_ptr + s * 3) : rp_ptr;
                const double* rr = batched_root ? (rr_ptr + s * 9) : rr_ptr;
                pos[0] << rp[0], rp[1], rp[2];
                rot[0] = Eigen::Map<const gp::Mat3>(rr);
                const double* qs = q_ptr + s * J;
                for (py::ssize_t i = 1; i < B; ++i) {
                    const int64_t p = par[i], j = jid[i], kind = knd[i];
                    const gp::Mat3 anchor_r = rot[static_cast<size_t>(p)] * a_rot[i];
                    gp::Vec3 anchor_p = pos[static_cast<size_t>(p)] + rot[static_cast<size_t>(p)] * a_pos[i];
                    gp::Mat3 moving_r = anchor_r;
                    if (kind == 1 || kind == 2) {
                        const double angle = qs[j];
                        moving_r.noalias() = anchor_r * (eye3 + std::sin(angle) * skew[i] + (1.0 - std::cos(angle)) * skew2[i]);
                    }
                    gp::Vec3 world_axis;
                    if (kind == 3 || (return_joints && j >= 0)) world_axis.noalias() = anchor_r * ax[i];
                    if (return_joints && j >= 0) {
                        Eigen::Map<gp::Vec3>(ojpos + (s * J + j) * 3) = anchor_p;
                        Eigen::Map<gp::Vec3>(ojax + (s * J + j) * 3) = world_axis;
                    }
                    if (kind == 3) anchor_p.noalias() += world_axis * qs[j];
                    pos[i].noalias() = anchor_p + moving_r * b_pos[i];
                    rot[i].noalias() = moving_r * b_rot[i];
                }
                for (py::ssize_t t = 0; t < T; ++t) {
                    const size_t idx = static_cast<size_t>(trk[t]);
                    Eigen::Map<gp::Vec3>(opos + (s * T + t) * 3) = pos[idx];
                    Eigen::Map<gp::Mat3>(orot + (s * T + t) * 9) = rot[idx];
                }
            }
        };
        gp::ThreadPool::instance().parallel_for(S, 64, work);
    }
    if (return_joints) return py::make_tuple(out_pos, out_rot, out_jpos, out_jaxis);
    return py::make_tuple(out_pos, out_rot);
}

void check_frames_shape(const Input& frames, int64_t expected_N) {
    if (frames.ndim() != 4 || frames.shape(1) != expected_N || frames.shape(2) != 3 || frames.shape(3) != 4) {
        throw std::invalid_argument(
            "frames shape mismatch: expected [S, " + std::to_string(expected_N) + ", 3, 4], got ndim="
            + std::to_string(frames.ndim())
            + (frames.ndim() >= 2 ? (" shape[1]=" + std::to_string(frames.shape(1))) : ""));
    }
}

void check_margins_shape(const Input& margins, int64_t expected_P) {
    if (margins.ndim() != 1 || margins.shape(0) != expected_P) {
        throw std::invalid_argument(
            "margins shape mismatch: expected [" + std::to_string(expected_P) + "], got size="
            + std::to_string(margins.size()));
    }
}

std::vector<std::pair<std::string, std::string>> snap_pair_labels(const gp::WorldSnapshot& snap) {
    std::vector<std::pair<std::string, std::string>> out;
    out.reserve(snap.pairs.size());
    for (const auto& pr : snap.pairs) {
        out.emplace_back(snap.shapes[static_cast<size_t>(pr.shape_a)].link_key,
                         snap.shapes[static_cast<size_t>(pr.shape_b)].link_key);
    }
    return out;
}

std::vector<std::pair<std::string, std::string>> snap_pair_shape_names(const gp::WorldSnapshot& snap) {
    std::vector<std::pair<std::string, std::string>> out;
    out.reserve(snap.pairs.size());
    for (const auto& pr : snap.pairs) {
        out.emplace_back(snap.shapes[static_cast<size_t>(pr.shape_a)].name,
                         snap.shapes[static_cast<size_t>(pr.shape_b)].name);
    }
    return out;
}

py::array_t<double> snap_pair_margins(const gp::WorldSnapshot& snap, double margin, double world_margin) {
    const auto m = snap.pair_margins(margin, world_margin);
    py::array_t<double> out(static_cast<py::ssize_t>(m.size()));
    if (m.size() > 0) std::copy(m.data(), m.data() + m.size(), out.mutable_data());
    return out;
}

py::tuple snap_export_warp_geometry(const gp::WorldSnapshot& snap) {
    const auto& shapes = snap.shapes;
    const auto& pairs = snap.pairs;
    const size_t N = shapes.size();
    const size_t P = pairs.size();
    std::vector<int> kinds(N);
    py::array_t<double> sizes({static_cast<py::ssize_t>(N), py::ssize_t(3)});
    py::array_t<int> spans({static_cast<py::ssize_t>(N), py::ssize_t(2)});
    std::vector<double> verts;
    for (size_t i = 0; i < N; ++i) {
        const auto& sh = *shapes[i].shape;
        kinds[i] = sh.kind;
        *spans.mutable_data(i, 0) = static_cast<int>(verts.size() / 3);
        *spans.mutable_data(i, 1) = static_cast<int>(sh.hull_points.size());
        for (const auto& v : sh.hull_points) {
            verts.push_back(v(0)); verts.push_back(v(1)); verts.push_back(v(2));
        }
        if (sh.kind == 1) {
            *sizes.mutable_data(i, 0) = sh.box_half_size(0);
            *sizes.mutable_data(i, 1) = sh.box_half_size(1);
            *sizes.mutable_data(i, 2) = sh.box_half_size(2);
        } else {
            *sizes.mutable_data(i, 0) = sh.radius;
            *sizes.mutable_data(i, 1) = 0.5 * sh.height;
            *sizes.mutable_data(i, 2) = 0.0;
        }
    }
    py::array_t<int> pair_arr({static_cast<py::ssize_t>(P), py::ssize_t(2)});
    for (size_t i = 0; i < P; ++i) {
        *pair_arr.mutable_data(i, 0) = pairs[i].shape_a;
        *pair_arr.mutable_data(i, 1) = pairs[i].shape_b;
    }
    py::array_t<float> vert_arr({static_cast<py::ssize_t>(verts.size() / 3), py::ssize_t(3)});
    for (size_t i = 0; i < verts.size(); ++i) vert_arr.mutable_data()[i] = static_cast<float>(verts[i]);
    return py::make_tuple(kinds, sizes, spans, vert_arr, pair_arr);
}

py::tuple snap_compute_frames(const gp::WorldSnapshot& snap, const Input& q_arr, bool return_joints) {
    if (q_arr.ndim() != 2 || q_arr.shape(1) != snap.kin.num_joints())
        throw std::invalid_argument("expected [S, J] joint array");
    const int64_t S = q_arr.shape(0);
    const int N = static_cast<int>(snap.shapes.size());
    const int J = snap.kin.num_joints();
    py::array_t<double> frames({S, static_cast<py::ssize_t>(N), py::ssize_t(3), py::ssize_t(4)});
    py::array_t<double> jorig = return_joints ? py::array_t<double>({S, static_cast<py::ssize_t>(J), py::ssize_t(3)}) : py::array_t<double>(0);
    py::array_t<double> jaxis = return_joints ? py::array_t<double>({S, static_cast<py::ssize_t>(J), py::ssize_t(3)}) : py::array_t<double>(0);
    {
        py::gil_scoped_release release;
        snap.compute_frames(q_arr.data(), S, frames.mutable_data(),
                            return_joints ? jorig.mutable_data() : nullptr,
                            return_joints ? jaxis.mutable_data() : nullptr);
    }
    if (return_joints) return py::make_tuple(frames, jorig, jaxis);
    return py::make_tuple(frames);
}

py::tuple snap_screen_aabbs(const gp::WorldSnapshot& snap, const Input& frames, const Input& thresholds,
                            bool verify_original, int64_t group_size) {
    check_frames_shape(frames, static_cast<int64_t>(snap.shapes.size()));
    check_margins_shape(thresholds, static_cast<int64_t>(snap.pairs.size()));
    std::vector<int64_t> rows, pids;
    {
        py::gil_scoped_release release;
        snap.screen_aabbs(frames.data(), frames.shape(0), thresholds.data(), verify_original, group_size, rows, pids);
    }
    py::array_t<int64_t> r_arr(static_cast<py::ssize_t>(rows.size()));
    py::array_t<int64_t> p_arr(static_cast<py::ssize_t>(pids.size()));
    if (!rows.empty()) {
        std::copy(rows.begin(), rows.end(), r_arr.mutable_data());
        std::copy(pids.begin(), pids.end(), p_arr.mutable_data());
    }
    return py::make_tuple(r_arr, p_arr);
}

py::array_t<double> snap_query_distances_indexed(const gp::WorldSnapshot& snap, const Input& frames,
                                                 const Indices& rows, const Indices& pair_ids, bool verify) {
    check_frames_shape(frames, static_cast<int64_t>(snap.shapes.size()));
    const int64_t K = rows.size();
    const int64_t width = verify ? 1 : 10;
    py::array_t<double> out({K, width});
    if (K > 0) {
        py::gil_scoped_release release;
        snap.query_distances_indexed(frames.data(), frames.shape(0), rows.data(), pair_ids.data(),
                                     K, verify, out.mutable_data());
    }
    return out;
}

py::object snap_first_violation_indexed(const gp::WorldSnapshot& snap, const Input& frames,
                                        const Indices& rows, const Indices& pair_ids,
                                        const Input& margins, bool verify) {
    check_frames_shape(frames, static_cast<int64_t>(snap.shapes.size()));
    check_margins_shape(margins, static_cast<int64_t>(snap.pairs.size()));
    std::optional<std::pair<int64_t, int64_t>> hit;
    {
        py::gil_scoped_release release;
        hit = snap.first_violation_indexed(frames.data(), frames.shape(0), rows.data(), pair_ids.data(),
                                           rows.size(), margins.data(), verify);
    }
    if (!hit.has_value()) return py::none();
    return py::make_tuple(hit->first, hit->second);
}

py::tuple snap_violating_pairs_indexed(const gp::WorldSnapshot& snap, const Input& frames,
                                       const Indices& rows, const Indices& pair_ids,
                                       const Input& margins, bool verify) {
    check_frames_shape(frames, static_cast<int64_t>(snap.shapes.size()));
    check_margins_shape(margins, static_cast<int64_t>(snap.pairs.size()));
    std::vector<int64_t> out_rows, out_pairs;
    if (rows.size() > 0) {
        py::gil_scoped_release release;
        snap.violating_pairs_indexed(frames.data(), frames.shape(0), rows.data(), pair_ids.data(),
                                     rows.size(), margins.data(), verify, out_rows, out_pairs);
    }
    py::array_t<int64_t> r_arr(static_cast<py::ssize_t>(out_rows.size()));
    py::array_t<int64_t> p_arr(static_cast<py::ssize_t>(out_pairs.size()));
    if (!out_rows.empty()) {
        std::copy(out_rows.begin(), out_rows.end(), r_arr.mutable_data());
        std::copy(out_pairs.begin(), out_pairs.end(), p_arr.mutable_data());
    }
    return py::make_tuple(r_arr, p_arr);
}

py::object snap_first_violation_continuous(const gp::WorldSnapshot& snap, const Input& q_waypoints,
                                           int substeps, const Input& frames, const Input& margins,
                                           bool verify, double disp_tol,
                                           py::object times_obj, py::object vel_obj, bool strict,
                                           py::object acc_obj = py::none(), int spline_order = 3) {
    if (spline_order != 3 && spline_order != 5)
        throw std::invalid_argument("spline_order must be 3 (cubic) or 5 (quintic)");
    if (substeps < 1)
        throw std::invalid_argument("substeps must be >= 1");
    const int J = snap.kin.num_joints();
    if (q_waypoints.ndim() != 2 || q_waypoints.shape(0) < 2 || q_waypoints.shape(1) != J)
        throw std::invalid_argument("q_waypoints must have shape [T>=2, J]");
    const int T = static_cast<int>(q_waypoints.shape(0));
    check_frames_shape(frames, static_cast<int64_t>(snap.shapes.size()));
    const int64_t expected_S = static_cast<int64_t>(T - 1) * static_cast<int64_t>(substeps) + 1;
    if (frames.shape(0) != expected_S)
        throw std::invalid_argument("frames row count must equal (T - 1) * substeps + 1");
    check_margins_shape(margins, static_cast<int64_t>(snap.pairs.size()));
    const bool has_t = !times_obj.is_none();
    const bool has_v = !vel_obj.is_none();
    const bool has_a = !acc_obj.is_none();
    if (has_t != has_v)
        throw std::invalid_argument("times and velocities must both be provided or both omitted");
    if (spline_order == 5 && (!has_t || !has_v || !has_a))
        throw std::invalid_argument("quintic continuous collision check requires times, velocities, and accelerations");
    const double* t_ptr = nullptr;
    const double* v_ptr = nullptr;
    const double* a_ptr = nullptr;
    Input t_arr, v_arr, a_arr;
    if (has_t) {
        t_arr = times_obj.cast<Input>();
        v_arr = vel_obj.cast<Input>();
        if (t_arr.ndim() != 1 || t_arr.shape(0) != T)
            throw std::invalid_argument("times must have shape [T]");
        if (v_arr.ndim() != 2 || v_arr.shape(0) != T || v_arr.shape(1) != J)
            throw std::invalid_argument("velocities must have shape [T, J]");
        t_ptr = t_arr.data();
        v_ptr = v_arr.data();
        if (has_a) {
            a_arr = acc_obj.cast<Input>();
            if (a_arr.ndim() != 2 || a_arr.shape(0) != T || a_arr.shape(1) != J)
                throw std::invalid_argument("accelerations must have shape [T, J]");
            a_ptr = a_arr.data();
        }
    }
    std::optional<gp::ContinuousViolation> hit;
    {
        py::gil_scoped_release release;
        hit = snap.first_violation_continuous(
            q_waypoints.data(), T,
            substeps, frames.data(), margins.data(), verify, disp_tol, t_ptr, v_ptr, strict, a_ptr, spline_order);
    }
    if (!hit.has_value()) return py::none();
    return py::make_tuple(hit->seg, hit->u, hit->pair_id, hit->distance);
}

py::tuple snap_compute_analytic_gradients(const gp::WorldSnapshot& snap, const Input& frames,
                                          const Input& jorig, const Input& jaxis,
                                          const Indices& rows, const Indices& pair_ids, const Input& data) {
    check_frames_shape(frames, static_cast<int64_t>(snap.shapes.size()));
    const int64_t K = rows.size();
    const int J = snap.kin.num_joints();
    py::array_t<double> grad({K, static_cast<py::ssize_t>(J)});
    py::array_t<double> wit({K, py::ssize_t(6)});
    if (K > 0) {
        py::gil_scoped_release release;
        snap.compute_analytic_gradients(frames.data(), jorig.data(), jaxis.data(),
                                        rows.data(), pair_ids.data(), data.data(),
                                        K, grad.mutable_data(), wit.mutable_data());
    }
    return py::make_tuple(grad, wit);
}

py::tuple snap_eval_collision_rows(const gp::WorldSnapshot& snap, const Input& q_arr,
                                   const Input& opt_margins, const Input& check_margins,
                                   double buffer, bool linearize, int64_t group_size, bool segment_minimum) {
    check_margins_shape(opt_margins, static_cast<int64_t>(snap.pairs.size()));
    check_margins_shape(check_margins, static_cast<int64_t>(snap.pairs.size()));
    const int64_t S = q_arr.shape(0);
    const int N = static_cast<int>(snap.shapes.size());
    const int J = snap.kin.num_joints();
    gp::CollisionEvalResult res;
    {
        py::gil_scoped_release release;
        res = snap.eval_collision_rows(q_arr.data(), S, opt_margins.data(), check_margins.data(),
                                       buffer, linearize, group_size, segment_minimum);
    }
    const py::ssize_t K = static_cast<py::ssize_t>(res.rows.size());
    py::array_t<int64_t> rows_arr(K), pids_arr(K);
    py::array_t<double> res_arr(K);
    if (K > 0) {
        std::copy(res.rows.begin(), res.rows.end(), rows_arr.mutable_data());
        std::copy(res.pair_ids.begin(), res.pair_ids.end(), pids_arr.mutable_data());
        std::copy(res.residuals.begin(), res.residuals.end(), res_arr.mutable_data());
    }
    py::object grad_obj = py::none();
    py::object wit_obj = py::none();
    if (linearize) {
        py::array_t<double> grad_arr({K, static_cast<py::ssize_t>(J)});
        py::array_t<double> wit_arr({K, py::ssize_t(6)});
        if (K > 0) {
            std::copy(res.gradients.begin(), res.gradients.end(), grad_arr.mutable_data());
            std::copy(res.witnesses.begin(), res.witnesses.end(), wit_arr.mutable_data());
        }
        grad_obj = std::move(grad_arr);
        wit_obj = std::move(wit_arr);
    }
    py::array_t<double> frames_arr({S, static_cast<py::ssize_t>(N), py::ssize_t(3), py::ssize_t(4)});
    if (!res.frames.empty()) std::copy(res.frames.begin(), res.frames.end(), frames_arr.mutable_data());
    py::array_t<bool> ok_arr(static_cast<py::ssize_t>(res.seg_ok.size()));
    for (size_t i = 0; i < res.seg_ok.size(); ++i) ok_arr.mutable_data()[i] = (res.seg_ok[i] != 0);
    return py::make_tuple(rows_arr, pids_arr, res_arr, grad_obj, frames_arr, ok_arr, wit_obj);
}

py::tuple snap_eval_pose_constraints(const gp::WorldSnapshot& snap, const Input& q_vec,
                                     const std::vector<gp::PoseConstraintSpec>& specs, bool linearize) {
    const int B = snap.kin.num_bodies();
    const int J = snap.kin.num_joints();
    std::vector<gp::Vec3> bpos(static_cast<size_t>(B)), jorig(linearize ? static_cast<size_t>(J) : 0),
        jaxis(linearize ? static_cast<size_t>(J) : 0);
    std::vector<gp::Mat3> brot(static_cast<size_t>(B));
    snap.kin.fk_single(q_vec.data(), bpos.data(), brot.data(),
                       linearize ? jorig.data() : nullptr, linearize ? jaxis.data() : nullptr);
    int total_dim = 0;
    for (const auto& s : specs) total_dim += s.residual_dim();
    py::array_t<double> res_arr(total_dim);
    py::object jac_obj = py::none();
    py::array_t<double> jac_arr;
    if (linearize) jac_arr = py::array_t<double>({static_cast<py::ssize_t>(total_dim), static_cast<py::ssize_t>(J)});
    bool all_ok = true;
    int offset = 0;
    Eigen::VectorXd r;
    Eigen::MatrixXd jc;
    for (const auto& s : specs) {
        const bool ok = snap.kin.eval_pose_constraint(s, bpos.data(), brot.data(),
                                                      linearize ? jorig.data() : nullptr,
                                                      linearize ? jaxis.data() : nullptr,
                                                      r, linearize ? &jc : nullptr);
        all_ok = all_ok && ok;
        for (int i = 0; i < r.size(); ++i) {
            res_arr.mutable_data()[offset + i] = r(i);
            if (linearize) {
                for (int j = 0; j < J; ++j) *jac_arr.mutable_data(offset + i, j) = jc(i, j);
            }
        }
        offset += static_cast<int>(r.size());
    }
    if (linearize) jac_obj = std::move(jac_arr);
    return py::make_tuple(res_arr, jac_obj, all_ok);
}

py::tuple snap_solve_ik_batch(const gp::WorldSnapshot& snap,
                              const std::vector<std::pair<std::vector<gp::PoseConstraintSpec>, Eigen::VectorXd>>& problems_in,
                              const MaskInput& active_mask_in,
                              double margin, double world_margin, int iterations, bool verify_original) {
    const int J = snap.kin.num_joints();
    std::vector<gp::IKProblem> problems;
    problems.reserve(problems_in.size());
    for (const auto& [targets, seed] : problems_in) problems.push_back({targets, seed});
    std::vector<uint8_t> mask(static_cast<size_t>(J), 1);
    if (active_mask_in.size() == J) {
        for (int j = 0; j < J; ++j) mask[static_cast<size_t>(j)] = active_mask_in.data()[j] ? 1 : 0;
    }
    std::vector<gp::IKResultNative> res;
    {
        py::gil_scoped_release release;
        res = snap.solve_ik_batch(problems, mask, margin, world_margin, iterations, verify_original);
    }
    const py::ssize_t B = static_cast<py::ssize_t>(res.size());
    py::array_t<double> q_out({B, static_cast<py::ssize_t>(J)});
    py::array_t<bool> ok_out(B);
    std::vector<py::object> reasons(static_cast<size_t>(B));
    for (py::ssize_t i = 0; i < B; ++i) {
        const auto& r = res[static_cast<size_t>(i)];
        ok_out.mutable_data()[i] = r.success;
        if (r.success) {
            for (py::ssize_t j = 0; j < J; ++j) *q_out.mutable_data(i, j) = r.q(j);
            reasons[static_cast<size_t>(i)] = py::none();
        } else {
            for (py::ssize_t j = 0; j < J; ++j) *q_out.mutable_data(i, j) = std::numeric_limits<double>::quiet_NaN();
            reasons[static_cast<size_t>(i)] = py::str(r.reason);
        }
    }
    return py::make_tuple(q_out, ok_out, reasons);
}

py::tuple snap_plan_birrt_batch(const gp::WorldSnapshot& snap,
                                const Input& starts, const Input& goals,
                                const MaskInput& active_mask_in,
                                double margin, double world_margin,
                                double step_size, int max_iterations, double timeout_s,
                                const std::vector<uint64_t>& seeds, bool verify_original,
                                int shortcut_iters, int output_waypoints) {
    const int64_t B = starts.shape(0);
    const int J = snap.kin.num_joints();
    std::vector<uint8_t> mask(static_cast<size_t>(J), 1);
    if (active_mask_in.size() == J) {
        for (int j = 0; j < J; ++j) mask[static_cast<size_t>(j)] = active_mask_in.data()[j] ? 1 : 0;
    }
    std::vector<gp::RRTResultNative> res(static_cast<size_t>(B));
    {
        py::gil_scoped_release release;
        gp::ThreadPool::instance().parallel_for(B, 1, [&](int64_t b, int64_t e) {
            for (int64_t i = b; i < e; ++i) {
                Eigen::Map<const Eigen::VectorXd> s(starts.data() + i * J, J);
                Eigen::Map<const Eigen::VectorXd> g(goals.data() + i * J, J);
                const uint64_t sd = (i < static_cast<int64_t>(seeds.size())) ? seeds[static_cast<size_t>(i)] : static_cast<uint64_t>(i + 1);
                res[static_cast<size_t>(i)] = snap.plan_birrt(s, g, mask, margin, world_margin, step_size,
                                                              max_iterations, timeout_s, sd, verify_original,
                                                              shortcut_iters, output_waypoints);
            }
        });
    }
    py::array_t<double> paths({B, static_cast<py::ssize_t>(output_waypoints), static_cast<py::ssize_t>(J)});
    py::array_t<bool> ok(B);
    for (int64_t i = 0; i < B; ++i) {
        ok.mutable_data()[i] = res[static_cast<size_t>(i)].success;
        if (res[static_cast<size_t>(i)].success) {
            for (int t = 0; t < output_waypoints; ++t) {
                for (int j = 0; j < J; ++j) {
                    *paths.mutable_data(i, t, j) = res[static_cast<size_t>(i)].waypoints[static_cast<size_t>(t)](j);
                }
            }
        } else {
            for (int t = 0; t < output_waypoints; ++t)
                for (int j = 0; j < J; ++j)
                    *paths.mutable_data(i, t, j) = std::numeric_limits<double>::quiet_NaN();
        }
    }
    return py::make_tuple(paths, ok);
}

py::tuple snap_finalize_trajectory(const gp::WorldSnapshot& snap, const Input& pos_arr, const py::dict& cfg_dict) {
    const int J = snap.kin.num_joints();
    if (pos_arr.ndim() != 2 || pos_arr.shape(1) != J)
        throw std::invalid_argument("positions must have shape [T, J]");
    gp::RowMatrixXd pos(pos_arr.shape(0), J);
    std::copy(pos_arr.data(), pos_arr.data() + pos_arr.shape(0) * J, pos.data());
    const auto cfg = parse_solver_config(cfg_dict);
    gp::FinalizeResultNative fin;
    {
        py::gil_scoped_release release;
        fin = gp::finalize_trajectory(snap, pos, cfg);
    }
    if (!fin.success) {
        return py::make_tuple(py::none(), fin.retime_ms, py::str(fin.reason));
    }
    return py::make_tuple(trajectory_to_dict(fin.trajectory), fin.retime_ms, py::none());
}

py::tuple snap_validate_trajectory_collision(
    const gp::WorldSnapshot& snap, const Input& q_waypoints, const Input& samples,
    int substeps, double margin, double world_margin, bool verify_original, double disp_tol,
    py::object times_obj, py::object vel_obj, bool strict_ccd,
    py::object acc_obj = py::none(), int spline_order = 3) {
    if (spline_order != 3 && spline_order != 5)
        throw std::invalid_argument("spline_order must be 3 (cubic) or 5 (quintic)");
    if (substeps < 1)
        throw std::invalid_argument("substeps must be >= 1");
    const int J = snap.kin.num_joints();
    if (q_waypoints.ndim() != 2 || q_waypoints.shape(0) < 2 || q_waypoints.shape(1) != J)
        throw std::invalid_argument("q_waypoints must have shape [T>=2, J]");
    const int T = static_cast<int>(q_waypoints.shape(0));
    const int64_t expected_S = static_cast<int64_t>(T - 1) * static_cast<int64_t>(substeps) + 1;
    if (samples.ndim() != 2 || samples.shape(0) != expected_S || samples.shape(1) != J)
        throw std::invalid_argument("samples must have shape [(T - 1) * substeps + 1, J]");
    const bool has_t = !times_obj.is_none();
    const bool has_v = !vel_obj.is_none();
    const bool has_a = !acc_obj.is_none();
    if (has_t != has_v)
        throw std::invalid_argument("times and velocities must both be provided or both omitted");
    if (spline_order == 5 && (!has_t || !has_v || !has_a))
        throw std::invalid_argument("quintic continuous collision check requires times, velocities, and accelerations");
    const double* t_ptr = nullptr;
    const double* v_ptr = nullptr;
    const double* a_ptr = nullptr;
    Input t_arr, v_arr, a_arr;
    if (has_t) {
        t_arr = times_obj.cast<Input>();
        v_arr = vel_obj.cast<Input>();
        if (t_arr.ndim() != 1 || t_arr.shape(0) != T)
            throw std::invalid_argument("times must have shape [T]");
        if (v_arr.ndim() != 2 || v_arr.shape(0) != T || v_arr.shape(1) != J)
            throw std::invalid_argument("velocities must have shape [T, J]");
        t_ptr = t_arr.data();
        v_ptr = v_arr.data();
        if (has_a) {
            a_arr = acc_obj.cast<Input>();
            if (a_arr.ndim() != 2 || a_arr.shape(0) != T || a_arr.shape(1) != J)
                throw std::invalid_argument("accelerations must have shape [T, J]");
            a_ptr = a_arr.data();
        }
    }
    gp::WorldSnapshot::ValidationResult res;
    {
        py::gil_scoped_release release;
        res = snap.validate_trajectory_collision(
            q_waypoints.data(), T,
            samples.data(), samples.shape(0), substeps,
            margin, world_margin, verify_original, disp_tol, t_ptr, v_ptr, strict_ccd, a_ptr, spline_order);
    }
    if (res.valid) return py::make_tuple(true, py::none());
    return py::make_tuple(false, py::str(res.reason));
}

gp::TrajectoryNative trajectory_from_arrays(const Input& times_in, const Input& pos_in,
                                            const Input& vel_in, const Input& acc_in,
                                            int spline_order) {
    if (pos_in.ndim() != 2 || pos_in.shape(0) < 2)
        throw std::invalid_argument("positions must have shape [T>=2, J]");
    const int T = static_cast<int>(pos_in.shape(0));
    const int J = static_cast<int>(pos_in.shape(1));
    if (times_in.ndim() != 1 || times_in.shape(0) != T)
        throw std::invalid_argument("times must have shape [T]");
    if (vel_in.ndim() != 2 || vel_in.shape(0) != T || vel_in.shape(1) != J)
        throw std::invalid_argument("velocities must have shape [T, J]");
    if (acc_in.ndim() != 2 || acc_in.shape(0) != T || acc_in.shape(1) != J)
        throw std::invalid_argument("accelerations must have shape [T, J]");
    if (spline_order != 3 && spline_order != 5)
        throw std::invalid_argument("spline_order must be 3 or 5");
    gp::TrajectoryNative traj;
    traj.spline_order = spline_order;
    traj.times.resize(T);
    std::copy(times_in.data(), times_in.data() + T, traj.times.data());
    traj.positions.resize(T, J);
    std::copy(pos_in.data(), pos_in.data() + T * J, traj.positions.data());
    traj.velocities.resize(T, J);
    std::copy(vel_in.data(), vel_in.data() + T * J, traj.velocities.data());
    traj.accelerations.resize(T, J);
    std::copy(acc_in.data(), acc_in.data() + T * J, traj.accelerations.data());
    return traj;
}

}  // namespace

PYBIND11_MODULE(_coal_native, module) {
    module.def("has_tbb", &gp::gobot_planning_has_tbb);
    module.def("abi_version", &gp::gobot_planning_abi_version);
    module.def("evaluate_trajectory", [](const Input& times_in, const Input& pos_in,
                                         const Input& vel_in, const Input& acc_in,
                                         const Input& query_times, int spline_order, bool return_jerk) -> py::tuple {
        const auto traj = trajectory_from_arrays(times_in, pos_in, vel_in, acc_in, spline_order);
        const int64_t S = query_times.size();
        const py::ssize_t J = traj.num_joints();
        py::array_t<double> out_p({static_cast<py::ssize_t>(S), J});
        py::array_t<double> out_v({static_cast<py::ssize_t>(S), J});
        py::array_t<double> out_a({static_cast<py::ssize_t>(S), J});
        py::array_t<double> out_j = return_jerk ? py::array_t<double>({static_cast<py::ssize_t>(S), J})
                                                : py::array_t<double>(0);
        if (S > 0) {
            py::gil_scoped_release release;
            traj.sample_batch(query_times.data(), S,
                              out_p.mutable_data(), out_v.mutable_data(), out_a.mutable_data(),
                              return_jerk ? out_j.mutable_data() : nullptr);
        }
        if (return_jerk) return py::make_tuple(out_p, out_v, out_a, out_j);
        return py::make_tuple(out_p, out_v, out_a);
    }, py::arg("times"), py::arg("positions"), py::arg("velocities"), py::arg("accelerations"),
       py::arg("query_times"), py::arg("spline_order") = 3, py::arg("return_jerk") = false);

    module.def("retime_trajectory", [](const Input& pos_in,
                                       const Eigen::VectorXd& vmax,
                                       const Eigen::VectorXd& amax,
                                       bool stop_at_endpoints,
                                       py::object dt_obj,
                                       py::object knot_scales_obj,
                                       py::object jmax_obj,
                                       int spline_order) {
        if (pos_in.ndim() != 2 || pos_in.shape(0) < 2)
            throw std::invalid_argument("positions must have shape [T>=2, J]");
        const int T = static_cast<int>(pos_in.shape(0));
        const int J = static_cast<int>(pos_in.shape(1));
        gp::RowMatrixXd pos(T, J);
        std::copy(pos_in.data(), pos_in.data() + T * J, pos.data());
        const bool has_dt = !dt_obj.is_none();
        const double dt_val = has_dt ? dt_obj.cast<double>() : 0.0;
        Eigen::VectorXd ks;
        const Eigen::VectorXd* ks_ptr = nullptr;
        if (!knot_scales_obj.is_none()) {
            ks = knot_scales_obj.cast<Eigen::VectorXd>();
            ks_ptr = &ks;
        }
        Eigen::VectorXd jmax;
        const Eigen::VectorXd* jmax_ptr = nullptr;
        if (!jmax_obj.is_none()) {
            jmax = jmax_obj.cast<Eigen::VectorXd>();
            jmax_ptr = &jmax;
        }
        gp::TrajectoryNative traj;
        {
            py::gil_scoped_release release;
            traj = gp::retime_trajectory(pos, vmax, amax, stop_at_endpoints, has_dt, dt_val,
                                         ks_ptr, jmax_ptr, spline_order);
        }
        return trajectory_to_dict(traj);
    }, py::arg("positions"), py::arg("velocity_limits"), py::arg("acceleration_limits"),
       py::arg("stop_at_endpoints") = true, py::arg("dt") = py::none(),
       py::arg("knot_velocity_scales") = py::none(), py::arg("jerk_limits") = py::none(),
       py::arg("spline_order") = 3);

    module.def("check_trajectory_bounds", [](const Input& times_in, const Input& pos_in,
                                             const Input& vel_in, const Input& acc_in,
                                             const Eigen::VectorXd& vmax,
                                             const Eigen::VectorXd& amax,
                                             bool check_velocity, bool check_acceleration, double tol,
                                             py::object lo_obj, py::object hi_obj,
                                             py::object jmax_obj, bool check_jerk, int spline_order) {
        const auto traj = trajectory_from_arrays(times_in, pos_in, vel_in, acc_in, spline_order);
        Eigen::VectorXd lo, hi, jmax;
        const Eigen::VectorXd* lo_ptr = nullptr;
        const Eigen::VectorXd* hi_ptr = nullptr;
        const Eigen::VectorXd* jmax_ptr = nullptr;
        if (!lo_obj.is_none()) {
            lo = lo_obj.cast<Eigen::VectorXd>();
            lo_ptr = &lo;
        }
        if (!hi_obj.is_none()) {
            hi = hi_obj.cast<Eigen::VectorXd>();
            hi_ptr = &hi;
        }
        if (!jmax_obj.is_none()) {
            jmax = jmax_obj.cast<Eigen::VectorXd>();
            jmax_ptr = &jmax;
        }
        gp::BoundsCheckResult res;
        {
            py::gil_scoped_release release;
            res = gp::check_trajectory_bounds(traj, vmax, amax, check_velocity, check_acceleration,
                                              tol, lo_ptr, hi_ptr, jmax_ptr, check_jerk);
        }
        return py::make_tuple(res.ok, res.ok ? py::none() : py::object(py::str(res.reason)), res.bad_segments);
    }, py::arg("times"), py::arg("positions"), py::arg("velocities"), py::arg("accelerations"),
       py::arg("velocity_limits"), py::arg("acceleration_limits"),
       py::arg("check_velocity") = true, py::arg("check_acceleration") = true, py::arg("tol") = 1e-5,
       py::arg("lower_limits") = py::none(), py::arg("upper_limits") = py::none(),
       py::arg("jerk_limits") = py::none(), py::arg("check_jerk") = true, py::arg("spline_order") = 3);
    module.def("forward_kinematics", &forward_kinematics,
               py::arg("parent"), py::arg("kinds"), py::arg("joint_idx"),
               py::arg("parent_joint"), py::arg("joint_child"), py::arg("axes"),
               py::arg("tracked_idx"), py::arg("root_pos"), py::arg("root_rot"),
               py::arg("q"), py::arg("return_joints") = false);

    module.def("solve_l1_box_qp", [](const Eigen::MatrixXd& H,
                                     const Eigen::VectorXd& g,
                                     const Eigen::MatrixXd& A,
                                     const Eigen::VectorXd& v,
                                     const Eigen::VectorXd& lower,
                                     const Eigen::VectorXd& upper,
                                     double penalty,
                                     double eps_abs,
                                     double eps_rel) {
        gp::L1BoxQPResult res;
        {
            py::gil_scoped_release release;
            res = gp::solve_l1_box_qp(H, g, A, v, lower, upper, penalty, eps_abs, eps_rel);
        }
        return py::make_tuple(res.x, res.solved, res.outer_iters, res.newton_iters, res.prim_res, res.dual_res);
    }, py::arg("H"), py::arg("g"), py::arg("A"), py::arg("v"),
       py::arg("lower"), py::arg("upper"), py::arg("penalty"),
       py::arg("eps_abs") = 1e-5, py::arg("eps_rel") = 1e-5);

    py::class_<gp::L1BoxQPSolver>(module, "L1BoxQPSolver")
        .def(py::init<>())
        .def("setup", [](gp::L1BoxQPSolver& s,
                         const Eigen::MatrixXd& H,
                         const Eigen::VectorXd& g,
                         const Eigen::MatrixXd& A,
                         const Eigen::VectorXd& v,
                         double penalty) {
            py::gil_scoped_release release;
            s.setup(H, g, A, v, penalty);
        }, py::arg("H"), py::arg("g"), py::arg("A"), py::arg("v"), py::arg("penalty"))
        .def("solve", [](gp::L1BoxQPSolver& s,
                         const Eigen::VectorXd& lower,
                         const Eigen::VectorXd& upper,
                         double eps_abs,
                         double eps_rel) -> py::object {
            gp::L1BoxQPResult res;
            {
                py::gil_scoped_release release;
                res = s.solve(lower, upper, eps_abs, eps_rel);
            }
            if (!res.solved) return py::none();
            return py::cast(res.x);
        }, py::arg("lower"), py::arg("upper"), py::arg("eps_abs") = 1e-5, py::arg("eps_rel") = 1e-5);

    py::class_<gp::CollisionShape, std::shared_ptr<gp::CollisionShape>>(module, "Shape")
        .def(py::init(&make_collision_shape))
        .def_readonly("label", &gp::CollisionShape::label)
        .def_readonly("kind", &gp::CollisionShape::kind)
        .def("bounds", [](const gp::CollisionShape& sh, bool original) {
            const auto& box = (original ? sh.original : sh.convex)->aabb_local;
            return std::make_pair(box.min_, box.max_);
        });

    py::class_<gp::PoseConstraintSpec>(module, "PoseConstraintSpec")
        .def(py::init<>())
        .def_readwrite("link_id", &gp::PoseConstraintSpec::link_id)
        .def_readwrite("has_position", &gp::PoseConstraintSpec::has_position)
        .def_readwrite("has_rotation", &gp::PoseConstraintSpec::has_rotation)
        .def_readwrite("position", &gp::PoseConstraintSpec::position)
        .def_readwrite("rotation", &gp::PoseConstraintSpec::rotation)
        .def_readwrite("pos_tol", &gp::PoseConstraintSpec::pos_tol)
        .def_readwrite("rot_tol", &gp::PoseConstraintSpec::rot_tol)
        .def_readwrite("weight", &gp::PoseConstraintSpec::weight)
        .def("residual_dim", &gp::PoseConstraintSpec::residual_dim);

    struct PyWorldSnapshot {
        std::shared_ptr<const gp::WorldSnapshot> ptr;
    };
    py::class_<PyWorldSnapshot>(module, "WorldSnapshot")
        .def_property_readonly("version", [](const PyWorldSnapshot& s) { return s.ptr->version; })
        .def_property_readonly("topology_version", [](const PyWorldSnapshot& s) { return s.ptr->topology_version; })
        .def_property_readonly("num_bodies", [](const PyWorldSnapshot& s) { return s.ptr->kin.num_bodies(); })
        .def_property_readonly("num_joints", [](const PyWorldSnapshot& s) { return s.ptr->kin.num_joints(); })
        .def_property_readonly("self_pair_count", [](const PyWorldSnapshot& s) { return s.ptr->self_pair_count; })
        .def_property_readonly("has_mesh", [](const PyWorldSnapshot& s) { return s.ptr->has_mesh; })
        .def_property_readonly("num_shapes", [](const PyWorldSnapshot& s) { return static_cast<int>(s.ptr->shapes.size()); })
        .def_property_readonly("num_pairs", [](const PyWorldSnapshot& s) { return static_cast<int>(s.ptr->pairs.size()); })
        .def_property_readonly("lower_limits", [](const PyWorldSnapshot& s) { return s.ptr->kin.lower(); })
        .def_property_readonly("upper_limits", [](const PyWorldSnapshot& s) { return s.ptr->kin.upper(); })
        .def_property_readonly("velocity_limits", [](const PyWorldSnapshot& s) { return s.ptr->kin.velocity_limit(); })
        .def_property_readonly("acceleration_limits", [](const PyWorldSnapshot& s) { return s.ptr->kin.acceleration_limit(); })
        .def_property_readonly("jerk_limits", [](const PyWorldSnapshot& s) { return s.ptr->kin.jerk_limit(); })
        .def("pair_labels", [](const PyWorldSnapshot& s) { return snap_pair_labels(*s.ptr); })
        .def("pair_shape_names", [](const PyWorldSnapshot& s) { return snap_pair_shape_names(*s.ptr); })
        .def("pair_margins", [](const PyWorldSnapshot& s, double margin, double world_margin) {
            return snap_pair_margins(*s.ptr, margin, world_margin);
        }, py::arg("margin"), py::arg("world_margin"))
        .def("export_warp_geometry", [](const PyWorldSnapshot& s) { return snap_export_warp_geometry(*s.ptr); })
        .def("compute_frames", [](const PyWorldSnapshot& s, const Input& q_arr, bool return_joints) {
            return snap_compute_frames(*s.ptr, q_arr, return_joints);
        }, py::arg("q"), py::arg("return_joints") = false)
        .def("screen_aabbs", [](const PyWorldSnapshot& s, const Input& frames, const Input& thresholds,
                                bool verify_original, int64_t group_size) {
            return snap_screen_aabbs(*s.ptr, frames, thresholds, verify_original, group_size);
        }, py::arg("frames"), py::arg("thresholds"), py::arg("verify_original") = false, py::arg("group_size") = 1)
        .def("query_distances_indexed", [](const PyWorldSnapshot& s, const Input& frames,
                                           const Indices& rows, const Indices& pair_ids, bool verify) {
            return snap_query_distances_indexed(*s.ptr, frames, rows, pair_ids, verify);
        }, py::arg("frames"), py::arg("rows"), py::arg("pair_ids"), py::arg("verify") = false)
        .def("first_violation_indexed", [](const PyWorldSnapshot& s, const Input& frames,
                                           const Indices& rows, const Indices& pair_ids,
                                           const Input& margins, bool verify) {
            return snap_first_violation_indexed(*s.ptr, frames, rows, pair_ids, margins, verify);
        }, py::arg("frames"), py::arg("rows"), py::arg("pair_ids"), py::arg("margins"), py::arg("verify") = false)
        .def("violating_pairs_indexed", [](const PyWorldSnapshot& s, const Input& frames,
                                           const Indices& rows, const Indices& pair_ids,
                                           const Input& margins, bool verify) {
            return snap_violating_pairs_indexed(*s.ptr, frames, rows, pair_ids, margins, verify);
        }, py::arg("frames"), py::arg("rows"), py::arg("pair_ids"), py::arg("margins"), py::arg("verify") = false)
        .def("first_violation_continuous", [](const PyWorldSnapshot& s, const Input& q_waypoints,
                                              int substeps, const Input& frames, const Input& margins,
                                              bool verify, double disp_tol,
                                              py::object times_obj, py::object vel_obj, bool strict,
                                              py::object acc_obj, int spline_order) {
            return snap_first_violation_continuous(*s.ptr, q_waypoints, substeps, frames, margins,
                                                   verify, disp_tol, times_obj, vel_obj, strict,
                                                   acc_obj, spline_order);
        }, py::arg("q_waypoints"), py::arg("substeps"), py::arg("frames"), py::arg("margins"),
           py::arg("verify") = false, py::arg("disp_tol") = 5e-4,
           py::arg("times") = py::none(), py::arg("velocities") = py::none(), py::arg("strict") = false,
           py::arg("accelerations") = py::none(), py::arg("spline_order") = 3)
        .def("compute_analytic_gradients", [](const PyWorldSnapshot& s, const Input& frames,
                                              const Input& jorig, const Input& jaxis,
                                              const Indices& rows, const Indices& pair_ids, const Input& data) {
            return snap_compute_analytic_gradients(*s.ptr, frames, jorig, jaxis, rows, pair_ids, data);
        })
        .def("eval_collision_rows", [](const PyWorldSnapshot& s, const Input& q_arr,
                                       const Input& opt_margins, const Input& check_margins,
                                       double buffer, bool linearize, int64_t group_size, bool segment_minimum) {
            return snap_eval_collision_rows(*s.ptr, q_arr, opt_margins, check_margins,
                                            buffer, linearize, group_size, segment_minimum);
        })
        .def("eval_pose_constraints", [](const PyWorldSnapshot& s, const Input& q_vec,
                                         const std::vector<gp::PoseConstraintSpec>& specs, bool linearize) {
            return snap_eval_pose_constraints(*s.ptr, q_vec, specs, linearize);
        })
        .def("solve_ik_batch", [](const PyWorldSnapshot& s,
                                  const std::vector<std::pair<std::vector<gp::PoseConstraintSpec>, Eigen::VectorXd>>& problems_in,
                                  const MaskInput& active_mask_in,
                                  double margin, double world_margin, int iterations, bool verify_original) {
            return snap_solve_ik_batch(*s.ptr, problems_in, active_mask_in,
                                       margin, world_margin, iterations, verify_original);
        })
        .def("plan_birrt_batch", [](const PyWorldSnapshot& s,
                                    const Input& starts, const Input& goals,
                                    const MaskInput& active_mask_in,
                                    double margin, double world_margin,
                                    double step_size, int max_iterations, double timeout_s,
                                    const std::vector<uint64_t>& seeds, bool verify_original,
                                    int shortcut_iters, int output_waypoints) {
            return snap_plan_birrt_batch(*s.ptr, starts, goals, active_mask_in,
                                         margin, world_margin, step_size, max_iterations, timeout_s,
                                         seeds, verify_original, shortcut_iters, output_waypoints);
        })
        .def("finalize_trajectory", [](const PyWorldSnapshot& s, const Input& pos_arr, const py::dict& cfg_dict) {
            return snap_finalize_trajectory(*s.ptr, pos_arr, cfg_dict);
        })
        .def("validate_trajectory_collision", [](const PyWorldSnapshot& s, const Input& q_waypoints,
                                                 const Input& samples, int substeps,
                                                 double margin, double world_margin, bool verify_original,
                                                 double disp_tol, py::object times_obj, py::object vel_obj,
                                                 bool strict_ccd, py::object acc_obj, int spline_order) {
            return snap_validate_trajectory_collision(
                *s.ptr, q_waypoints, samples, substeps, margin, world_margin,
                verify_original, disp_tol, times_obj, vel_obj, strict_ccd, acc_obj, spline_order);
        }, py::arg("q_waypoints"), py::arg("samples"), py::arg("substeps"),
           py::arg("margin"), py::arg("world_margin"), py::arg("verify_original") = true,
           py::arg("disp_tol") = 5e-4, py::arg("times") = py::none(), py::arg("velocities") = py::none(),
           py::arg("strict_ccd") = false, py::arg("accelerations") = py::none(), py::arg("spline_order") = 3);

    py::class_<gp::PlanningWorld, std::shared_ptr<gp::PlanningWorld>>(module, "PlanningWorld")
        .def(py::init<>())
        .def("snapshot", [](const gp::PlanningWorld& w) { return PyWorldSnapshot{w.snapshot()}; })
        .def_property_readonly("version", &gp::PlanningWorld::version)
        .def_property_readonly("topology_version", &gp::PlanningWorld::topology_version)
        .def_property_readonly("num_bodies", &gp::PlanningWorld::num_bodies)
        .def_property_readonly("num_joints", &gp::PlanningWorld::num_joints)
        .def_property_readonly("self_pair_count", &gp::PlanningWorld::self_pair_count)
        .def_property_readonly("has_mesh", &gp::PlanningWorld::has_mesh)
        .def_property_readonly("num_shapes", &gp::PlanningWorld::num_shapes)
        .def_property_readonly("num_pairs", &gp::PlanningWorld::num_pairs)
        .def("init_kinematics", [](gp::PlanningWorld& w,
                                   std::vector<int> parent, std::vector<int> kinds, std::vector<int> joint_idx,
                                   const Input& parent_joint, const Input& joint_child, const Input& axes,
                                   std::vector<std::tuple<std::string, int, gp::Vec3, gp::Mat3>> branches_in,
                                   std::vector<std::string> body_names, std::vector<std::string> joint_names,
                                   const Eigen::VectorXd& lower, const Eigen::VectorXd& upper,
                                   const Eigen::VectorXd& vel, const Eigen::VectorXd& acc,
                                   const Eigen::VectorXd& init_q,
                                   std::unordered_map<std::string, int> link_index) {
            const size_t B = parent.size();
            std::vector<gp::Mat3> a_rot(B), b_rot(B);
            std::vector<gp::Vec3> a_pos(B), b_pos(B), ax(B);
            const double* pj = parent_joint.data();
            const double* jc = joint_child.data();
            const double* ap = axes.data();
            for (size_t i = 0; i < B; ++i) {
                const double* a = pj + i * 16;
                const double* b = jc + i * 16;
                a_rot[i] << a[0], a[1], a[2], a[4], a[5], a[6], a[8], a[9], a[10];
                a_pos[i] << a[3], a[7], a[11];
                b_rot[i] << b[0], b[1], b[2], b[4], b[5], b[6], b[8], b[9], b[10];
                b_pos[i] << b[3], b[7], b[11];
                ax[i] << ap[i * 3 + 0], ap[i * 3 + 1], ap[i * 3 + 2];
            }
            std::vector<gp::KinematicsModel::Branch> branches;
            for (auto& [rname, rbody, rpos, rrot] : branches_in) {
                branches.push_back({std::move(rname), rbody, rpos, rrot});
            }
            w.init_kinematics(std::move(parent), std::move(kinds), std::move(joint_idx),
                              std::move(a_rot), std::move(a_pos), std::move(b_rot), std::move(b_pos),
                              std::move(ax), std::move(branches), std::move(body_names), std::move(joint_names),
                              lower, upper, vel, acc, init_q, std::move(link_index));
        })
        .def("add_shape", [](gp::PlanningWorld& w, const std::string& name, const std::string& link_key,
                             int body_id, const Input& local_frame, int layer, int mask,
                             std::shared_ptr<gp::CollisionShape> shape) {
            gp::WorldShape ws;
            ws.name = name;
            ws.link_key = link_key;
            ws.body_id = body_id;
            ws.local_frame = frame_from_array(local_frame);
            ws.layer = layer;
            ws.mask = mask;
            ws.shape = std::move(shape);
            w.add_shape(std::move(ws));
        })
        .def("update_obstacle_pose", [](gp::PlanningWorld& w, const std::string& name, const Input& frame) {
            w.update_obstacle_pose(name, frame_from_array(frame));
        })
        .def("remove_obstacle", &gp::PlanningWorld::remove_obstacle)
        .def("attach_object", [](gp::PlanningWorld& w, const std::string& name, const std::string& link_key,
                                 const Input& local_frame, const std::vector<std::string>& touch_links) {
            w.attach_object(name, link_key, frame_from_array(local_frame), touch_links);
        })
        .def("detach_object", &gp::PlanningWorld::detach_object)
        .def("set_auto_collision_disabled", &gp::PlanningWorld::set_auto_collision_disabled)
        .def("set_collision_enabled", &gp::PlanningWorld::set_collision_enabled)
        .def("set_robot_base_pose", [](gp::PlanningWorld& w, int branch_idx, const gp::Vec3& pos, const gp::Mat3& rot) {
            w.set_robot_base_pose(branch_idx, pos, rot);
        })
        .def("set_jerk_limits", &gp::PlanningWorld::set_jerk_limits)
        .def("attached_shape_names", &gp::PlanningWorld::attached_shape_names)
        .def("shape_names", &gp::PlanningWorld::shape_names)
        .def("apply_sync_batch", [](gp::PlanningWorld& w,
                                    const py::list& base_updates,
                                    const py::list& pose_updates,
                                    const std::vector<std::string>& remove_names,
                                    const py::list& replace_shapes,
                                    const py::list& add_shapes) {
            gp::SyncBatch batch;
            batch.base_updates.reserve(base_updates.size());
            for (const auto& item : base_updates) {
                auto tup = item.cast<py::tuple>();
                gp::SyncBaseUpdate bu;
                bu.branch_idx = tup[0].cast<int>();
                bu.pos = tup[1].cast<gp::Vec3>();
                bu.rot = tup[2].cast<gp::Mat3>();
                batch.base_updates.push_back(std::move(bu));
            }
            batch.pose_updates.reserve(pose_updates.size());
            for (const auto& item : pose_updates) {
                auto tup = item.cast<py::tuple>();
                gp::SyncPoseUpdate pu;
                pu.name = tup[0].cast<std::string>();
                pu.world_frame = frame_from_array(tup[1].cast<Input>());
                batch.pose_updates.push_back(std::move(pu));
            }
            batch.remove_names = remove_names;
            batch.replace_shapes.reserve(replace_shapes.size());
            for (const auto& item : replace_shapes) {
                auto tup = item.cast<py::tuple>();
                gp::SyncShapeReplacement rep;
                rep.name = tup[0].cast<std::string>();
                rep.link_key = tup[1].cast<std::string>();
                rep.world_frame = frame_from_array(tup[2].cast<Input>());
                rep.layer = tup[3].cast<int>();
                rep.mask = tup[4].cast<int>();
                if (!tup[5].is_none()) {
                    rep.shape = tup[5].cast<std::shared_ptr<gp::CollisionShape>>();
                }
                batch.replace_shapes.push_back(std::move(rep));
            }
            batch.add_shapes.reserve(add_shapes.size());
            for (const auto& item : add_shapes) {
                auto tup = item.cast<py::tuple>();
                gp::WorldShape ws;
                ws.name = tup[0].cast<std::string>();
                ws.link_key = tup[1].cast<std::string>();
                ws.body_id = -1;
                ws.local_frame = frame_from_array(tup[2].cast<Input>());
                ws.layer = tup[3].cast<int>();
                ws.mask = tup[4].cast<int>();
                ws.shape = tup[5].cast<std::shared_ptr<gp::CollisionShape>>();
                batch.add_shapes.push_back(std::move(ws));
            }
            gp::SyncResultNative res;
            {
                py::gil_scoped_release release;
                res = w.apply_sync_batch(batch);
            }
            py::dict d;
            d["version"] = res.version;
            d["topology_version"] = res.topology_version;
            d["changed"] = res.changed;
            d["poses_updated"] = res.poses_updated;
            d["shapes_added"] = res.shapes_added;
            d["shapes_removed"] = res.shapes_removed;
            d["shapes_replaced"] = res.shapes_replaced;
            d["robot_bases_updated"] = res.robot_bases_updated;
            return d;
        })
        .def("pair_labels", [](const gp::PlanningWorld& w) { return snap_pair_labels(*w.snapshot()); })
        .def("pair_shape_names", [](const gp::PlanningWorld& w) { return snap_pair_shape_names(*w.snapshot()); })
        .def("pair_margins", [](const gp::PlanningWorld& w, double margin, double world_margin) {
            return snap_pair_margins(*w.snapshot(), margin, world_margin);
        }, py::arg("margin"), py::arg("world_margin"))
        .def("export_warp_geometry", [](const gp::PlanningWorld& w) { return snap_export_warp_geometry(*w.snapshot()); })
        .def("forward_kinematics", [](const gp::PlanningWorld& w, const Input& q_arr) {
            auto snap = w.snapshot();
            if (q_arr.ndim() != 2 || q_arr.shape(1) != snap->kin.num_joints())
                throw std::invalid_argument("expected [S, J] joint array");
            const int64_t S = q_arr.shape(0);
            const int B = snap->kin.num_bodies();
            const int J = snap->kin.num_joints();
            py::array_t<double> pos({S, static_cast<py::ssize_t>(B), py::ssize_t(3)});
            py::array_t<double> rot({S, static_cast<py::ssize_t>(B), py::ssize_t(3), py::ssize_t(3)});
            const double* qp = q_arr.data();
            double* pp = pos.mutable_data();
            double* rp = rot.mutable_data();
            {
                py::gil_scoped_release release;
                gp::ThreadPool::instance().parallel_for(S, 32, [&](int64_t b, int64_t e) {
                    std::vector<gp::Vec3> bp(static_cast<size_t>(B));
                    std::vector<gp::Mat3> br(static_cast<size_t>(B));
                    for (int64_t s = b; s < e; ++s) {
                        snap->kin.fk_single(qp + s * J, bp.data(), br.data());
                        for (int i = 0; i < B; ++i) {
                            Eigen::Map<gp::Vec3>(pp + (s * B + i) * 3) = bp[static_cast<size_t>(i)];
                            Eigen::Map<gp::Mat3>(rp + (s * B + i) * 9) = br[static_cast<size_t>(i)];
                        }
                    }
                });
            }
            return py::make_tuple(pos, rot);
        })
        .def("compute_frames", [](const gp::PlanningWorld& w, const Input& q_arr, bool return_joints) {
            return snap_compute_frames(*w.snapshot(), q_arr, return_joints);
        }, py::arg("q"), py::arg("return_joints") = false)
        .def("screen_aabbs", [](const gp::PlanningWorld& w, const Input& frames, const Input& thresholds,
                                bool verify_original, int64_t group_size) {
            return snap_screen_aabbs(*w.snapshot(), frames, thresholds, verify_original, group_size);
        }, py::arg("frames"), py::arg("thresholds"), py::arg("verify_original") = false, py::arg("group_size") = 1)
        .def("query_distances_indexed", [](const gp::PlanningWorld& w, const Input& frames,
                                           const Indices& rows, const Indices& pair_ids, bool verify) {
            return snap_query_distances_indexed(*w.snapshot(), frames, rows, pair_ids, verify);
        }, py::arg("frames"), py::arg("rows"), py::arg("pair_ids"), py::arg("verify") = false)
        .def("query_all_distances", [](const gp::PlanningWorld& w, const Input& q_arr, bool verify) {
            auto snap = w.snapshot();
            if (q_arr.ndim() != 2 || q_arr.shape(1) != snap->kin.num_joints())
                throw std::invalid_argument("expected [S, J] joint array");
            const int64_t S = q_arr.shape(0);
            const auto& pairs = snap->pairs;
            const int64_t P = static_cast<int64_t>(pairs.size());
            const int64_t N = static_cast<int64_t>(snap->shapes.size());
            py::array_t<double> out({S, P});
            if (S == 0 || P == 0) return out;
            std::vector<double> frames(static_cast<size_t>(S * N * 12));
            double* op = out.mutable_data();
            {
                py::gil_scoped_release release;
                snap->compute_frames(q_arr.data(), S, frames.data());
                const auto& shapes = snap->shapes;
                gp::ThreadPool::instance().parallel_for(S * P, 64, [&](int64_t b, int64_t e) {
                    double tmp[10];
                    for (int64_t k = b; k < e; ++k) {
                        const int64_t s = k / P;
                        const int64_t p = k % P;
                        const auto& pr = pairs[static_cast<size_t>(p)];
                        gp::eval_coal_distance(*shapes[static_cast<size_t>(pr.shape_a)].shape,
                                               *shapes[static_cast<size_t>(pr.shape_b)].shape,
                                               frames.data() + (s * N + pr.shape_a) * 12,
                                               frames.data() + (s * N + pr.shape_b) * 12,
                                               verify, std::numeric_limits<double>::infinity(), tmp);
                        op[k] = tmp[0];
                    }
                });
            }
            return out;
        }, py::arg("q"), py::arg("verify") = false)
        .def("first_violation_indexed", [](const gp::PlanningWorld& w, const Input& frames,
                                           const Indices& rows, const Indices& pair_ids,
                                           const Input& margins, bool verify) -> py::object {
            return snap_first_violation_indexed(*w.snapshot(), frames, rows, pair_ids, margins, verify);
        }, py::arg("frames"), py::arg("rows"), py::arg("pair_ids"), py::arg("margins"), py::arg("verify") = false)
        .def("violating_pairs_indexed", [](const gp::PlanningWorld& w, const Input& frames,
                                           const Indices& rows, const Indices& pair_ids,
                                           const Input& margins, bool verify) {
            return snap_violating_pairs_indexed(*w.snapshot(), frames, rows, pair_ids, margins, verify);
        }, py::arg("frames"), py::arg("rows"), py::arg("pair_ids"), py::arg("margins"), py::arg("verify") = false)
        .def("first_violation_continuous", [](const gp::PlanningWorld& w, const Input& q_waypoints,
                                              int substeps, const Input& frames, const Input& margins,
                                              bool verify, double disp_tol,
                                              py::object times_obj, py::object vel_obj, bool strict,
                                              py::object acc_obj, int spline_order) -> py::object {
            return snap_first_violation_continuous(
                *w.snapshot(), q_waypoints, substeps, frames, margins, verify, disp_tol,
                times_obj, vel_obj, strict, acc_obj, spline_order);
        }, py::arg("q_waypoints"), py::arg("substeps"), py::arg("frames"), py::arg("margins"),
           py::arg("verify") = false, py::arg("disp_tol") = 5e-4,
           py::arg("times") = py::none(), py::arg("velocities") = py::none(), py::arg("strict") = false,
           py::arg("accelerations") = py::none(), py::arg("spline_order") = 3)
        .def("compute_analytic_gradients", [](const gp::PlanningWorld& w, const Input& frames,
                                              const Input& jorig, const Input& jaxis,
                                              const Indices& rows, const Indices& pair_ids, const Input& data) {
            return snap_compute_analytic_gradients(*w.snapshot(), frames, jorig, jaxis, rows, pair_ids, data);
        })
        .def("eval_collision_rows", [](const gp::PlanningWorld& w, const Input& q_arr,
                                       const Input& opt_margins, const Input& check_margins,
                                       double buffer, bool linearize, int64_t group_size, bool segment_minimum) {
            return snap_eval_collision_rows(*w.snapshot(), q_arr, opt_margins, check_margins,
                                            buffer, linearize, group_size, segment_minimum);
        })
        .def("eval_pose_constraints", [](const gp::PlanningWorld& w, const Input& q_vec,
                                         const std::vector<gp::PoseConstraintSpec>& specs, bool linearize) {
            return snap_eval_pose_constraints(*w.snapshot(), q_vec, specs, linearize);
        })
        .def("solve_ik_batch", [](const gp::PlanningWorld& w,
                                  const std::vector<std::pair<std::vector<gp::PoseConstraintSpec>, Eigen::VectorXd>>& problems_in,
                                  const MaskInput& active_mask_in,
                                  double margin, double world_margin, int iterations, bool verify_original) {
            return snap_solve_ik_batch(*w.snapshot(), problems_in, active_mask_in,
                                       margin, world_margin, iterations, verify_original);
        })
        .def("plan_birrt_batch", [](const gp::PlanningWorld& w,
                                    const Input& starts, const Input& goals,
                                    const MaskInput& active_mask_in,
                                    double margin, double world_margin,
                                    double step_size, int max_iterations, double timeout_s,
                                    const std::vector<uint64_t>& seeds, bool verify_original,
                                    int shortcut_iters, int output_waypoints) {
            return snap_plan_birrt_batch(*w.snapshot(), starts, goals, active_mask_in,
                                         margin, world_margin, step_size, max_iterations, timeout_s,
                                         seeds, verify_original, shortcut_iters, output_waypoints);
        })
        .def("finalize_trajectory", [](const gp::PlanningWorld& w, const Input& pos_arr, const py::dict& cfg_dict) {
            return snap_finalize_trajectory(*w.snapshot(), pos_arr, cfg_dict);
        })
        .def("validate_trajectory_collision", [](const gp::PlanningWorld& w, const Input& q_waypoints,
                                                 const Input& samples, int substeps,
                                                 double margin, double world_margin, bool verify_original,
                                                 double disp_tol, py::object times_obj, py::object vel_obj,
                                                 bool strict_ccd, py::object acc_obj, int spline_order) {
            return snap_validate_trajectory_collision(
                *w.snapshot(), q_waypoints, samples, substeps, margin, world_margin,
                verify_original, disp_tol, times_obj, vel_obj, strict_ccd, acc_obj, spline_order);
        }, py::arg("q_waypoints"), py::arg("samples"), py::arg("substeps"),
           py::arg("margin"), py::arg("world_margin"), py::arg("verify_original") = true,
           py::arg("disp_tol") = 5e-4, py::arg("times") = py::none(), py::arg("velocities") = py::none(),
           py::arg("strict_ccd") = false, py::arg("accelerations") = py::none(), py::arg("spline_order") = 3)
        .def("solve_sqp_batch", [](const gp::PlanningWorld& w, const py::list& jobs_in, bool race) {
            auto snap = w.snapshot();
            const int J = snap->kin.num_joints();
            std::vector<gp::SQPJobSpecNative> specs;
            specs.reserve(jobs_in.size());
            for (const auto& item : jobs_in) {
                const py::dict d = item.cast<py::dict>();
                gp::SQPJobSpecNative sp;
                const Input init_arr = d["initial"].cast<Input>();
                if (init_arr.ndim() != 2 || init_arr.shape(1) != J)
                    throw std::invalid_argument("SQP initial trajectory must have shape [T, J]");
                const py::ssize_t T = init_arr.shape(0);
                sp.initial.resize(T, J);
                std::copy(init_arr.data(), init_arr.data() + T * J, sp.initial.data());
                sp.fixed = d["fixed"].cast<int>();
                if (d.contains("dt_vel") && !d["dt_vel"].is_none()) {
                    sp.has_dt_vel = true;
                    sp.dt_vel = d["dt_vel"].cast<double>();
                }
                sp.dt_smooth = d["dt_smooth"].cast<double>();
                const MaskInput mask_in = d["active_mask"].cast<MaskInput>();
                sp.active_mask.assign(static_cast<size_t>(J), 1);
                if (mask_in.size() == J) {
                    for (int j = 0; j < J; ++j) sp.active_mask[static_cast<size_t>(j)] = mask_in.data()[j] ? 1 : 0;
                }
                if (d.contains("pose_constraints") && !d["pose_constraints"].is_none()) {
                    const py::dict pc_dict = d["pose_constraints"].cast<py::dict>();
                    for (const auto& kv : pc_dict) {
                        auto vec = kv.second.cast<std::vector<gp::PoseConstraintSpec>>();
                        if (!vec.empty()) sp.pose_constraints.emplace_back(kv.first.cast<int>(), std::move(vec));
                    }
                }
                if (d.contains("joint_targets") && !d["joint_targets"].is_none()) {
                    const py::dict jt_dict = d["joint_targets"].cast<py::dict>();
                    for (const auto& kv : jt_dict) {
                        sp.joint_targets.emplace_back(kv.first.cast<int>(), kv.second.cast<Eigen::VectorXd>());
                    }
                }
                sp.margin = d["margin"].cast<double>();
                sp.world_margin = d["world_margin"].cast<double>();
                sp.collision_substeps = d["collision_substeps"].cast<int>();
                if (d.contains("max_segment_length") && !d["max_segment_length"].is_none()) {
                    sp.max_segment_length = d["max_segment_length"].cast<double>();
                }
                sp.iterations = d["iterations"].cast<int>();
                sp.first_feasible = d["first_feasible"].cast<bool>();
                sp.velocity_scale = d["velocity_scale"].cast<double>();
                sp.seed_strategy = d["seed_strategy"].cast<std::string>();
                sp.goal_index = d["goal_index"].cast<int>();
                specs.push_back(std::move(sp));
            }
            std::vector<gp::SQPJobResultNative> res;
            {
                py::gil_scoped_release release;
                res = gp::solve_sqp_batch(*snap, specs, race);
            }
            py::list out_list;
            for (const auto& r : res) {
                py::dict item;
                item["success"] = r.success;
                if (r.success) {
                    const py::ssize_t T = r.positions.rows();
                    py::array_t<double> pos_arr({T, static_cast<py::ssize_t>(J)});
                    std::copy(r.positions.data(), r.positions.data() + T * J, pos_arr.mutable_data());
                    item["positions"] = std::move(pos_arr);
                    item["reason"] = py::none();
                } else {
                    item["positions"] = py::none();
                    item["reason"] = py::str(r.reason);
                }
                item["qp_calls"] = r.qp_calls;
                item["sqp_iterations"] = r.sqp_iterations;
                item["sqp_ms"] = r.sqp_ms;
                item["validate_ms"] = r.validate_ms;
                item["cost"] = r.cost;
                item["seed_strategy"] = r.seed_strategy;
                item["goal_index"] = r.goal_index;
                if (!r.witnesses.empty()) {
                    const py::ssize_t K = static_cast<py::ssize_t>(r.witnesses.size() / 6);
                    py::array_t<double> wit_arr({K, py::ssize_t(6)});
                    std::copy(r.witnesses.begin(), r.witnesses.end(), wit_arr.mutable_data());
                    item["witnesses"] = std::move(wit_arr);
                } else {
                    item["witnesses"] = py::none();
                }
                out_list.append(std::move(item));
            }
            return out_list;
        }, py::arg("jobs"), py::arg("race") = false);

    py::class_<gp::MotionPlannerNative>(module, "MotionPlanner")
        .def(py::init<std::shared_ptr<gp::PlanningWorld>>())
        .def("plan", [](const gp::MotionPlannerNative& planner, const py::dict& req_dict) {
            const int J = planner.world()->num_joints();
            const auto req = parse_plan_request(req_dict, J);
            gp::PlanResultNative res;
            {
                py::gil_scoped_release release;
                res = planner.plan(req);
            }
            return plan_result_to_dict(res);
        })
        .def("plan_batch", [](const gp::MotionPlannerNative& planner, const py::list& req_list, int chunk_size) {
            const int J = planner.world()->num_joints();
            std::vector<gp::MotionPlanRequestNative> reqs;
            reqs.reserve(req_list.size());
            for (const auto& item : req_list) {
                reqs.push_back(parse_plan_request(item.cast<py::dict>(), J));
            }
            std::vector<gp::PlanResultNative> res;
            {
                py::gil_scoped_release release;
                res = planner.plan_batch(reqs, chunk_size);
            }
            py::list out;
            for (const auto& r : res) out.append(plan_result_to_dict(r));
            return out;
        }, py::arg("requests"), py::arg("chunk_size") = 64);
}
