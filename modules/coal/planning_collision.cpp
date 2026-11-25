#include "planning_world.hpp"

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
namespace gp = gobot::planning;

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

}  // namespace

PYBIND11_MODULE(_coal_native, module) {
    module.def("forward_kinematics", &forward_kinematics,
               py::arg("parent"), py::arg("kinds"), py::arg("joint_idx"),
               py::arg("parent_joint"), py::arg("joint_child"), py::arg("axes"),
               py::arg("tracked_idx"), py::arg("root_pos"), py::arg("root_rot"),
               py::arg("q"), py::arg("return_joints") = false);

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

    py::class_<gp::PlanningWorld, std::shared_ptr<gp::PlanningWorld>>(module, "PlanningWorld")
        .def(py::init<>())
        .def_property_readonly("version", &gp::PlanningWorld::version)
        .def_property_readonly("topology_version", &gp::PlanningWorld::topology_version)
        .def_property_readonly("num_bodies", [](const gp::PlanningWorld& w) { return w.kin.num_bodies(); })
        .def_property_readonly("num_joints", [](const gp::PlanningWorld& w) { return w.kin.num_joints(); })
        .def_property_readonly("self_pair_count", &gp::PlanningWorld::self_pair_count)
        .def_property_readonly("has_mesh", &gp::PlanningWorld::has_mesh)
        .def_property_readonly("num_shapes", [](const gp::PlanningWorld& w) { return static_cast<int>(w.shapes().size()); })
        .def_property_readonly("num_pairs", [](const gp::PlanningWorld& w) { return static_cast<int>(w.active_pairs().size()); })
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
            w.kin.initialize(std::move(parent), std::move(kinds), std::move(joint_idx),
                             std::move(a_rot), std::move(a_pos), std::move(b_rot), std::move(b_pos),
                             std::move(ax), std::move(branches), std::move(body_names), std::move(joint_names),
                             lower, upper, vel, acc, init_q);
            w.set_link_index(std::move(link_index));
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
        .def("set_collision_enabled", &gp::PlanningWorld::set_collision_enabled)
        .def("set_robot_base_pose", [](gp::PlanningWorld& w, int branch_idx, const gp::Vec3& pos, const gp::Mat3& rot) {
            w.set_robot_base_pose(branch_idx, pos, rot);
        })
        .def("pair_labels", [](const gp::PlanningWorld& w) {
            const auto& shapes = w.shapes();
            const auto& pairs = w.active_pairs();
            std::vector<std::pair<std::string, std::string>> out;
            out.reserve(pairs.size());
            for (const auto& pr : pairs) {
                out.emplace_back(shapes[pr.shape_a].link_key, shapes[pr.shape_b].link_key);
            }
            return out;
        })
        .def("pair_shape_names", [](const gp::PlanningWorld& w) {
            const auto& shapes = w.shapes();
            const auto& pairs = w.active_pairs();
            std::vector<std::pair<std::string, std::string>> out;
            out.reserve(pairs.size());
            for (const auto& pr : pairs) {
                out.emplace_back(shapes[pr.shape_a].name, shapes[pr.shape_b].name);
            }
            return out;
        })
        .def("export_warp_geometry", [](const gp::PlanningWorld& w) {
            // Export compact geometry metadata for MuJoCo Warp GPU backend.
            const auto& shapes = w.shapes();
            const auto& pairs = w.active_pairs();
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
        })
        .def("forward_kinematics", [](const gp::PlanningWorld& w, const Input& q_arr) {
            if (q_arr.ndim() != 2 || q_arr.shape(1) != w.kin.num_joints())
                throw std::invalid_argument("expected [S, J] joint array");
            const int64_t S = q_arr.shape(0);
            const int B = w.kin.num_bodies();
            const int J = w.kin.num_joints();
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
                        w.kin.fk_single(qp + s * J, bp.data(), br.data());
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
            if (q_arr.ndim() != 2 || q_arr.shape(1) != w.kin.num_joints())
                throw std::invalid_argument("expected [S, J] joint array");
            const int64_t S = q_arr.shape(0);
            const int N = static_cast<int>(w.shapes().size());
            const int J = w.kin.num_joints();
            py::array_t<double> frames({S, static_cast<py::ssize_t>(N), py::ssize_t(3), py::ssize_t(4)});
            py::array_t<double> jorig = return_joints ? py::array_t<double>({S, static_cast<py::ssize_t>(J), py::ssize_t(3)}) : py::array_t<double>(0);
            py::array_t<double> jaxis = return_joints ? py::array_t<double>({S, static_cast<py::ssize_t>(J), py::ssize_t(3)}) : py::array_t<double>(0);
            {
                py::gil_scoped_release release;
                w.compute_frames(q_arr.data(), S, frames.mutable_data(),
                                 return_joints ? jorig.mutable_data() : nullptr,
                                 return_joints ? jaxis.mutable_data() : nullptr);
            }
            if (return_joints) return py::make_tuple(frames, jorig, jaxis);
            return py::make_tuple(frames);
        }, py::arg("q"), py::arg("return_joints") = false)
        .def("screen_aabbs", [](const gp::PlanningWorld& w, const Input& frames, const Input& thresholds,
                                bool verify_original, int64_t group_size) {
            std::vector<int64_t> rows, pids;
            {
                py::gil_scoped_release release;
                w.screen_aabbs(frames.data(), frames.shape(0), thresholds.data(), verify_original, group_size, rows, pids);
            }
            py::array_t<int64_t> r_arr(static_cast<py::ssize_t>(rows.size()));
            py::array_t<int64_t> p_arr(static_cast<py::ssize_t>(pids.size()));
            if (!rows.empty()) {
                std::copy(rows.begin(), rows.end(), r_arr.mutable_data());
                std::copy(pids.begin(), pids.end(), p_arr.mutable_data());
            }
            return py::make_tuple(r_arr, p_arr);
        }, py::arg("frames"), py::arg("thresholds"), py::arg("verify_original") = false, py::arg("group_size") = 1)
        .def("query_distances_indexed", [](const gp::PlanningWorld& w, const Input& frames,
                                           const Indices& rows, const Indices& pair_ids, bool verify) {
            const int64_t K = rows.size();
            const int64_t width = verify ? 1 : 10;
            py::array_t<double> out({K, width});
            if (K > 0) {
                py::gil_scoped_release release;
                w.query_distances_indexed(frames.data(), frames.shape(0), rows.data(), pair_ids.data(),
                                          K, verify, out.mutable_data());
            }
            return out;
        }, py::arg("frames"), py::arg("rows"), py::arg("pair_ids"), py::arg("verify") = false)
        .def("query_all_distances", [](const gp::PlanningWorld& w, const Input& q_arr, bool verify) {
            if (q_arr.ndim() != 2 || q_arr.shape(1) != w.kin.num_joints())
                throw std::invalid_argument("expected [S, J] joint array");
            const int64_t S = q_arr.shape(0);
            const auto& pairs = w.active_pairs();
            const int64_t P = static_cast<int64_t>(pairs.size());
            const int64_t N = static_cast<int64_t>(w.shapes().size());
            py::array_t<double> out({S, P});
            if (S == 0 || P == 0) return out;
            std::vector<double> frames(static_cast<size_t>(S * N * 12));
            double* op = out.mutable_data();
            {
                py::gil_scoped_release release;
                w.compute_frames(q_arr.data(), S, frames.data());
                const auto& shapes = w.shapes();
                gp::ThreadPool::instance().parallel_for(S * P, 64, [&](int64_t b, int64_t e) {
                    double tmp[10];
                    for (int64_t k = b; k < e; ++k) {
                        const int64_t s = k / P;
                        const int64_t p = k % P;
                        const auto& pr = pairs[static_cast<size_t>(p)];
                        gp::eval_coal_distance(*shapes[pr.shape_a].shape, *shapes[pr.shape_b].shape,
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
            std::optional<std::pair<int64_t, int64_t>> hit;
            {
                py::gil_scoped_release release;
                hit = w.first_violation_indexed(frames.data(), frames.shape(0), rows.data(), pair_ids.data(),
                                                rows.size(), margins.data(), verify);
            }
            if (!hit.has_value()) return py::none();
            return py::make_tuple(hit->first, hit->second);
        }, py::arg("frames"), py::arg("rows"), py::arg("pair_ids"), py::arg("margins"), py::arg("verify") = false)
        .def("compute_analytic_gradients", [](const gp::PlanningWorld& w, const Input& frames,
                                              const Input& jorig, const Input& jaxis,
                                              const Indices& rows, const Indices& pair_ids, const Input& data) {
            const int64_t K = rows.size();
            const int J = w.kin.num_joints();
            py::array_t<double> grad({K, static_cast<py::ssize_t>(J)});
            py::array_t<double> wit({K, py::ssize_t(6)});
            if (K > 0) {
                py::gil_scoped_release release;
                w.compute_analytic_gradients(frames.data(), jorig.data(), jaxis.data(),
                                             rows.data(), pair_ids.data(), data.data(),
                                             K, grad.mutable_data(), wit.mutable_data());
            }
            return py::make_tuple(grad, wit);
        })
        .def("eval_collision_rows", [](const gp::PlanningWorld& w, const Input& q_arr,
                                       const Input& opt_margins, const Input& check_margins,
                                       double buffer, bool linearize, int64_t group_size, bool segment_minimum) {
            const int64_t S = q_arr.shape(0);
            const int N = static_cast<int>(w.shapes().size());
            const int J = w.kin.num_joints();
            gp::CollisionEvalResult res;
            {
                py::gil_scoped_release release;
                res = w.eval_collision_rows(q_arr.data(), S, opt_margins.data(), check_margins.data(),
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
        })
        .def("eval_pose_constraints", [](const gp::PlanningWorld& w, const Input& q_vec,
                                         const std::vector<gp::PoseConstraintSpec>& specs, bool linearize) {
            const int B = w.kin.num_bodies();
            const int J = w.kin.num_joints();
            std::vector<gp::Vec3> bpos(static_cast<size_t>(B)), jorig(linearize ? static_cast<size_t>(J) : 0),
                jaxis(linearize ? static_cast<size_t>(J) : 0);
            std::vector<gp::Mat3> brot(static_cast<size_t>(B));
            w.kin.fk_single(q_vec.data(), bpos.data(), brot.data(),
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
                const bool ok = w.kin.eval_pose_constraint(s, bpos.data(), brot.data(),
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
        })
        .def("solve_ik_batch", [](const gp::PlanningWorld& w,
                                  const std::vector<std::pair<std::vector<gp::PoseConstraintSpec>, Eigen::VectorXd>>& problems_in,
                                  const MaskInput& active_mask_in,
                                  double margin, double world_margin, int iterations, bool verify_original) {
            std::vector<gp::IKProblem> problems;
            problems.reserve(problems_in.size());
            for (const auto& [targets, seed] : problems_in) problems.push_back({targets, seed});
            std::vector<uint8_t> mask(static_cast<size_t>(w.kin.num_joints()), 1);
            if (active_mask_in.size() == w.kin.num_joints()) {
                for (int j = 0; j < w.kin.num_joints(); ++j) mask[static_cast<size_t>(j)] = active_mask_in.data()[j] ? 1 : 0;
            }
            std::vector<gp::IKResultNative> res;
            {
                py::gil_scoped_release release;
                res = w.solve_ik_batch(problems, mask, margin, world_margin, iterations, verify_original);
            }
            const py::ssize_t B = static_cast<py::ssize_t>(res.size());
            const py::ssize_t J = w.kin.num_joints();
            py::array_t<double> q_out({B, J});
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
        })
        .def("plan_birrt_batch", [](const gp::PlanningWorld& w,
                                    const Input& starts, const Input& goals,
                                    const MaskInput& active_mask_in,
                                    double margin, double world_margin,
                                    double step_size, int max_iterations, double timeout_s,
                                    const std::vector<uint64_t>& seeds, bool verify_original,
                                    int shortcut_iters, int output_waypoints) {
            const int64_t B = starts.shape(0);
            const int J = w.kin.num_joints();
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
                        res[static_cast<size_t>(i)] = w.plan_birrt(s, g, mask, margin, world_margin, step_size,
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
        });
}
