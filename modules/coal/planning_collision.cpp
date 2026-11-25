#include <coal/BV/OBBRSS.h>
#include <coal/BVH/BVH_model.h>
#include <coal/collision.h>
#include <coal/distance.h>
#include <coal/shape/convex.h>
#include <coal/shape/geometric_shapes.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

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
#include <vector>

namespace py = pybind11;
namespace {
using Geometry = std::shared_ptr<coal::CollisionGeometry>;
using Frame = Eigen::Matrix<double, 3, 4, Eigen::RowMajor>;
using Input = py::array_t<double, py::array::c_style | py::array::forcecast>;
using Indices = py::array_t<int64_t, py::array::c_style | py::array::forcecast>;
using MaskInput = py::array_t<bool, py::array::c_style | py::array::forcecast>;

class ThreadPool {
public:
    static ThreadPool& instance() {
        static ThreadPool pool;
        return pool;
    }

    unsigned int max_threads() const { return num_threads_; }

    void parallel_for(py::ssize_t count, py::ssize_t min_chunk,
                      const std::function<void(py::ssize_t, py::ssize_t)>& fn) {
        if (count <= 0) return;
        if (num_threads_ <= 1 || count < min_chunk * 2) {
            fn(0, count);
            return;
        }
        // If another Python thread is already using the pool, run inline to avoid contention or deadlock.
        std::unique_lock<std::mutex> dispatch_lock(dispatch_mutex_, std::try_to_lock);
        if (!dispatch_lock.owns_lock()) {
            fn(0, count);
            return;
        }
        const py::ssize_t max_chunks = std::min<py::ssize_t>(num_threads_, (count + min_chunk - 1) / min_chunk);
        if (max_chunks <= 1) {
            fn(0, count);
            return;
        }
        const py::ssize_t chunk = (count + max_chunks - 1) / max_chunks;
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

        // Execute chunk 0 on the calling thread.
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
    }

private:
    ThreadPool() {
        const unsigned int hw = std::thread::hardware_concurrency();
        num_threads_ = std::min(8u, std::max(1u, hw));
        workers_.reserve(num_threads_ > 1 ? num_threads_ - 1 : 0);
        for (unsigned int id = 1; id < num_threads_; ++id) {
            workers_.emplace_back([this, id] { worker_loop(static_cast<py::ssize_t>(id)); });
        }
    }

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            stop_ = true;
            ++generation_;
        }
        cv_start_.notify_all();
        for (auto& worker : workers_) {
            if (worker.joinable()) worker.join();
        }
    }

    void worker_loop(py::ssize_t id) {
        uint64_t seen = 0;
        while (true) {
            const std::function<void(py::ssize_t, py::ssize_t)>* fn = nullptr;
            std::vector<std::exception_ptr>* errors = nullptr;
            py::ssize_t begin = 0, end = 0;
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
    const std::function<void(py::ssize_t, py::ssize_t)>* active_fn_ = nullptr;
    std::vector<std::exception_ptr>* active_errors_ = nullptr;
    py::ssize_t total_count_ = 0;
    py::ssize_t chunk_size_ = 0;
    py::ssize_t active_chunks_ = 0;
    py::ssize_t pending_workers_ = 0;
    uint64_t generation_ = 0;
    bool stop_ = false;
};

struct Shape {
    std::string label;
    Geometry convex, original;

    Shape(const std::string& name, const py::dict& spec) : label(name) {
        switch (spec["type"].cast<int>()) {
            case 1: convex = std::make_shared<coal::Box>(spec["size"].cast<coal::Vec3s>()); break;
            case 2: convex = std::make_shared<coal::Sphere>(spec["radius"].cast<double>()); break;
            case 3: convex = std::make_shared<coal::Cylinder>(spec["radius"].cast<double>(), spec["height"].cast<double>()); break;
            case 4: convex = std::make_shared<coal::Capsule>(spec["radius"].cast<double>(), spec["height"].cast<double>()); break;
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
                original = mesh;
                auto hull = std::make_shared<std::vector<coal::Vec3s>>(spec["hull_points"].cast<std::vector<coal::Vec3s>>());
                auto faces = std::make_shared<std::vector<coal::Triangle>>();
                for (const auto& face : spec["hull_faces"].cast<std::vector<std::array<size_t, 3>>>()) {
                    if (face[0] >= hull->size() || face[1] >= hull->size() || face[2] >= hull->size())
                        throw std::invalid_argument(name + ": invalid convex hull face");
                    faces->emplace_back(face[0], face[1], face[2]);
                }
                if (faces->empty()) throw std::invalid_argument(name + ": empty convex hull");
                convex = std::make_shared<coal::Convex<coal::Triangle>>(hull, hull->size(), faces, faces->size());
                break;
            }
            case 6: convex = std::make_shared<coal::Halfspace>(coal::Vec3s::UnitZ(), 0); break;
            default: throw std::invalid_argument(name + ": unsupported collision shape");
        }
        if (!original) original = convex;
        convex->computeLocalAABB();
        original->computeLocalAABB();
    }
};

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

inline void eval_pair_distance(const Shape& a, const Shape& b, const double* fa, const double* fb,
                               bool verify, double clearance, double* out) {
    const auto ta = transform_from(fa);
    const auto tb = transform_from(fb);
    const auto& ga = verify ? a.original : a.convex;
    const auto& gb = verify ? b.original : b.convex;
    // During final validation a bounding-box gap above the requested
    // clearance already proves that the original shapes pass this pair.
    if (verify && std::isfinite(clearance) && ga->getNodeType() != coal::GEOM_HALFSPACE &&
        gb->getNodeType() != coal::GEOM_HALFSPACE) {
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
        !result.normal.allFinite() || std::abs(result.normal.norm() - 1) > 1e-5)
        throw std::runtime_error("invalid witnesses or separation normal");
    Eigen::Map<coal::Vec3s> local_a(out + 1), local_b(out + 4), normal(out + 7);
    local_a = ta.inverseTransform(result.nearest_points[0]);
    local_b = tb.inverseTransform(result.nearest_points[1]);
    normal = -result.normal;
}

inline bool eval_pair_clearance(const Shape& a, const Shape& b, const double* fa, const double* fb,
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

    // Fast security-margin collision test with early exit in BVH / GJK.
    coal::CollisionRequest req(coal::NO_REQUEST, 1);
    req.security_margin = threshold;
    req.distance_upper_bound = threshold;
    coal::CollisionResult hit;
    coal::collide(ga.get(), ta, gb.get(), tb, req, hit);
    if (!hit.isCollision()) {
        return true;
    }

    // Confirm exact boundary semantics on the rare colliding/near-margin pair.
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

// Fast C++ broadphase AABB screening (segment-level union + per-sample filtering).
py::tuple screen_aabbs(const Input& frames, const Indices& endpoints, const MaskInput& planes,
                       const Indices& moving, const Input& local_center, const Input& local_half,
                       const Input& fixed_lo, const Input& fixed_hi, const Input& thresholds,
                       py::ssize_t group_size) {
    if (frames.ndim() != 4 || frames.shape(2) != 3 || frames.shape(3) != 4)
        throw std::invalid_argument("expected samples × shapes × 3 × 4 frames");
    const py::ssize_t S = frames.shape(0);
    const py::ssize_t N = frames.shape(1);
    const py::ssize_t P = endpoints.shape(0);
    const py::ssize_t M = moving.shape(0);
    if (S == 0 || P == 0) {
        return py::make_tuple(py::array_t<int64_t>(0), py::array_t<int64_t>(0));
    }

    const double* f_ptr = frames.data();
    const int64_t* ep_ptr = endpoints.data();
    const bool* pl_ptr = planes.data();
    const int64_t* mv_ptr = moving.data();
    const double* lc_ptr = local_center.data();
    const double* lh_ptr = local_half.data();
    const double* flo_ptr = fixed_lo.data();
    const double* fhi_ptr = fixed_hi.data();
    const double* thr_ptr = thresholds.data();

    std::vector<int64_t> out_rows;
    std::vector<int64_t> out_pairs;
    {
        py::gil_scoped_release release;

        // Map shape index -> moving slot (-1 if static).
        std::vector<int64_t> move_slot(static_cast<size_t>(N), -1);
        for (py::ssize_t m = 0; m < M; ++m) {
            move_slot[static_cast<size_t>(mv_ptr[m])] = m;
        }

        // Compute world AABBs only for moving shapes: [S, M, 3].
        std::vector<double> mlo(static_cast<size_t>(S * M * 3));
        std::vector<double> mhi(static_cast<size_t>(S * M * 3));
        const double eps16 = 16.0 * std::numeric_limits<double>::epsilon();

        auto compute_moving_aabb = [&](py::ssize_t s_begin, py::ssize_t s_end) {
            for (py::ssize_t s = s_begin; s < s_end; ++s) {
                const double* sf = f_ptr + s * N * 12;
                double* lo_s = mlo.data() + s * M * 3;
                double* hi_s = mhi.data() + s * M * 3;
                for (py::ssize_t m = 0; m < M; ++m) {
                    const double* fr = sf + mv_ptr[m] * 12;
                    const double cx = lc_ptr[m * 3 + 0], cy = lc_ptr[m * 3 + 1], cz = lc_ptr[m * 3 + 2];
                    const double hx = lh_ptr[m * 3 + 0], hy = lh_ptr[m * 3 + 1], hz = lh_ptr[m * 3 + 2];
                    const double acx = std::abs(cx), acy = std::abs(cy), acz = std::abs(cz);
                    for (int a = 0; a < 3; ++a) {
                        const double r0 = fr[a * 4 + 0], r1 = fr[a * 4 + 1], r2 = fr[a * 4 + 2], t = fr[a * 4 + 3];
                        const double ar0 = std::abs(r0), ar1 = std::abs(r1), ar2 = std::abs(r2);
                        const double center = r0 * cx + r1 * cy + r2 * cz + t;
                        const double extent0 = ar0 * hx + ar1 * hy + ar2 * hz;
                        const double scale = std::abs(t) + (ar0 * acx + ar1 * acy + ar2 * acz) + extent0;
                        const double extent = extent0 + eps16 * (1.0 + scale);
                        lo_s[m * 3 + a] = center - extent;
                        hi_s[m * 3 + a] = center + extent;
                    }
                }
            }
        };
        ThreadPool::instance().parallel_for(S, 32, compute_moving_aabb);

        auto pair_overlaps = [&](const double* a_lo, const double* a_hi,
                                 const double* b_lo, const double* b_hi, double thr) -> bool {
            for (int axis = 0; axis < 3; ++axis) {
                const double g0 = a_lo[axis] - b_hi[axis];
                const double g1 = b_lo[axis] - a_hi[axis];
                if ((g0 > g1 ? g0 : g1) > thr) return false;
            }
            return true;
        };

        auto get_bounds = [&](py::ssize_t s, int64_t shape_id, const double*& lo, const double*& hi) {
            const int64_t slot = move_slot[static_cast<size_t>(shape_id)];
            if (slot >= 0) {
                const size_t off = static_cast<size_t>((s * M + slot) * 3);
                lo = mlo.data() + off;
                hi = mhi.data() + off;
            } else {
                const size_t off = static_cast<size_t>(shape_id * 3);
                lo = flo_ptr + off;
                hi = fhi_ptr + off;
            }
        };

        if (group_size <= 1 || S < 2) {
            for (py::ssize_t s = 0; s < S; ++s) {
                for (py::ssize_t p = 0; p < P; ++p) {
                    if (pl_ptr[p]) {
                        out_rows.push_back(s);
                        out_pairs.push_back(p);
                        continue;
                    }
                    const double *alo, *ahi, *blo, *bhi;
                    get_bounds(s, ep_ptr[p * 2 + 0], alo, ahi);
                    get_bounds(s, ep_ptr[p * 2 + 1], blo, bhi);
                    if (pair_overlaps(alo, ahi, blo, bhi, thr_ptr[p])) {
                        out_rows.push_back(s);
                        out_pairs.push_back(p);
                    }
                }
            }
        } else {
            // Segment union bounds for moving shapes.
            std::vector<py::ssize_t> starts;
            for (py::ssize_t s = 0; s < S - 1; s += group_size) starts.push_back(s);
            const py::ssize_t G = static_cast<py::ssize_t>(starts.size());
            std::vector<double> glo(static_cast<size_t>(M * 3));
            std::vector<double> ghi(static_cast<size_t>(M * 3));
            std::vector<py::ssize_t> selected;
            selected.reserve(static_cast<size_t>(P));

            for (py::ssize_t g = 0; g < G; ++g) {
                const py::ssize_t start = starts[static_cast<size_t>(g)];
                const py::ssize_t stop = (g + 1 < G) ? starts[static_cast<size_t>(g + 1)] : S;
                // Union moving bounds across [start, stop).
                const double* first_lo = mlo.data() + start * M * 3;
                const double* first_hi = mhi.data() + start * M * 3;
                std::copy(first_lo, first_lo + M * 3, glo.begin());
                std::copy(first_hi, first_hi + M * 3, ghi.begin());
                for (py::ssize_t s = start + 1; s < stop; ++s) {
                    const double* slo = mlo.data() + s * M * 3;
                    const double* shi = mhi.data() + s * M * 3;
                    for (py::ssize_t k = 0; k < M * 3; ++k) {
                        if (slo[k] < glo[static_cast<size_t>(k)]) glo[static_cast<size_t>(k)] = slo[k];
                        if (shi[k] > ghi[static_cast<size_t>(k)]) ghi[static_cast<size_t>(k)] = shi[k];
                    }
                }
                selected.clear();
                for (py::ssize_t p = 0; p < P; ++p) {
                    if (pl_ptr[p]) {
                        selected.push_back(p);
                        continue;
                    }
                    const int64_t ia = ep_ptr[p * 2 + 0], ib = ep_ptr[p * 2 + 1];
                    const int64_t sa = move_slot[static_cast<size_t>(ia)];
                    const int64_t sb = move_slot[static_cast<size_t>(ib)];
                    const double* alo = sa >= 0 ? glo.data() + sa * 3 : flo_ptr + ia * 3;
                    const double* ahi = sa >= 0 ? ghi.data() + sa * 3 : fhi_ptr + ia * 3;
                    const double* blo = sb >= 0 ? glo.data() + sb * 3 : flo_ptr + ib * 3;
                    const double* bhi = sb >= 0 ? ghi.data() + sb * 3 : fhi_ptr + ib * 3;
                    if (pair_overlaps(alo, ahi, blo, bhi, thr_ptr[p])) {
                        selected.push_back(p);
                    }
                }
                if (selected.empty()) continue;
                for (py::ssize_t s = start; s < stop; ++s) {
                    for (const py::ssize_t p : selected) {
                        if (pl_ptr[p]) {
                            out_rows.push_back(s);
                            out_pairs.push_back(p);
                            continue;
                        }
                        const double *alo, *ahi, *blo, *bhi;
                        get_bounds(s, ep_ptr[p * 2 + 0], alo, ahi);
                        get_bounds(s, ep_ptr[p * 2 + 1], blo, bhi);
                        if (pair_overlaps(alo, ahi, blo, bhi, thr_ptr[p])) {
                            out_rows.push_back(s);
                            out_pairs.push_back(p);
                        }
                    }
                }
            }
        }
    }

    py::array_t<int64_t> rows_arr(static_cast<py::ssize_t>(out_rows.size()));
    py::array_t<int64_t> pairs_arr(static_cast<py::ssize_t>(out_pairs.size()));
    if (!out_rows.empty()) {
        std::copy(out_rows.begin(), out_rows.end(), rows_arr.mutable_data());
        std::copy(out_pairs.begin(), out_pairs.end(), pairs_arr.mutable_data());
    }
    return py::make_tuple(rows_arr, pairs_arr);
}

// Fast batched CPU forward kinematics (matching SceneKinematics._evaluate).
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
        using Mat3 = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>;
        using Vec3 = Eigen::Vector3d;

        std::vector<Mat3> a_rot(static_cast<size_t>(B)), b_rot(static_cast<size_t>(B));
        std::vector<Vec3> a_pos(static_cast<size_t>(B)), b_pos(static_cast<size_t>(B)), ax(static_cast<size_t>(B));
        std::vector<Mat3> skew(static_cast<size_t>(B)), skew2(static_cast<size_t>(B));
        for (py::ssize_t i = 0; i < B; ++i) {
            const double* a = pj_ptr + i * 16;
            const double* b = jc_ptr + i * 16;
            a_rot[i] << a[0], a[1], a[2], a[4], a[5], a[6], a[8], a[9], a[10];
            a_pos[i] << a[3], a[7], a[11];
            b_rot[i] << b[0], b[1], b[2], b[4], b[5], b[6], b[8], b[9], b[10];
            b_pos[i] << b[3], b[7], b[11];
            ax[i] << ax_ptr[i * 3 + 0], ax_ptr[i * 3 + 1], ax_ptr[i * 3 + 2];
            Mat3 s;
            s << 0.0, -ax[i](2), ax[i](1),
                 ax[i](2), 0.0, -ax[i](0),
                 -ax[i](1), ax[i](0), 0.0;
            skew[i] = s;
            skew2[i] = s * s;
        }
        const Mat3 eye3 = Mat3::Identity();

        auto work = [&](py::ssize_t s_begin, py::ssize_t s_end) {
            std::vector<Vec3> pos(static_cast<size_t>(B));
            std::vector<Mat3> rot(static_cast<size_t>(B));
            for (py::ssize_t s = s_begin; s < s_end; ++s) {
                const double* rp = batched_root ? (rp_ptr + s * 3) : rp_ptr;
                const double* rr = batched_root ? (rr_ptr + s * 9) : rr_ptr;
                pos[0] << rp[0], rp[1], rp[2];
                rot[0] = Eigen::Map<const Mat3>(rr);
                const double* qs = q_ptr + s * J;
                for (py::ssize_t i = 1; i < B; ++i) {
                    const int64_t p = par[i];
                    const int64_t j = jid[i];
                    const int64_t kind = knd[i];
                    const Mat3 anchor_r = rot[static_cast<size_t>(p)] * a_rot[i];
                    Vec3 anchor_p = pos[static_cast<size_t>(p)] + rot[static_cast<size_t>(p)] * a_pos[i];
                    Mat3 moving_r = anchor_r;
                    if (kind == 1 || kind == 2) {
                        const double angle = qs[j];
                        const double sn = std::sin(angle);
                        const double cs = 1.0 - std::cos(angle);
                        moving_r.noalias() = anchor_r * (eye3 + sn * skew[i] + cs * skew2[i]);
                    }
                    Vec3 world_axis;
                    if (kind == 3 || (return_joints && j >= 0)) {
                        world_axis.noalias() = anchor_r * ax[i];
                    }
                    if (return_joints && j >= 0) {
                        Eigen::Map<Vec3>(ojpos + (s * J + j) * 3) = anchor_p;
                        Eigen::Map<Vec3>(ojax + (s * J + j) * 3) = world_axis;
                    }
                    if (kind == 3) {
                        anchor_p.noalias() += world_axis * qs[j];
                    }
                    pos[i].noalias() = anchor_p + moving_r * b_pos[i];
                    rot[i].noalias() = moving_r * b_rot[i];
                }
                for (py::ssize_t t = 0; t < T; ++t) {
                    const size_t idx = static_cast<size_t>(trk[t]);
                    Eigen::Map<Vec3>(opos + (s * T + t) * 3) = pos[idx];
                    Eigen::Map<Mat3>(orot + (s * T + t) * 9) = rot[idx];
                }
            }
        };
        ThreadPool::instance().parallel_for(S, 64, work);
    }
    if (return_joints) {
        return py::make_tuple(out_pos, out_rot, out_jpos, out_jaxis);
    }
    return py::make_tuple(out_pos, out_rot);
}

// Store the immutable geometries once. Query results and GJK state belong to the caller's batch.
using Pair = std::array<std::shared_ptr<Shape>, 2>;
class Pairs {
public:
    explicit Pairs(std::vector<Pair> pairs) : pairs_(std::move(pairs)) {}

    py::array_t<double> query(const Input& frames, bool verify, double clearance, const py::object& ids) const {
        const py::ssize_t count = static_cast<py::ssize_t>(pairs_.size());
        const bool selected = !ids.is_none();
        Indices indices;
        if (selected) indices = ids.cast<Indices>();
        if (frames.ndim() != 5 || frames.shape(1) != (selected ? 1 : count) || frames.shape(2) != 2 ||
            frames.shape(3) != 3 || frames.shape(4) != 4)
            throw std::invalid_argument("expected batch × pairs × 2 × 3 × 4 frames");
        if (selected && (indices.ndim() != 1 || indices.size() != frames.shape(0)))
            throw std::invalid_argument("expected one pair index per frame row");
        const py::ssize_t rows = frames.shape(0);
        const py::ssize_t cols = frames.shape(1);
        const py::ssize_t width = verify ? 1 : 10;
        py::array_t<double> output({rows, cols, width});
        const double* f_ptr = frames.data();
        const int64_t* idx_ptr = selected ? indices.data() : nullptr;
        double* out_ptr = output.mutable_data();
        py::gil_scoped_release release;

        const py::ssize_t total = rows * cols;
        auto work = [&](py::ssize_t begin, py::ssize_t end) {
            for (py::ssize_t k = begin; k < end; ++k) {
                const py::ssize_t row = k / cols;
                const py::ssize_t i = k % cols;
                const auto id = selected ? idx_ptr[row] : i;
                if (id < 0 || id >= count) throw std::out_of_range("invalid collision pair index");
                const auto& [a, b] = pairs_[static_cast<size_t>(id)];
                try {
                    const double* fa = f_ptr + (k * 2 + 0) * 12;
                    const double* fb = f_ptr + (k * 2 + 1) * 12;
                    eval_pair_distance(*a, *b, fa, fb, verify, clearance, out_ptr + k * width);
                } catch (const std::exception& error) {
                    throw std::runtime_error(a->label + " / " + b->label + " (sample " + std::to_string(row) + "): " + error.what());
                }
            }
        };
        ThreadPool::instance().parallel_for(total, 64, work);
        return output;
    }

    py::array_t<double> query_indexed(const Input& frames, const Indices& endpoints,
                                      const Indices& rows, const Indices& pair_ids, bool verify) const {
        if (frames.ndim() != 4 || frames.shape(2) != 3 || frames.shape(3) != 4)
            throw std::invalid_argument("expected samples × shapes × 3 × 4 frames");
        const py::ssize_t count = static_cast<py::ssize_t>(pairs_.size());
        const py::ssize_t S = frames.shape(0);
        const py::ssize_t N = frames.shape(1);
        const py::ssize_t K = rows.size();
        if (pair_ids.size() != K) throw std::invalid_argument("rows and pair_ids length mismatch");
        const py::ssize_t width = verify ? 1 : 10;
        py::array_t<double> output({K, width});
        if (K == 0) return output;

        const double* f_ptr = frames.data();
        const int64_t* ep_ptr = endpoints.data();
        const int64_t* r_ptr = rows.data();
        const int64_t* p_ptr = pair_ids.data();
        double* out_ptr = output.mutable_data();
        py::gil_scoped_release release;

        auto work = [&](py::ssize_t begin, py::ssize_t end) {
            for (py::ssize_t k = begin; k < end; ++k) {
                const int64_t row = r_ptr[k];
                const int64_t id = p_ptr[k];
                if (row < 0 || row >= S || id < 0 || id >= count)
                    throw std::out_of_range("invalid indexed collision query");
                const auto& [a, b] = pairs_[static_cast<size_t>(id)];
                const int64_t sa = ep_ptr[id * 2 + 0];
                const int64_t sb = ep_ptr[id * 2 + 1];
                try {
                    const double* fa = f_ptr + (row * N + sa) * 12;
                    const double* fb = f_ptr + (row * N + sb) * 12;
                    eval_pair_distance(*a, *b, fa, fb, verify, std::numeric_limits<double>::infinity(),
                                       out_ptr + k * width);
                } catch (const std::exception& error) {
                    throw std::runtime_error(a->label + " / " + b->label + " (sample " + std::to_string(row) + "): " + error.what());
                }
            }
        };
        ThreadPool::instance().parallel_for(K, 64, work);
        return output;
    }

    py::object first_violation(const Input& frames, const Indices& endpoints,
                               const Indices& rows, const Indices& pair_ids,
                               const Input& margins, bool verify) const {
        if (frames.ndim() != 4 || frames.shape(2) != 3 || frames.shape(3) != 4)
            throw std::invalid_argument("expected samples × shapes × 3 × 4 frames");
        const py::ssize_t count = static_cast<py::ssize_t>(pairs_.size());
        const py::ssize_t S = frames.shape(0);
        const py::ssize_t N = frames.shape(1);
        const py::ssize_t K = rows.size();
        if (K == 0) return py::none();
        if (pair_ids.size() != K || margins.size() != count)
            throw std::invalid_argument("invalid first_violation input shapes");

        const double* f_ptr = frames.data();
        const int64_t* ep_ptr = endpoints.data();
        const int64_t* r_ptr = rows.data();
        const int64_t* p_ptr = pair_ids.data();
        const double* m_ptr = margins.data();

        std::atomic<py::ssize_t> earliest(K);
        {
            py::gil_scoped_release release;
            auto work = [&](py::ssize_t begin, py::ssize_t end) {
                for (py::ssize_t k = begin; k < end; ++k) {
                    if (k >= earliest.load(std::memory_order_relaxed)) break;
                    const int64_t row = r_ptr[k];
                    const int64_t id = p_ptr[k];
                    if (row < 0 || row >= S || id < 0 || id >= count)
                        throw std::out_of_range("invalid clearance check index");
                    const auto& [a, b] = pairs_[static_cast<size_t>(id)];
                    const int64_t sa = ep_ptr[id * 2 + 0];
                    const int64_t sb = ep_ptr[id * 2 + 1];
                    try {
                        const double* fa = f_ptr + (row * N + sa) * 12;
                        const double* fb = f_ptr + (row * N + sb) * 12;
                        if (!eval_pair_clearance(*a, *b, fa, fb, verify, m_ptr[id])) {
                            py::ssize_t prev = earliest.load(std::memory_order_relaxed);
                            while (k < prev && !earliest.compare_exchange_weak(prev, k, std::memory_order_relaxed)) {}
                            break;
                        }
                    } catch (const std::exception& error) {
                        throw std::runtime_error(a->label + " / " + b->label + " (sample " + std::to_string(row) + "): " + error.what());
                    }
                }
            };
            // For small batches run sequentially to hit earliest failure immediately; parallelize large checks.
            ThreadPool::instance().parallel_for(K, 128, work);
        }
        const py::ssize_t hit = earliest.load();
        if (hit >= K) return py::none();
        return py::make_tuple(r_ptr[hit], p_ptr[hit]);
    }

    py::array_t<bool> clearance_mask_indexed(const Input& frames, const Indices& endpoints,
                                             const Indices& rows, const Indices& pair_ids,
                                             const Input& margins, bool verify) const {
        if (frames.ndim() != 4 || frames.shape(2) != 3 || frames.shape(3) != 4)
            throw std::invalid_argument("expected samples × shapes × 3 × 4 frames");
        const py::ssize_t count = static_cast<py::ssize_t>(pairs_.size());
        const py::ssize_t S = frames.shape(0);
        const py::ssize_t N = frames.shape(1);
        const py::ssize_t K = rows.size();
        py::array_t<bool> output(K);
        if (K == 0) return output;
        if (pair_ids.size() != K || margins.size() != count)
            throw std::invalid_argument("invalid clearance_mask_indexed input shapes");

        const double* f_ptr = frames.data();
        const int64_t* ep_ptr = endpoints.data();
        const int64_t* r_ptr = rows.data();
        const int64_t* p_ptr = pair_ids.data();
        const double* m_ptr = margins.data();
        bool* out_ptr = output.mutable_data();

        py::gil_scoped_release release;
        auto work = [&](py::ssize_t begin, py::ssize_t end) {
            for (py::ssize_t k = begin; k < end; ++k) {
                const int64_t row = r_ptr[k];
                const int64_t id = p_ptr[k];
                if (row < 0 || row >= S || id < 0 || id >= count)
                    throw std::out_of_range("invalid clearance check index");
                const auto& [a, b] = pairs_[static_cast<size_t>(id)];
                const int64_t sa = ep_ptr[id * 2 + 0];
                const int64_t sb = ep_ptr[id * 2 + 1];
                try {
                    const double* fa = f_ptr + (row * N + sa) * 12;
                    const double* fb = f_ptr + (row * N + sb) * 12;
                    out_ptr[k] = eval_pair_clearance(*a, *b, fa, fb, verify, m_ptr[id]);
                } catch (const std::exception& error) {
                    throw std::runtime_error(a->label + " / " + b->label + " (sample " + std::to_string(row) + "): " + error.what());
                }
            }
        };
        ThreadPool::instance().parallel_for(K, 64, work);
        return output;
    }

private:
    std::vector<Pair> pairs_;
};
}  // namespace

PYBIND11_MODULE(_coal_native, module) {
    module.def("screen_aabbs", &screen_aabbs,
               py::arg("frames"), py::arg("endpoints"), py::arg("planes"),
               py::arg("moving"), py::arg("local_center"), py::arg("local_half"),
               py::arg("fixed_lo"), py::arg("fixed_hi"), py::arg("thresholds"),
               py::arg("group_size") = 1);
    module.def("forward_kinematics", &forward_kinematics,
               py::arg("parent"), py::arg("kinds"), py::arg("joint_idx"),
               py::arg("parent_joint"), py::arg("joint_child"), py::arg("axes"),
               py::arg("tracked_idx"), py::arg("root_pos"), py::arg("root_rot"),
               py::arg("q"), py::arg("return_joints") = false);
    py::class_<Shape, std::shared_ptr<Shape>>(module, "Shape").def(py::init<const std::string&, const py::dict&>())
        .def("bounds", [](const Shape& shape, bool original) {
            const auto& box = (original ? shape.original : shape.convex)->aabb_local;
            return std::make_pair(box.min_, box.max_);
        });
    py::class_<Pairs>(module, "Pairs")
        .def(py::init<std::vector<Pair>>())
        .def("query", &Pairs::query,
             py::arg("frames"), py::arg("verify"), py::arg("clearance") = std::numeric_limits<double>::infinity(),
             py::arg("pair_indices") = py::none())
        .def("query_indexed", &Pairs::query_indexed,
             py::arg("frames"), py::arg("endpoints"), py::arg("rows"), py::arg("pair_ids"),
             py::arg("verify") = false)
        .def("first_violation", &Pairs::first_violation,
             py::arg("frames"), py::arg("endpoints"), py::arg("rows"), py::arg("pair_ids"),
             py::arg("margins"), py::arg("verify") = false)
        .def("clearance_mask_indexed", &Pairs::clearance_mask_indexed,
             py::arg("frames"), py::arg("endpoints"), py::arg("rows"), py::arg("pair_ids"),
             py::arg("margins"), py::arg("verify") = false);
}
