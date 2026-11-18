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

#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;
namespace {
using Geometry = std::shared_ptr<coal::CollisionGeometry>;
using Frame = Eigen::Matrix<double, 3, 4, Eigen::RowMajor>;
using Input = py::array_t<double, py::array::c_style | py::array::forcecast>;

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

coal::Transform3s transform(const double* data) {
    Eigen::Map<const Frame> frame(data);
    if (!frame.allFinite()) throw std::invalid_argument("nonfinite shape transform");
    return coal::Transform3s(frame.leftCols<3>(), frame.col(3));
}

// Store the immutable geometries once. Query results and GJK state belong to the caller's batch.
using Pair = std::array<std::shared_ptr<Shape>, 2>;
class Pairs {
public:
    explicit Pairs(std::vector<Pair> pairs) : pairs_(std::move(pairs)) {}

    py::array_t<double> query(const Input& frames, bool verify, double clearance) const {
        const py::ssize_t count = static_cast<py::ssize_t>(pairs_.size());
        if (frames.ndim() != 5 || frames.shape(1) != count || frames.shape(2) != 2 ||
            frames.shape(3) != 3 || frames.shape(4) != 4) throw std::invalid_argument("expected batch × pairs × 2 × 3 × 4 frames");
        const py::ssize_t width = verify ? 1 : 10;
        py::array_t<double> output({frames.shape(0), count, width});
        py::gil_scoped_release release;
        for (py::ssize_t row = 0; row < frames.shape(0); ++row) {
            for (py::ssize_t i = 0; i < count; ++i) {
                const auto& [a, b] = pairs_[i];
                double* out = output.mutable_data(row, i, 0);
                try {
                    const auto ta = transform(frames.data(row, i, 0, 0, 0));
                    const auto tb = transform(frames.data(row, i, 1, 0, 0));
                    const auto& ga = verify ? a->original : a->convex;
                    const auto& gb = verify ? b->original : b->convex;
                    // During final validation a bounding-box gap above the requested
                    // clearance already proves that the original shapes pass this pair.
                    if (verify && ga->getNodeType() != coal::GEOM_HALFSPACE && gb->getNodeType() != coal::GEOM_HALFSPACE) {
                        const auto world_bounds = [](const Geometry& geometry, const coal::Transform3s& pose) {
                            const auto& local = geometry->aabb_local;
                            const coal::Vec3s center = pose.transform(local.center());
                            const coal::Vec3s half = pose.getRotation().cwiseAbs() * ((local.max_ - local.min_) * 0.5);
                            return coal::AABB(center - half, center + half);
                        };
                        const auto ba = world_bounds(ga, ta);
                        const auto bb = world_bounds(gb, tb);
                        double gap = ba.distance(bb);
                        if (gap > clearance) {
                            out[0] = gap;
                            continue;
                        }
                    }
                    if (verify && (ga != a->convex || gb != b->convex)) {
                        coal::CollisionResult hit;
                        coal::collide(ga.get(), ta, gb.get(), tb, coal::CollisionRequest(), hit);
                        if (hit.isCollision()) {
                            out[0] = -std::numeric_limits<double>::infinity();
                            continue;
                        }
                    }
                    coal::DistanceResult result;
                    coal::distance(ga.get(), ta, gb.get(), tb, coal::DistanceRequest(), result);
                    out[0] = result.min_distance;
                    if (!std::isfinite(out[0]) || out[0] == std::numeric_limits<double>::max())
                        throw std::runtime_error("invalid distance");
                    if (verify) continue;
                    if (!result.nearest_points[0].allFinite() || !result.nearest_points[1].allFinite() ||
                        !result.normal.allFinite() || std::abs(result.normal.norm() - 1) > 1e-5)
                        throw std::runtime_error("invalid witnesses or separation normal");
                    Eigen::Map<coal::Vec3s> local_a(out + 1), local_b(out + 4), normal(out + 7);
                    local_a = ta.inverseTransform(result.nearest_points[0]);
                    local_b = tb.inverseTransform(result.nearest_points[1]);
                    normal = -result.normal;
                } catch (const std::exception& error) {
                    throw std::runtime_error(a->label + " / " + b->label + " (sample " + std::to_string(row) + "): " + error.what());
                }
            }
        }
        return output;
    }
private:
    std::vector<Pair> pairs_;
};
}

PYBIND11_MODULE(_coal_native, module) {
    py::class_<Shape, std::shared_ptr<Shape>>(module, "Shape").def(py::init<const std::string&, const py::dict&>());
    py::class_<Pairs>(module, "Pairs").def(py::init<std::vector<Pair>>()).def("query", &Pairs::query,
        py::arg("frames"), py::arg("verify"), py::arg("clearance") = std::numeric_limits<double>::infinity());
}
