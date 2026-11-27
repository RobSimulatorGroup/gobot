#pragma once

#include "planning_types.hpp"

#include <Eigen/SVD>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace gobot::planning {

namespace detail {

struct EdgeKey {
    int a = 0;
    int b = 0;
    bool operator==(const EdgeKey& o) const noexcept { return a == o.a && b == o.b; }
};

struct EdgeKeyHash {
    size_t operator()(const EdgeKey& k) const noexcept {
        return (static_cast<size_t>(static_cast<uint32_t>(k.a)) << 32) ^
               static_cast<size_t>(static_cast<uint32_t>(k.b));
    }
};

struct HullFace {
    int a = 0;
    int b = 0;
    int c = 0;
    Vec3 normal = Vec3::Zero();
    double offset = 0.0;  // normal.dot(v) - offset > 0 means strictly outside
    bool alive = true;
};

inline bool make_oriented_face(const std::vector<Vec3>& pts,
                               int a, int b, int c,
                               const Vec3& interior,
                               double area_eps,
                               HullFace& out) {
    const Vec3& pa = pts[static_cast<size_t>(a)];
    const Vec3& pb = pts[static_cast<size_t>(b)];
    const Vec3& pc = pts[static_cast<size_t>(c)];
    Vec3 n = (pb - pa).cross(pc - pa);
    const double len = n.norm();
    if (!std::isfinite(len) || len <= area_eps) {
        return false;
    }
    n /= len;
    if (n.dot(pa - interior) < 0.0) {
        std::swap(b, c);
        n = -n;
    }
    out.a = a;
    out.b = b;
    out.c = c;
    out.normal = n;
    out.offset = n.dot(pa);
    out.alive = true;
    return true;
}

// Compute exact 2D convex hull (Andrew's monotone chain) for coplanar point set.
inline void compute_2d_convex_hull(const std::vector<Vec3>& pts,
                                   const Vec3& centroid,
                                   const Eigen::Matrix<double, 3, 2>& basis2,
                                   std::vector<coal::Vec3s>& out_hull_pts,
                                   std::vector<coal::Triangle>& out_faces) {
    struct Pt2 {
        double x = 0.0;
        double y = 0.0;
        int idx = 0;
    };
    const int n = static_cast<int>(pts.size());
    std::vector<Pt2> p2(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        const Eigen::Vector2d q = basis2.transpose() * (pts[static_cast<size_t>(i)] - centroid);
        p2[static_cast<size_t>(i)] = {q.x(), q.y(), i};
    }
    std::sort(p2.begin(), p2.end(), [](const Pt2& u, const Pt2& v) {
        if (u.x != v.x) return u.x < v.x;
        return u.y < v.y;
    });
    auto cross2 = [](const Pt2& o, const Pt2& a, const Pt2& b) {
        return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
    };
    std::vector<Pt2> lower, upper;
    for (const auto& pt : p2) {
        while (lower.size() >= 2 && cross2(lower[lower.size() - 2], lower.back(), pt) <= 0.0) {
            lower.pop_back();
        }
        lower.push_back(pt);
    }
    for (int i = n - 1; i >= 0; --i) {
        const auto& pt = p2[static_cast<size_t>(i)];
        while (upper.size() >= 2 && cross2(upper[upper.size() - 2], upper.back(), pt) <= 0.0) {
            upper.pop_back();
        }
        upper.push_back(pt);
    }
    lower.pop_back();
    upper.pop_back();
    lower.insert(lower.end(), upper.begin(), upper.end());
    if (lower.size() < 3) {
        throw std::invalid_argument("degenerate coplanar hull");
    }
    out_hull_pts.clear();
    out_hull_pts.reserve(lower.size());
    for (const auto& pt : lower) {
        out_hull_pts.push_back(pts[static_cast<size_t>(pt.idx)]);
    }
    out_faces.clear();
    out_faces.reserve(lower.size() - 2);
    for (size_t i = 1; i + 1 < lower.size(); ++i) {
        out_faces.emplace_back(0, static_cast<coal::Triangle::index_type>(i),
                               static_cast<coal::Triangle::index_type>(i + 1));
    }
}

// Compute exact 3D convex hull via incremental horizon-edge expansion (furthest-point order).
inline void compute_3d_convex_hull(const std::vector<Vec3>& pts,
                                   double scale,
                                   std::vector<coal::Vec3s>& out_hull_pts,
                                   std::vector<coal::Triangle>& out_faces) {
    const int n = static_cast<int>(pts.size());
    if (n < 4) {
        throw std::invalid_argument("3D convex hull requires at least 4 points");
    }

    // 1. Pick extreme initial tetrahedron maximizing distance at each step.
    int i0 = 0, i1 = 0;
    for (int i = 1; i < n; ++i) {
        if (pts[static_cast<size_t>(i)].x() < pts[static_cast<size_t>(i0)].x()) i0 = i;
        if (pts[static_cast<size_t>(i)].x() > pts[static_cast<size_t>(i1)].x()) i1 = i;
    }
    if (i0 == i1) {
        double best_d2 = -1.0;
        for (int i = 1; i < n; ++i) {
            const double d2 = (pts[static_cast<size_t>(i)] - pts[static_cast<size_t>(i0)]).squaredNorm();
            if (d2 > best_d2) {
                best_d2 = d2;
                i1 = i;
            }
        }
    }
    const Vec3 d01 = pts[static_cast<size_t>(i1)] - pts[static_cast<size_t>(i0)];
    const double d01_norm = d01.norm();
    if (d01_norm <= 1e-12 * std::max(1.0, scale)) {
        throw std::invalid_argument("degenerate 3D mesh points");
    }
    const Vec3 dir01 = d01 / d01_norm;

    int i2 = -1;
    double best_line_d2 = -1.0;
    for (int i = 0; i < n; ++i) {
        if (i == i0 || i == i1) continue;
        const Vec3 v = pts[static_cast<size_t>(i)] - pts[static_cast<size_t>(i0)];
        const double d2 = (v - dir01 * v.dot(dir01)).squaredNorm();
        if (d2 > best_line_d2) {
            best_line_d2 = d2;
            i2 = i;
        }
    }
    if (i2 < 0 || best_line_d2 <= 0.0) {
        throw std::invalid_argument("collinear 3D mesh points");
    }
    Vec3 plane_n = (pts[static_cast<size_t>(i1)] - pts[static_cast<size_t>(i0)])
                       .cross(pts[static_cast<size_t>(i2)] - pts[static_cast<size_t>(i0)]);
    const double plane_n_norm = plane_n.norm();
    if (plane_n_norm <= 0.0) {
        throw std::invalid_argument("collinear 3D mesh points");
    }
    plane_n /= plane_n_norm;

    int i3 = -1;
    double best_plane_d = -1.0;
    for (int i = 0; i < n; ++i) {
        if (i == i0 || i == i1 || i == i2) continue;
        const double d = std::abs(plane_n.dot(pts[static_cast<size_t>(i)] - pts[static_cast<size_t>(i0)]));
        if (d > best_plane_d) {
            best_plane_d = d;
            i3 = i;
        }
    }
    if (i3 < 0 || best_plane_d <= 0.0) {
        throw std::invalid_argument("coplanar 3D mesh points");
    }

    const Vec3 interior = 0.25 * (pts[static_cast<size_t>(i0)] + pts[static_cast<size_t>(i1)] +
                                  pts[static_cast<size_t>(i2)] + pts[static_cast<size_t>(i3)]);
    const double plane_eps = 1e-10 * std::max(1.0, scale);
    const double area_eps = 1e-14 * std::max(1.0, scale * scale);

    std::vector<HullFace> faces;
    faces.reserve(static_cast<size_t>(std::min(n * 4, 65536)));
    const std::array<std::array<int, 3>, 4> init_tris = {{
        {i0, i1, i2}, {i0, i1, i3}, {i1, i2, i3}, {i2, i0, i3}
    }};
    for (const auto& t : init_tris) {
        HullFace f;
        if (!make_oriented_face(pts, t[0], t[1], t[2], interior, area_eps, f)) {
            throw std::invalid_argument("failed to construct initial tetrahedron");
        }
        faces.push_back(f);
    }

    // Track remaining candidate points (furthest-outside point per step, QuickHull-style).
    std::vector<int> remaining;
    remaining.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        if (i != i0 && i != i1 && i != i2 && i != i3) {
            remaining.push_back(i);
        }
    }

    std::vector<int> visible;
    std::unordered_map<EdgeKey, int, EdgeKeyHash> edge_counts;
    std::vector<int> next_remaining;
    next_remaining.reserve(remaining.size());

    while (!remaining.empty()) {
        int best_pt = -1;
        double best_dist = plane_eps;
        next_remaining.clear();
        for (int idx : remaining) {
            const Vec3& p = pts[static_cast<size_t>(idx)];
            double max_d = -std::numeric_limits<double>::infinity();
            for (const auto& f : faces) {
                if (!f.alive) continue;
                const double d = f.normal.dot(p) - f.offset;
                if (d > max_d) max_d = d;
            }
            if (max_d > plane_eps) {
                next_remaining.push_back(idx);
                if (max_d > best_dist) {
                    best_dist = max_d;
                    best_pt = idx;
                }
            }
        }
        remaining.swap(next_remaining);
        if (best_pt < 0) {
            break;
        }
        // Remove best_pt from remaining.
        remaining.erase(std::remove(remaining.begin(), remaining.end(), best_pt), remaining.end());

        const Vec3& p = pts[static_cast<size_t>(best_pt)];
        visible.clear();
        for (int fi = 0; fi < static_cast<int>(faces.size()); ++fi) {
            if (!faces[static_cast<size_t>(fi)].alive) continue;
            const double d = faces[static_cast<size_t>(fi)].normal.dot(p) - faces[static_cast<size_t>(fi)].offset;
            if (d > plane_eps) {
                visible.push_back(fi);
            }
        }
        if (visible.empty()) {
            continue;
        }

        // Count directed edges of visible faces; an edge (u, v) is on the horizon iff (v, u) is not visible.
        edge_counts.clear();
        edge_counts.reserve(visible.size() * 3);
        for (int fi : visible) {
            auto& f = faces[static_cast<size_t>(fi)];
            f.alive = false;
            edge_counts[{f.a, f.b}] += 1;
            edge_counts[{f.b, f.c}] += 1;
            edge_counts[{f.c, f.a}] += 1;
        }

        for (const auto& [ek, cnt] : edge_counts) {
            if (cnt != 1) continue;
            if (edge_counts.find({ek.b, ek.a}) != edge_counts.end()) continue;
            HullFace nf;
            if (make_oriented_face(pts, ek.a, ek.b, best_pt, interior, area_eps, nf)) {
                faces.push_back(nf);
            }
        }
    }

    // Compact live vertices and triangles.
    std::vector<int> remap(static_cast<size_t>(n), -1);
    out_hull_pts.clear();
    out_faces.clear();
    for (const auto& f : faces) {
        if (!f.alive) continue;
        std::array<int, 3> idx = {f.a, f.b, f.c};
        for (int k = 0; k < 3; ++k) {
            int& r = remap[static_cast<size_t>(idx[static_cast<size_t>(k)])];
            if (r < 0) {
                r = static_cast<int>(out_hull_pts.size());
                out_hull_pts.push_back(pts[static_cast<size_t>(idx[static_cast<size_t>(k)])]);
            }
            idx[static_cast<size_t>(k)] = r;
        }
        out_faces.emplace_back(static_cast<coal::Triangle::index_type>(idx[0]),
                               static_cast<coal::Triangle::index_type>(idx[1]),
                               static_cast<coal::Triangle::index_type>(idx[2]));
    }
    if (out_hull_pts.size() < 4 || out_faces.size() < 4) {
        throw std::invalid_argument("failed to produce valid 3D convex hull");
    }
}

}  // namespace detail

inline std::shared_ptr<CollisionShape> make_box_shape(const std::string& label, const Vec3& size) {
    if (!size.allFinite() || (size.array() < 0.0).any()) {
        throw std::invalid_argument("invalid size on " + label);
    }
    auto sh = std::make_shared<CollisionShape>();
    sh->label = label;
    sh->kind = 1;
    sh->convex = std::make_shared<coal::Box>(size);
    sh->box_half_size = size * 0.5;
    sh->finalize_bounds();
    return sh;
}

inline std::shared_ptr<CollisionShape> make_sphere_shape(const std::string& label, double radius) {
    if (!std::isfinite(radius) || radius < 0.0) {
        throw std::invalid_argument("invalid radius on " + label);
    }
    auto sh = std::make_shared<CollisionShape>();
    sh->label = label;
    sh->kind = 2;
    sh->radius = radius;
    sh->convex = std::make_shared<coal::Sphere>(radius);
    sh->finalize_bounds();
    return sh;
}

inline std::shared_ptr<CollisionShape> make_cylinder_shape(const std::string& label, double radius, double height) {
    if (!std::isfinite(radius) || radius < 0.0 || !std::isfinite(height) || height < 0.0) {
        throw std::invalid_argument("invalid cylinder radius/height on " + label);
    }
    auto sh = std::make_shared<CollisionShape>();
    sh->label = label;
    sh->kind = 3;
    sh->radius = radius;
    sh->height = height;
    sh->convex = std::make_shared<coal::Cylinder>(radius, height);
    sh->finalize_bounds();
    return sh;
}

inline std::shared_ptr<CollisionShape> make_capsule_shape(const std::string& label, double radius, double height) {
    if (!std::isfinite(radius) || radius < 0.0 || !std::isfinite(height) || height < 0.0) {
        throw std::invalid_argument("invalid capsule radius/height on " + label);
    }
    auto sh = std::make_shared<CollisionShape>();
    sh->label = label;
    sh->kind = 4;
    sh->radius = radius;
    sh->height = height;
    sh->convex = std::make_shared<coal::Capsule>(radius, height);
    sh->finalize_bounds();
    return sh;
}

inline std::shared_ptr<CollisionShape> make_plane_shape(const std::string& label) {
    auto sh = std::make_shared<CollisionShape>();
    sh->label = label;
    sh->kind = 6;
    sh->convex = std::make_shared<coal::Halfspace>(coal::Vec3s::UnitZ(), 0.0);
    sh->finalize_bounds();
    return sh;
}

// Build a mesh CollisionShape matching Python world.py:_prepare_mesh_spec semantics:
// - builds exact BVHModel<OBBRSS> as `original`
// - SVD degeneracy check: collinear -> throws, coplanar -> exact 2D convex hull,
//   full-rank 3D -> exact 3D convex hull (never silently falls back to a bloated OBB).
inline std::shared_ptr<CollisionShape> make_mesh_shape(const std::string& label,
                                                       const std::vector<Vec3>& vertices,
                                                       const std::vector<uint32_t>& indices_in) {
    if (vertices.size() < 3) {
        throw std::invalid_argument("invalid vertices on " + label);
    }
    for (const auto& v : vertices) {
        if (!v.allFinite()) {
            throw std::invalid_argument("invalid vertices on " + label);
        }
    }

    std::vector<uint32_t> indices = indices_in;
    if (indices.empty() && vertices.size() % 3 == 0) {
        indices.resize(vertices.size());
        std::iota(indices.begin(), indices.end(), 0u);
    }
    if (indices.empty() || (indices.size() % 3) != 0) {
        throw std::invalid_argument("invalid triangle indices on " + label);
    }
    std::vector<coal::Triangle> triangles;
    triangles.reserve(indices.size() / 3);
    for (size_t i = 0; i < indices.size(); i += 3) {
        for (size_t k = 0; k < 3; ++k) {
            if (indices[i + k] >= vertices.size()) {
                throw std::invalid_argument("invalid triangle indices on " + label);
            }
        }
        triangles.emplace_back(indices[i], indices[i + 1], indices[i + 2]);
    }

    // Deduplicate referenced vertices deterministically (matches np.unique(vertices[indices], axis=0)).
    std::vector<Vec3> unique_pts;
    unique_pts.reserve(vertices.size());
    std::vector<uint8_t> used(vertices.size(), 0);
    for (uint32_t idx : indices) {
        used[idx] = 1;
    }
    for (size_t i = 0; i < vertices.size(); ++i) {
        if (used[i]) unique_pts.push_back(vertices[i]);
    }
    std::sort(unique_pts.begin(), unique_pts.end(), [](const Vec3& a, const Vec3& b) {
        if (a.x() != b.x()) return a.x() < b.x();
        if (a.y() != b.y()) return a.y() < b.y();
        return a.z() < b.z();
    });
    unique_pts.erase(std::unique(unique_pts.begin(), unique_pts.end(), [](const Vec3& a, const Vec3& b) {
        return a.x() == b.x() && a.y() == b.y() && a.z() == b.z();
    }), unique_pts.end());

    if (unique_pts.size() < 3) {
        throw std::invalid_argument("degenerate mesh on " + label);
    }

    Vec3 centroid = Vec3::Zero();
    for (const auto& p : unique_pts) centroid += p;
    centroid /= static_cast<double>(unique_pts.size());

    Eigen::MatrixXd centered(static_cast<Eigen::Index>(unique_pts.size()), 3);
    for (size_t i = 0; i < unique_pts.size(); ++i) {
        centered.row(static_cast<Eigen::Index>(i)) = (unique_pts[i] - centroid).transpose();
    }
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(centered, Eigen::ComputeThinV);
    const Eigen::Vector3d sv = svd.singularValues();
    const double tol = sv(0) * static_cast<double>(std::max<size_t>(unique_pts.size(), 3)) *
                       std::numeric_limits<double>::epsilon();
    if (sv(1) <= tol) {
        throw std::invalid_argument("collinear mesh on " + label);
    }

    auto hull_pts = std::make_shared<std::vector<coal::Vec3s>>();
    auto hull_faces = std::make_shared<std::vector<coal::Triangle>>();
    if (sv(2) <= tol) {
        const Eigen::Matrix<double, 3, 2> basis2 = svd.matrixV().leftCols<2>();
        detail::compute_2d_convex_hull(unique_pts, centroid, basis2, *hull_pts, *hull_faces);
    } else {
        detail::compute_3d_convex_hull(unique_pts, sv(0), *hull_pts, *hull_faces);
    }

    std::vector<coal::Vec3s> coal_verts(vertices.begin(), vertices.end());
    auto bvh = std::make_shared<coal::BVHModel<coal::OBBRSS>>();
    if (bvh->beginModel() != 0 || bvh->addSubModel(coal_verts, triangles) != 0 || bvh->endModel() != 0) {
        throw std::invalid_argument("failed to build triangle BVH on " + label);
    }

    auto sh = std::make_shared<CollisionShape>();
    sh->label = label;
    sh->kind = 5;
    sh->original = bvh;
    sh->hull_points.assign(hull_pts->begin(), hull_pts->end());
    sh->convex = std::make_shared<coal::Convex<coal::Triangle>>(
        hull_pts, hull_pts->size(), hull_faces, hull_faces->size());
    sh->finalize_bounds();
    return sh;
}

}  // namespace gobot::planning
