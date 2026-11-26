#pragma once

#include "planning_types.hpp"

#include <Eigen/Eigenvalues>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace gobot::planning {

struct LinkSphere {
    int body_id = -1;
    int shape_id = -1;       // source WorldShape index (-1 if user-provided custom sphere)
    int ref_shape_id = -1;   // any WorldShape index on same body_id for fast frame reuse (-1 if none)
    std::string link_key;    // canonical "robot::link" or "::attached_obj"
    std::string shape_name;  // diagnostic display label
    Vec3 local_center = Vec3::Zero();       // center in body_id frame
    Vec3 ref_shape_center = Vec3::Zero();   // center in ref_shape_id frame (valid when ref_shape_id >= 0)
    double radius = 0.0;
    double body_reach_radius = 0.0;         // ||local_center||_2 for exact kinematic-chain Lipschitz bound
    int layer = 1;
    int mask = 1;
};

struct CustomLinkSpheres {
    std::vector<std::pair<Vec3, double>> spheres;
    int layer = 1;
    int mask = 1;
};

namespace esdf_detail {

constexpr int64_t kMaxVoxelCount = 100'000'000LL;  // ~400 MB float safety cap

inline int64_t checked_voxel_count(int nx, int ny, int nz) {
    int64_t count = 1;
    for (int n : {nx, ny, nz}) {
        if (n < 1) throw std::invalid_argument("ESDF grid dimensions must be >= 1");
        if (n > kMaxVoxelCount / count)
            throw std::invalid_argument("ESDF grid voxel count exceeds safety limit (100M voxels)");
        count *= n;
    }
    return count;
}

// Exact O(n) 1D squared Euclidean distance transform (Felzenszwalb & Huttenlocher, 2012).
// Computes out[q] = min_{0 <= p < n} (f[p] + (q - p)^2).
// All-INF input lines produce all-INF output lines cleanly without overflow.
inline void dt1d_squared(const double* f, int n, double* out, int* v, double* z) {
    constexpr double kInf = 1e30;
    int k = -1;
    for (int q = 0; q < n; ++q) {
        if (f[q] >= kInf * 0.5) continue;
        if (k < 0) {
            k = 0;
            v[0] = q;
            z[0] = -std::numeric_limits<double>::infinity();
            z[1] = std::numeric_limits<double>::infinity();
            continue;
        }
        double s = 0.0;
        while (k >= 0) {
            const int vk = v[k];
            const double num = (f[q] + static_cast<double>(q) * static_cast<double>(q)) -
                               (f[vk] + static_cast<double>(vk) * static_cast<double>(vk));
            const double den = 2.0 * static_cast<double>(q - vk);
            s = num / den;
            if (s <= z[k]) {
                --k;
            } else {
                break;
            }
        }
        ++k;
        v[k] = q;
        z[k] = (k == 0) ? -std::numeric_limits<double>::infinity() : s;
        z[k + 1] = std::numeric_limits<double>::infinity();
    }
    if (k < 0) {
        std::fill(out, out + n, kInf);
        return;
    }
    int idx = 0;
    for (int q = 0; q < n; ++q) {
        const double qd = static_cast<double>(q);
        while (z[idx + 1] < qd) ++idx;
        const double diff = qd - static_cast<double>(v[idx]);
        out[q] = diff * diff + f[v[idx]];
    }
}

// Separable 3-pass exact 3D squared Euclidean distance transform in voxel units.
// Input `data` [nx * ny * nz] (C-order) contains 0.0 for seeds and kInf for non-seeds.
// Output overwrites `data` in place with exact squared voxel distance to nearest seed.
inline void edt3d_squared_inplace(std::vector<double>& data, int nx, int ny, int nz) {
    const int64_t nyz = static_cast<int64_t>(ny) * static_cast<int64_t>(nz);

    // Pass 1: along Z (contiguous in memory).
    if (nz > 1) {
        ThreadPool::instance().parallel_for(static_cast<int64_t>(nx) * static_cast<int64_t>(ny), 64,
            [&](int64_t b, int64_t e) {
                std::vector<double> f(static_cast<size_t>(nz));
                std::vector<double> d(static_cast<size_t>(nz));
                std::vector<int> v(static_cast<size_t>(nz));
                std::vector<double> z(static_cast<size_t>(nz + 1));
                for (int64_t xy = b; xy < e; ++xy) {
                    double* ptr = data.data() + xy * nz;
                    std::copy(ptr, ptr + nz, f.data());
                    dt1d_squared(f.data(), nz, d.data(), v.data(), z.data());
                    std::copy(d.data(), d.data() + nz, ptr);
                }
            });
    }

    // Pass 2: along Y.
    if (ny > 1) {
        ThreadPool::instance().parallel_for(static_cast<int64_t>(nx), 1,
            [&](int64_t xb, int64_t xe) {
                std::vector<double> f(static_cast<size_t>(ny));
                std::vector<double> d(static_cast<size_t>(ny));
                std::vector<int> v(static_cast<size_t>(ny));
                std::vector<double> z(static_cast<size_t>(ny + 1));
                for (int64_t x = xb; x < xe; ++x) {
                    const int64_t xoff = x * nyz;
                    for (int k = 0; k < nz; ++k) {
                        for (int y = 0; y < ny; ++y) {
                            f[static_cast<size_t>(y)] = data[static_cast<size_t>(xoff + static_cast<int64_t>(y) * nz + k)];
                        }
                        dt1d_squared(f.data(), ny, d.data(), v.data(), z.data());
                        for (int y = 0; y < ny; ++y) {
                            data[static_cast<size_t>(xoff + static_cast<int64_t>(y) * nz + k)] = d[static_cast<size_t>(y)];
                        }
                    }
                }
            });
    }

    // Pass 3: along X.
    if (nx > 1) {
        ThreadPool::instance().parallel_for(nyz, 64,
            [&](int64_t yzb, int64_t yze) {
                std::vector<double> f(static_cast<size_t>(nx));
                std::vector<double> d(static_cast<size_t>(nx));
                std::vector<int> v(static_cast<size_t>(nx));
                std::vector<double> z(static_cast<size_t>(nx + 1));
                for (int64_t yz = yzb; yz < yze; ++yz) {
                    for (int x = 0; x < nx; ++x) {
                        f[static_cast<size_t>(x)] = data[static_cast<size_t>(static_cast<int64_t>(x) * nyz + yz)];
                    }
                    dt1d_squared(f.data(), nx, d.data(), v.data(), z.data());
                    for (int x = 0; x < nx; ++x) {
                        data[static_cast<size_t>(static_cast<int64_t>(x) * nyz + yz)] = d[static_cast<size_t>(x)];
                    }
                }
            });
    }
}

// Subdivide a local box with half-extents `half` into conservative enclosing spheres.
inline void append_box_spheres(const Vec3& center_offset, const Mat3& frame_rot,
                               const Vec3& half, int max_spheres,
                               std::vector<std::pair<Vec3, double>>& out) {
    const double hx = std::max(0.0, half(0));
    const double hy = std::max(0.0, half(1));
    const double hz = std::max(0.0, half(2));
    const double max_h = std::max({hx, hy, hz});
    if (max_h <= 1e-9) {
        out.emplace_back(center_offset, 1e-4);
        return;
    }
    // Determine balanced subdivisions so sub-cells are roughly cubic while total <= max_spheres.
    const double pos_min = std::min({hx > 1e-6 ? hx : max_h, hy > 1e-6 ? hy : max_h, hz > 1e-6 ? hz : max_h});
    double cell_half = std::max(pos_min, 0.015);
    std::array<int, 3> n{1, 1, 1};
    auto compute_n = [&](double ch) {
        for (int axis = 0; axis < 3; ++axis) {
            n[axis] = static_cast<int>(std::clamp(std::round(half(axis) / ch), 1.0,
                                                  static_cast<double>(max_spheres) + 1.0));
        }
    };
    compute_n(cell_half);
    while (n[0] * n[1] * n[2] > max_spheres) {
        cell_half *= 1.15;
        compute_n(cell_half);
        if (cell_half > max_h * 2.0) {
            n = {1, 1, 1};
            break;
        }
    }
    const double sx = hx / static_cast<double>(n[0]);
    const double sy = hy / static_cast<double>(n[1]);
    const double sz = hz / static_cast<double>(n[2]);
    const double r = std::sqrt(sx * sx + sy * sy + sz * sz);
    for (int ix = 0; ix < n[0]; ++ix) {
        const double cx = (n[0] == 1) ? 0.0 : (-hx + (2.0 * static_cast<double>(ix) + 1.0) * sx);
        for (int iy = 0; iy < n[1]; ++iy) {
            const double cy = (n[1] == 1) ? 0.0 : (-hy + (2.0 * static_cast<double>(iy) + 1.0) * sy);
            for (int iz = 0; iz < n[2]; ++iz) {
                const double cz = (n[2] == 1) ? 0.0 : (-hz + (2.0 * static_cast<double>(iz) + 1.0) * sz);
                out.emplace_back(center_offset + frame_rot * Vec3(cx, cy, cz), r);
            }
        }
    }
}

// Generate conservative enclosing spheres in shape-local coordinates for any supported CollisionShape.
inline std::vector<std::pair<Vec3, double>> generate_shape_sphere_cover(
    const CollisionShape& sh, int max_spheres = 12) {
    std::vector<std::pair<Vec3, double>> out;
    if (sh.kind == 6) return out;  // infinite halfspace is never a mobile link shape

    if (sh.kind == 2) {
        // Sphere: exact single sphere.
        out.emplace_back(Vec3::Zero(), std::max(0.0, sh.radius));
        return out;
    }
    if (sh.kind == 4) {
        // Capsule: segment along local Z in [-height/2, +height/2] with radius R.
        const double R = std::max(0.0, sh.radius);
        const double H = std::max(0.0, sh.height);
        if (H == 0.0) {
            out.emplace_back(Vec3::Zero(), R);
            return out;
        }
        const double bloat_tol = std::max(0.005, 0.1 * R);
        const double max_delta = std::sqrt(std::max(1e-8, (R + bloat_tol) * (R + bloat_tol) - R * R));
        const int N = static_cast<int>(std::clamp(std::ceil(H / (2.0 * max_delta)) + 1.0, 2.0, static_cast<double>(std::max(2, max_spheres))));
        const double half_step = H / (2.0 * static_cast<double>(N - 1));
        const double r = std::sqrt(R * R + half_step * half_step);
        for (int k = 0; k < N; ++k) {
            const double frac = static_cast<double>(k) / static_cast<double>(N - 1);
            const double z = -0.5 * H + frac * H;
            out.emplace_back(Vec3(0.0, 0.0, z), r);
        }
        return out;
    }
    if (sh.kind == 3) {
        // Cylinder: flat caps at z = +-height/2 with radius R.
        const double R = std::max(0.0, sh.radius);
        const double H = std::max(0.0, sh.height);
        if (H == 0.0) {
            out.emplace_back(Vec3::Zero(), R);
            return out;
        }
        const double target_half_slice = std::max(0.015, 0.5 * R);
        const int N = static_cast<int>(std::clamp(std::ceil(H / (2.0 * target_half_slice)), 1.0, static_cast<double>(std::max(1, max_spheres))));
        const double half_slice = H / (2.0 * static_cast<double>(N));
        const double r = std::sqrt(R * R + half_slice * half_slice);
        for (int k = 0; k < N; ++k) {
            const double z = -0.5 * H + (2.0 * static_cast<double>(k) + 1.0) * half_slice;
            out.emplace_back(Vec3(0.0, 0.0, z), r);
        }
        return out;
    }
    if (sh.kind == 1) {
        // Box: balanced conservative sub-box cover.
        append_box_spheres(Vec3::Zero(), Mat3::Identity(), sh.box_half_size, std::max(1, max_spheres), out);
        return out;
    }
    if (sh.kind == 5) {
        // Mesh: PCA-aligned OBB on convex hull vertices, subdivided into conservative enclosing spheres.
        if (sh.hull_points.size() >= 3) {
            Vec3 mean = Vec3::Zero();
            for (const auto& p : sh.hull_points) mean += p;
            mean /= static_cast<double>(sh.hull_points.size());

            Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
            for (const auto& p : sh.hull_points) {
                const Vec3 d = p - mean;
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
            for (const auto& p : sh.hull_points) {
                const Vec3 lp = basis.transpose() * (p - mean);
                lo = lo.cwiseMin(lp);
                hi = hi.cwiseMax(lp);
            }
            const Vec3 obb_local_c = 0.5 * (lo + hi);
            const Vec3 obb_half = 0.5 * (hi - lo);
            const Vec3 obb_center = mean + basis * obb_local_c;
            append_box_spheres(obb_center, basis, obb_half, std::max(1, max_spheres), out);
            // If only 1 sphere was emitted, tighten its radius to the exact hull bounding sphere around obb_center.
            if (out.size() == 1) {
                double max_r2 = 0.0;
                for (const auto& p : sh.hull_points) {
                    max_r2 = std::max(max_r2, (p - obb_center).squaredNorm());
                }
                out[0].second = std::min(out[0].second, std::sqrt(max_r2));
            }
            return out;
        }
        // Fallback to local convex AABB if hull_points are unavailable.
        append_box_spheres(sh.convex_center, Mat3::Identity(), sh.convex_half, std::max(1, max_spheres), out);
        return out;
    }
    return out;
}

}  // namespace esdf_detail

// Immutable 3D Euclidean Signed Distance Field on a uniform axis-aligned voxel grid.
// Voxel (ix, iy, iz) spans [origin + (ix, iy, iz) * resolution, origin + (ix+1, iy+1, iy+1) * resolution]
// with voxel center at origin + (ix + 0.5, iy + 0.5, iz + 0.5) * resolution.
// Positive distances lie in free space; negative distances lie inside obstacles.
struct VoxelESDF {
    Vec3 origin = Vec3::Zero();
    double resolution = 0.05;
    double inv_resolution = 20.0;
    int nx = 0;
    int ny = 0;
    int nz = 0;
    int collision_layer = 1;
    int collision_mask = 1;
    bool is_empty = true;          // true when built from empty occupancy/point cloud (distance = +inf)
    double lipschitz_bound = 1.0;  // certified global Lipschitz constant |f(p1)-f(p0)| <= L * ||p1-p0||
    bool from_occupancy = false;
    double inflation_radius = 0.0;
    double max_distance = std::numeric_limits<double>::infinity();
    double min_value = std::numeric_limits<double>::infinity();
    std::vector<float> sdf;        // C-order [ix * ny * nz + iy * nz + iz]

    bool valid() const {
        return resolution > 0.0 && nx > 0 && ny > 0 && nz > 0 &&
               sdf.size() == static_cast<size_t>(static_cast<int64_t>(nx) * ny * nz);
    }

    Vec3 bounds_min() const { return origin; }
    Vec3 bounds_max() const {
        return origin + Vec3(static_cast<double>(nx), static_cast<double>(ny), static_cast<double>(nz)) * resolution;
    }

    inline double at_grid(int ix, int iy, int iz) const {
        const size_t idx = static_cast<size_t>((static_cast<int64_t>(ix) * ny + iy) * nz + iz);
        return static_cast<double>(sdf[idx]);
    }

    // Bound the gradient norm of the trilinear interpolant and its exterior extension.
    void finalize_lipschitz() {
        inv_resolution = 1.0 / resolution;
        if (is_empty || !valid()) {
            lipschitz_bound = 1.0;
            min_value = std::numeric_limits<double>::infinity();
            return;
        }
        double min_v = std::numeric_limits<double>::infinity();
        bool boundary_negative = false;
        const int64_t nyz = static_cast<int64_t>(ny) * static_cast<int64_t>(nz);
        for (int ix = 0; ix < nx; ++ix) {
            const bool bx = (ix == 0 || ix == nx - 1);
            for (int iy = 0; iy < ny; ++iy) {
                const bool by = bx || (iy == 0 || iy == ny - 1);
                const float* row = sdf.data() + static_cast<int64_t>(ix) * nyz + static_cast<int64_t>(iy) * nz;
                for (int iz = 0; iz < nz; ++iz) {
                    const double v = static_cast<double>(row[iz]);
                    if (v < min_v) min_v = v;
                    if ((by || iz == 0 || iz == nz - 1) && v < 0.0) {
                        boundary_negative = true;
                    }
                }
            }
        }
        min_value = std::clamp(min_v - inflation_radius, -max_distance, max_distance);

        // Per-cell certified max gradient norm upper bound for trilinear interpolation.
        double max_gradient = 0.0;
        const int cx_end = std::max(1, nx - 1);
        const int cy_end = std::max(1, ny - 1);
        const int cz_end = std::max(1, nz - 1);
        for (int ix = 0; ix < cx_end; ++ix) {
            const int ix1 = std::min(ix + 1, nx - 1);
            for (int iy = 0; iy < cy_end; ++iy) {
                const int iy1 = std::min(iy + 1, ny - 1);
                for (int iz = 0; iz < cz_end; ++iz) {
                    const int iz1 = std::min(iz + 1, nz - 1);
                    const double c000 = at_grid(ix, iy, iz);
                    const double c001 = at_grid(ix, iy, iz1);
                    const double c010 = at_grid(ix, iy1, iz);
                    const double c011 = at_grid(ix, iy1, iz1);
                    const double c100 = at_grid(ix1, iy, iz);
                    const double c101 = at_grid(ix1, iy, iz1);
                    const double c110 = at_grid(ix1, iy1, iz);
                    const double c111 = at_grid(ix1, iy1, iz1);

                    double gx = 0.0, gy = 0.0, gz = 0.0;
                    if (nx > 1) {
                        gx = std::max({std::abs(c100 - c000), std::abs(c101 - c001),
                                       std::abs(c110 - c010), std::abs(c111 - c011)}) * inv_resolution;
                    }
                    if (ny > 1) {
                        gy = std::max({std::abs(c010 - c000), std::abs(c011 - c001),
                                       std::abs(c110 - c100), std::abs(c111 - c101)}) * inv_resolution;
                    }
                    if (nz > 1) {
                        gz = std::max({std::abs(c001 - c000), std::abs(c011 - c010),
                                       std::abs(c101 - c100), std::abs(c111 - c110)}) * inv_resolution;
                    }
                    const double norm = std::hypot(gx, gy, gz);
                    if (!std::isfinite(norm)) throw std::invalid_argument("ESDF gradient exceeds numeric range");
                    max_gradient = std::max(max_gradient, norm);
                }
            }
        }
        // Exterior extension:
        // - When boundary_val >= 0: f(p) = sqrt(s^2 + d_out^2) has ||grad f|| <= max(1, L_grid).
        // - When boundary_val < 0: f(p) = s + d_out has ||grad f|| <= sqrt(1 + L_grid^2).
        if (boundary_negative) {
            lipschitz_bound = std::hypot(1.0, max_gradient);
        } else {
            lipschitz_bound = std::max(1.0, max_gradient);
        }
        if (!std::isfinite(lipschitz_bound))
            throw std::invalid_argument("ESDF gradient exceeds numeric range");
    }

    // Apply inflation and truncation after interpolation and exterior extension.
    double query(const Vec3& p, Vec3* grad = nullptr, bool fallback_zero_grad = false) const {
        const double value = interpolate(p, grad, fallback_zero_grad);
        return is_empty ? value : truncate(value - inflation_radius, grad);
    }

    // Lower bound on free-space clearance to the occupied cubes or the supplied field's zero set.
    double collision_distance(const Vec3& p, Vec3* grad = nullptr, bool fallback_zero_grad = false) const {
        const double value = interpolate(p, grad, fallback_zero_grad);
        if (!from_occupancy) {
            // A sampled field can have a slope greater than one between centers.
            if (grad) *grad /= lipschitz_bound;
            return value / lipschitz_bound;
        }
        if (is_empty) return value;
        // For occupied center o and interpolation corner V with trilinear weights,
        // E|V-o| <= sqrt(|p-o|^2 + 3h^2/4). The orthogonal extension below preserves this bound.
        // For distance D to the cube, |p-o| <= D+sqrt(3)h/2. Hence
        // value+h/2 <= sqrt((D+sqrt(3)h/2)^2 + 3h^2/4) <= D+sqrt(3/2)h.
        const double voxel_allowance = (std::sqrt(1.5) - 0.5) * resolution;
        return truncate(value - voxel_allowance - inflation_radius, grad);
    }

private:
    double truncate(double value, Vec3* grad) const {
        if (grad && std::abs(value) > max_distance) grad->setZero();
        return std::clamp(value, -max_distance, max_distance);
    }

    double interpolate(const Vec3& p, Vec3* out_grad, bool fallback_zero_grad) const {
        if (!p.allFinite()) throw std::invalid_argument("ESDF query point must be finite");
        if (is_empty || !valid()) {
            if (out_grad) *out_grad = Vec3::Zero();
            return std::numeric_limits<double>::infinity();
        }
        const double gx = (p(0) - origin(0)) * inv_resolution - 0.5;
        const double gy = (p(1) - origin(1)) * inv_resolution - 0.5;
        const double gz = (p(2) - origin(2)) * inv_resolution - 0.5;

        const double max_gx = static_cast<double>(nx - 1);
        const double max_gy = static_cast<double>(ny - 1);
        const double max_gz = static_cast<double>(nz - 1);

        const double cgx = std::clamp(gx, 0.0, max_gx);
        const double cgy = std::clamp(gy, 0.0, max_gy);
        const double cgz = std::clamp(gz, 0.0, max_gz);

        const bool out_x = (gx != cgx);
        const bool out_y = (gy != cgy);
        const bool out_z = (gz != cgz);

        int ix0 = 0, ix1 = 0;
        double tx = 0.0;
        if (nx > 1) {
            ix0 = std::min(static_cast<int>(std::floor(cgx)), nx - 2);
            ix1 = ix0 + 1;
            tx = cgx - static_cast<double>(ix0);
        }
        int iy0 = 0, iy1 = 0;
        double ty = 0.0;
        if (ny > 1) {
            iy0 = std::min(static_cast<int>(std::floor(cgy)), ny - 2);
            iy1 = iy0 + 1;
            ty = cgy - static_cast<double>(iy0);
        }
        int iz0 = 0, iz1 = 0;
        double tz = 0.0;
        if (nz > 1) {
            iz0 = std::min(static_cast<int>(std::floor(cgz)), nz - 2);
            iz1 = iz0 + 1;
            tz = cgz - static_cast<double>(iz0);
        }

        const double c000 = at_grid(ix0, iy0, iz0);
        const double c001 = at_grid(ix0, iy0, iz1);
        const double c010 = at_grid(ix0, iy1, iz0);
        const double c011 = at_grid(ix0, iy1, iz1);
        const double c100 = at_grid(ix1, iy0, iz0);
        const double c101 = at_grid(ix1, iy0, iz1);
        const double c110 = at_grid(ix1, iy1, iz0);
        const double c111 = at_grid(ix1, iy1, iz1);

        const double wx0 = 1.0 - tx, wx1 = tx;
        const double wy0 = 1.0 - ty, wy1 = ty;
        const double wz0 = 1.0 - tz, wz1 = tz;

        const double c00 = c000 * wz0 + c001 * wz1;
        const double c01 = c010 * wz0 + c011 * wz1;
        const double c10 = c100 * wz0 + c101 * wz1;
        const double c11 = c110 * wz0 + c111 * wz1;

        const double c0 = c00 * wy0 + c01 * wy1;
        const double c1 = c10 * wy0 + c11 * wy1;
        const double s = c0 * wx0 + c1 * wx1;

        Vec3 g_in = Vec3::Zero();
        if (out_grad != nullptr) {
            if (nx > 1) g_in(0) = (c1 - c0) * inv_resolution;
            if (ny > 1) g_in(1) = ((c01 - c00) * wx0 + (c11 - c10) * wx1) * inv_resolution;
            if (nz > 1) {
                const double dz00 = c001 - c000;
                const double dz01 = c011 - c010;
                const double dz10 = c101 - c100;
                const double dz11 = c111 - c110;
                g_in(2) = ((dz00 * wy0 + dz01 * wy1) * wx0 + (dz10 * wy0 + dz11 * wy1) * wx1) * inv_resolution;
            }
            if (fallback_zero_grad && !(out_x || out_y || out_z) && g_in.squaredNorm() < 1e-12) {
                // Symmetric saddle/medial-axis fallback via broader grid central difference so SQP never stalls.
                const int cx = std::clamp(static_cast<int>(std::round(cgx)), 0, nx - 1);
                const int cy = std::clamp(static_cast<int>(std::round(cgy)), 0, ny - 1);
                const int cz = std::clamp(static_cast<int>(std::round(cgz)), 0, nz - 1);
                for (int step = 1; step <= 3 && g_in.squaredNorm() < 1e-12; ++step) {
                    const int xm = std::max(0, cx - step), xp = std::min(nx - 1, cx + step);
                    const int ym = std::max(0, cy - step), yp = std::min(ny - 1, cy + step);
                    const int zm = std::max(0, cz - step), zp = std::min(nz - 1, cz + step);
                    if (xp > xm) g_in(0) = (at_grid(xp, cy, cz) - at_grid(xm, cy, cz)) / (static_cast<double>(xp - xm) * resolution);
                    if (yp > ym) g_in(1) = (at_grid(cx, yp, cz) - at_grid(cx, ym, cz)) / (static_cast<double>(yp - ym) * resolution);
                    if (zp > zm) g_in(2) = (at_grid(cx, cy, zp) - at_grid(cx, cy, zm)) / (static_cast<double>(zp - zm) * resolution);
                }
            }
        }

        if (!out_x && !out_y && !out_z) {
            if (out_grad) *out_grad = g_in;
            return s;
        }

        const Vec3 delta((gx - cgx) * resolution, (gy - cgy) * resolution, (gz - cgz) * resolution);
        const double d_out = delta.stableNorm();
        // Occupancy samples bound distance to voxel centers after adding h/2.
        // All occupied centers lie inside the clamping box.
        const double offset = from_occupancy ? 0.5 * resolution : 0.0;
        const double edge = s + offset;
        if (edge >= 0.0) {
            const double dist = std::hypot(edge, d_out);
            if (out_grad) {
                if (dist > 1e-12) {
                    const double inv_d = 1.0 / dist;
                    (*out_grad)(0) = out_x ? (delta(0) * inv_d) : (edge * inv_d * g_in(0));
                    (*out_grad)(1) = out_y ? (delta(1) * inv_d) : (edge * inv_d * g_in(1));
                    (*out_grad)(2) = out_z ? (delta(2) * inv_d) : (edge * inv_d * g_in(2));
                } else {
                    *out_grad = Vec3::Zero();
                }
            }
            return dist - offset;
        } else {
            const double dist = s + d_out;
            if (out_grad) {
                const double inv_do = (d_out > 1e-12) ? (1.0 / d_out) : 0.0;
                (*out_grad)(0) = out_x ? (delta(0) * inv_do) : g_in(0);
                (*out_grad)(1) = out_y ? (delta(1) * inv_do) : g_in(1);
                (*out_grad)(2) = out_z ? (delta(2) * inv_do) : g_in(2);
            }
            return dist;
        }
    }

public:
    // Build a signed 3D ESDF from a binary occupancy grid [nx, ny, nz] (non-zero = occupied).
    // Free voxel center:     sdf =  d_out - 0.5 * resolution - inflation_radius
    // Occupied voxel center: sdf = -(d_in - 0.5 * resolution) - inflation_radius
    // Treats the exterior of the grid as free space so boundary occupied voxels have exact face depth -0.5*h.
    static std::shared_ptr<VoxelESDF> from_occupancy_grid(
        const uint8_t* occupied, int nx, int ny, int nz,
        const Vec3& origin, double resolution,
        double max_distance = std::numeric_limits<double>::infinity(),
        double inflation_radius = 0.0,
        int collision_layer = 1, int collision_mask = 1) {
        if (!origin.allFinite()) {
            throw std::invalid_argument("ESDF origin must be finite");
        }
        if (!std::isfinite(resolution) || resolution <= 0.0 || !std::isfinite(1.0 / resolution)) {
            throw std::invalid_argument("ESDF resolution must be finite and positive");
        }
        const int64_t total = esdf_detail::checked_voxel_count(nx, ny, nz);
        if (std::isnan(max_distance) || max_distance <= 0.0) {
            throw std::invalid_argument("max_distance must be positive (or inf)");
        }
        if (!std::isfinite(inflation_radius) || inflation_radius < 0.0) {
            throw std::invalid_argument("inflation_radius must be finite and non-negative");
        }

        auto grid = std::make_shared<VoxelESDF>();
        grid->origin = origin;
        grid->resolution = resolution;
        grid->inv_resolution = 1.0 / resolution;
        grid->nx = nx;
        grid->ny = ny;
        grid->nz = nz;
        grid->collision_layer = collision_layer;
        grid->collision_mask = collision_mask;
        grid->sdf.resize(static_cast<size_t>(total));

        grid->from_occupancy = true;
        grid->inflation_radius = inflation_radius;
        grid->max_distance = max_distance;
        constexpr double kInf = 1e30;
        std::vector<double> d_out(static_cast<size_t>(total), kInf);
        std::vector<double> d_in(static_cast<size_t>(total), kInf);
        int64_t occ_count = 0;
        for (int64_t i = 0; i < total; ++i) {
            if (occupied[i] != 0) {
                d_out[static_cast<size_t>(i)] = 0.0;
                ++occ_count;
            } else {
                d_in[static_cast<size_t>(i)] = 0.0;
            }
        }

        if (occ_count == 0) {
            grid->is_empty = true;
            std::fill(grid->sdf.begin(), grid->sdf.end(), std::numeric_limits<float>::infinity());
            grid->finalize_lipschitz();
            return grid;
        }

        grid->is_empty = false;
        esdf_detail::edt3d_squared_inplace(d_out, nx, ny, nz);
        if (occ_count < total) {
            esdf_detail::edt3d_squared_inplace(d_in, nx, ny, nz);
        }

        const int64_t nyz = static_cast<int64_t>(ny) * static_cast<int64_t>(nz);
        ThreadPool::instance().parallel_for(static_cast<int64_t>(nx), 1, [&](int64_t xb, int64_t xe) {
            for (int64_t ix = xb; ix < xe; ++ix) {
                const int64_t dx_ext = std::min<int64_t>(ix + 1, static_cast<int64_t>(nx) - ix);
                const double dx_ext2 = static_cast<double>(dx_ext * dx_ext);
                for (int iy = 0; iy < ny; ++iy) {
                    const int64_t dy_ext = std::min<int64_t>(iy + 1, ny - iy);
                    const double dxy_ext2 = std::min(dx_ext2, static_cast<double>(dy_ext * dy_ext));
                    const int64_t row_off = ix * nyz + static_cast<int64_t>(iy) * nz;
                    for (int iz = 0; iz < nz; ++iz) {
                        const size_t idx = static_cast<size_t>(row_off + iz);
                        double val = 0.0;
                        if (occupied[idx] != 0) {
                            const int64_t dz_ext = std::min<int64_t>(iz + 1, nz - iz);
                            const double ext2 = std::min(dxy_ext2, static_cast<double>(dz_ext * dz_ext));
                            const double min_in2 = std::min(d_in[idx], ext2);
                            val = (0.5 - std::sqrt(min_in2)) * resolution;
                        } else {
                            val = (std::sqrt(d_out[idx]) - 0.5) * resolution;
                        }
                        if (std::abs(val) > std::numeric_limits<float>::max())
                            throw std::invalid_argument("occupancy distance exceeds ESDF float range");
                        const float sample = static_cast<float>(val);
                        // Round down to preserve the clearance bound in float storage.
                        grid->sdf[idx] = sample > val ? std::nextafter(sample, -std::numeric_limits<float>::infinity()) : sample;
                    }
                }
            }
        });
        grid->finalize_lipschitz();
        return grid;
    }

    // Voxelize a 3D point cloud and build its center-sampled distance field [num_points, 3].
    static std::shared_ptr<VoxelESDF> from_point_cloud(
        const double* points_xyz, int64_t num_points, double resolution,
        const std::optional<Vec3>& origin_opt = std::nullopt,
        const std::optional<std::array<int, 3>>& shape_opt = std::nullopt,
        double padding = 0.2,
        double point_radius = 0.0,
        double max_distance = std::numeric_limits<double>::infinity(),
        int collision_layer = 1, int collision_mask = 1) {
        if (num_points < 0) {
            throw std::invalid_argument("num_points must be non-negative");
        }
        if (!std::isfinite(resolution) || resolution <= 0.0 || !std::isfinite(1.0 / resolution)) {
            throw std::invalid_argument("ESDF resolution must be finite and positive");
        }
        if (!std::isfinite(padding) || padding < 0.0) {
            throw std::invalid_argument("padding must be finite and non-negative");
        }
        if (!std::isfinite(point_radius) || point_radius < 0.0) {
            throw std::invalid_argument("point_radius must be finite and non-negative");
        }
        if (origin_opt.has_value() && !origin_opt->allFinite()) {
            throw std::invalid_argument("ESDF origin must be finite");
        }
        if (shape_opt.has_value()) {
            for (int a = 0; a < 3; ++a) {
                if ((*shape_opt)[static_cast<size_t>(a)] < 1) {
                    throw std::invalid_argument("ESDF grid shape dimensions must be >= 1");
                }
            }
        }
        if (num_points == 0) {
            const Vec3 org = origin_opt.value_or(Vec3::Zero());
            const auto shp = shape_opt.value_or(std::array<int, 3>{1, 1, 1});
            std::vector<uint8_t> empty_occ(
                static_cast<size_t>(esdf_detail::checked_voxel_count(shp[0], shp[1], shp[2])), 0);
            return from_occupancy_grid(empty_occ.data(), shp[0], shp[1], shp[2], org, resolution,
                                       max_distance, point_radius, collision_layer, collision_mask);
        }

        Vec3 pmin = Vec3::Constant(std::numeric_limits<double>::infinity());
        Vec3 pmax = Vec3::Constant(-std::numeric_limits<double>::infinity());
        for (int64_t i = 0; i < num_points; ++i) {
            const double x = points_xyz[i * 3 + 0];
            const double y = points_xyz[i * 3 + 1];
            const double z = points_xyz[i * 3 + 2];
            if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
                throw std::invalid_argument("point cloud contains non-finite coordinates");
            }
            if (x < pmin(0)) pmin(0) = x;
            if (y < pmin(1)) pmin(1) = y;
            if (z < pmin(2)) pmin(2) = z;
            if (x > pmax(0)) pmax(0) = x;
            if (y > pmax(1)) pmax(1) = y;
            if (z > pmax(2)) pmax(2) = z;
        }

        const double eff_pad = std::max(padding, 2.0 * resolution);
        const double pad_cells = std::ceil(eff_pad / resolution);
        std::array<int, 3> shp;
        Vec3 origin;
        if (origin_opt.has_value() && shape_opt.has_value()) {
            origin = *origin_opt;
            shp = *shape_opt;
        } else if (shape_opt.has_value()) {
            shp = *shape_opt;
            const Vec3 mid = 0.5 * pmin + 0.5 * pmax;
            origin = mid - 0.5 * Vec3(static_cast<double>(shp[0]), static_cast<double>(shp[1]),
                                      static_cast<double>(shp[2])) * resolution;
        } else {
            origin = origin_opt.value_or(
                pmin - Vec3::Constant((pad_cells + 0.5) * resolution));
            const Vec3 hi = pmax + Vec3::Constant((pad_cells + 0.5) * resolution);
            for (int a = 0; a < 3; ++a) {
                const double span = hi(a) - origin(a);
                const double cells = std::max(std::ceil(span / resolution), 1.0);
                if (!(cells <= esdf_detail::kMaxVoxelCount))
                    throw std::invalid_argument("point cloud ESDF grid exceeds safety limit");
                shp[static_cast<size_t>(a)] = static_cast<int>(cells);
            }
        }
        const int nx = shp[0], ny = shp[1], nz = shp[2];
        const int64_t total = esdf_detail::checked_voxel_count(nx, ny, nz);

        std::vector<uint8_t> occ(static_cast<size_t>(total), 0);
        const double inv_res = 1.0 / resolution;
        const int64_t nyz = static_cast<int64_t>(ny) * static_cast<int64_t>(nz);
        for (int64_t i = 0; i < num_points; ++i) {
            const Vec3 cell = (Eigen::Map<const Vec3>(points_xyz + i * 3) - origin) * inv_res;
            if (cell.minCoeff() >= 0.0 && cell.x() < nx && cell.y() < ny && cell.z() < nz) {
                const int ix = static_cast<int>(cell.x());
                const int iy = static_cast<int>(cell.y());
                const int iz = static_cast<int>(cell.z());
                occ[static_cast<size_t>(static_cast<int64_t>(ix) * nyz + static_cast<int64_t>(iy) * nz + iz)] = 1;
            }
        }
        return from_occupancy_grid(occ.data(), nx, ny, nz, origin, resolution,
                                   max_distance, point_radius, collision_layer, collision_mask);
    }

    // Build an immutable VoxelESDF directly from a precomputed signed distance array [nx, ny, nz].
    static std::shared_ptr<VoxelESDF> from_esdf_array(
        const double* values, int nx, int ny, int nz,
        const Vec3& origin, double resolution,
        int collision_layer = 1, int collision_mask = 1) {
        if (!origin.allFinite()) {
            throw std::invalid_argument("ESDF origin must be finite");
        }
        if (!std::isfinite(resolution) || resolution <= 0.0 || !std::isfinite(1.0 / resolution)) {
            throw std::invalid_argument("ESDF resolution must be finite and positive");
        }
        const int64_t total = esdf_detail::checked_voxel_count(nx, ny, nz);
        auto grid = std::make_shared<VoxelESDF>();
        grid->origin = origin;
        grid->resolution = resolution;
        grid->inv_resolution = 1.0 / resolution;
        grid->nx = nx;
        grid->ny = ny;
        grid->nz = nz;
        grid->collision_layer = collision_layer;
        grid->collision_mask = collision_mask;
        grid->is_empty = false;
        grid->sdf.resize(static_cast<size_t>(total));
        for (int64_t i = 0; i < total; ++i) {
            if (!(std::abs(values[i]) <= std::numeric_limits<float>::max())) {
                throw std::invalid_argument("ESDF values must be finite and representable as floats");
            }
            grid->sdf[static_cast<size_t>(i)] = static_cast<float>(values[i]);
        }
        grid->finalize_lipschitz();
        return grid;
    }
};

}  // namespace gobot::planning
