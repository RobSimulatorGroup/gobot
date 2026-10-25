#include "gobot/simulation/tactile_sensor_processor.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>
#include "gobot/core/math/mesh_validation.hpp"

namespace gobot {
namespace {
// Deterministic triangle rasterization at pixel centers. No graphics context required.
template <class F>
void Pixels(const Vector3 &a, const Vector3 &b, const Vector3 &c, std::uint32_t width, std::uint32_t height,
            F visit) {
    const double determinant = (b.y() - c.y()) * (a.x() - c.x()) + (c.x() - b.x()) * (a.y() - c.y());
    if (std::abs(determinant) < 1e-12)
        return;
    const int left = std::max(0, int(std::floor(std::min({a.x(), b.x(), c.x()}))));
    const int right = std::min(int(width) - 1, int(std::ceil(std::max({a.x(), b.x(), c.x()}))));
    const int top = std::max(0, int(std::floor(std::min({a.y(), b.y(), c.y()}))));
    const int bottom = std::min(int(height) - 1, int(std::ceil(std::max({a.y(), b.y(), c.y()}))));
    for (int y = top; y <= bottom; ++y)
        for (int x = left; x <= right; ++x) {
            const double u =
                ((b.y() - c.y()) * (x + 0.5 - c.x()) + (c.x() - b.x()) * (y + 0.5 - c.y())) / determinant;
            const double v =
                ((c.y() - a.y()) * (x + 0.5 - c.x()) + (a.x() - c.x()) * (y + 0.5 - c.y())) / determinant;
            if (u >= -1e-9 && v >= -1e-9 && u + v <= 1 + 1e-9)
                visit(std::uint32_t(y) * width + x, Vector3{RealType(u), RealType(v), RealType(1 - u - v)},
                      std::abs(determinant) * 0.5);
        }
}
std::uint8_t Byte(double v) { return std::uint8_t(std::lround(std::clamp(v, 0.0, 1.0) * 255)); }
} // namespace

TactileSensorProcessor::TactileSensorProcessor(TactileSensorParameters parameters,
                                               std::vector<Vector3> vertices,
                                               std::vector<std::uint32_t> tetrahedra,
                                               std::vector<std::uint32_t> surface,
                                               TactileOpticalData calibration)
    : parameters_(std::move(parameters)), calibration_(std::move(calibration)),
      rest_vertices_(std::move(vertices)), tetrahedra_(std::move(tetrahedra)) {
    std::string error;
    if (!ValidateTetrahedralMesh(rest_vertices_, tetrahedra_, surface, &error) ||
        !parameters_.Validate(rest_vertices_.size(), tetrahedra_.size() / 4, &error))
        throw std::invalid_argument(error);
    if (parameters_.image_width > 4096 || parameters_.image_height > 4096)
        throw std::invalid_argument("tactile image resolution exceeds 4096");
    if (parameters_.rgb_model != "gobot_deterministic_v1" && parameters_.rgb_model != "taxim_v1")
        throw std::invalid_argument("unsupported tactile optical model: " + parameters_.rgb_model);
    if (parameters_.rgb_model == "taxim_v1" &&
        (!calibration_.Validate(&error) || calibration_.width != parameters_.image_width ||
         calibration_.height != parameters_.image_height))
        throw std::invalid_argument("invalid TAXIM calibration: " + error);
    std::vector<bool> active(rest_vertices_.size(), parameters_.coat_vertex_indices.empty());
    for (auto i : parameters_.coat_vertex_indices)
        active[i] = true;
    for (std::size_t i = 0; i < surface.size(); i += 3)
        if (active[surface[i]] && active[surface[i + 1]] && active[surface[i + 2]])
            surface_.insert(surface_.end(), surface.begin() + i, surface.begin() + i + 3);
    if (surface_.empty())
        throw std::invalid_argument("tactile sensing surface is empty");
    coat_ = surface_;
    std::sort(coat_.begin(), coat_.end());
    coat_.erase(std::unique(coat_.begin(), coat_.end()), coat_.end());
    const std::size_t count = std::size_t(parameters_.image_width) * parameters_.image_height;
    reference_normals_ = SurfaceNormals(rest_vertices_);
    rest_depth_.assign(count, parameters_.far_plane);
    Rasterize(rest_vertices_, rest_depth_);
    pixel_areas_.assign(count, 0);
    vertex_weights_.assign(rest_vertices_.size(), 0);
    for (std::size_t i = 0; i < surface_.size(); i += 3) {
        const auto a = surface_[i], b = surface_[i + 1], c = surface_[i + 2];
        const double area =
            (rest_vertices_[b] - rest_vertices_[a]).cross(rest_vertices_[c] - rest_vertices_[a]).norm() * 0.5;
        Pixels(Project(rest_vertices_[a]), Project(rest_vertices_[b]), Project(rest_vertices_[c]),
               parameters_.image_width, parameters_.image_height,
               [&](auto pixel, const Vector3 &w, double projected_area) {
                   samples_.push_back({pixel, a, b, c, w});
                   pixel_areas_[pixel] += float(area / projected_area);
                   vertex_weights_[a] += w.x();
                   vertex_weights_[b] += w.y();
                   vertex_weights_[c] += w.z();
               });
    }
    for (auto index : coat_)
        if (vertex_weights_[index] <= 0)
            throw std::invalid_argument("tactile sensing surface lies outside the camera or is undersampled");
    for (std::size_t i = 0; i < parameters_.marker_tetrahedra.size(); ++i) {
        Vector3 p = Vector3::Zero();
        for (int j = 0; j < 4; ++j)
            p += rest_vertices_[tetrahedra_[parameters_.marker_tetrahedra[i] * 4 + j]] *
                 parameters_.marker_barycentric[i][j];
        rest_markers_.push_back(Project(p).head<2>());
    }
}

Vector3 TactileSensorProcessor::Project(const Vector3 &point) const {
    const Vector3 p = parameters_.camera_transform.inverse() * point;
    const auto &k = parameters_.camera_intrinsics;
    if (k.x() > 0) {
        if (p.z() <= 0)
            throw std::invalid_argument("tactile surface is behind the camera");
        return {k.x() * p.x() / p.z() + k.z(), -k.y() * p.y() / p.z() + k.w(), p.z()};
    }
    return {p.x() / parameters_.pixel_size + parameters_.image_width * RealType(.5),
            -p.y() / parameters_.pixel_size + parameters_.image_height * RealType(.5), p.z()};
}

std::vector<Vector3> TactileSensorProcessor::SurfaceNormals(std::span<const Vector3> vertices) const {
    std::vector<Vector3> normals(vertices.size(), Vector3::Zero());
    for (std::size_t i = 0; i < surface_.size(); i += 3) {
        const auto a = surface_[i], b = surface_[i + 1], c = surface_[i + 2];
        const Vector3 face = (vertices[b] - vertices[a]).cross(vertices[c] - vertices[a]);
        normals[a] += face;
        normals[b] += face;
        normals[c] += face;
    }
    for (auto &n : normals) {
        if (n.squaredNorm() > 0)
            n = parameters_.camera_transform.linear().transpose() * n.normalized();
        else
            n = Vector3::UnitZ();
    }
    return normals;
}

void TactileSensorProcessor::Rasterize(std::span<const Vector3> vertices, std::vector<float> &depth,
                                       std::vector<Vector2> *gradients) const {
    std::vector<Vector3> projected;
    projected.reserve(vertices.size());
    for (const auto &v : vertices)
        projected.push_back(Project(v));
    const auto normals = gradients ? SurfaceNormals(vertices) : std::vector<Vector3>{};
    for (std::size_t i = 0; i < surface_.size(); i += 3) {
        const auto ia = surface_[i], ib = surface_[i + 1], ic = surface_[i + 2];
        const auto &a = projected[ia];
        const auto &b = projected[ib];
        const auto &c = projected[ic];
        Pixels(a, b, c, parameters_.image_width, parameters_.image_height,
               [&](auto pixel, const Vector3 &weights, double) {
                   Vector3 w = weights;
                   const bool perspective = parameters_.camera_intrinsics.x() > 0;
                   const double z = perspective ? 1 / (w.x() / a.z() + w.y() / b.z() + w.z() / c.z())
                                                : w.dot(Vector3{a.z(), b.z(), c.z()});
                   if (z < parameters_.near_plane || z >= depth[pixel])
                       return;
                   depth[pixel] = float(z);
                   if (!gradients)
                       return;
                   if (perspective) {
                       w.x() *= z / a.z();
                       w.y() *= z / b.z();
                       w.z() *= z / c.z();
                   }
                   const Vector3 n = normals[ia] * w.x() + normals[ib] * w.y() + normals[ic] * w.z();
                   const Vector3 r = reference_normals_[ia] * w.x() + reference_normals_[ib] * w.y() +
                                     reference_normals_[ic] * w.z();
                   // Reconstruct a continuous optical normal field from FEM vertices.
                   // Depth and force outputs keep the unfiltered physical discretization.
                   const double nz = std::copysign(std::max(std::abs(double(n.z())), 1e-4), n.z());
                   const double rz = std::copysign(std::max(std::abs(double(r.z())), 1e-4), r.z());
                   (*gradients)[pixel] =
                       Vector2{RealType(n.x() / nz - r.x() / rz), RealType(-n.y() / nz + r.y() / rz)};
               });
    }
}

namespace {
// Gaussian support and contact reinsertion follow UniVTAC's TAXIM implementation.
// The FEM state is untouched; this models the optical gel height seen by the camera.
void BlurHeight(std::vector<float> &image, int width, int height, double sigma) {
    const double argument = 1e-5 * std::sqrt(2 * std::numbers::pi) * sigma;
    const int radius = std::max(1, int(std::round(std::sqrt(-2 * std::log(argument)) * sigma)) / 2);
    std::vector<double> kernel(radius * 2 + 1);
    double total = 0;
    for (int i = -radius; i <= radius; ++i) {
        kernel[i + radius] = std::exp(-.5 * i * i / (sigma * sigma));
        total += kernel[i + radius];
    }
    for (auto &v : kernel)
        v /= total;
    std::vector<float> temp(image.size());
    const auto reflect = [](int i, int n) {
        if (n <= 1)
            return 0;
        while (i < 0 || i >= n)
            i = i < 0 ? -i : 2 * n - 2 - i;
        return i;
    };
    for (int pass = 0; pass < 2; ++pass) {
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x) {
                double value = 0;
                for (int k = -radius; k <= radius; ++k) {
                    const int sx = reflect(x + (pass == 0 ? k : 0), width),
                              sy = reflect(y + (pass == 1 ? k : 0), height);
                    value += image[sy * width + sx] * kernel[k + radius];
                }
                temp[y * width + x] = float(value);
            }
        image.swap(temp);
    }
}
} // namespace

void TactileSensorProcessor::ContactOptics(std::span<const Vector3> vertices,
                                           std::span<const std::uint32_t> triangles,
                                           TactileSensorFrame &frame, std::vector<Vector2> &gradients) const {
    if (triangles.size() % 3)
        throw std::invalid_argument("tactile collider triangles are incomplete");
    std::vector<Vector3> projected(vertices.size());
    std::vector<bool> valid(vertices.size());
    const auto inverse_camera = parameters_.camera_transform.inverse();
    for (std::size_t i = 0; i < vertices.size(); ++i) {
        const auto camera = inverse_camera * vertices[i];
        valid[i] = camera.z() > parameters_.near_plane;
        if (valid[i])
            projected[i] = Project(vertices[i]);
    }
    for (std::size_t i = 0; i < triangles.size(); i += 3) {
        const auto ia = triangles[i], ib = triangles[i + 1], ic = triangles[i + 2];
        if (ia >= vertices.size() || ib >= vertices.size() || ic >= vertices.size())
            throw std::invalid_argument("tactile collider triangle index is out of range");
        if (!valid[ia] || !valid[ib] || !valid[ic])
            continue;
        const auto &a = projected[ia];
        const auto &b = projected[ib];
        const auto &c = projected[ic];
        Pixels(a, b, c, frame.width, frame.height, [&](auto pixel, const Vector3 &w, double) {
            const double z = parameters_.camera_intrinsics.x() > 0
                                 ? 1 / (w.x() / a.z() + w.y() / b.z() + w.z() / c.z())
                                 : w.dot(Vector3{a.z(), b.z(), c.z()});
            if (z >= parameters_.near_plane && z < frame.contact_depth[pixel])
                frame.contact_depth[pixel] = float(z);
        });
    }
    std::vector<float> overlap(frame.depth.size(), 0);
    float maximum = 0;
    for (std::size_t i = 0; i < overlap.size(); ++i) {
        if (rest_depth_[i] < parameters_.far_plane)
            overlap[i] = std::max(0.f, rest_depth_[i] - frame.contact_depth[i]);
        maximum = std::max(maximum, overlap[i]);
    }
    auto optical = overlap;
    if (maximum > 0) {
        constexpr double sigma_rel[] = {.04765625, .02421875, .0125, .00546875, .003125, .00171875};
        for (double relative : sigma_rel) {
            BlurHeight(optical, frame.width, frame.height, relative * frame.width);
            for (std::size_t i = 0; i < optical.size(); ++i)
                if (overlap[i] > .4f * maximum)
                    optical[i] = overlap[i];
        }
        BlurHeight(optical, frame.width, frame.height, .003125 * frame.width);
    }
    frame.optical_indentation = std::move(optical);
    const double pixel_x =
        parameters_.rgb_model == "taxim_v1"
            ? double(calibration_.pixel_size) * calibration_.calibration_width / frame.width
            : parameters_.pixel_size;
    const double pixel_y =
        parameters_.rgb_model == "taxim_v1"
            ? double(calibration_.pixel_size) * calibration_.calibration_height / frame.height
            : parameters_.pixel_size;
    for (int y = 0; y < int(frame.height); ++y)
        for (int x = 0; x < int(frame.width); ++x) {
            const int left = std::max(0, x - 1), right = std::min(x + 1, int(frame.width) - 1);
            const int top = std::max(0, y - 1), bottom = std::min(y + 1, int(frame.height) - 1);
            const auto &h = frame.optical_indentation;
            gradients[y * frame.width + x] =
                Vector2{RealType((h[y * frame.width + right] - h[y * frame.width + left]) /
                                 (std::max(1, right - left) * pixel_x)),
                        RealType((h[bottom * frame.width + x] - h[top * frame.width + x]) /
                                 (std::max(1, bottom - top) * pixel_y))};
        }
}

std::shared_ptr<const TactileSensorFrame>
TactileSensorProcessor::Process(std::span<const Vector3> world_vertices,
                                std::span<const Vector3> world_forces, const Affine3 &sensor_to_world,
                                std::uint64_t tick, double time, std::span<const Vector3> contact_vertices,
                                std::span<const std::uint32_t> contact_triangles) const {
    if (world_vertices.size() != rest_vertices_.size() || world_forces.size() != world_vertices.size() ||
        !std::isfinite(time) || !sensor_to_world.matrix().allFinite() ||
        !(sensor_to_world.linear().transpose() * sensor_to_world.linear())
             .isApprox(Matrix3::Identity(), 1e-5))
        throw std::invalid_argument("invalid tactile input state");
    auto frame = std::make_shared<TactileSensorFrame>();
    frame->width = parameters_.image_width;
    frame->height = parameters_.image_height;
    frame->tick = tick;
    frame->simulation_time = time;
    const auto count = std::size_t(frame->width) * frame->height;
    const Affine3 inverse = sensor_to_world.inverse();
    std::vector<Vector3> forces;
    forces.reserve(world_forces.size());
    for (std::size_t i = 0; i < world_vertices.size(); ++i) {
        if (!world_vertices[i].allFinite() || !world_forces[i].allFinite())
            throw std::invalid_argument("non-finite tactile input state");
        frame->local_vertices.push_back(inverse * world_vertices[i]);
        forces.push_back(inverse.linear() * world_forces[i]);
    }
    frame->depth.assign(count, parameters_.far_plane);
    std::vector<Vector2> gradients(count, Vector2::Zero());
    Rasterize(frame->local_vertices, frame->depth, &gradients);
    // Optical point-spread filtering (TAXIM also smooths before the normal/RGB
    // conversion). Smooth the reconstructed normals, never the force or depth data.
    constexpr double sigma = 1.25;
    double kernel[9], sum = 0;
    for (int i = -4; i <= 4; ++i) {
        kernel[i + 4] = std::exp(-.5 * i * i / (sigma * sigma));
        sum += kernel[i + 4];
    }
    for (auto &weight : kernel)
        weight /= sum;
    std::vector<Vector2> scratch(count, Vector2::Zero());
    for (int pass = 0; pass < 2; ++pass) {
        for (int y = 0; y < int(frame->height); ++y)
            for (int x = 0; x < int(frame->width); ++x) {
                Vector2 value = Vector2::Zero();
                for (int k = -4; k <= 4; ++k) {
                    const int sx = std::clamp(x + (pass == 0 ? k : 0), 0, int(frame->width) - 1);
                    const int sy = std::clamp(y + (pass == 1 ? k : 0), 0, int(frame->height) - 1);
                    value += gradients[sy * frame->width + sx] * RealType(kernel[k + 4]);
                }
                scratch[y * frame->width + x] = value;
            }
        gradients.swap(scratch);
    }
    frame->indentation.resize(count);
    frame->rgb.resize(count * 3);
    frame->normal_pressure.assign(count, 0);
    frame->shear_stress.assign(count * 2, 0);
    for (auto i : coat_)
        frame->total_force += forces[i];
    // Conservative mass-lumped deposition in the reference gel's material grid.
    for (const auto &sample : samples_) {
        const Vector3 force = forces[sample.a] * (sample.weights.x() / vertex_weights_[sample.a]) +
                              forces[sample.b] * (sample.weights.y() / vertex_weights_[sample.b]) +
                              forces[sample.c] * (sample.weights.z() / vertex_weights_[sample.c]);
        const double area = pixel_areas_[sample.pixel];
        frame->normal_pressure[sample.pixel] += float(-force.z() / area);
        frame->shear_stress[sample.pixel * 2] += float(force.x() / area);
        frame->shear_stress[sample.pixel * 2 + 1] += float(force.y() / area);
    }
    for (std::size_t i = 0; i < count; ++i) {
        frame->indentation[i] =
            (rest_depth_[i] < parameters_.far_plane && frame->depth[i] < parameters_.far_plane)
                ? rest_depth_[i] - frame->depth[i]
                : 0;
        frame->normal_pressure[i] = std::max(0.0f, frame->normal_pressure[i]);
    }
    frame->contact_depth.assign(count, parameters_.far_plane);
    frame->optical_indentation = frame->indentation;
    if (!contact_vertices.empty() || !contact_triangles.empty()) {
        std::vector<Vector3> local;
        local.reserve(contact_vertices.size());
        for (const auto &v : contact_vertices) {
            if (!v.allFinite())
                throw std::invalid_argument("non-finite tactile collider vertex");
            local.push_back(inverse * v);
        }
        ContactOptics(local, contact_triangles, *frame, gradients);
    }
    const double pi = std::numbers::pi;
    for (std::uint32_t y = 0; y < frame->height; ++y)
        for (std::uint32_t x = 0; x < frame->width; ++x) {
            const auto p = y * frame->width + x;
            const double dx = gradients[p].x(), dy = gradients[p].y();
            if (parameters_.rgb_model == "taxim_v1") {
                const auto bins = calibration_.bins;
                const double magnitude = std::clamp(std::atan(std::hypot(dx, dy)) / (0.5 * pi / (bins - 1)),
                                                    0.0, double(bins - 1));
                const double direction =
                    std::clamp((std::atan2(dy, dx) + pi) / (2 * pi / (bins - 1)), 0.0, double(bins - 1));
                const auto m0 = std::uint32_t(magnitude), d0 = std::uint32_t(direction);
                const auto m1 = std::min(m0 + 1, bins - 1), d1 = std::min(d0 + 1, bins - 1);
                const double fm = magnitude - m0, fd = direction - d0;
                const double cx = double(x) * calibration_.calibration_width / frame->width;
                const double cy = double(y) * calibration_.calibration_height / frame->height;
                const double features[6] = {cx * cx, cy * cy, cx * cy, cx, cy, 1};
                for (int channel = 0; channel < 3; ++channel) {
                    double value = calibration_.background[p * 3 + channel];
                    for (int j = 0; j < 6; ++j) {
                        const auto at = [&](auto m, auto d) {
                            return double(
                                calibration_
                                    .coefficients[((std::size_t(channel) * bins + m) * bins + d) * 6 + j]);
                        };
                        const double coefficient = (1 - fm) * ((1 - fd) * at(m0, d0) + fd * at(m0, d1)) +
                                                   fm * ((1 - fd) * at(m1, d0) + fd * at(m1, d1));
                        value += coefficient * features[j];
                    }
                    frame->rgb[p * 3 + channel] = Byte(value);
                }
            } else {
                frame->rgb[p * 3] = Byte(.5 + dx * .35);
                frame->rgb[p * 3 + 1] = Byte(.5 + dy * .35);
                frame->rgb[p * 3 + 2] = Byte(.5 - std::hypot(dx, dy) * .2);
            }
        }
    frame->rgb_marker = frame->rgb;
    for (std::size_t i = 0; i < rest_markers_.size(); ++i) {
        Vector3 p = Vector3::Zero();
        for (int j = 0; j < 4; ++j)
            p += frame->local_vertices[tetrahedra_[parameters_.marker_tetrahedra[i] * 4 + j]] *
                 parameters_.marker_barycentric[i][j];
        const Vector2 uv = Project(p).head<2>();
        frame->marker_positions.push_back(uv);
        frame->marker_motion.push_back(uv - rest_markers_[i]);
        const int cx = int(std::floor(uv.x())), cy = int(std::floor(uv.y()));
        for (int dy = -4; dy <= 4; ++dy)
            for (int dx = -4; dx <= 4; ++dx) {
                const int x = cx + dx, y = cy + dy;
                if (x < 0 || y < 0 || x >= int(frame->width) || y >= int(frame->height))
                    continue;
                const double coverage =
                    std::clamp(3.5 - std::hypot(x + .5 - uv.x(), y + .5 - uv.y()), 0.0, 1.0);
                const auto pixel = (y * frame->width + x) * 3;
                for (int c = 0; c < 3; ++c)
                    frame->rgb_marker[pixel + c] =
                        std::uint8_t(std::lround(frame->rgb_marker[pixel + c] * (1 - .85 * coverage)));
            }
    }
    return frame;
}
} // namespace gobot
