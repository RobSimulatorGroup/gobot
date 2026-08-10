/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 *
 * This file is a Gobot adapter for libuipc. It was written independently
 * against libuipc's public C++ API at revision GOBOT_LIBUIPC_REVISION.
 */

#include "gobot/physics/ipc_solver_module_api.hpp"
#include "gobot/physics/ipc_batch_solver_module_api.hpp"

#include <uipc/uipc.h>
#include <uipc/builtin/attribute_name.h>
#include <uipc/constitution/affine_body_constitution.h>
#include <uipc/constitution/affine_body_driving_prismatic_joint.h>
#include <uipc/constitution/affine_body_driving_revolute_joint.h>
#include <uipc/constitution/affine_body_prismatic_joint.h>
#include <uipc/constitution/affine_body_revolute_joint.h>
#include <uipc/constitution/elastic_moduli.h>
#include <uipc/constitution/soft_transform_constraint.h>
#include <uipc/constitution/stable_neo_hookean.h>
#include <uipc/core/affine_body_state_accessor_feature.h>
#include <uipc/core/contact_system_feature.h>
#include <uipc/core/finite_element_state_accessor_feature.h>
#include <uipc/geometry/utils/affine_body/affine_body_from_rigid_body.h>
#include <uipc/geometry/utils/factory.h>
#include <uipc/geometry/utils/label_surface.h>
#include <uipc/geometry/utils/label_triangle_orient.h>

#include <nlohmann/json.hpp>
#include <cuda_runtime_api.h>
#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <numbers>
#include <optional>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace gobot::libuipc_solver {
namespace {

using Json = nlohmann::json;
using uipc::IndexT;
using uipc::Float;
using uipc::Matrix3x3;
using uipc::Matrix4x4;
using uipc::S;
using uipc::Vector2;
using uipc::Vector3;
using uipc::Vector3i;
using uipc::Vector4i;
using uipc::constitution::AffineBodyConstitution;
using uipc::constitution::AffineBodyDrivingPrismaticJoint;
using uipc::constitution::AffineBodyDrivingRevoluteJoint;
using uipc::constitution::AffineBodyPrismaticJoint;
using uipc::constitution::AffineBodyRevoluteJoint;
using uipc::constitution::ElasticModuli;
using uipc::constitution::SoftTransformConstraint;
using uipc::constitution::StableNeoHookean;
using uipc::core::AffineBodyStateAccessorFeature;
using uipc::core::ContactSystemFeature;
using uipc::core::ContactElement;
using uipc::core::Engine;
using uipc::core::FiniteElementStateAccessorFeature;
using uipc::core::Scene;
using uipc::core::SubsceneElement;
using uipc::core::World;
using uipc::geometry::SimplicialComplex;
using uipc::geometry::SimplicialComplexSlot;
using uipc::geometry::Geometry;
using uipc::geometry::label_surface;
using uipc::geometry::label_triangle_orient;
using uipc::geometry::tetmesh;
using uipc::geometry::trimesh;
using uipc::geometry::view;

constexpr std::string_view kTetEncoding = "gobot.tetrahedral-mesh.le.v1";
constexpr std::string_view kTetMagic = "GOBTIPC1";
constexpr std::string_view kTriangleEncoding = "gobot.triangle-mesh.le.v1";
constexpr std::string_view kTriangleMagic = "GOBTTRI1";

struct TetMeshData {
    std::vector<Vector3> vertices;
    std::vector<Vector4i> tetrahedra;
};

struct TriangleMeshData {
    std::vector<Vector3> vertices;
    std::vector<Vector3i> triangles;
};

struct BodyRecord {
    std::string path;
    std::size_t offset{0};
    std::size_t count{0};
};

struct DeformableContactRange {
    std::size_t output_offset{0};
    std::size_t vertex_count{0};
    S<SimplicialComplexSlot> geometry;
    IndexT global_vertex_offset{-1};
};

struct ContactGradientBuffer {
    std::string primitive_type;
    std::unique_ptr<Geometry> geometry;
};

struct ContactElementRecord {
    double friction{0.0};
    uipc::core::ContactElement element;
    std::uint32_t contact_type{0};
    std::uint32_t contact_affinity{0};
    bool always_enabled{false};
};

struct WorkspaceLease {
    std::filesystem::path path;
    bool owned{false};

    ~WorkspaceLease() {
        if (!owned) {
            return;
        }
        std::error_code error;
        std::filesystem::remove_all(path, error);
    }
};

std::string MakeBatchWorkspaceSuffix() {
    static std::atomic<std::uint64_t> serial{0};
    const auto timestamp = std::chrono::steady_clock::now()
                                   .time_since_epoch()
                                   .count();
    return fmt::format("batch_session_{:x}_{:x}", timestamp,
                       serial.fetch_add(1, std::memory_order_relaxed));
}

bool ContactMasksMatch(const ContactElementRecord& first,
                       const ContactElementRecord& second) {
    return first.always_enabled || second.always_enabled ||
           (first.contact_type & second.contact_affinity) != 0U ||
           (second.contact_type & first.contact_affinity) != 0U;
}

struct AffineTarget {
    Matrix4x4 initial{Matrix4x4::Identity()};
    Matrix4x4 value{Matrix4x4::Identity()};
};

struct JointTarget {
    double initial{0.0};
    double value{0.0};
    std::string state_attribute;
    std::string aim_attribute;
    S<SimplicialComplexSlot> geometry;
};

void WriteError(char* destination, std::size_t capacity, std::string_view message) {
    if (destination == nullptr || capacity == 0) {
        return;
    }
    const std::size_t count = std::min(capacity - 1, message.size());
    std::memcpy(destination, message.data(), count);
    destination[count] = '\0';
}

template <typename Function>
bool Guard(char* error, std::size_t error_size, Function&& function) {
    try {
        function();
        return true;
    } catch (const std::exception& exception) {
        WriteError(error, error_size, exception.what());
        return false;
    } catch (...) {
        WriteError(error, error_size, "libuipc solver raised an unknown exception");
        return false;
    }
}

std::uint32_t ReadU32(std::span<const std::uint8_t> data, std::size_t* cursor) {
    if (*cursor > data.size() || data.size() - *cursor < 4) {
        throw std::runtime_error("IPC tetrahedral mesh blob is truncated");
    }
    std::uint32_t value = 0;
    for (std::size_t byte = 0; byte < 4; ++byte) {
        value |= static_cast<std::uint32_t>(data[*cursor + byte]) << (byte * 8U);
    }
    *cursor += 4;
    return value;
}

double ReadF64(std::span<const std::uint8_t> data, std::size_t* cursor) {
    if (*cursor > data.size() || data.size() - *cursor < 8) {
        throw std::runtime_error("IPC tetrahedral mesh blob is truncated");
    }
    std::uint64_t bits = 0;
    for (std::size_t byte = 0; byte < 8; ++byte) {
        bits |= static_cast<std::uint64_t>(data[*cursor + byte]) << (byte * 8U);
    }
    *cursor += 8;
    return std::bit_cast<double>(bits);
}

TetMeshData DecodeTetMesh(std::span<const std::uint8_t> data) {
    if (data.size() < 24 ||
        std::string_view(reinterpret_cast<const char*>(data.data()), 8) != kTetMagic) {
        throw std::runtime_error("IPC tetrahedral mesh blob has invalid magic");
    }
    std::size_t cursor = 8;
    const std::uint32_t version = ReadU32(data, &cursor);
    const std::uint32_t vertex_count = ReadU32(data, &cursor);
    const std::uint32_t tetrahedron_count = ReadU32(data, &cursor);
    const std::uint32_t surface_triangle_count = ReadU32(data, &cursor);
    if (version != 1 || vertex_count == 0 || tetrahedron_count == 0) {
        throw std::runtime_error("IPC tetrahedral mesh blob has invalid metadata");
    }
    const std::uint64_t expected = 24ULL + static_cast<std::uint64_t>(vertex_count) * 24ULL +
                                   static_cast<std::uint64_t>(tetrahedron_count) * 16ULL +
                                   static_cast<std::uint64_t>(surface_triangle_count) * 12ULL;
    if (expected != data.size()) {
        throw std::runtime_error("IPC tetrahedral mesh blob byte length is inconsistent");
    }

    TetMeshData mesh;
    mesh.vertices.reserve(vertex_count);
    mesh.tetrahedra.reserve(tetrahedron_count);
    for (std::uint32_t index = 0; index < vertex_count; ++index) {
        Vector3 vertex{ReadF64(data, &cursor), ReadF64(data, &cursor), ReadF64(data, &cursor)};
        if (!vertex.allFinite()) {
            throw std::runtime_error("IPC tetrahedral mesh contains a non-finite vertex");
        }
        mesh.vertices.push_back(vertex);
    }
    for (std::uint32_t index = 0; index < tetrahedron_count; ++index) {
        Vector4i tetrahedron;
        for (std::size_t corner = 0; corner < 4; ++corner) {
            const std::uint32_t vertex = ReadU32(data, &cursor);
            if (vertex >= vertex_count) {
                throw std::runtime_error("IPC tetrahedral mesh contains an invalid index");
            }
            tetrahedron[static_cast<Eigen::Index>(corner)] = static_cast<IndexT>(vertex);
        }
        mesh.tetrahedra.push_back(tetrahedron);
    }
    cursor += static_cast<std::size_t>(surface_triangle_count) * 12;
    return mesh;
}

TriangleMeshData DecodeTriangleMesh(std::span<const std::uint8_t> data) {
    if (data.size() < 20 ||
        std::string_view(reinterpret_cast<const char*>(data.data()), 8) !=
                kTriangleMagic) {
        throw std::runtime_error("IPC triangle mesh blob has invalid magic");
    }
    std::size_t cursor = 8;
    const std::uint32_t version = ReadU32(data, &cursor);
    const std::uint32_t vertex_count = ReadU32(data, &cursor);
    const std::uint32_t triangle_count = ReadU32(data, &cursor);
    if (version != 1 || vertex_count == 0 || triangle_count == 0) {
        throw std::runtime_error("IPC triangle mesh blob has invalid metadata");
    }
    const std::uint64_t expected =
            20ULL + static_cast<std::uint64_t>(vertex_count) * 24ULL +
            static_cast<std::uint64_t>(triangle_count) * 12ULL;
    if (expected != data.size()) {
        throw std::runtime_error(
                "IPC triangle mesh blob byte length is inconsistent");
    }

    TriangleMeshData mesh;
    mesh.vertices.reserve(vertex_count);
    mesh.triangles.reserve(triangle_count);
    for (std::uint32_t index = 0; index < vertex_count; ++index) {
        Vector3 vertex{ReadF64(data, &cursor), ReadF64(data, &cursor),
                       ReadF64(data, &cursor)};
        if (!vertex.allFinite()) {
            throw std::runtime_error(
                    "IPC triangle mesh contains a non-finite vertex");
        }
        mesh.vertices.push_back(vertex);
    }
    for (std::uint32_t index = 0; index < triangle_count; ++index) {
        Vector3i triangle;
        for (std::size_t corner = 0; corner < 3; ++corner) {
            const std::uint32_t vertex = ReadU32(data, &cursor);
            if (vertex >= vertex_count) {
                throw std::runtime_error(
                        "IPC triangle mesh contains an invalid index");
            }
            triangle[static_cast<Eigen::Index>(corner)] =
                    static_cast<IndexT>(vertex);
        }
        mesh.triangles.push_back(triangle);
    }
    return mesh;
}

Matrix4x4 ParseTransform(const Json& value, std::string_view description) {
    if (!value.is_object() || !value.contains("matrix_row_major") ||
        !value["matrix_row_major"].is_array() ||
        value["matrix_row_major"].size() != 16) {
        throw std::runtime_error(std::string(description) + " must contain a 4x4 row-major matrix");
    }
    Matrix4x4 result;
    for (std::size_t index = 0; index < 16; ++index) {
        const double scalar = value["matrix_row_major"][index].get<double>();
        if (!std::isfinite(scalar)) {
            throw std::runtime_error(std::string(description) + " contains a non-finite value");
        }
        result(static_cast<Eigen::Index>(index / 4),
               static_cast<Eigen::Index>(index % 4)) = scalar;
    }
    return result;
}

Vector3 ParseVector3(const Json& value, std::string_view description) {
    if (!value.is_array() || value.size() != 3) {
        throw std::runtime_error(std::string(description) + " must contain three values");
    }
    Vector3 result{value[0].get<double>(), value[1].get<double>(), value[2].get<double>()};
    if (!result.allFinite()) {
        throw std::runtime_error(std::string(description) + " contains a non-finite value");
    }
    return result;
}

Vector3 TransformPoint(const Matrix4x4& transform, const Vector3& point) {
    const Eigen::Vector4d homogeneous = transform * Eigen::Vector4d{point.x(), point.y(), point.z(), 1.0};
    return homogeneous.head<3>();
}

TriangleMeshData MakeBox(const Vector3& size) {
    if (!size.allFinite() || (size.array() <= 0.0).any()) {
        throw std::runtime_error("IPC box collision shape has invalid size");
    }
    const Vector3 h = size * 0.5;
    TriangleMeshData mesh;
    mesh.vertices = {
            {-h.x(), -h.y(), -h.z()}, {h.x(), -h.y(), -h.z()},
            {h.x(), h.y(), -h.z()},   {-h.x(), h.y(), -h.z()},
            {-h.x(), -h.y(), h.z()},  {h.x(), -h.y(), h.z()},
            {h.x(), h.y(), h.z()},    {-h.x(), h.y(), h.z()}};
    mesh.triangles = {
            {0, 2, 1}, {0, 3, 2}, {4, 5, 6}, {4, 6, 7},
            {0, 1, 5}, {0, 5, 4}, {3, 7, 6}, {3, 6, 2},
            {0, 4, 7}, {0, 7, 3}, {1, 2, 6}, {1, 6, 5}};
    return mesh;
}

TriangleMeshData MakeSphere(double radius) {
    if (!std::isfinite(radius) || radius <= 0.0) {
        throw std::runtime_error("IPC sphere collision shape has invalid radius");
    }
    constexpr int kSegments = 16;
    constexpr int kRings = 8;
    TriangleMeshData mesh;
    mesh.vertices.push_back(Vector3{0.0, 0.0, radius});
    for (int ring = 1; ring < kRings; ++ring) {
        const double phi = std::numbers::pi * static_cast<double>(ring) /
                           static_cast<double>(kRings);
        for (int segment = 0; segment < kSegments; ++segment) {
            const double theta = 2.0 * std::numbers::pi * static_cast<double>(segment) /
                                 static_cast<double>(kSegments);
            mesh.vertices.push_back(Vector3{radius * std::sin(phi) * std::cos(theta),
                                            radius * std::sin(phi) * std::sin(theta),
                                            radius * std::cos(phi)});
        }
    }
    const IndexT bottom = static_cast<IndexT>(mesh.vertices.size());
    mesh.vertices.push_back(Vector3{0.0, 0.0, -radius});
    for (int segment = 0; segment < kSegments; ++segment) {
        const IndexT next = static_cast<IndexT>((segment + 1) % kSegments);
        mesh.triangles.push_back(Vector3i{0, 1 + segment, 1 + next});
    }
    for (int ring = 0; ring < kRings - 2; ++ring) {
        const IndexT current = 1 + ring * kSegments;
        const IndexT next_ring = current + kSegments;
        for (int segment = 0; segment < kSegments; ++segment) {
            const IndexT next = static_cast<IndexT>((segment + 1) % kSegments);
            mesh.triangles.push_back(
                    Vector3i{current + segment, next_ring + segment, current + next});
            mesh.triangles.push_back(
                    Vector3i{current + next, next_ring + segment, next_ring + next});
        }
    }
    const IndexT last_ring = 1 + (kRings - 2) * kSegments;
    for (int segment = 0; segment < kSegments; ++segment) {
        const IndexT next = static_cast<IndexT>((segment + 1) % kSegments);
        mesh.triangles.push_back(
                Vector3i{last_ring + next, last_ring + segment, bottom});
    }
    return mesh;
}

void AppendShape(TriangleMeshData* destination,
                 TriangleMeshData source,
                 const Matrix4x4& transform) {
    if (destination->vertices.size() + source.vertices.size() >
        static_cast<std::size_t>(std::numeric_limits<IndexT>::max())) {
        throw std::runtime_error("IPC affine collision mesh exceeds libuipc index capacity");
    }
    const IndexT offset = static_cast<IndexT>(destination->vertices.size());
    for (const Vector3& vertex : source.vertices) {
        destination->vertices.push_back(TransformPoint(transform, vertex));
    }
    for (const Vector3i& triangle : source.triangles) {
        destination->triangles.push_back((triangle.array() + offset).matrix());
    }
}

class Session final {
public:
    Session(const IpcSolverArtifactView& artifact,
            const IpcSolverModuleConfig& config,
            std::size_t environment_count = 1,
            bool external_affine_proxies = false,
            std::string workspace_suffix = {})
        : environment_count_(environment_count),
          device_index_(config.device_index),
          external_affine_proxies_(external_affine_proxies) {
        ValidateConfig(config);
        if (environment_count_ == 0) {
            throw std::runtime_error(
                    "libuipc environment count must be positive");
        }
        ActivateDevice();
        ConfigureLibuipc(config);
        const Json manifest = Json::parse(
                std::string_view(artifact.manifest, artifact.manifest_size));
        if (artifact.schema_version != 2 ||
            manifest.value("schema_version", 0) != 2 ||
            manifest.value("format", std::string{}) != "gobot-ipc") {
            throw std::runtime_error(
                    "libuipc module requires a Gobot IPC schema v2 artifact with explicit PhysicsCoupling entries");
        }
        for (std::size_t index = 0; index < artifact.blob_count; ++index) {
            const IpcSolverArtifactBlobView& blob = artifact.blobs[index];
            if (blob.id == nullptr || blob.encoding == nullptr || blob.data == nullptr) {
                throw std::runtime_error("IPC artifact contains an invalid blob view");
            }
            blobs_.emplace(blob.id, blob);
        }

        auto engine_config = Engine::default_config();
        engine_config["gpu"]["device"] = config.device_index;
        std::filesystem::path workspace = ResolveWorkspace(config);
        if (!workspace_suffix.empty()) {
            workspace /= workspace_suffix;
        }
        std::error_code workspace_error;
        std::filesystem::create_directories(workspace, workspace_error);
        if (workspace_error) {
            throw std::runtime_error(
                    "cannot create libuipc workspace '" + workspace.string() +
                    "': " + workspace_error.message());
        }
        engine_ = std::make_unique<Engine>("cuda", workspace.string(), engine_config);
        world_ = std::make_unique<World>(*engine_);

        auto scene_config = Scene::default_config();
        scene_config["dt"] = config.fixed_time_step;
        scene_config["gravity"] = Vector3{config.gravity[0], config.gravity[1], config.gravity[2]};
        scene_config["contact"]["enable"] = true;
        scene_config["contact"]["d_hat"] = config.contact_activation_distance;
        scene_config["contact"]["friction"]["enable"] =
                config.friction_coefficient > 0.0;
        scene_config["contact"]["constitution"] = "ipc";
        // libuipc's strict offline defaults can spend hundreds of Newton
        // iterations resolving driven affine joints.  These are the same
        // tolerances used by its interactive affine-body examples.
        scene_config["newton"]["max_iter"] = 16;
        scene_config["newton"]["velocity_tol"] = 0.1;
        scene_config["newton"]["transrate_tol"] = 10.0;
        scene_config["newton"]["ccd_tol"] = 5.0e-4;
        scene_config["line_search"]["max_iter"] = 8;
        scene_config["linear_system"]["tol_rate"] = 1.0e-3;
        scene_ = std::make_unique<Scene>(scene_config);
        scene_->contact_tabular().default_model(
                config.friction_coefficient, config.contact_resistance);
        const auto contact = scene_->contact_tabular().default_element();

        std::vector<SubsceneElement> subscenes;
        subscenes.reserve(environment_count_);
        if (environment_count_ == 1) {
            subscenes.push_back(scene_->subscene_tabular().default_element());
        } else {
            for (std::size_t environment = 0;
                 environment < environment_count_; ++environment) {
                subscenes.push_back(scene_->subscene_tabular().create(
                        fmt::format("environment_{}", environment)));
            }
        }

        BuildDeformables(manifest.value("deformable_bodies", Json::array()),
                         contact, subscenes);
        const Json affine_robots = external_affine_proxies_
                                           ? SelectExternalAffineRobots(manifest)
                                           : manifest.value("robots", Json::array());
        BuildAffineBodies(affine_robots, contact, config, subscenes);
        if (deformable_bodies_.empty()) {
            throw std::runtime_error("libuipc module requires at least one deformable body");
        }

        world_->init(*scene_);
        if (!world_->is_valid()) {
            throw std::runtime_error("libuipc rejected the compiled Gobot scene");
        }
        InitializeAccessors();
        if (external_affine_proxies_) {
            if (world_->frame() != 0 || !world_->dump()) {
                throw std::runtime_error(
                        "libuipc batch solver could not snapshot its initial state");
            }
            initial_state_dumped_ = true;
        }
        if (!external_affine_proxies_) {
            InitializeContactForceExport(config.fixed_time_step);
            Refresh(false);
        }
    }

    void Step(std::uint32_t steps) {
        if (steps == 0) {
            throw std::runtime_error("libuipc step count must be positive");
        }
        const auto start = std::chrono::steady_clock::now();
        for (std::uint32_t index = 0; index < steps; ++index) {
            world_->advance();
            if (!world_->is_valid()) {
                throw std::runtime_error("libuipc world became invalid while stepping");
            }
        }
        world_->retrieve();
        frame_ += steps;
        Refresh(true);
        const auto end = std::chrono::steady_clock::now();
        last_step_latency_ms_ =
                std::chrono::duration<double, std::milli>(end - start).count();
    }

    void Reset() {
        if (fem_accessor_ != nullptr) {
            auto position = fem_state_->vertices().find<Vector3>(uipc::builtin::position);
            auto velocity = fem_state_->vertices().find<Vector3>(uipc::builtin::velocity);
            std::ranges::copy(initial_fem_positions_, view(*position).begin());
            std::ranges::fill(view(*velocity), Vector3::Zero());
            fem_accessor_->copy_from(*fem_state_);
        }
        if (affine_accessor_ != nullptr) {
            auto transform = affine_state_->instances().find<Matrix4x4>(uipc::builtin::transform);
            auto velocity = affine_state_->instances().find<Matrix4x4>(uipc::builtin::velocity);
            std::ranges::copy(initial_affine_transforms_, view(*transform).begin());
            std::ranges::fill(view(*velocity), Matrix4x4::Zero());
            for (const auto& [path, target] : affine_targets_) {
                target->value = target->initial;
            }
            for (const auto& [path, target] : joint_targets_) {
                target->value = target->initial;
                auto geometry = target->geometry->geometry().as<SimplicialComplex>();
                auto state = geometry->edges().find<Float>(target->state_attribute);
                auto aim = geometry->edges().find<Float>(target->aim_attribute);
                view(*state)[0] = target->initial;
                view(*aim)[0] = target->initial;
            }
            affine_accessor_->copy_from(*affine_state_);
        }
        world_->retrieve();
        frame_ = 0;
        Refresh(false);
    }

    void SetAffineTarget(std::string_view path, const double* row_major) {
        const auto found = affine_targets_.find(std::string(path));
        if (found == affine_targets_.end()) {
            throw std::runtime_error("libuipc scene has no affine body '" + std::string(path) + "'");
        }
        Matrix4x4 transform;
        for (std::size_t index = 0; index < 16; ++index) {
            if (!std::isfinite(row_major[index])) {
                throw std::runtime_error("libuipc affine target contains a non-finite value");
            }
            transform(static_cast<Eigen::Index>(index / 4),
                      static_cast<Eigen::Index>(index % 4)) = row_major[index];
        }
        found->second->value = transform;
    }

    void SetJointTarget(std::string_view path, double position) {
        const auto found = joint_targets_.find(std::string(path));
        if (found == joint_targets_.end()) {
            throw std::runtime_error("libuipc scene has no driven joint '" +
                                     std::string(path) + "'");
        }
        if (!std::isfinite(position)) {
            throw std::runtime_error("libuipc joint target contains a non-finite value");
        }
        found->second->value = position;
    }

    const std::vector<BodyRecord>& DeformableBodies() const { return deformable_bodies_; }
    const std::vector<BodyRecord>& AffineBodies() const { return affine_bodies_; }
    const std::vector<double>& Positions() const { return positions_; }
    const std::vector<double>& Velocities() const { return velocities_; }
    const std::vector<double>& ContactForces() const { return contact_forces_; }
    const std::vector<double>& AffineTransforms() const { return affine_transforms_; }
    std::uint64_t Frame() const { return frame_; }
    double LastStepLatencyMs() const { return last_step_latency_ms_; }
    bool IsValid() const { return world_->is_valid(); }
    std::size_t EnvironmentCount() const { return environment_count_; }

    void BindDeviceBuffers(const IpcBatchSolverModuleBuffers& buffers) {
        if (!external_affine_proxies_) {
            throw std::runtime_error(
                    "device buffers require external affine proxy mode");
        }
        ActivateDevice();
        device_buffers_ = buffers;
        device_buffers_bound_ = true;
        WriteDeviceState();
    }

    void StepDevice(std::uint32_t steps) {
        if (!device_buffers_bound_) {
            throw std::runtime_error(
                    "libuipc batch device buffers are not bound");
        }
        if (steps == 0) {
            throw std::runtime_error("libuipc step count must be positive");
        }
        ActivateDevice();
        const auto start = std::chrono::steady_clock::now();
        for (std::uint32_t index = 0; index < steps; ++index) {
            UploadAffineTargets();
            world_->advance();
            if (!world_->is_valid()) {
                throw std::runtime_error(
                        "libuipc batch world became invalid while stepping");
            }
        }
        frame_ = world_->frame();
        WriteDeviceState();
        const auto end = std::chrono::steady_clock::now();
        last_step_latency_ms_ =
                std::chrono::duration<double, std::milli>(end - start).count();
    }

    void ResetDevice() {
        if (!device_buffers_bound_) {
            throw std::runtime_error(
                    "libuipc batch device buffers are not bound");
        }
        ActivateDevice();
        if (!initial_state_dumped_ || !world_->recover(0)) {
            throw std::runtime_error(
                    "libuipc batch solver could not recover its initial state");
        }
        frame_ = world_->frame();
        if (frame_ != 0) {
            throw std::runtime_error(
                    "libuipc batch solver recovered an invalid frame");
        }
        WriteDeviceState();
    }

    void SynchronizeDevice() {
        ActivateDevice();
        world_->sync();
    }

private:
    static void RequireCuda(cudaError_t result, std::string_view operation) {
        if (result != cudaSuccess) {
            throw std::runtime_error(
                    std::string(operation) + ": " +
                    cudaGetErrorString(result));
        }
    }

    void ActivateDevice() const {
        RequireCuda(cudaSetDevice(static_cast<int>(device_index_)),
                    "selecting the libuipc CUDA device");
    }

    static uipc::backend::BufferView DeviceView(
            const IpcSolverDeviceBufferView& buffer,
            std::size_t element_count,
            std::size_t element_size) {
        return uipc::backend::BufferView{
                static_cast<uipc::backend::HandleT>(
                        reinterpret_cast<std::uintptr_t>(buffer.data)),
                0,
                element_count,
                element_size,
                element_size,
                "cuda"};
    }

    void UploadAffineTargets() {
        const std::size_t body_count = initial_affine_transforms_.size();
        if (body_count == 0) {
            return;
        }
        target_row_major_.resize(body_count * 16);
        RequireCuda(
                cudaMemcpy(target_row_major_.data(),
                           device_buffers_.affine_targets.data,
                           target_row_major_.size() * sizeof(double),
                           cudaMemcpyDeviceToHost),
                "copying MuJoCo affine targets to libuipc");
        auto transform = affine_state_->instances().find<Matrix4x4>(
                uipc::builtin::transform);
        auto velocity = affine_state_->instances().find<Matrix4x4>(
                uipc::builtin::velocity);
        auto transforms = view(*transform);
        for (std::size_t body = 0; body < body_count; ++body) {
            Matrix4x4 value;
            for (std::size_t index = 0; index < 16; ++index) {
                value(static_cast<Eigen::Index>(index / 4),
                      static_cast<Eigen::Index>(index % 4)) =
                        target_row_major_[body * 16 + index];
            }
            transforms[body] = value;
        }
        std::ranges::fill(view(*velocity), Matrix4x4::Zero());
        affine_accessor_->copy_from(*affine_state_);
    }

    void WriteDeviceState() {
        const std::size_t vertex_count = initial_fem_positions_.size();
        fem_accessor_->copy_position_to(
                DeviceView(device_buffers_.deformable_positions,
                           vertex_count, sizeof(Vector3)));
        fem_accessor_->copy_velocity_to(
                DeviceView(device_buffers_.deformable_velocities,
                           vertex_count, sizeof(Vector3)));
        world_->sync();

        const std::size_t affine_count = initial_affine_transforms_.size();
        if (affine_count != 0) {
            affine_accessor_->copy_transform_to(
                    DeviceView(device_buffers_.affine_transforms,
                               affine_count, sizeof(Matrix4x4)));
        }

        const std::size_t deformable_force_scalars = vertex_count * 3;
        if (deformable_force_scalars != 0) {
            RequireCuda(
                    cudaMemset(device_buffers_.deformable_contact_forces.data,
                               0,
                               deformable_force_scalars * sizeof(double)),
                    "clearing batched deformable contact forces");
        }
        const std::size_t affine_wrench_scalars = affine_count * 6;
        if (affine_wrench_scalars != 0) {
            RequireCuda(
                    cudaMemset(device_buffers_.affine_contact_wrenches.data, 0,
                               affine_wrench_scalars * sizeof(double)),
                    "clearing batched affine contact wrenches");
        }
        RequireCuda(cudaDeviceSynchronize(),
                    "synchronizing libuipc batch device state");
    }

    static void ValidateConfig(const IpcSolverModuleConfig& config) {
        if (!std::isfinite(config.fixed_time_step) || config.fixed_time_step <= 0.0 ||
            !std::isfinite(config.friction_coefficient) || config.friction_coefficient < 0.0 ||
            !std::isfinite(config.contact_activation_distance) ||
            config.contact_activation_distance <= 0.0 ||
            !std::isfinite(config.contact_resistance) || config.contact_resistance <= 0.0 ||
            !std::isfinite(config.affine_stiffness) || config.affine_stiffness <= 0.0 ||
            !std::isfinite(config.kinematic_strength) || config.kinematic_strength <= 0.0) {
            throw std::runtime_error("libuipc solver configuration contains invalid values");
        }
        for (double gravity : config.gravity) {
            if (!std::isfinite(gravity)) {
                throw std::runtime_error("libuipc gravity contains a non-finite value");
            }
        }
    }

    static std::string ResolveWorkspace(const IpcSolverModuleConfig& config) {
        if (config.workspace != nullptr && config.workspace[0] != '\0') {
            return config.workspace;
        }
        return (std::filesystem::temp_directory_path() / "gobot-libuipc").string();
    }

    static void ConfigureLibuipc(const IpcSolverModuleConfig& config) {
        static std::mutex mutex;
        static std::optional<std::string> configured_directory;
        std::lock_guard lock(mutex);
        std::string directory;
        if (config.backend_module_directory != nullptr &&
            config.backend_module_directory[0] != '\0') {
            directory = config.backend_module_directory;
        } else {
            directory = GOBOT_LIBUIPC_BACKEND_BUILD_DIR;
        }
        directory = std::filesystem::absolute(directory).lexically_normal().string();
        if (configured_directory.has_value() && *configured_directory != directory) {
            throw std::runtime_error(
                    "libuipc backend module directory cannot change after the first session");
        }
        if (!configured_directory.has_value()) {
            uipc::init(Json{{"module_dir", directory}});
            uipc::logger::set_level(spdlog::level::warn);
            configured_directory = directory;
        }
    }

    const IpcSolverArtifactBlobView& FindBlob(
            std::string_view id,
            std::string_view expected_encoding,
            std::string_view description) const {
        const auto found = blobs_.find(std::string(id));
        if (found == blobs_.end()) {
            throw std::runtime_error("IPC artifact is missing mesh blob '" + std::string(id) + "'");
        }
        if (std::string_view(found->second.encoding) != expected_encoding) {
            throw std::runtime_error(std::string(description) +
                                     " blob has unsupported encoding");
        }
        return found->second;
    }

    static Json SelectExternalAffineRobots(const Json& manifest) {
        const Json& robots = manifest.at("robots");
        const Json& couplings = manifest.at("couplings");
        if (!robots.is_array() || !couplings.is_array()) {
            throw std::runtime_error(
                    "IPC schema v2 robot and PhysicsCoupling tables must be arrays");
        }
        if (couplings.empty()) {
            throw std::runtime_error(
                    "combined MuJoCo/libuipc simulation requires at least one enabled PhysicsCoupling");
        }

        struct RobotLinkRecord {
            const Json* robot;
            const Json* link;
        };
        std::unordered_map<std::string, RobotLinkRecord> links_by_path;
        for (const Json& robot : robots) {
            if (!robot.contains("links") || !robot.at("links").is_array()) {
                throw std::runtime_error("IPC robot table contains invalid links");
            }
            for (const Json& link : robot.at("links")) {
                const std::string path = link.at("path").get<std::string>();
                if (!links_by_path.emplace(
                            path, RobotLinkRecord{&robot, &link}).second) {
                    throw std::runtime_error(
                            "IPC schema v2 contains duplicate Robot3D Link3D paths");
                }
            }
        }

        Json selected = Json::array();
        std::unordered_set<std::string> coupling_paths;
        std::unordered_set<std::string> link_paths;
        std::string previous_coupling_path;
        for (std::size_t proxy_index = 0; proxy_index < couplings.size();
             ++proxy_index) {
            const Json& coupling = couplings.at(proxy_index);
            const std::string coupling_path =
                    coupling.at("coupling_path").get<std::string>();
            const std::string link_path =
                    coupling.at("link_path").get<std::string>();
            if (coupling_path.empty() || link_path.empty() ||
                !coupling_paths.insert(coupling_path).second ||
                !link_paths.insert(link_path).second) {
                throw std::runtime_error(
                        "IPC schema v2 PhysicsCoupling paths must be non-empty and unique");
            }
            if (proxy_index != 0 && previous_coupling_path > coupling_path) {
                throw std::runtime_error(
                        "IPC schema v2 PhysicsCoupling table is not canonically sorted");
            }
            previous_coupling_path = coupling_path;
            if (coupling.at("proxy_index").get<std::size_t>() != proxy_index) {
                throw std::runtime_error(
                        "IPC schema v2 PhysicsCoupling proxy indices must be contiguous");
            }
            const std::string mode = coupling.at("mode").get<std::string>();
            const double force_scale = coupling.at("force_scale").get<double>();
            const double torque_scale = coupling.at("torque_scale").get<double>();
            if ((mode != "OneWay" && mode != "TwoWay") ||
                !std::isfinite(force_scale) || force_scale < 0.0 ||
                !std::isfinite(torque_scale) || torque_scale < 0.0) {
                throw std::runtime_error(
                        "IPC schema v2 PhysicsCoupling mode or wrench scale is invalid");
            }
            const auto found = links_by_path.find(link_path);
            if (found == links_by_path.end()) {
                throw std::runtime_error(
                        "IPC schema v2 PhysicsCoupling references an unknown Link3D path");
            }
            const Json& robot = *found->second.robot;
            const Json& link = *found->second.link;
            if (coupling.at("robot_name").get<std::string>() !=
                        robot.at("name").get<std::string>() ||
                coupling.at("link_name").get<std::string>() !=
                        link.at("name").get<std::string>()) {
                throw std::runtime_error(
                        "IPC schema v2 PhysicsCoupling names do not match its Link3D path");
            }
            const bool has_enabled_collision = std::ranges::any_of(
                    link.at("collision_shapes"), [](const Json& shape) {
                        return !shape.value("disabled", false);
                    });
            if (!has_enabled_collision) {
                throw std::runtime_error(
                        "IPC schema v2 PhysicsCoupling target has no enabled collision shape");
            }
            selected.push_back({
                    {"joints", Json::array()},
                    {"links", Json::array({link})},
                    {"name", robot.at("name")},
                    {"path", robot.at("path")},
                    {"root_link_paths", Json::array({link_path})},
                    {"transform", robot.at("transform")}});
        }
        return selected;
    }

    template <typename ContactElement>
    void BuildDeformables(
            const Json& bodies,
            const ContactElement& contact,
            const std::vector<SubsceneElement>& subscenes) {
        if (!bodies.is_array()) {
            throw std::runtime_error("IPC deformable body table must be an array");
        }
        StableNeoHookean material;
        auto object = scene_->objects().create("gobot_deformables");
        std::size_t global_vertex_offset = 0;
        for (std::size_t environment = 0; environment < environment_count_;
             ++environment) {
            std::size_t environment_vertex_offset = 0;
            for (const Json& body : bodies) {
                const std::string path = body.at("path").get<std::string>();
                const std::string blob_id =
                        body.at("mesh_blob").get<std::string>();
                const IpcSolverArtifactBlobView& blob =
                        FindBlob(blob_id, kTetEncoding, "IPC deformable mesh");
                TetMeshData decoded = DecodeTetMesh(
                        std::span<const std::uint8_t>(blob.data, blob.size));
                const Matrix4x4 transform =
                        ParseTransform(body.at("transform"), path);
                for (Vector3& vertex : decoded.vertices) {
                    vertex = TransformPoint(transform, vertex);
                }
                SimplicialComplex mesh =
                        tetmesh(decoded.vertices, decoded.tetrahedra);
                label_surface(mesh);
                label_triangle_orient(mesh);
                material.apply_to(
                        mesh,
                        ElasticModuli::youngs_poisson(
                                body.at("young_modulus").get<double>(),
                                body.at("poisson_ratio").get<double>()),
                        body.at("density").get<double>());
                contact.apply_to(mesh);
                subscenes[environment].apply_to(mesh);
                auto self_collision = mesh.meta().find<IndexT>(
                        uipc::builtin::self_collision);
                view(*self_collision)[0] =
                        body.value("self_collision", true) ? 1 : 0;
                if (body.value("kinematic", false)) {
                    auto is_fixed = mesh.vertices().find<IndexT>(
                            uipc::builtin::is_fixed);
                    std::ranges::fill(view(*is_fixed), 1);
                }
                auto created = object->geometries().create(mesh);
                if (environment == 0) {
                    deformable_bodies_.push_back(BodyRecord{
                            path, environment_vertex_offset,
                            decoded.vertices.size()});
                }
                if (!external_affine_proxies_) {
                    deformable_contact_ranges_.push_back(
                            DeformableContactRange{
                                    global_vertex_offset,
                                    decoded.vertices.size(), created.geometry});
                }
                initial_fem_positions_.insert(initial_fem_positions_.end(),
                                              decoded.vertices.begin(),
                                              decoded.vertices.end());
                environment_vertex_offset += decoded.vertices.size();
                global_vertex_offset += decoded.vertices.size();
            }
        }
    }

    template <typename ContactElement>
    void BuildAffineBodies(const Json& robots,
                           const ContactElement& contact,
                           const IpcSolverModuleConfig& config,
                           const std::vector<SubsceneElement>& subscenes) {
        if (!robots.is_array()) {
            throw std::runtime_error("IPC robot table must be an array");
        }
        std::size_t global_body_offset = 0;
        for (std::size_t environment = 0; environment < environment_count_;
             ++environment) {
            BuildAffineEnvironment(robots, contact, config,
                                   subscenes[environment], environment,
                                   &global_body_offset);
        }
    }

    template <typename ContactElement>
    void BuildAffineEnvironment(const Json& robots,
                                const ContactElement& contact,
                                const IpcSolverModuleConfig& config,
                                const SubsceneElement& subscene,
                                std::size_t environment,
                                std::size_t* global_body_offset) {
        AffineBodyConstitution affine;
        SoftTransformConstraint constraint;
        std::size_t environment_body_offset = 0;
        std::vector<ContactElementRecord> contact_elements;
        contact_elements.push_back(
                ContactElementRecord{1.0, contact, 0U, 0U, true});
        for (const Json& robot : robots) {
            const std::size_t external_contact_count = contact_elements.size();
            const Json& joints = robot.at("joints");
            const bool articulated =
                    !external_affine_proxies_ && !joints.empty();
            std::unordered_set<std::string> root_paths;
            for (const Json& path : robot.at("root_link_paths")) {
                root_paths.insert(path.get<std::string>());
            }
            std::unordered_map<std::string, S<SimplicialComplexSlot>> link_slots;

            for (const Json& link : robot.at("links")) {
                const std::string path = link.at("path").get<std::string>();
                TriangleMeshData combined;
                double sliding_friction = 0.0;
                bool has_collision_friction = false;
                std::uint32_t contact_type = 0U;
                std::uint32_t contact_affinity = 0U;
                for (const Json& shape : link.at("collision_shapes")) {
                    if (shape.value("disabled", false)) {
                        continue;
                    }
                    const Json& friction = shape.at("friction");
                    if (!friction.is_array() || friction.size() != 3) {
                        throw std::runtime_error(
                                "libuipc collision shape has invalid friction: '" +
                                shape.at("path").get<std::string>() + "'");
                    }
                    const double shape_sliding_friction = friction.at(0).get<double>();
                    if (!std::isfinite(shape_sliding_friction) ||
                        shape_sliding_friction < 0.0) {
                        throw std::runtime_error(
                                "libuipc collision shape has invalid sliding friction: '" +
                                shape.at("path").get<std::string>() + "'");
                    }
                    sliding_friction = std::max(sliding_friction,
                                                shape_sliding_friction);
                    has_collision_friction = true;
                    const std::int64_t shape_contact_type =
                            shape.at("contact_type").get<std::int64_t>();
                    const std::int64_t shape_contact_affinity =
                            shape.at("contact_affinity").get<std::int64_t>();
                    if (shape_contact_type < 0 || shape_contact_affinity < 0 ||
                        shape_contact_type >
                                std::numeric_limits<std::uint32_t>::max() ||
                        shape_contact_affinity >
                                std::numeric_limits<std::uint32_t>::max()) {
                        throw std::runtime_error(
                                "libuipc collision shape has an invalid contact mask: '" +
                                shape.at("path").get<std::string>() + "'");
                    }
                    contact_type |= static_cast<std::uint32_t>(shape_contact_type);
                    contact_affinity |=
                            static_cast<std::uint32_t>(shape_contact_affinity);
                    const std::string type = shape.at("shape_type").get<std::string>();
                    TriangleMeshData primitive;
                    if (type == "box") {
                        primitive = MakeBox(ParseVector3(shape.at("size"), "box size"));
                    } else if (type == "sphere") {
                        primitive = MakeSphere(shape.at("radius").get<double>());
                    } else if (type == "triangle_mesh") {
                        const std::string blob_id =
                                shape.at("mesh_blob").get<std::string>();
                        const IpcSolverArtifactBlobView& blob = FindBlob(
                                blob_id, kTriangleEncoding,
                                "IPC robot triangle mesh");
                        primitive = DecodeTriangleMesh(
                                std::span<const std::uint8_t>(blob.data, blob.size));
                        if (primitive.vertices.size() !=
                                    shape.at("vertex_count").get<std::size_t>() ||
                            primitive.triangles.size() !=
                                    shape.at("triangle_count").get<std::size_t>()) {
                            throw std::runtime_error(
                                    "IPC robot triangle mesh metadata does not match its blob");
                        }
                    } else {
                        throw std::runtime_error(
                                "libuipc Gobot adapter does not yet support robot collision shape '" +
                                type + "'");
                    }
                    AppendShape(&combined, std::move(primitive),
                                ParseTransform(shape.at("link_transform"), "collision transform"));
                }
                if (combined.vertices.empty()) {
                    continue;
                }
                if (!has_collision_friction) {
                    throw std::runtime_error(
                            "libuipc affine body has no enabled collision friction: '" +
                            path + "'");
                }
                SimplicialComplex mesh = trimesh(combined.vertices, combined.triangles);
                label_surface(mesh);

                const double mass = std::max(link.value("mass", 0.0), 1.0e-6);
                const Vector3 center = ParseVector3(
                        link.value("center_of_mass", Json::array({0.0, 0.0, 0.0})),
                        "link center of mass");
                const Vector3 inertia_diagonal = ParseVector3(
                        link.value("inertia_diagonal", Json::array({1.0e-6, 1.0e-6, 1.0e-6})),
                        "link inertia diagonal");
                Matrix3x3 inertia = inertia_diagonal.asDiagonal();
                const Vector3 inertia_off_diagonal = ParseVector3(
                        link.value("inertia_off_diagonal", Json::array({0.0, 0.0, 0.0})),
                        "link inertia off diagonal");
                inertia(0, 1) = inertia(1, 0) = inertia_off_diagonal.x();
                inertia(0, 2) = inertia(2, 0) = inertia_off_diagonal.y();
                inertia(1, 2) = inertia(2, 1) = inertia_off_diagonal.z();
                const auto mass_matrix =
                        uipc::geometry::affine_body::from_rigid_body(mass, center, inertia);
                affine.apply_to(mesh, config.affine_stiffness, mass_matrix,
                                std::max(mass / 1000.0, 1.0e-9));
                if (articulated) {
                    auto fixed = mesh.instances().find<IndexT>(uipc::builtin::is_fixed);
                    view(*fixed)[0] = root_paths.contains(
                            link.at("path").get<std::string>()) ? 1 : 0;
                } else if (!external_affine_proxies_) {
                    constraint.apply_to(
                            mesh, Vector2{config.kinematic_strength,
                                          config.kinematic_strength});
                }
                auto link_contact = scene_->contact_tabular().create(
                        environment_count_ == 1
                                ? path
                                : fmt::format("{}@environment_{}", path,
                                              environment));
                const ContactElementRecord link_contact_record{
                        sliding_friction, link_contact, contact_type,
                        contact_affinity, false};
                for (std::size_t index = 0; index < external_contact_count; ++index) {
                    const ContactElementRecord& other = contact_elements[index];
                    const bool contact_enabled =
                            external_affine_proxies_ && index != 0
                                    ? false
                                    : ContactMasksMatch(link_contact_record,
                                                        other);
                    scene_->contact_tabular().insert(
                            link_contact,
                            other.element,
                            config.friction_coefficient *
                                    std::sqrt(sliding_friction * other.friction),
                            config.contact_resistance,
                            contact_enabled);
                }
                // Match the source robot model's disabled self-collision policy.
                // Adjacent detailed link meshes can overlap at their joints.
                for (std::size_t index = external_contact_count;
                     index < contact_elements.size(); ++index) {
                    scene_->contact_tabular().insert(
                            link_contact, contact_elements[index].element,
                            0.0, 0.0, false);
                }
                scene_->contact_tabular().insert(
                        link_contact, link_contact, 0.0, 0.0, false);
                link_contact.apply_to(mesh);
                subscene.apply_to(mesh);
                contact_elements.push_back(link_contact_record);
                const Matrix4x4 initial = ParseTransform(link.at("transform"), "link transform");
                view(mesh.transforms())[0] = initial;

                if (environment == 0) {
                    affine_bodies_.push_back(
                            BodyRecord{path, environment_body_offset, 1});
                }
                initial_affine_transforms_.push_back(initial);
                ++environment_body_offset;
                ++(*global_body_offset);

                auto object = scene_->objects().create(
                        environment_count_ == 1
                                ? path
                                : fmt::format("{}@environment_{}", path,
                                              environment));
                auto created = object->geometries().create(mesh);
                link_slots.emplace(path, created.geometry);

                if (!articulated && !external_affine_proxies_) {
                    auto target = std::make_shared<AffineTarget>();
                    target->initial = initial;
                    target->value = initial;
                    affine_targets_.emplace(path, target);
                    scene_->animator().insert(
                            *object,
                            [target](uipc::core::Animation::UpdateInfo& info) {
                                auto geometry =
                                        info.geo_slots()[0]->geometry().as<SimplicialComplex>();
                                auto constrained = geometry->instances().find<IndexT>(
                                        uipc::builtin::is_constrained);
                                auto aim = geometry->instances().find<Matrix4x4>(
                                        uipc::builtin::aim_transform);
                                view(*constrained)[0] = 1;
                                view(*aim)[0] = target->value;
                            });
                }
            }

            if (external_affine_proxies_) {
                continue;
            }

            for (const Json& joint : joints) {
                const std::string path = joint.at("path").get<std::string>();
                const std::string parent_path =
                        joint.at("parent_link_path").get<std::string>();
                const std::string child_path =
                        joint.at("child_link_path").get<std::string>();
                const auto parent = link_slots.find(parent_path);
                const auto child = link_slots.find(child_path);
                if (parent == link_slots.end() || child == link_slots.end()) {
                    throw std::runtime_error(
                            "libuipc articulated joints require collision geometry on both links: '" +
                            path + "'");
                }

                const Matrix4x4 transform =
                        ParseTransform(joint.at("transform"), "joint transform");
                Vector3 axis = transform.block<3, 3>(0, 0) *
                               ParseVector3(joint.at("axis"), "joint axis");
                if (!axis.allFinite() || axis.squaredNorm() <= 1.0e-12) {
                    throw std::runtime_error("libuipc joint has an invalid world axis: '" +
                                             path + "'");
                }
                axis.normalize();
                const Vector3 origin = transform.block<3, 1>(0, 3);
                const std::array<Vector3, 1> position0{origin - axis * 0.05};
                const std::array<Vector3, 1> position1{origin + axis * 0.05};
                std::array<S<SimplicialComplexSlot>, 1> parent_slots{
                        parent->second};
                std::array<S<SimplicialComplexSlot>, 1> child_slots{
                        child->second};
                std::array<IndexT, 1> instance_ids{0};
                std::array<Float, 1> strengths{
                        static_cast<Float>(config.kinematic_strength)};

                std::string state_attribute;
                std::string aim_attribute;
                const int joint_type = joint.at("joint_type").get<int>();
                SimplicialComplex joint_geometry = [&] {
                    if (joint_type == 1 || joint_type == 2) {
                        AffineBodyRevoluteJoint revolute;
                        auto geometry = revolute.create_geometry(
                                position0, position1, parent_slots, instance_ids,
                                child_slots, instance_ids, strengths);
                        AffineBodyDrivingRevoluteJoint driving;
                        driving.apply_to(geometry, strengths);
                        state_attribute = "angle";
                        aim_attribute = "aim_angle";
                        return geometry;
                    }
                    if (joint_type == 3) {
                        AffineBodyPrismaticJoint prismatic;
                        auto geometry = prismatic.create_geometry(
                                position0, position1, parent_slots, instance_ids,
                                child_slots, instance_ids, strengths);
                        AffineBodyDrivingPrismaticJoint driving;
                        driving.apply_to(geometry, strengths);
                        state_attribute = "distance";
                        aim_attribute = "aim_distance";
                        return geometry;
                    }
                    throw std::runtime_error(
                            "libuipc Gobot adapter supports only revolute, continuous, and "
                            "prismatic robot joints; joint '" + path + "' is unsupported");
                }();

                const double initial = joint.value("joint_position", 0.0);
                auto state = joint_geometry.edges().find<Float>(state_attribute);
                auto initial_state = joint_geometry.edges().find<Float>(
                        state_attribute == "angle" ? "init_angle" : "init_distance");
                auto aim = joint_geometry.edges().find<Float>(aim_attribute);
                auto constrained = joint_geometry.edges().find<IndexT>(
                        "driving/is_constrained");
                auto passive = joint_geometry.edges().find<IndexT>("is_passive");
                view(*state)[0] = initial;
                view(*initial_state)[0] = initial;
                view(*aim)[0] = initial;
                view(*constrained)[0] = 1;
                view(*passive)[0] = 0;

                auto target = std::make_shared<JointTarget>();
                target->initial = initial;
                target->value = initial;
                target->state_attribute = state_attribute;
                target->aim_attribute = aim_attribute;
                auto object = scene_->objects().create(path);
                auto created = object->geometries().create(joint_geometry);
                target->geometry = created.geometry;
                if (!joint_targets_.emplace(path, target).second) {
                    throw std::runtime_error("libuipc scene contains duplicate joint path '" +
                                             path + "'");
                }
                scene_->animator().insert(
                        *object,
                        [target](uipc::core::Animation::UpdateInfo& info) {
                            auto geometry =
                                    info.geo_slots()[0]->geometry().as<SimplicialComplex>();
                            auto constrained = geometry->edges().find<IndexT>(
                                    "driving/is_constrained");
                            auto passive = geometry->edges().find<IndexT>("is_passive");
                            auto aim = geometry->edges().find<Float>(target->aim_attribute);
                            view(*constrained)[0] = 1;
                            view(*passive)[0] = 0;
                            view(*aim)[0] = target->value;
                        });
            }
        }
    }

    void InitializeAccessors() {
        fem_accessor_ = world_->features().find<FiniteElementStateAccessorFeature>();
        if (fem_accessor_ == nullptr) {
            throw std::runtime_error("libuipc CUDA backend has no FEM state accessor");
        }
        fem_state_ = std::make_unique<SimplicialComplex>(fem_accessor_->create_geometry());
        fem_state_->vertices().create<Vector3>(uipc::builtin::position);
        fem_state_->vertices().create<Vector3>(uipc::builtin::velocity);

        if (!affine_bodies_.empty()) {
            affine_accessor_ = world_->features().find<AffineBodyStateAccessorFeature>();
            if (affine_accessor_ == nullptr) {
                throw std::runtime_error("libuipc CUDA backend has no affine-body state accessor");
            }
            affine_state_ = std::make_unique<SimplicialComplex>(
                    affine_accessor_->create_geometry());
            affine_state_->instances().create<Matrix4x4>(uipc::builtin::transform);
            affine_state_->instances().create<Matrix4x4>(uipc::builtin::velocity);
        }
    }

    void InitializeContactForceExport(double fixed_time_step) {
        contact_forces_.assign(initial_fem_positions_.size() * 3, 0.0);
        inverse_time_step_squared_ =
                1.0 / (fixed_time_step * fixed_time_step);

        contact_system_ = world_->features().find<ContactSystemFeature>();
        if (contact_system_ == nullptr) {
            throw std::runtime_error(
                    "libuipc CUDA backend has no contact-system feature");
        }

        for (DeformableContactRange& range : deformable_contact_ranges_) {
            auto global_offset = range.geometry->geometry().meta().find<IndexT>(
                    uipc::builtin::global_vertex_offset);
            if (global_offset == nullptr) {
                throw std::runtime_error(
                        "libuipc deformable geometry has no global vertex offset");
            }
            const auto offsets = view(*global_offset);
            if (offsets.size() != 1 || offsets[0] < 0) {
                throw std::runtime_error(
                        "libuipc deformable geometry has an invalid global vertex offset");
            }
            range.global_vertex_offset = offsets[0];
        }

        constexpr std::array<std::string_view, 8> kSimplexContactTypes{
                "PT+N", "EE+N", "PE+N", "PP+N",
                "PT+F", "EE+F", "PE+F", "PP+F"};
        for (const std::string& primitive_type :
             contact_system_->contact_primitive_types()) {
            if (std::ranges::find(kSimplexContactTypes, primitive_type) ==
                kSimplexContactTypes.end()) {
                continue;
            }
            contact_gradient_buffers_.push_back(ContactGradientBuffer{
                    primitive_type, std::make_unique<Geometry>()});
        }
        if (contact_gradient_buffers_.empty()) {
            throw std::runtime_error(
                    "libuipc CUDA backend has no simplex contact-gradient exporters");
        }
    }

    std::optional<std::size_t> FindDeformableVertex(
            IndexT global_vertex) const {
        for (const DeformableContactRange& range : deformable_contact_ranges_) {
            if (global_vertex < range.global_vertex_offset) {
                continue;
            }
            const IndexT relative = global_vertex - range.global_vertex_offset;
            if (static_cast<std::size_t>(relative) < range.vertex_count) {
                return range.output_offset + static_cast<std::size_t>(relative);
            }
        }
        return std::nullopt;
    }

    void RefreshContactForces() {
        std::ranges::fill(contact_forces_, 0.0);
        for (ContactGradientBuffer& buffer : contact_gradient_buffers_) {
            contact_system_->contact_gradient(buffer.primitive_type,
                                               *buffer.geometry);
            auto indices = buffer.geometry->instances().find<IndexT>("i");
            auto gradients = buffer.geometry->instances().find<Vector3>("grad");
            if (indices == nullptr || gradients == nullptr) {
                throw std::runtime_error(
                        "libuipc contact-gradient exporter returned incomplete data");
            }
            const auto index_values = view(*indices);
            const auto gradient_values = view(*gradients);
            if (index_values.size() != gradient_values.size()) {
                throw std::runtime_error(
                        "libuipc contact-gradient exporter returned inconsistent data");
            }
            for (std::size_t index = 0; index < index_values.size(); ++index) {
                const auto output_vertex = FindDeformableVertex(index_values[index]);
                if (!output_vertex.has_value()) {
                    continue;
                }
                // libuipc assembles dt^2 times the physical IPC potential.
                const Vector3 force =
                        -gradient_values[index] * inverse_time_step_squared_;
                if (!force.allFinite()) {
                    throw std::runtime_error(
                            "libuipc contact-gradient exporter returned a non-finite force");
                }
                for (std::size_t axis = 0; axis < 3; ++axis) {
                    double& value = contact_forces_[*output_vertex * 3 + axis];
                    value += force[static_cast<Eigen::Index>(axis)];
                    if (!std::isfinite(value)) {
                        throw std::runtime_error(
                                "libuipc accumulated a non-finite deformable contact force");
                    }
                }
            }
        }
    }

    void Refresh(bool refresh_contact_forces) {
        fem_accessor_->copy_to(*fem_state_);
        const auto positions = view(*fem_state_->vertices().find<Vector3>(uipc::builtin::position));
        const auto velocities = view(*fem_state_->vertices().find<Vector3>(uipc::builtin::velocity));
        positions_.resize(positions.size() * 3);
        velocities_.resize(velocities.size() * 3);
        for (std::size_t index = 0; index < positions.size(); ++index) {
            for (std::size_t axis = 0; axis < 3; ++axis) {
                positions_[index * 3 + axis] = positions[index][static_cast<Eigen::Index>(axis)];
                velocities_[index * 3 + axis] = velocities[index][static_cast<Eigen::Index>(axis)];
            }
        }
        if (affine_accessor_ != nullptr) {
            affine_accessor_->copy_to(*affine_state_);
            const auto transforms = view(
                    *affine_state_->instances().find<Matrix4x4>(uipc::builtin::transform));
            affine_transforms_.resize(transforms.size() * 16);
            for (std::size_t body = 0; body < transforms.size(); ++body) {
                for (std::size_t index = 0; index < 16; ++index) {
                    affine_transforms_[body * 16 + index] =
                            transforms[body](static_cast<Eigen::Index>(index / 4),
                                             static_cast<Eigen::Index>(index % 4));
                }
            }
        }
        if (refresh_contact_forces) {
            RefreshContactForces();
        } else {
            std::ranges::fill(contact_forces_, 0.0);
        }
    }

    std::unordered_map<std::string, IpcSolverArtifactBlobView> blobs_;
    std::unique_ptr<Engine> engine_;
    std::unique_ptr<World> world_;
    std::unique_ptr<Scene> scene_;
    std::shared_ptr<FiniteElementStateAccessorFeature> fem_accessor_;
    std::shared_ptr<AffineBodyStateAccessorFeature> affine_accessor_;
    std::shared_ptr<ContactSystemFeature> contact_system_;
    std::unique_ptr<SimplicialComplex> fem_state_;
    std::unique_ptr<SimplicialComplex> affine_state_;
    std::vector<BodyRecord> deformable_bodies_;
    std::vector<DeformableContactRange> deformable_contact_ranges_;
    std::vector<BodyRecord> affine_bodies_;
    std::vector<ContactGradientBuffer> contact_gradient_buffers_;
    std::unordered_map<std::string, std::shared_ptr<AffineTarget>> affine_targets_;
    std::unordered_map<std::string, std::shared_ptr<JointTarget>> joint_targets_;
    std::vector<Vector3> initial_fem_positions_;
    std::vector<Matrix4x4> initial_affine_transforms_;
    std::vector<double> positions_;
    std::vector<double> velocities_;
    std::vector<double> contact_forces_;
    std::vector<double> affine_transforms_;
    std::vector<double> target_row_major_;
    IpcBatchSolverModuleBuffers device_buffers_{};
    double inverse_time_step_squared_{0.0};
    std::uint64_t frame_{0};
    double last_step_latency_ms_{0.0};
    std::size_t environment_count_{1};
    std::uint32_t device_index_{0};
    bool external_affine_proxies_{false};
    bool device_buffers_bound_{false};
    bool initial_state_dumped_{false};
};

IpcSolverDeviceBufferView OffsetDeviceBuffer(
        IpcSolverDeviceBufferView buffer,
        std::size_t scalar_offset,
        std::size_t local_environment_count) {
    if (buffer.data != nullptr) {
        buffer.data = static_cast<double*>(buffer.data) + scalar_offset;
    }
    if (buffer.rank != 0) {
        buffer.shape[0] = local_environment_count;
    }
    return buffer;
}

class BatchSession final {
public:
    BatchSession(const IpcSolverArtifactView& artifact,
                 const IpcBatchSolverModuleConfig& config)
        : environment_count_(config.environment_count),
          environments_per_shard_(config.environments_per_shard),
          workspace_suffix_(MakeBatchWorkspaceSuffix()) {
        if (environment_count_ == 0 || environments_per_shard_ == 0 ||
            environment_count_ % environments_per_shard_ != 0) {
            throw std::runtime_error(
                    "libuipc batch environment count must be a positive "
                    "multiple of environments_per_shard");
        }
        if (!config.external_affine_proxies) {
            throw std::runtime_error(
                    "libuipc batch v1 requires external affine proxies");
        }
        const std::filesystem::path workspace_root =
                config.solver.workspace != nullptr &&
                                config.solver.workspace[0] != '\0'
                        ? std::filesystem::path(config.solver.workspace)
                        : std::filesystem::temp_directory_path() /
                                  "gobot-libuipc";
        std::error_code workspace_error;
        std::filesystem::create_directories(workspace_root, workspace_error);
        if (workspace_error) {
            throw std::runtime_error(
                    "cannot create libuipc batch workspace root '" +
                    workspace_root.string() + "': " +
                    workspace_error.message());
        }
        workspace_lease_.path = workspace_root / workspace_suffix_;
        const bool workspace_created = std::filesystem::create_directory(
                workspace_lease_.path, workspace_error);
        if (workspace_error || !workspace_created) {
            throw std::runtime_error(
                    "cannot create an exclusive libuipc batch workspace '" +
                    workspace_lease_.path.string() + "'" +
                    (workspace_error ? ": " + workspace_error.message() : ""));
        }
        workspace_lease_.owned = true;
        const std::size_t shard_count =
                environment_count_ / environments_per_shard_;
        shards_.reserve(shard_count);
        for (std::size_t shard = 0; shard < shard_count; ++shard) {
            shards_.push_back(std::make_unique<Session>(
                    artifact, config.solver, environments_per_shard_, true,
                    (std::filesystem::path(workspace_suffix_) /
                     fmt::format("shard_{}", shard))
                            .string()));
        }
        const auto& first_deformables = shards_.front()->DeformableBodies();
        const auto& first_affines = shards_.front()->AffineBodies();
        deformable_bodies_ = first_deformables;
        affine_bodies_ = first_affines;
        for (const auto& shard : shards_) {
            if (shard->DeformableBodies().size() !=
                        deformable_bodies_.size() ||
                shard->AffineBodies().size() != affine_bodies_.size()) {
                throw std::runtime_error(
                        "libuipc batch shards produced inconsistent layouts");
            }
        }
        for (const BodyRecord& body : deformable_bodies_) {
            deformable_vertex_count_per_environment_ += body.count;
        }
    }

    void BindDeviceBuffers(const IpcBatchSolverModuleBuffers& buffers) {
        const std::size_t vertices =
                deformable_vertex_count_per_environment_;
        const std::size_t affines = affine_bodies_.size();
        for (std::size_t shard = 0; shard < shards_.size(); ++shard) {
            const std::size_t environment_offset =
                    shard * environments_per_shard_;
            IpcBatchSolverModuleBuffers local = buffers;
            local.deformable_positions = OffsetDeviceBuffer(
                    buffers.deformable_positions,
                    environment_offset * vertices * 3,
                    environments_per_shard_);
            local.deformable_velocities = OffsetDeviceBuffer(
                    buffers.deformable_velocities,
                    environment_offset * vertices * 3,
                    environments_per_shard_);
            local.deformable_contact_forces = OffsetDeviceBuffer(
                    buffers.deformable_contact_forces,
                    environment_offset * vertices * 3,
                    environments_per_shard_);
            local.affine_targets = OffsetDeviceBuffer(
                    buffers.affine_targets,
                    environment_offset * affines * 16,
                    environments_per_shard_);
            local.affine_transforms = OffsetDeviceBuffer(
                    buffers.affine_transforms,
                    environment_offset * affines * 16,
                    environments_per_shard_);
            local.affine_contact_wrenches = OffsetDeviceBuffer(
                    buffers.affine_contact_wrenches,
                    environment_offset * affines * 6,
                    environments_per_shard_);
            shards_[shard]->BindDeviceBuffers(local);
        }
        buffers_bound_ = true;
    }

    void Step(std::uint32_t steps) {
        if (!buffers_bound_) {
            throw std::runtime_error(
                    "libuipc batch device buffers are not bound");
        }
        const auto start = std::chrono::steady_clock::now();
        for (const auto& shard : shards_) {
            shard->StepDevice(steps);
        }
        frame_ += steps;
        const auto end = std::chrono::steady_clock::now();
        last_step_latency_ms_ =
                std::chrono::duration<double, std::milli>(end - start).count();
    }

    void Reset() {
        if (!buffers_bound_) {
            throw std::runtime_error(
                    "libuipc batch device buffers are not bound");
        }
        for (const auto& shard : shards_) {
            shard->ResetDevice();
        }
        frame_ = 0;
    }

    void Synchronize() {
        for (const auto& shard : shards_) {
            shard->SynchronizeDevice();
        }
        RequireCuda(cudaDeviceSynchronize(),
                    "synchronizing libuipc batch shards");
    }

    const std::vector<BodyRecord>& DeformableBodies() const {
        return deformable_bodies_;
    }
    const std::vector<BodyRecord>& AffineBodies() const {
        return affine_bodies_;
    }
    std::size_t EnvironmentCount() const { return environment_count_; }
    std::size_t ShardCount() const { return shards_.size(); }
    std::size_t DeformableVertexCountPerEnvironment() const {
        return deformable_vertex_count_per_environment_;
    }
    std::uint64_t Frame() const { return frame_; }
    double LastStepLatencyMs() const { return last_step_latency_ms_; }
    bool IsValid() const {
        return std::ranges::all_of(
                shards_, [](const auto& shard) { return shard->IsValid(); });
    }

private:
    static void RequireCuda(cudaError_t result, std::string_view operation) {
        if (result != cudaSuccess) {
            throw std::runtime_error(
                    std::string(operation) + ": " +
                    cudaGetErrorString(result));
        }
    }

    std::size_t environment_count_{0};
    std::size_t environments_per_shard_{0};
    std::size_t deformable_vertex_count_per_environment_{0};
    std::string workspace_suffix_;
    WorkspaceLease workspace_lease_;
    std::vector<std::unique_ptr<Session>> shards_;
    std::vector<BodyRecord> deformable_bodies_;
    std::vector<BodyRecord> affine_bodies_;
    std::uint64_t frame_{0};
    double last_step_latency_ms_{0.0};
    bool buffers_bound_{false};
};

Session* Cast(void* session) {
    if (session == nullptr) {
        throw std::runtime_error("libuipc solver session is null");
    }
    return static_cast<Session*>(session);
}

void* Create(const IpcSolverArtifactView* artifact,
             const IpcSolverModuleConfig* config,
             char* error,
             std::size_t error_size) {
    if (artifact == nullptr || config == nullptr || artifact->manifest == nullptr) {
        WriteError(error, error_size, "libuipc solver requires an artifact and configuration");
        return nullptr;
    }
    try {
        return new Session(*artifact, *config);
    } catch (const std::exception& exception) {
        WriteError(error, error_size, exception.what());
        return nullptr;
    }
}

void Destroy(void* session) {
    delete static_cast<Session*>(session);
}

bool Step(void* session, std::uint32_t steps, char* error, std::size_t error_size) {
    return Guard(error, error_size, [&] { Cast(session)->Step(steps); });
}

bool Reset(void* session, char* error, std::size_t error_size) {
    return Guard(error, error_size, [&] { Cast(session)->Reset(); });
}

std::size_t DeformableBodyCount(void* session) {
    try {
        return Cast(session)->DeformableBodies().size();
    } catch (...) {
        return 0;
    }
}

bool BodyInfo(const std::vector<BodyRecord>& bodies,
              std::size_t index,
              IpcSolverModuleBodyInfo* info,
              char* error,
              std::size_t error_size) {
    return Guard(error, error_size, [&] {
        if (info == nullptr || index >= bodies.size()) {
            throw std::runtime_error("libuipc body index is out of range");
        }
        const BodyRecord& body = bodies[index];
        *info = IpcSolverModuleBodyInfo{body.path.c_str(), body.offset, body.count};
    });
}

bool DeformableBodyInfo(void* session,
                        std::size_t index,
                        IpcSolverModuleBodyInfo* info,
                        char* error,
                        std::size_t error_size) {
    try {
        return BodyInfo(Cast(session)->DeformableBodies(), index, info, error, error_size);
    } catch (const std::exception& exception) {
        WriteError(error, error_size, exception.what());
        return false;
    }
}

bool CopyVector(const std::vector<double>& source,
                double* destination,
                std::size_t scalar_count,
                char* error,
                std::size_t error_size) {
    return Guard(error, error_size, [&] {
        if (scalar_count != source.size() ||
            (destination == nullptr && scalar_count != 0)) {
            throw std::runtime_error("libuipc state destination has the wrong size");
        }
        if (!source.empty()) {
            std::ranges::copy(source, destination);
        }
    });
}

bool CopyDeformablePositions(void* session,
                             double* destination,
                             std::size_t scalar_count,
                             char* error,
                             std::size_t error_size) {
    try {
        return CopyVector(Cast(session)->Positions(), destination, scalar_count, error, error_size);
    } catch (const std::exception& exception) {
        WriteError(error, error_size, exception.what());
        return false;
    }
}

bool CopyDeformableVelocities(void* session,
                              double* destination,
                              std::size_t scalar_count,
                              char* error,
                              std::size_t error_size) {
    try {
        return CopyVector(Cast(session)->Velocities(), destination, scalar_count, error, error_size);
    } catch (const std::exception& exception) {
        WriteError(error, error_size, exception.what());
        return false;
    }
}

bool CopyDeformableContactForces(void* session,
                                 double* destination,
                                 std::size_t scalar_count,
                                 char* error,
                                 std::size_t error_size) {
    try {
        return CopyVector(Cast(session)->ContactForces(), destination,
                          scalar_count, error, error_size);
    } catch (const std::exception& exception) {
        WriteError(error, error_size, exception.what());
        return false;
    }
}

std::size_t AffineBodyCount(void* session) {
    try {
        return Cast(session)->AffineBodies().size();
    } catch (...) {
        return 0;
    }
}

bool AffineBodyInfo(void* session,
                    std::size_t index,
                    IpcSolverModuleBodyInfo* info,
                    char* error,
                    std::size_t error_size) {
    try {
        return BodyInfo(Cast(session)->AffineBodies(), index, info, error, error_size);
    } catch (const std::exception& exception) {
        WriteError(error, error_size, exception.what());
        return false;
    }
}

bool CopyAffineTransforms(void* session,
                          double* destination,
                          std::size_t scalar_count,
                          char* error,
                          std::size_t error_size) {
    try {
        return CopyVector(Cast(session)->AffineTransforms(), destination,
                          scalar_count, error, error_size);
    } catch (const std::exception& exception) {
        WriteError(error, error_size, exception.what());
        return false;
    }
}

bool SetAffineTarget(void* session,
                     const char* path,
                     const double* transform,
                     char* error,
                     std::size_t error_size) {
    return Guard(error, error_size, [&] {
        if (path == nullptr || transform == nullptr) {
            throw std::runtime_error("libuipc affine target arguments are null");
        }
        Cast(session)->SetAffineTarget(path, transform);
    });
}

bool SetJointTarget(void* session,
                    const char* path,
                    double position,
                    char* error,
                    std::size_t error_size) {
    return Guard(error, error_size, [&] {
        if (path == nullptr) {
            throw std::runtime_error("libuipc joint target path is null");
        }
        Cast(session)->SetJointTarget(path, position);
    });
}

bool Diagnostics(void* session,
                 IpcSolverModuleDiagnostics* diagnostics,
                 char* error,
                 std::size_t error_size) {
    return Guard(error, error_size, [&] {
        if (diagnostics == nullptr) {
            throw std::runtime_error("libuipc diagnostics output is null");
        }
        Session* value = Cast(session);
        std::size_t vertex_count = 0;
        for (const BodyRecord& body : value->DeformableBodies()) {
            vertex_count += body.count;
        }
        *diagnostics = IpcSolverModuleDiagnostics{
                value->Frame(), value->DeformableBodies().size(), vertex_count,
                value->AffineBodies().size(), value->LastStepLatencyMs(), value->IsValid()};
    });
}

BatchSession* CastBatch(void* session) {
    if (session == nullptr) {
        throw std::runtime_error("libuipc batch solver session is null");
    }
    return static_cast<BatchSession*>(session);
}

void* BatchCreate(const IpcSolverArtifactView* artifact,
                  const IpcBatchSolverModuleConfig* config,
                  char* error,
                  std::size_t error_size) {
    if (artifact == nullptr || config == nullptr ||
        artifact->manifest == nullptr) {
        WriteError(error, error_size,
                   "libuipc batch solver requires an artifact and configuration");
        return nullptr;
    }
    try {
        return new BatchSession(*artifact, *config);
    } catch (const std::exception& exception) {
        WriteError(error, error_size, exception.what());
        return nullptr;
    }
}

void BatchDestroy(void* session) {
    delete static_cast<BatchSession*>(session);
}

bool BatchBindDeviceBuffers(void* session,
                            const IpcBatchSolverModuleBuffers* buffers,
                            char* error,
                            std::size_t error_size) {
    return Guard(error, error_size, [&] {
        if (buffers == nullptr) {
            throw std::runtime_error(
                    "libuipc batch device buffer table is null");
        }
        CastBatch(session)->BindDeviceBuffers(*buffers);
    });
}

bool BatchStep(void* session,
               std::uint32_t steps,
               char* error,
               std::size_t error_size) {
    return Guard(error, error_size,
                 [&] { CastBatch(session)->Step(steps); });
}

bool BatchResetFull(void* session, char* error, std::size_t error_size) {
    return Guard(error, error_size,
                 [&] { CastBatch(session)->Reset(); });
}

bool BatchSynchronize(void* session, char* error, std::size_t error_size) {
    return Guard(error, error_size,
                 [&] { CastBatch(session)->Synchronize(); });
}

std::size_t BatchDeformableBodyCount(void* session) {
    try {
        return CastBatch(session)->DeformableBodies().size();
    } catch (...) {
        return 0;
    }
}

bool BatchDeformableBodyInfo(void* session,
                             std::size_t index,
                             IpcSolverModuleBodyInfo* info,
                             char* error,
                             std::size_t error_size) {
    try {
        return BodyInfo(CastBatch(session)->DeformableBodies(), index, info,
                        error, error_size);
    } catch (const std::exception& exception) {
        WriteError(error, error_size, exception.what());
        return false;
    }
}

std::size_t BatchAffineBodyCount(void* session) {
    try {
        return CastBatch(session)->AffineBodies().size();
    } catch (...) {
        return 0;
    }
}

bool BatchAffineBodyInfo(void* session,
                         std::size_t index,
                         IpcSolverModuleBodyInfo* info,
                         char* error,
                         std::size_t error_size) {
    try {
        return BodyInfo(CastBatch(session)->AffineBodies(), index, info,
                        error, error_size);
    } catch (const std::exception& exception) {
        WriteError(error, error_size, exception.what());
        return false;
    }
}

bool BatchDiagnostics(void* session,
                      IpcBatchSolverModuleDiagnostics* diagnostics,
                      char* error,
                      std::size_t error_size) {
    return Guard(error, error_size, [&] {
        if (diagnostics == nullptr) {
            throw std::runtime_error(
                    "libuipc batch diagnostics output is null");
        }
        BatchSession* value = CastBatch(session);
        *diagnostics = IpcBatchSolverModuleDiagnostics{
                value->Frame(),
                value->EnvironmentCount(),
                value->ShardCount(),
                value->DeformableBodies().size(),
                value->DeformableVertexCountPerEnvironment(),
                value->AffineBodies().size(),
                value->LastStepLatencyMs(),
                value->IsValid()};
    });
}

const IpcSolverModuleApi kApi{
        GOBOT_IPC_SOLVER_MODULE_ABI_VERSION,
        "libuipc",
        &Create,
        &Destroy,
        &Step,
        &Reset,
        &DeformableBodyCount,
        &DeformableBodyInfo,
        &CopyDeformablePositions,
        &CopyDeformableVelocities,
        &CopyDeformableContactForces,
        &AffineBodyCount,
        &AffineBodyInfo,
        &CopyAffineTransforms,
        &SetAffineTarget,
        &SetJointTarget,
        &Diagnostics};

const IpcBatchSolverModuleApi kBatchApi{
        GOBOT_IPC_BATCH_SOLVER_MODULE_ABI_VERSION,
        "libuipc-batch",
        &BatchCreate,
        &BatchDestroy,
        &BatchBindDeviceBuffers,
        &BatchStep,
        &BatchResetFull,
        &BatchSynchronize,
        &BatchDeformableBodyCount,
        &BatchDeformableBodyInfo,
        &BatchAffineBodyCount,
        &BatchAffineBodyInfo,
        &BatchDiagnostics};

} // namespace

extern "C" __attribute__((visibility("default")))
const IpcSolverModuleApi* gobot_ipc_solver_get_api() {
    return &kApi;
}

extern "C" __attribute__((visibility("default")))
const IpcBatchSolverModuleApi* gobot_ipc_solver_get_batch_api() {
    return &kBatchApi;
}

} // namespace gobot::libuipc_solver
