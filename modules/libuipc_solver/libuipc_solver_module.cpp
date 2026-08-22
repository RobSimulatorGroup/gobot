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
#include "device_coupling_workspace.hpp"

#include <uipc/uipc.h>
#include <uipc/builtin/attribute_name.h>
#include <uipc/constitution/affine_body_constitution.h>
#include <uipc/constitution/affine_body_driving_prismatic_joint.h>
#include <uipc/constitution/affine_body_driving_revolute_joint.h>
#include <uipc/constitution/affine_body_prismatic_joint.h>
#include <uipc/constitution/affine_body_revolute_joint.h>
#include <uipc/constitution/elastic_moduli.h>
#include <uipc/constitution/discrete_shell_bending.h>
#include <uipc/constitution/finite_element_external_force.h>
#include <uipc/constitution/strain_limiting_baraff_witkin.h>
#include <uipc/constitution/soft_position_constraint.h>
#include <uipc/constitution/soft_transform_constraint.h>
#include <uipc/constitution/stable_neo_hookean.h>
#include <uipc/core/affine_body_state_accessor_feature.h>
#include <uipc/core/contact_system_feature.h>
#include <uipc/core/finite_element_state_accessor_feature.h>
#include <uipc/core/finite_element_external_force_accessor_feature.h>
#include <uipc/core/soft_position_constraint_accessor_feature.h>
#include <uipc/core/soft_transform_constraint_accessor_feature.h>
#include <uipc/core/solver_control_feature.h>
#include <uipc/core/solver_diagnostics_feature.h>
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
#include <cstdlib>
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
using uipc::constitution::ElasticModuli2D;
using uipc::constitution::DiscreteShellBending;
using uipc::constitution::FiniteElementExternalForce;
using uipc::constitution::SoftPositionConstraint;
using uipc::constitution::SoftTransformConstraint;
using uipc::constitution::StableNeoHookean;
using uipc::constitution::StrainLimitingBaraffWitkinShell;
using uipc::core::AffineBodyStateAccessorFeature;
using uipc::core::ContactSystemFeature;
using uipc::core::ContactElement;
using uipc::core::Engine;
using uipc::core::FiniteElementStateAccessorFeature;
using uipc::core::FiniteElementExternalForceAccessorFeature;
using uipc::core::Scene;
using uipc::core::SoftPositionConstraintAccessorFeature;
using uipc::core::SoftTransformConstraintAccessorFeature;
using uipc::core::SolverControlFeature;
using uipc::core::SolverDiagnostics;
using uipc::core::SolverDiagnosticsFeature;
using uipc::core::SolverRuntimeOptions;
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

static_assert(sizeof(IndexT) == sizeof(std::int32_t));
static_assert(sizeof(Vector3) == 3 * sizeof(double));
static_assert(sizeof(Matrix4x4) == 16 * sizeof(double));

constexpr std::string_view kTetEncoding = "gobot.tetrahedral-mesh.le.v1";
constexpr std::string_view kTetMagic = "GOBTIPC1";
constexpr std::string_view kTriangleEncoding = "gobot.triangle-mesh.le.v1";

using SteadyClock = std::chrono::steady_clock;

double ElapsedMilliseconds(SteadyClock::time_point start) {
    return std::chrono::duration<double, std::milli>(
                   SteadyClock::now() - start)
            .count();
}

bool HasOutputFlag(std::uint32_t flags,
                   IpcBatchSolverOutputFlag flag) {
    return (flags & static_cast<std::uint32_t>(flag)) != 0;
}

bool DeviceNativeCouplingEnabled() {
    const char* value = std::getenv("GOBOT_LIBUIPC_DEVICE_NATIVE_COUPLING");
    if (value == nullptr) {
        return true;
    }
    const std::string_view text{value};
    return text != "0" && text != "false" && text != "off";
}

void ValidateOutputFlags(std::uint32_t flags) {
    if ((flags & ~IpcBatchSolverOutputAll) != 0) {
        throw std::runtime_error(
                "libuipc batch output flags contain unknown bits");
    }
}

class PinnedDoubleBuffer final {
public:
    ~PinnedDoubleBuffer() {
        if (data_ != nullptr) {
            cudaFreeHost(data_);
        }
    }

    PinnedDoubleBuffer(const PinnedDoubleBuffer&) = delete;
    PinnedDoubleBuffer& operator=(const PinnedDoubleBuffer&) = delete;
    PinnedDoubleBuffer() = default;

    void resize(std::size_t size) {
        if (size == size_) {
            return;
        }
        if (data_ != nullptr) {
            cudaFreeHost(data_);
            data_ = nullptr;
            size_ = 0;
        }
        if (size == 0) {
            return;
        }
        const cudaError_t result = cudaHostAlloc(
                reinterpret_cast<void**>(&data_), size * sizeof(double),
                cudaHostAllocPortable);
        if (result != cudaSuccess) {
            throw std::runtime_error(
                    std::string("allocating pinned libuipc staging: ") +
                    cudaGetErrorString(result));
        }
        size_ = size;
    }

    double* data() { return data_; }
    const double* data() const { return data_; }
    std::size_t size() const { return size_; }
    double& operator[](std::size_t index) { return data_[index]; }
    const double& operator[](std::size_t index) const { return data_[index]; }

private:
    double* data_{nullptr};
    std::size_t size_{0};
};
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

struct AffineContactRange {
    std::size_t output_offset{0};
    std::size_t vertex_count{0};
    S<SimplicialComplexSlot> geometry;
    IndexT global_vertex_offset{-1};
    std::vector<Vector3> local_vertices;
    Vector3 local_center_of_mass{Vector3::Zero()};
};

struct DeformableAttachmentSpec {
    std::string path;
    std::string deformable_path;
    std::string rigid_link_path;
    std::size_t proxy_index{0};
    double strength_rate{0.0};
    std::vector<IndexT> vertex_indices;
};

struct DeformableAttachmentAim {
    IndexT local_vertex{0};
    std::string rigid_link_path;
    std::size_t environment{0};
    std::size_t affine_output_offset{0};
    Vector3 link_local_position{Vector3::Zero()};
};

struct DeformableAttachmentVertex {
    std::size_t deformable_output_offset{0};
    std::string rigid_link_path;
    std::size_t environment{0};
    std::size_t affine_output_offset{0};
    Vector3 link_local_position{Vector3::Zero()};
    Vector3 link_local_center_of_mass{Vector3::Zero()};
    double vertex_mass{0.0};
    double strength_rate{0.0};
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

TriangleMeshData MakeCylinder(double radius, double height) {
    if (!std::isfinite(radius) || radius <= 0.0 ||
        !std::isfinite(height) || height <= 0.0) {
        throw std::runtime_error(
                "IPC cylinder collision shape has invalid dimensions");
    }
    constexpr int kSegments = 16;
    const double half_height = height * 0.5;
    TriangleMeshData mesh;
    mesh.vertices.reserve(kSegments * 2 + 2);
    for (int end = 0; end < 2; ++end) {
        const double z = end == 0 ? half_height : -half_height;
        for (int segment = 0; segment < kSegments; ++segment) {
            const double angle = 2.0 * std::numbers::pi *
                                 static_cast<double>(segment) /
                                 static_cast<double>(kSegments);
            mesh.vertices.push_back(
                    Vector3{radius * std::cos(angle),
                            radius * std::sin(angle), z});
        }
    }
    const IndexT top_center = static_cast<IndexT>(mesh.vertices.size());
    mesh.vertices.push_back(Vector3{0.0, 0.0, half_height});
    const IndexT bottom_center = static_cast<IndexT>(mesh.vertices.size());
    mesh.vertices.push_back(Vector3{0.0, 0.0, -half_height});
    for (int segment = 0; segment < kSegments; ++segment) {
        const IndexT next = static_cast<IndexT>((segment + 1) % kSegments);
        const IndexT top = static_cast<IndexT>(segment);
        const IndexT top_next = next;
        const IndexT bottom = static_cast<IndexT>(kSegments + segment);
        const IndexT bottom_next = static_cast<IndexT>(kSegments) + next;
        mesh.triangles.push_back(Vector3i{top_center, top, top_next});
        mesh.triangles.push_back(
                Vector3i{bottom_center, bottom_next, bottom});
        mesh.triangles.push_back(Vector3i{top, bottom, top_next});
        mesh.triangles.push_back(Vector3i{top_next, bottom, bottom_next});
    }
    return mesh;
}

TriangleMeshData MakeCapsule(double radius, double height) {
    if (!std::isfinite(radius) || radius <= 0.0 ||
        !std::isfinite(height) || height <= 0.0) {
        throw std::runtime_error(
                "IPC capsule collision shape has invalid dimensions");
    }
    constexpr int kSegments = 16;
    constexpr int kHemisphereRings = 4;
    const double half_height = height * 0.5;
    TriangleMeshData mesh;
    mesh.vertices.push_back(Vector3{0.0, 0.0, half_height + radius});

    std::vector<std::pair<double, double>> rings;
    for (int ring = 1; ring <= kHemisphereRings; ++ring) {
        const double angle = std::numbers::pi * 0.5 *
                             static_cast<double>(ring) /
                             static_cast<double>(kHemisphereRings);
        rings.emplace_back(radius * std::sin(angle),
                           half_height + radius * std::cos(angle));
    }
    rings.emplace_back(radius, -half_height);
    for (int ring = kHemisphereRings - 1; ring >= 1; --ring) {
        const double angle = std::numbers::pi * 0.5 *
                             static_cast<double>(ring) /
                             static_cast<double>(kHemisphereRings);
        rings.emplace_back(radius * std::sin(angle),
                           -half_height - radius * std::cos(angle));
    }
    for (const auto& [ring_radius, z] : rings) {
        for (int segment = 0; segment < kSegments; ++segment) {
            const double angle = 2.0 * std::numbers::pi *
                                 static_cast<double>(segment) /
                                 static_cast<double>(kSegments);
            mesh.vertices.push_back(
                    Vector3{ring_radius * std::cos(angle),
                            ring_radius * std::sin(angle), z});
        }
    }
    const IndexT bottom = static_cast<IndexT>(mesh.vertices.size());
    mesh.vertices.push_back(Vector3{0.0, 0.0, -half_height - radius});
    for (int segment = 0; segment < kSegments; ++segment) {
        const IndexT next = static_cast<IndexT>((segment + 1) % kSegments);
        mesh.triangles.push_back(Vector3i{0, 1 + segment, 1 + next});
    }
    for (std::size_t ring = 0; ring + 1 < rings.size(); ++ring) {
        const IndexT current = 1 + static_cast<IndexT>(ring * kSegments);
        const IndexT next_ring = current + kSegments;
        for (int segment = 0; segment < kSegments; ++segment) {
            const IndexT next = static_cast<IndexT>((segment + 1) % kSegments);
            mesh.triangles.push_back(Vector3i{
                    current + segment, next_ring + segment, current + next});
            mesh.triangles.push_back(Vector3i{
                    current + next, next_ring + segment, next_ring + next});
        }
    }
    const IndexT last_ring =
            1 + static_cast<IndexT>((rings.size() - 1) * kSegments);
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
            std::string workspace_suffix = {},
            std::string contact_constitution = "ipc",
            double al_ipc_mu_scale_fem = 5.0e7,
            double al_ipc_mu_scale_abd = 1.0e5,
            double al_ipc_toi_threshold = 0.1,
            double al_ipc_alpha_lower_bound = 1.0e-6,
            double al_ipc_decay_factor = 0.3,
            std::uint32_t newton_max_iterations = 16,
            std::uint32_t line_search_max_iterations = 8,
            double linear_system_tolerance_rate = 1.0e-3,
            bool strict_convergence = false,
            std::uint32_t output_flags = IpcBatchSolverOutputAll)
        : environment_count_(environment_count),
          device_index_(config.device_index),
          external_affine_proxies_(external_affine_proxies),
          contact_constitution_(std::move(contact_constitution)),
          exact_contact_wrench_(contact_constitution_ == "ipc"),
          device_native_coupling_enabled_(DeviceNativeCouplingEnabled()),
          output_flags_(output_flags) {
        ValidateConfig(config);
        ValidateOutputFlags(output_flags_);
        if (newton_max_iterations == 0 ||
            line_search_max_iterations == 0 ||
            !std::isfinite(linear_system_tolerance_rate) ||
            linear_system_tolerance_rate <= 0.0) {
            throw std::runtime_error(
                    "libuipc solver iteration parameters are invalid");
        }
        if (contact_constitution_ != "ipc" &&
            contact_constitution_ != "al-ipc") {
            throw std::runtime_error(
                    "libuipc contact constitution must be 'ipc' or 'al-ipc'");
        }
        const std::array<double, 5> al_parameters{
                al_ipc_mu_scale_fem, al_ipc_mu_scale_abd,
                al_ipc_toi_threshold, al_ipc_alpha_lower_bound,
                al_ipc_decay_factor};
        if (!std::ranges::all_of(al_parameters, [](double value) {
                return std::isfinite(value) && value > 0.0;
            }) ||
            al_ipc_toi_threshold > 1.0 ||
            al_ipc_alpha_lower_bound > 1.0 || al_ipc_decay_factor > 1.0) {
            throw std::runtime_error("libuipc AL-IPC parameters are invalid");
        }
        if (environment_count_ == 0) {
            throw std::runtime_error(
                    "libuipc environment count must be positive");
        }
        ActivateDevice();
        ConfigureLibuipc(config);
        const Json manifest = Json::parse(
                std::string_view(artifact.manifest, artifact.manifest_size));
        if ((artifact.schema_version != 3 && artifact.schema_version != 4 &&
             artifact.schema_version != 5) ||
            manifest.value("schema_version", 0) != artifact.schema_version ||
            manifest.value("format", std::string{}) != "gobot-ipc") {
            throw std::runtime_error(
                    "libuipc module requires a Gobot IPC schema v3, v4, or v5 artifact");
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
        scene_config["contact"]["constitution"] = contact_constitution_;
        scene_config["contact"]["al-ipc"]["mu_scale_fem"] =
                al_ipc_mu_scale_fem;
        scene_config["contact"]["al-ipc"]["mu_scale_abd"] =
                al_ipc_mu_scale_abd;
        scene_config["contact"]["al-ipc"]["toi_threshold"] =
                al_ipc_toi_threshold;
        scene_config["contact"]["al-ipc"]["alpha_lower_bound"] =
                al_ipc_alpha_lower_bound;
        scene_config["contact"]["al-ipc"]["decay_factor"] =
                al_ipc_decay_factor;
        // libuipc's strict offline defaults can spend hundreds of Newton
        // iterations resolving driven affine joints.  These are the same
        // tolerances used by its interactive affine-body examples.
        scene_config["newton"]["max_iter"] = newton_max_iterations;
        scene_config["newton"]["velocity_tol"] = 0.1;
        scene_config["newton"]["transrate_tol"] = 10.0;
        scene_config["newton"]["ccd_tol"] = 5.0e-4;
        scene_config["line_search"]["max_iter"] =
                line_search_max_iterations;
        scene_config["linear_system"]["tol_rate"] =
                linear_system_tolerance_rate;
        scene_config["extras"]["strict_mode"]["enable"] =
                strict_convergence;
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

        const Json deformable_attachments =
                manifest.value("deformable_attachments", Json::array());
        const std::vector<DeformableAttachmentSpec> attachment_specs =
                ParseDeformableAttachments(
                        manifest.value("deformable_bodies", Json::array()),
                        manifest.value("couplings", Json::array()),
                        deformable_attachments);
        BuildDeformables(manifest.value("deformable_bodies", Json::array()),
                         attachment_specs, contact, subscenes);
        const Json affine_robots = external_affine_proxies_
                                           ? SelectExternalAffineRobots(manifest)
                                           : manifest.value("robots", Json::array());
        BuildAffineBodies(affine_robots, contact, config, subscenes);
        BuildStaticColliders(
                manifest.value("static_colliders", Json::array()), contact,
                config, subscenes);
        ValidateDeformableAttachmentMappings(attachment_specs);
        if (deformable_bodies_.empty()) {
            throw std::runtime_error("libuipc module requires at least one deformable body");
        }

        world_->init(*scene_);
        if (!world_->is_valid()) {
            throw std::runtime_error("libuipc rejected the compiled Gobot scene");
        }
        InitializeAccessors();
        InitializeContactForceExport(config.fixed_time_step);
        if (external_affine_proxies_) {
            if (world_->frame() != 0 || !world_->dump()) {
                throw std::runtime_error(
                        "libuipc batch solver could not snapshot its initial state");
            }
            initial_state_dumped_ = true;
        }
        if (!external_affine_proxies_) {
            Refresh(false);
        }
    }

    ~Session() {
        try {
            world_->sync();
            if (external_force_accessor_ != nullptr &&
                external_force_accessor_->has_bound_external_forces()) {
                external_force_accessor_->unbind_external_forces();
            }
            if (soft_position_accessor_ != nullptr &&
                soft_position_accessor_->has_bound_aim_positions()) {
                soft_position_accessor_->unbind_aim_positions();
            }
            if (soft_transform_accessor_ != nullptr &&
                soft_transform_accessor_->has_bound_aim_transforms()) {
                soft_transform_accessor_->unbind_aim_transforms();
            }
        } catch (...) {
            // Destruction must not propagate through the C ABI.
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
            attachment_target_transforms_ = initial_affine_transforms_;
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
        const auto attachment_body = affine_output_offsets_.find(
                AffineEnvironmentKey(0, path));
        if (attachment_body != affine_output_offsets_.end()) {
            attachment_target_transforms_.at(attachment_body->second) = transform;
        }
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
    const std::string& ContactConstitution() const {
        return contact_constitution_;
    }
    bool ExactContactWrench() const { return exact_contact_wrench_; }
    SolverDiagnostics GetSolverDiagnostics() const {
        return solver_diagnostics_->diagnostics();
    }
    void SetRuntimeOptions(const IpcBatchSolverRuntimeOptions& options) {
        solver_control_->set_options(SolverRuntimeOptions{
                options.newton_max_iterations,
                options.line_search_max_iterations,
                options.linear_system_tolerance_rate,
                options.strict_convergence});
    }
    std::size_t StaticColliderCountPerEnvironment() const {
        return static_collider_count_per_environment_;
    }

    void BindDeviceBuffers(const IpcBatchSolverModuleBuffers& buffers) {
        if (!external_affine_proxies_) {
            throw std::runtime_error(
                    "device buffers require external affine proxy mode");
        }
        ActivateDevice();
        device_buffers_ = buffers;
        if (device_native_coupling_enabled_) {
            InitializeDeviceCouplingWorkspace();
            UploadAffineTargetsAndTwists();
            if (!initial_affine_transforms_.empty()) {
                if (soft_transform_accessor_ == nullptr) {
                    throw std::runtime_error(
                            "libuipc CUDA backend has no soft-transform device accessor");
                }
                soft_transform_accessor_->bind_aim_transforms(DeviceView(
                        device_workspace_->target_transforms(),
                        initial_affine_transforms_.size(), sizeof(Matrix4x4)));
            }
            if (!deformable_attachment_vertices_.empty()) {
                if (soft_position_accessor_ == nullptr) {
                    throw std::runtime_error(
                            "libuipc CUDA backend has no soft-position device accessor");
                }
                soft_position_accessor_->bind_aim_positions(
                        DeviceView(device_workspace_->attachment_aim_positions(),
                                   initial_fem_positions_.size(),
                                   sizeof(Vector3)));
            }
        } else {
            UploadAffineTargetsAndTwistsHost();
        }
        if (external_force_accessor_ == nullptr) {
            throw std::runtime_error(
                    "libuipc CUDA backend has no FEM external-force accessor");
        }
        external_force_accessor_->bind_external_forces(DeviceView(
                buffers.deformable_external_forces,
                initial_fem_positions_.size(), sizeof(Vector3)));
        device_buffers_bound_ = true;
        WriteDeviceState(IpcBatchSolverOutputAll);
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
        const auto start = SteadyClock::now();
        last_target_staging_latency_ms_ = 0.0;
        last_ipc_advance_latency_ms_ = 0.0;
        last_reaction_export_latency_ms_ = 0.0;
        last_state_sync_latency_ms_ = 0.0;
        for (std::uint32_t index = 0; index < steps; ++index) {
            const auto staging_start = SteadyClock::now();
            UploadAffineTargetsAndTwists();
            last_target_staging_latency_ms_ +=
                    ElapsedMilliseconds(staging_start);
            const auto advance_start = SteadyClock::now();
            world_->advance();
            last_ipc_advance_latency_ms_ +=
                    ElapsedMilliseconds(advance_start);
            if (!world_->is_valid()) {
                throw std::runtime_error(
                        "libuipc batch world became invalid while stepping");
            }
        }
        frame_ = world_->frame();
        WriteDeviceState(output_flags_);
        last_step_latency_ms_ = ElapsedMilliseconds(start);
    }

    void ResetDevice() {
        if (!device_buffers_bound_) {
            throw std::runtime_error(
                    "libuipc batch device buffers are not bound");
        }
        ActivateDevice();
        checkpoint_frame_.reset();
        checkpoint_affine_targets_.clear();
        checkpoint_attachment_targets_.clear();
        if (!initial_state_dumped_ || !world_->recover(0)) {
            throw std::runtime_error(
                    "libuipc batch solver could not recover its initial state");
        }
        frame_ = world_->frame();
        if (frame_ != 0) {
            throw std::runtime_error(
                    "libuipc batch solver recovered an invalid frame");
        }
        WriteDeviceState(IpcBatchSolverOutputAll);
    }

    void CaptureDeviceCheckpoint() {
        if (!device_buffers_bound_) {
            throw std::runtime_error(
                    "libuipc batch device buffers are not bound");
        }
        if (checkpoint_frame_.has_value()) {
            throw std::runtime_error(
                    "libuipc batch checkpoint slot is already active");
        }
        ActivateDevice();
        world_->sync();
        if (!world_->dump_memory()) {
            throw std::runtime_error(
                    "libuipc batch solver could not capture an in-memory checkpoint");
        }
        checkpoint_frame_ = world_->frame();
        checkpoint_affine_targets_.clear();
        checkpoint_affine_targets_.reserve(external_affine_targets_.size());
        for (const auto& target : external_affine_targets_) {
            checkpoint_affine_targets_.push_back(target->value);
        }
        checkpoint_attachment_targets_ = attachment_target_transforms_;
    }

    void RewindDeviceCheckpoint() {
        if (!checkpoint_frame_.has_value()) {
            throw std::runtime_error(
                    "libuipc batch checkpoint slot is not active");
        }
        ActivateDevice();
        if (!world_->recover_memory(*checkpoint_frame_)) {
            throw std::runtime_error(
                    "libuipc batch solver could not rewind its in-memory checkpoint");
        }
        if (checkpoint_affine_targets_.size() !=
            external_affine_targets_.size()) {
            throw std::runtime_error(
                    "libuipc batch checkpoint affine target layout changed");
        }
        for (std::size_t index = 0; index < external_affine_targets_.size();
             ++index) {
            external_affine_targets_[index]->value =
                    checkpoint_affine_targets_[index];
        }
        attachment_target_transforms_ = checkpoint_attachment_targets_;
        frame_ = world_->frame();
        WriteDeviceState(IpcBatchSolverOutputAll);
    }

    void CommitDeviceCheckpoint() {
        if (!checkpoint_frame_.has_value()) {
            throw std::runtime_error(
                    "libuipc batch checkpoint slot is not active");
        }
        checkpoint_frame_.reset();
        checkpoint_affine_targets_.clear();
        checkpoint_attachment_targets_.clear();
    }

    bool CheckpointActive() const { return checkpoint_frame_.has_value(); }

    void SynchronizeDevice() {
        ActivateDevice();
        world_->sync();
    }

    void SetOutputFlags(std::uint32_t output_flags) {
        ValidateOutputFlags(output_flags);
        output_flags_ = output_flags;
    }

    void RefreshOutputs(std::uint32_t output_flags) {
        if (!device_buffers_bound_) {
            throw std::runtime_error(
                    "libuipc batch device buffers are not bound");
        }
        ValidateOutputFlags(output_flags);
        if (output_flags == IpcBatchSolverOutputNone) {
            throw std::runtime_error(
                    "libuipc batch refresh output flags are empty");
        }
        ActivateDevice();
        last_reaction_export_latency_ms_ = 0.0;
        last_state_sync_latency_ms_ = 0.0;
        WriteDeviceState(output_flags);
    }

    std::uint32_t OutputFlags() const { return output_flags_; }
    std::uint64_t DeformableContactForceFrame() const {
        return deformable_contact_force_frame_;
    }
    double LastTargetStagingLatencyMs() const {
        return last_target_staging_latency_ms_;
    }
    double LastIpcAdvanceLatencyMs() const {
        return last_ipc_advance_latency_ms_;
    }
    double LastReactionExportLatencyMs() const {
        return last_reaction_export_latency_ms_;
    }
    double LastStateSyncLatencyMs() const {
        return last_state_sync_latency_ms_;
    }
    bool DeviceNativeCoupling() const { return device_workspace_ != nullptr; }
    std::size_t DeviceWorkspaceAllocationCount() const {
        return device_workspace_ != nullptr
                       ? device_workspace_->allocation_count()
                       : 0;
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
        return DeviceView(buffer.data, element_count, element_size);
    }

    static uipc::backend::BufferView DeviceView(void* data,
                                                std::size_t element_count,
                                                std::size_t element_size) {
        return uipc::backend::BufferView{
                static_cast<uipc::backend::HandleT>(
                        reinterpret_cast<std::uintptr_t>(data)),
                0,
                element_count,
                element_size,
                element_size,
                "cuda"};
    }

    void UploadAffineTargetsAndTwists() {
        if (!device_native_coupling_enabled_) {
            UploadAffineTargetsAndTwistsHost();
            return;
        }
        const std::size_t body_count = initial_affine_transforms_.size();
        if (body_count == 0) {
            return;
        }
        if (external_affine_targets_.size() != body_count ||
            device_workspace_ == nullptr) {
            throw std::runtime_error(
                    "libuipc external affine target layout changed after initialization");
        }
        device_workspace_->StageTargets(
                static_cast<const double*>(device_buffers_.affine_targets.data),
                static_cast<const double*>(
                        device_buffers_.affine_target_twists.data));
        affine_accessor_->copy_velocity_from(
                DeviceView(device_workspace_->target_velocities(), body_count,
                           sizeof(Matrix4x4)),
                0, body_count);
    }

    void UploadAffineTargetsAndTwistsHost() {
        const std::size_t body_count = initial_affine_transforms_.size();
        if (body_count == 0) {
            return;
        }
        if (external_affine_targets_.size() != body_count) {
            throw std::runtime_error(
                    "libuipc external affine target layout changed after initialization");
        }
        target_row_major_.resize(body_count * 16);
        target_twists_.resize(body_count * 6);
        RequireCuda(
                cudaMemcpy(target_row_major_.data(),
                           device_buffers_.affine_targets.data,
                           target_row_major_.size() * sizeof(double),
                           cudaMemcpyDeviceToHost),
                "copying MuJoCo affine targets to libuipc");
        RequireCuda(
                cudaMemcpy(target_twists_.data(),
                           device_buffers_.affine_target_twists.data,
                           target_twists_.size() * sizeof(double),
                           cudaMemcpyDeviceToHost),
                "copying MuJoCo affine target twists to libuipc");
        affine_accessor_->copy_to(*affine_state_);
        auto velocities = affine_state_->instances().find<Matrix4x4>(
                uipc::builtin::velocity);
        if (velocities == nullptr || view(*velocities).size() != body_count) {
            throw std::runtime_error(
                    "libuipc affine target velocity layout changed after initialization");
        }
        auto velocity_values = view(*velocities);
        for (std::size_t body = 0; body < body_count; ++body) {
            Matrix4x4 value;
            for (std::size_t index = 0; index < 16; ++index) {
                value(static_cast<Eigen::Index>(index / 4),
                      static_cast<Eigen::Index>(index % 4)) =
                        target_row_major_[body * 16 + index];
            }
            for (std::size_t index = 0; index < 6; ++index) {
                if (!std::isfinite(target_twists_[body * 6 + index])) {
                    throw std::runtime_error(
                            "libuipc affine target twist contains a non-finite value");
                }
            }
            external_affine_targets_[body]->value = value;
            attachment_target_transforms_.at(body) = value;

            const Vector3 linear{
                    target_twists_[body * 6],
                    target_twists_[body * 6 + 1],
                    target_twists_[body * 6 + 2]};
            const Vector3 angular{
                    target_twists_[body * 6 + 3],
                    target_twists_[body * 6 + 4],
                    target_twists_[body * 6 + 5]};
            Matrix3x3 angular_cross;
            angular_cross << 0.0, -angular.z(), angular.y(),
                    angular.z(), 0.0, -angular.x(),
                    -angular.y(), angular.x(), 0.0;
            Matrix4x4 velocity = Matrix4x4::Zero();
            velocity.block<3, 3>(0, 0) =
                    angular_cross * value.block<3, 3>(0, 0);
            velocity.block<3, 1>(0, 3) = linear;
            velocity_values[body] = velocity;
        }
        affine_accessor_->copy_from(*affine_state_);
    }

    void WriteDeviceState(std::uint32_t output_flags) {
        ValidateOutputFlags(output_flags);
        const std::size_t vertex_count = initial_fem_positions_.size();
        const std::size_t affine_count = initial_affine_transforms_.size();
        const bool export_deformable_state = HasOutputFlag(
                output_flags, IpcBatchSolverOutputDeformableState);
        const bool export_affine_state = HasOutputFlag(
                output_flags, IpcBatchSolverOutputAffineState);
        bool outputs_synchronized = false;
        if (export_deformable_state || export_affine_state) {
            if (export_deformable_state) {
                fem_accessor_->copy_position_to(
                        DeviceView(device_buffers_.deformable_positions,
                                   vertex_count, sizeof(Vector3)),
                        0, vertex_count);
                fem_accessor_->copy_velocity_to(
                        DeviceView(device_buffers_.deformable_velocities,
                                   vertex_count, sizeof(Vector3)),
                        0, vertex_count);
            }
            if (export_affine_state && affine_count != 0) {
                affine_accessor_->copy_transform_to(
                        DeviceView(device_buffers_.affine_transforms,
                                   affine_count, sizeof(Matrix4x4)),
                        0, affine_count);
            }
        }

        const std::size_t deformable_force_scalars = vertex_count * 3;
        const std::size_t affine_wrench_scalars = affine_count * 6;
        const bool export_deformable_contact = HasOutputFlag(
                output_flags,
                IpcBatchSolverOutputDeformableContactForces);
        const bool export_affine_wrench = HasOutputFlag(
                output_flags,
                IpcBatchSolverOutputAffineContactWrenches);
        if (export_deformable_contact || export_affine_wrench) {
            const auto reaction_start = SteadyClock::now();
            if (frame_ != 0 && exact_contact_wrench_) {
                if (device_workspace_ != nullptr) {
                    if (export_affine_wrench && affine_count != 0) {
                        affine_accessor_->copy_transform_to(
                                DeviceView(
                                        device_workspace_->current_affine_transforms(),
                                        affine_count, sizeof(Matrix4x4)),
                                0, affine_count);
                    }
                    if (export_affine_wrench &&
                        !deformable_attachment_vertices_.empty()) {
                        fem_accessor_->copy_position_to(
                                DeviceView(
                                        device_workspace_->current_deformable_positions(),
                                        vertex_count, sizeof(Vector3)),
                                0, vertex_count);
                    }
                    device_contact_gradient_views_.clear();
                    for (const ContactGradientBuffer& buffer :
                         contact_gradient_buffers_) {
                        const auto view =
                                contact_system_->contact_gradient_device_view(
                                        buffer.primitive_type);
                        if (!view.supported ||
                            view.vertex_indices.size() != view.gradients.size() ||
                            view.vertex_indices.element_size() != sizeof(IndexT) ||
                            view.vertex_indices.element_stride() != sizeof(IndexT) ||
                            view.gradients.element_size() != sizeof(Vector3) ||
                            view.gradients.element_stride() != sizeof(Vector3)) {
                            throw std::runtime_error(
                                    "libuipc contact-gradient device view has an invalid layout");
                        }
                        const auto* indices =
                                reinterpret_cast<const IndexT*>(
                                        view.vertex_indices.handle()) +
                                view.vertex_indices.offset();
                        const auto* gradients =
                                reinterpret_cast<const Vector3*>(
                                        view.gradients.handle()) +
                                view.gradients.offset();
                        device_contact_gradient_views_.push_back(
                                DeviceContactGradientView{
                                        indices, gradients,
                                        view.gradients.size()});
                    }
                    device_workspace_->ExportReactions(
                            device_contact_gradient_views_,
                            inverse_time_step_squared_,
                            static_cast<double*>(
                                    device_buffers_.deformable_contact_forces.data),
                            static_cast<double*>(
                                    device_buffers_.affine_contact_wrenches.data),
                            export_deformable_contact,
                            export_affine_wrench);
                } else {
                    if (!contact_forces_current_ ||
                        contact_force_host_frame_ != frame_) {
                        if (!deformable_attachment_vertices_.empty()) {
                            fem_accessor_->copy_to(*fem_state_);
                        }
                        if (affine_count != 0) {
                            affine_accessor_->copy_to(*affine_state_);
                        }
                        const auto sync_start = SteadyClock::now();
                        world_->sync();
                        last_state_sync_latency_ms_ +=
                                ElapsedMilliseconds(sync_start);
                        outputs_synchronized = true;
                        RefreshContactForces();
                        contact_force_host_frame_ = frame_;
                        contact_forces_current_ = true;
                    }
                    if (export_deformable_contact &&
                        deformable_force_scalars != 0) {
                        RequireCuda(
                                cudaMemcpy(
                                        device_buffers_.deformable_contact_forces.data,
                                        contact_forces_.data(),
                                        deformable_force_scalars * sizeof(double),
                                        cudaMemcpyHostToDevice),
                                "uploading batched deformable contact forces");
                    }
                    if (export_affine_wrench && affine_wrench_scalars != 0) {
                        RequireCuda(
                                cudaMemcpy(
                                        device_buffers_.affine_contact_wrenches.data,
                                        affine_contact_wrenches_.data(),
                                        affine_wrench_scalars * sizeof(double),
                                        cudaMemcpyHostToDevice),
                                "uploading batched affine contact wrenches");
                    }
                }
                if (export_deformable_contact) {
                    deformable_contact_force_frame_ = frame_;
                }
            } else {
                if (export_deformable_contact &&
                    deformable_force_scalars != 0) {
                    RequireCuda(
                            cudaMemset(
                                    device_buffers_.deformable_contact_forces.data,
                                    0,
                                    deformable_force_scalars * sizeof(double)),
                            "clearing batched deformable contact forces");
                    deformable_contact_force_frame_ = frame_;
                }
                if (export_affine_wrench && affine_wrench_scalars != 0) {
                    RequireCuda(
                            cudaMemset(
                                    device_buffers_.affine_contact_wrenches.data,
                                    0,
                                    affine_wrench_scalars * sizeof(double)),
                            "clearing batched affine contact wrenches");
                }
                contact_force_host_frame_ = frame_;
                contact_forces_current_ = frame_ == 0;
            }
            last_reaction_export_latency_ms_ +=
                    ElapsedMilliseconds(reaction_start);
        }
        if (!outputs_synchronized) {
            const auto sync_start = SteadyClock::now();
            world_->sync();
            last_state_sync_latency_ms_ += ElapsedMilliseconds(sync_start);
        }
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

    static std::string AffineEnvironmentKey(
            std::size_t environment, std::string_view path) {
        return fmt::format("{}#{}", environment, path);
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

    TriangleMeshData DecodeCollisionShape(
            const Json& shape, std::string_view description) const {
        const std::string type = shape.at("shape_type").get<std::string>();
        if (type == "box") {
            return MakeBox(ParseVector3(shape.at("size"), "box size"));
        }
        if (type == "sphere") {
            return MakeSphere(shape.at("radius").get<double>());
        }
        if (type == "capsule") {
            return MakeCapsule(shape.at("radius").get<double>(),
                               shape.at("height").get<double>());
        }
        if (type == "cylinder") {
            return MakeCylinder(shape.at("radius").get<double>(),
                                shape.at("height").get<double>());
        }
        if (type == "triangle_mesh") {
            const std::string blob_id =
                    shape.at("mesh_blob").get<std::string>();
            const IpcSolverArtifactBlobView& blob = FindBlob(
                    blob_id, kTriangleEncoding, description);
            TriangleMeshData mesh = DecodeTriangleMesh(
                    std::span<const std::uint8_t>(blob.data, blob.size));
            if (mesh.vertices.size() !=
                        shape.at("vertex_count").get<std::size_t>() ||
                mesh.triangles.size() !=
                        shape.at("triangle_count").get<std::size_t>()) {
                throw std::runtime_error(
                        std::string(description) +
                        " metadata does not match its blob");
            }
            return mesh;
        }
        throw std::runtime_error(
                "libuipc Gobot adapter does not support collision shape '" +
                type + "'");
    }

    static Json SelectExternalAffineRobots(const Json& manifest) {
        const Json& robots = manifest.at("robots");
        const Json& couplings = manifest.at("couplings");
        if (!robots.is_array() || !couplings.is_array()) {
            throw std::runtime_error(
                    "IPC robot and PhysicsCoupling tables must be arrays");
        }
        if (couplings.empty()) {
            return Json::array();
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
                            "IPC schema v3 contains duplicate Robot3D Link3D paths");
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
                        "IPC schema v3 PhysicsCoupling paths must be non-empty and unique");
            }
            if (proxy_index != 0 && previous_coupling_path > coupling_path) {
                throw std::runtime_error(
                        "IPC schema v3 PhysicsCoupling table is not canonically sorted");
            }
            previous_coupling_path = coupling_path;
            if (coupling.at("proxy_index").get<std::size_t>() != proxy_index) {
                throw std::runtime_error(
                        "IPC schema v3 PhysicsCoupling proxy indices must be contiguous");
            }
            const std::string mode = coupling.at("mode").get<std::string>();
            const double force_scale = coupling.at("force_scale").get<double>();
            const double torque_scale = coupling.at("torque_scale").get<double>();
            if ((mode != "OneWay" && mode != "TwoWay") ||
                !std::isfinite(force_scale) || force_scale < 0.0 ||
                !std::isfinite(torque_scale) || torque_scale < 0.0) {
                throw std::runtime_error(
                        "IPC schema v3 PhysicsCoupling mode or wrench scale is invalid");
            }
            const auto found = links_by_path.find(link_path);
            if (found == links_by_path.end()) {
                throw std::runtime_error(
                        "IPC schema v3 PhysicsCoupling references an unknown Link3D path");
            }
            const Json& robot = *found->second.robot;
            Json link = *found->second.link;
            if (coupling.at("robot_name").get<std::string>() !=
                        robot.at("name").get<std::string>() ||
                coupling.at("link_name").get<std::string>() !=
                        link.at("name").get<std::string>()) {
                throw std::runtime_error(
                        "IPC schema v3 PhysicsCoupling names do not match its Link3D path");
            }
            const bool has_enabled_collision = std::ranges::any_of(
                    link.at("collision_shapes"), [](const Json& shape) {
                        return !shape.value("disabled", false);
                    });
            if (!has_enabled_collision) {
                throw std::runtime_error(
                        "IPC schema v3 PhysicsCoupling target has no enabled collision shape");
            }
            link["external_proxy_mode"] = mode;
            selected.push_back({
                    {"joints", Json::array()},
                    {"kind", robot.value("kind", "articulation")},
                    {"links", Json::array({std::move(link)})},
                    {"name", robot.at("name")},
                    {"path", robot.at("path")},
                    {"root_link_paths", Json::array({link_path})},
                    {"transform", robot.at("transform")}});
        }
        return selected;
    }

    static std::vector<DeformableAttachmentSpec> ParseDeformableAttachments(
            const Json& bodies,
            const Json& couplings,
            const Json& attachments) {
        if (!attachments.is_array()) {
            throw std::runtime_error(
                    "IPC deformable attachment table must be an array");
        }
        if (attachments.empty()) {
            return {};
        }
        if (!bodies.is_array() || !couplings.is_array()) {
            throw std::runtime_error(
                    "IPC deformable attachments require body and coupling tables");
        }

        std::unordered_map<std::string, std::size_t> body_vertex_counts;
        for (const Json& body : bodies) {
            body_vertex_counts.emplace(
                    body.at("path").get<std::string>(),
                    body.at("vertex_count").get<std::size_t>());
        }
        std::unordered_map<std::string, std::size_t> proxy_indices;
        for (const Json& coupling : couplings) {
            proxy_indices.emplace(
                    coupling.at("link_path").get<std::string>(),
                    coupling.at("proxy_index").get<std::size_t>());
        }

        std::vector<DeformableAttachmentSpec> result;
        result.reserve(attachments.size());
        std::unordered_set<std::string> paths;
        std::unordered_set<std::string> attached_vertices;
        std::string previous_path;
        for (const Json& attachment : attachments) {
            const std::string path =
                    attachment.at("attachment_path").get<std::string>();
            const std::string body_path =
                    attachment.at("deformable_body_path").get<std::string>();
            const std::string link_path =
                    attachment.at("rigid_link_path").get<std::string>();
            const auto body = body_vertex_counts.find(body_path);
            const auto proxy = proxy_indices.find(link_path);
            if (path.empty() || !paths.insert(path).second ||
                (!previous_path.empty() && previous_path > path)) {
                throw std::runtime_error(
                        "IPC deformable attachment paths must be unique and canonically sorted");
            }
            previous_path = path;
            if (body == body_vertex_counts.end() || proxy == proxy_indices.end()) {
                throw std::runtime_error(
                        "IPC deformable attachment references an unknown body or coupling");
            }
            const std::size_t proxy_index =
                    attachment.at("proxy_index").get<std::size_t>();
            const double strength_rate =
                    attachment.at("strength_rate").get<double>();
            const Json& indices = attachment.at("vertex_indices");
            if (proxy_index != proxy->second || !std::isfinite(strength_rate) ||
                strength_rate <= 0.0 || !indices.is_array() || indices.empty()) {
                throw std::runtime_error(
                        "IPC deformable attachment has invalid proxy, strength, or vertices");
            }

            DeformableAttachmentSpec spec;
            spec.path = path;
            spec.deformable_path = body_path;
            spec.rigid_link_path = link_path;
            spec.proxy_index = proxy_index;
            spec.strength_rate = strength_rate;
            IndexT previous_vertex = -1;
            for (const Json& value : indices) {
                const std::size_t vertex = value.get<std::size_t>();
                if (vertex >= body->second ||
                    (previous_vertex >= 0 &&
                     vertex <= static_cast<std::size_t>(previous_vertex))) {
                    throw std::runtime_error(
                            "IPC deformable attachment vertices must be sorted, unique, and in range");
                }
                const std::string vertex_key =
                        fmt::format("{}#{}", body_path, vertex);
                if (!attached_vertices.insert(vertex_key).second) {
                    throw std::runtime_error(
                            "IPC deformable vertex is assigned to multiple attachments");
                }
                previous_vertex = static_cast<IndexT>(vertex);
                spec.vertex_indices.push_back(previous_vertex);
            }
            result.push_back(std::move(spec));
        }
        return result;
    }

    static std::vector<double> ComputeVertexMasses(
            const TetMeshData& mesh, double density) {
        std::vector<double> masses(mesh.vertices.size(), 0.0);
        for (const Vector4i& tetrahedron : mesh.tetrahedra) {
            const Vector3& p0 = mesh.vertices[static_cast<std::size_t>(tetrahedron[0])];
            const Vector3& p1 = mesh.vertices[static_cast<std::size_t>(tetrahedron[1])];
            const Vector3& p2 = mesh.vertices[static_cast<std::size_t>(tetrahedron[2])];
            const Vector3& p3 = mesh.vertices[static_cast<std::size_t>(tetrahedron[3])];
            const double volume =
                    (p1 - p0).dot((p2 - p0).cross(p3 - p0)) / 6.0;
            if (!std::isfinite(volume) || volume <= 0.0) {
                throw std::runtime_error(
                        "IPC deformable attachment mesh has a non-positive tetrahedron");
            }
            const double vertex_mass = density * volume / 4.0;
            for (const IndexT vertex : tetrahedron) {
                masses[static_cast<std::size_t>(vertex)] += vertex_mass;
            }
        }
        return masses;
    }

    static std::vector<double> ComputeVertexMasses(
            const TriangleMeshData& mesh,
            double density,
            double thickness) {
        std::vector<double> masses(mesh.vertices.size(), 0.0);
        for (const Vector3i& triangle : mesh.triangles) {
            const Vector3& p0 = mesh.vertices[static_cast<std::size_t>(triangle[0])];
            const Vector3& p1 = mesh.vertices[static_cast<std::size_t>(triangle[1])];
            const Vector3& p2 = mesh.vertices[static_cast<std::size_t>(triangle[2])];
            const double area = 0.5 * (p1 - p0).cross(p2 - p0).norm();
            if (!std::isfinite(area) || area <= 0.0) {
                throw std::runtime_error(
                        "IPC thin-shell mesh has a non-positive triangle");
            }
            const double vertex_mass = density * thickness * area / 3.0;
            for (const IndexT vertex : triangle) {
                masses[static_cast<std::size_t>(vertex)] += vertex_mass;
            }
        }
        return masses;
    }

    template <typename ContactElement>
    void BuildDeformables(
            const Json& bodies,
            const std::vector<DeformableAttachmentSpec>& attachment_specs,
            const ContactElement& contact,
            const std::vector<SubsceneElement>& subscenes) {
        if (!bodies.is_array()) {
            throw std::runtime_error("IPC deformable body table must be an array");
        }
        StableNeoHookean material;
        StrainLimitingBaraffWitkinShell shell_material;
        DiscreteShellBending shell_bending;
        FiniteElementExternalForce external_force;
        SoftPositionConstraint attachment_constraint;
        auto object = scene_->objects().create("gobot_deformables");
        attachment_aims_by_geometry_.clear();
        std::size_t global_vertex_offset = 0;
        for (std::size_t environment = 0; environment < environment_count_;
             ++environment) {
            std::size_t environment_vertex_offset = 0;
            for (const Json& body : bodies) {
                const std::string path = body.at("path").get<std::string>();
                const std::string blob_id =
                        body.at("mesh_blob").get<std::string>();
                const Matrix4x4 transform =
                        ParseTransform(body.at("transform"), path);
                const std::string model =
                        body.value("model", std::string{"volumetric"});
                std::vector<Vector3> decoded_vertices;
                std::vector<double> vertex_masses;
                std::unique_ptr<SimplicialComplex> mesh_storage;
                if (model == "thin_shell") {
                    const IpcSolverArtifactBlobView& blob = FindBlob(
                            blob_id, kTriangleEncoding,
                            "IPC thin-shell deformable mesh");
                    TriangleMeshData decoded = DecodeTriangleMesh(
                            std::span<const std::uint8_t>(blob.data, blob.size));
                    for (Vector3& vertex : decoded.vertices) {
                        vertex = TransformPoint(transform, vertex);
                    }
                    const double thickness = body.at("thickness").get<double>();
                    if (!std::isfinite(thickness) || thickness <= 0.0) {
                        throw std::runtime_error(
                                "IPC thin-shell thickness must be finite and positive");
                    }
                    decoded_vertices = decoded.vertices;
                    vertex_masses = ComputeVertexMasses(
                            decoded, body.at("density").get<double>(), thickness);
                    mesh_storage = std::make_unique<SimplicialComplex>(
                            trimesh(decoded.vertices, decoded.triangles));
                    label_surface(*mesh_storage);
                    shell_material.apply_to(
                            *mesh_storage,
                            ElasticModuli2D::youngs_poisson(
                                    body.at("young_modulus").get<double>(),
                                    body.at("poisson_ratio").get<double>()),
                            body.at("density").get<double>(),
                            0.5 * thickness);
                    shell_bending.apply_to(
                            *mesh_storage,
                            body.at("bending_stiffness").get<double>());
                } else if (model == "volumetric") {
                    const IpcSolverArtifactBlobView& blob = FindBlob(
                            blob_id, kTetEncoding, "IPC deformable mesh");
                    TetMeshData decoded = DecodeTetMesh(
                            std::span<const std::uint8_t>(blob.data, blob.size));
                    for (Vector3& vertex : decoded.vertices) {
                        vertex = TransformPoint(transform, vertex);
                    }
                    decoded_vertices = decoded.vertices;
                    vertex_masses = ComputeVertexMasses(
                            decoded, body.at("density").get<double>());
                    mesh_storage = std::make_unique<SimplicialComplex>(
                            tetmesh(decoded.vertices, decoded.tetrahedra));
                    label_surface(*mesh_storage);
                    label_triangle_orient(*mesh_storage);
                    material.apply_to(
                            *mesh_storage,
                            ElasticModuli::youngs_poisson(
                                    body.at("young_modulus").get<double>(),
                                    body.at("poisson_ratio").get<double>()),
                            body.at("density").get<double>());
                } else {
                    throw std::runtime_error(
                            "IPC deformable body has unsupported model '" +
                            model + "'");
                }
                SimplicialComplex& mesh = *mesh_storage;
                external_force.apply_to(mesh, Vector3::Zero());
                std::vector<DeformableAttachmentAim> geometry_aims;
                bool has_attachment_constraint = false;
                for (const DeformableAttachmentSpec& attachment :
                     attachment_specs) {
                    if (attachment.deformable_path != path) {
                        continue;
                    }
                    if (!has_attachment_constraint) {
                        attachment_constraint.apply_to(mesh,
                                                       attachment.strength_rate);
                        has_attachment_constraint = true;
                    }
                    auto constrained = mesh.vertices().find<IndexT>(
                            uipc::builtin::is_constrained);
                    auto strengths = mesh.vertices().find<Float>(
                            "strength_ratio");
                    for (const IndexT vertex : attachment.vertex_indices) {
                        view(*constrained)[static_cast<std::size_t>(vertex)] = 1;
                        view(*strengths)[static_cast<std::size_t>(vertex)] =
                                attachment.strength_rate;
                        geometry_aims.push_back(DeformableAttachmentAim{
                                vertex, attachment.rigid_link_path, environment,
                                attachment.proxy_index,
                                decoded_vertices[static_cast<std::size_t>(vertex)]});
                        deformable_attachment_vertices_.push_back(
                                DeformableAttachmentVertex{
                                        global_vertex_offset +
                                                static_cast<std::size_t>(vertex),
                                        attachment.rigid_link_path,
                                        environment,
                                        attachment.proxy_index,
                                        decoded_vertices[static_cast<std::size_t>(vertex)],
                                        Vector3::Zero(),
                                        vertex_masses[static_cast<std::size_t>(vertex)],
                                        attachment.strength_rate});
                    }
                }
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
                attachment_aims_by_geometry_.push_back(
                        std::move(geometry_aims));
                if (environment == 0) {
                    deformable_bodies_.push_back(BodyRecord{
                            path, environment_vertex_offset,
                            decoded_vertices.size()});
                }
                deformable_contact_ranges_.push_back(
                        DeformableContactRange{
                                global_vertex_offset,
                                decoded_vertices.size(), created.geometry});
                initial_fem_positions_.insert(initial_fem_positions_.end(),
                                              decoded_vertices.begin(),
                                              decoded_vertices.end());
                environment_vertex_offset += decoded_vertices.size();
                global_vertex_offset += decoded_vertices.size();
            }
        }
        if (!deformable_attachment_vertices_.empty()) {
            scene_->animator().insert(
                    *object,
                    [this](uipc::core::Animation::UpdateInfo& info) {
                        const auto slots = info.geo_slots();
                        if (slots.size() != attachment_aims_by_geometry_.size()) {
                            throw std::runtime_error(
                                    "libuipc deformable attachment geometry layout changed");
                        }
                        for (std::size_t geometry_index = 0;
                             geometry_index < slots.size(); ++geometry_index) {
                            if (attachment_aims_by_geometry_[geometry_index].empty()) {
                                continue;
                            }
                            auto geometry = slots[geometry_index]
                                                    ->geometry()
                                                    .as<SimplicialComplex>();
                            auto aim = geometry->vertices().find<Vector3>(
                                    uipc::builtin::aim_position);
                            auto constrained = geometry->vertices().find<IndexT>(
                                    uipc::builtin::is_constrained);
                            for (const DeformableAttachmentAim& attachment :
                                 attachment_aims_by_geometry_[geometry_index]) {
                                view(*constrained)[static_cast<std::size_t>(
                                        attachment.local_vertex)] = 1;
                                view(*aim)[static_cast<std::size_t>(
                                        attachment.local_vertex)] =
                                        TransformPoint(
                                                attachment_target_transforms_.at(
                                                        attachment.affine_output_offset),
                                                attachment.link_local_position);
                            }
                        }
                    });
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
        proxy_contact_elements_by_environment_.clear();
        proxy_contact_elements_by_environment_.resize(environment_count_);
        std::size_t global_body_offset = 0;
        for (std::size_t environment = 0; environment < environment_count_;
             ++environment) {
            BuildAffineEnvironment(robots, contact, config,
                                   subscenes[environment], environment,
                                   &global_body_offset);
        }
    }

    void BuildStaticColliders(
            const Json& colliders,
            const uipc::core::ContactElement& deformable_contact,
            const IpcSolverModuleConfig& config,
            const std::vector<SubsceneElement>& subscenes) {
        if (!colliders.is_array()) {
            throw std::runtime_error("IPC static collider table must be an array");
        }
        AffineBodyConstitution affine;
        const auto mass_matrix = uipc::geometry::affine_body::from_rigid_body(
                1.0, Vector3::Zero(), Matrix3x3::Identity());
        std::size_t enabled_count = 0;
        for (const Json& collider : colliders) {
            if (!collider.value("disabled", false)) {
                ++enabled_count;
            }
        }
        static_collider_count_per_environment_ = enabled_count;

        for (std::size_t environment = 0; environment < environment_count_;
             ++environment) {
            const auto& proxy_contacts =
                    proxy_contact_elements_by_environment_.at(environment);
            for (const Json& collider : colliders) {
                if (collider.value("disabled", false)) {
                    continue;
                }
                const std::string path =
                        collider.at("path").get<std::string>();
                TriangleMeshData decoded = DecodeCollisionShape(
                        collider, "IPC static triangle mesh");
                SimplicialComplex mesh =
                        trimesh(decoded.vertices, decoded.triangles);
                label_surface(mesh);
                affine.apply_to(mesh, config.affine_stiffness, mass_matrix,
                                1.0e-3);
                auto fixed = mesh.instances().find<IndexT>(
                        uipc::builtin::is_fixed);
                view(*fixed)[0] = 1;

                const Json& material = collider.at("material");
                const double sliding_friction =
                        material.at("sliding_friction").get<double>();
                const std::int64_t layer =
                        collider.at("collision_layer").get<std::int64_t>();
                const std::int64_t mask =
                        collider.at("collision_mask").get<std::int64_t>();
                if (!std::isfinite(sliding_friction) ||
                    sliding_friction < 0.0 || layer < 0 || mask < 0 ||
                    layer > std::numeric_limits<std::uint32_t>::max() ||
                    mask > std::numeric_limits<std::uint32_t>::max()) {
                    throw std::runtime_error(
                            "IPC static collider has invalid material or masks: '" +
                            path + "'");
                }
                auto static_contact = scene_->contact_tabular().create(
                        environment_count_ == 1
                                ? path
                                : fmt::format("{}@environment_{}", path,
                                              environment));
                const ContactElementRecord static_record{
                        sliding_friction, static_contact,
                        static_cast<std::uint32_t>(layer),
                        static_cast<std::uint32_t>(mask), false};
                for (const ContactElementRecord& other : proxy_contacts) {
                    // MuJoCo owns rigid-static pairs. The fixed collider is
                    // present in IPC only for deformable-static contact.
                    const bool enabled =
                            other.always_enabled &&
                            ContactMasksMatch(static_record, other);
                    scene_->contact_tabular().insert(
                            static_contact, other.element,
                            config.friction_coefficient *
                                    std::sqrt(sliding_friction *
                                              other.friction),
                            config.contact_resistance, enabled);
                }
                scene_->contact_tabular().insert(
                        static_contact, static_contact, 0.0, 0.0, false);
                static_contact.apply_to(mesh);
                subscenes.at(environment).apply_to(mesh);
                view(mesh.transforms())[0] = ParseTransform(
                        collider.at("transform"), "static collider transform");

                auto object = scene_->objects().create(
                        environment_count_ == 1
                                ? path
                                : fmt::format("{}@environment_{}", path,
                                              environment));
                object->geometries().create(mesh);
            }
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
                    const Json& material = shape.at("material");
                    if (!material.is_object() ||
                        !material.contains("sliding_friction")) {
                        throw std::runtime_error(
                                "libuipc collision shape has invalid physics material: '" +
                                shape.at("path").get<std::string>() + "'");
                    }
                    const double shape_sliding_friction =
                            material.at("sliding_friction").get<double>();
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
                            shape.at("collision_layer").get<std::int64_t>();
                    const std::int64_t shape_contact_affinity =
                            shape.at("collision_mask").get<std::int64_t>();
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
                    TriangleMeshData primitive = DecodeCollisionShape(
                            shape, "IPC robot triangle mesh");
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
                const bool one_way_proxy =
                        external_affine_proxies_ &&
                        link.value("external_proxy_mode", "TwoWay") == "OneWay";
                affine.apply_to(mesh, config.affine_stiffness, mass_matrix,
                                std::max(mass / 1000.0, 1.0e-9));
                if (articulated) {
                    auto fixed = mesh.instances().find<IndexT>(uipc::builtin::is_fixed);
                    view(*fixed)[0] = root_paths.contains(
                            link.at("path").get<std::string>()) ? 1 : 0;
                } else if (one_way_proxy) {
                    // Keep OneWay proxies non-fixed and constrain them from
                    // q_prev toward this frame's target. SoftTransformConstraint
                    // interpolates that endpoint over libuipc substeps, so CCD
                    // sees the complete sweep while contact cannot displace the
                    // massless external body instead of the deformable.
                    constraint.apply_to(
                            mesh, Vector2{config.kinematic_strength,
                                          config.kinematic_strength});
                    auto external_kinetic = mesh.instances().find<IndexT>(
                            uipc::builtin::external_kinetic);
                    view(*external_kinetic)[0] = 1;
                } else {
                    constraint.apply_to(
                            mesh, Vector2{config.kinematic_strength,
                                          config.kinematic_strength});
                    if (external_affine_proxies_) {
                        auto external_kinetic = mesh.instances().find<IndexT>(
                                uipc::builtin::external_kinetic);
                        view(*external_kinetic)[0] = 1;
                    }
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
                const std::size_t affine_output_offset = *global_body_offset;
                initial_affine_transforms_.push_back(initial);
                affine_output_offsets_.emplace(
                        AffineEnvironmentKey(environment, path),
                        affine_output_offset);
                ++environment_body_offset;
                ++(*global_body_offset);

                auto object = scene_->objects().create(
                        environment_count_ == 1
                                ? path
                                : fmt::format("{}@environment_{}", path,
                                              environment));
                auto created = object->geometries().create(mesh);
                link_slots.emplace(path, created.geometry);
                affine_contact_ranges_.push_back(AffineContactRange{
                        affine_output_offset,
                        combined.vertices.size(),
                        created.geometry,
                        -1,
                        combined.vertices,
                        center});

                if (!articulated) {
                    auto target = std::make_shared<AffineTarget>();
                    target->initial = initial;
                    target->value = initial;
                    if (external_affine_proxies_) {
                        if (external_affine_targets_.size() !=
                            affine_output_offset) {
                            throw std::runtime_error(
                                    "libuipc external affine target layout is not contiguous");
                        }
                        external_affine_targets_.push_back(target);
                    } else {
                        affine_targets_.emplace(path, target);
                    }
                    scene_->animator().insert(
                            *object,
                            [target](uipc::core::Animation::UpdateInfo& info) {
                                auto geometry = info.geo_slots()[0]
                                                        ->geometry()
                                                        .as<SimplicialComplex>();
                                auto constrained =
                                        geometry->instances().find<IndexT>(
                                                uipc::builtin::is_constrained);
                                auto aim =
                                        geometry->instances().find<Matrix4x4>(
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
        proxy_contact_elements_by_environment_.at(environment) =
                std::move(contact_elements);
    }

    void ValidateDeformableAttachmentMappings(
            const std::vector<DeformableAttachmentSpec>& attachment_specs) {
        // Affine target upload shares this storage with attachment animation.
        // Keep it sized even for ordinary contact-only scenes with no
        // DeformableAttachment3D nodes.
        attachment_target_transforms_ = initial_affine_transforms_;
        if (attachment_specs.empty()) {
            return;
        }
        if (initial_affine_transforms_.empty()) {
            throw std::runtime_error(
                    "libuipc deformable attachments require affine proxies");
        }
        for (std::vector<DeformableAttachmentAim>& geometry_aims :
             attachment_aims_by_geometry_) {
            for (DeformableAttachmentAim& attachment : geometry_aims) {
                const auto found = affine_output_offsets_.find(
                        AffineEnvironmentKey(attachment.environment,
                                             attachment.rigid_link_path));
                if (found == affine_output_offsets_.end()) {
                    throw std::runtime_error(
                            "libuipc deformable attachment has no affine proxy mapping");
                }
                attachment.affine_output_offset = found->second;
                const Matrix4x4& initial =
                        initial_affine_transforms_[found->second];
                attachment.link_local_position =
                        TransformPoint(initial.inverse(),
                                       attachment.link_local_position);
            }
        }
        for (DeformableAttachmentVertex& attachment :
             deformable_attachment_vertices_) {
            const auto found = affine_output_offsets_.find(
                    AffineEnvironmentKey(attachment.environment,
                                         attachment.rigid_link_path));
            if (found == affine_output_offsets_.end()) {
                throw std::runtime_error(
                        "libuipc deformable attachment reaction has no affine proxy mapping");
            }
            attachment.affine_output_offset = found->second;
            const Matrix4x4& initial = initial_affine_transforms_[found->second];
            attachment.link_local_position =
                    TransformPoint(initial.inverse(),
                                   attachment.link_local_position);
            attachment.link_local_center_of_mass =
                    affine_contact_ranges_.at(found->second)
                            .local_center_of_mass;
        }
    }

    void InitializeAccessors() {
        solver_diagnostics_ =
                world_->features().find<SolverDiagnosticsFeature>();
        solver_control_ = world_->features().find<SolverControlFeature>();
        if (solver_diagnostics_ == nullptr || solver_control_ == nullptr) {
            throw std::runtime_error(
                    "libuipc CUDA backend has no solver diagnostics/control feature");
        }
        fem_accessor_ = world_->features().find<FiniteElementStateAccessorFeature>();
        if (fem_accessor_ == nullptr) {
            throw std::runtime_error("libuipc CUDA backend has no FEM state accessor");
        }
        fem_state_ = std::make_unique<SimplicialComplex>(fem_accessor_->create_geometry());
        fem_state_->vertices().create<Vector3>(uipc::builtin::position);
        fem_state_->vertices().create<Vector3>(uipc::builtin::velocity);
        external_force_accessor_ = world_->features().find<
                FiniteElementExternalForceAccessorFeature>();
        if (external_force_accessor_ == nullptr) {
            throw std::runtime_error(
                    "libuipc CUDA backend has no FEM external-force accessor");
        }

        if (!affine_bodies_.empty()) {
            affine_accessor_ = world_->features().find<AffineBodyStateAccessorFeature>();
            if (affine_accessor_ == nullptr) {
                throw std::runtime_error("libuipc CUDA backend has no affine-body state accessor");
            }
            affine_state_ = std::make_unique<SimplicialComplex>(
                    affine_accessor_->create_geometry(
                            0, initial_affine_transforms_.size()));
            affine_state_->instances().create<Matrix4x4>(uipc::builtin::transform);
            affine_state_->instances().create<Matrix4x4>(uipc::builtin::velocity);
            if (external_affine_proxies_) {
                soft_transform_accessor_ =
                        world_->features()
                                .find<SoftTransformConstraintAccessorFeature>();
                if (soft_transform_accessor_ == nullptr) {
                    throw std::runtime_error(
                            "libuipc CUDA backend has no soft-transform device accessor");
                }
            }
        }
        if (!deformable_attachment_vertices_.empty()) {
            soft_position_accessor_ =
                    world_->features()
                            .find<SoftPositionConstraintAccessorFeature>();
        }
    }

    void InitializeDeviceCouplingWorkspace() {
        std::vector<DeviceDeformableContactRange> deformable_ranges;
        if (exact_contact_wrench_) {
            deformable_ranges.reserve(deformable_contact_ranges_.size());
            for (const DeformableContactRange& range :
                 deformable_contact_ranges_) {
                deformable_ranges.push_back(DeviceDeformableContactRange{
                        range.output_offset, range.vertex_count,
                        static_cast<std::int32_t>(range.global_vertex_offset)});
            }
        }

        std::vector<DeviceAffineContactRange> affine_ranges;
        if (exact_contact_wrench_) {
            affine_ranges.reserve(affine_contact_ranges_.size());
            for (const AffineContactRange& range : affine_contact_ranges_) {
                DeviceAffineContactRange value;
                value.output_offset = range.output_offset;
                value.global_vertex_offset =
                        static_cast<std::int32_t>(range.global_vertex_offset);
                value.local_vertices.reserve(range.local_vertices.size() * 3);
                for (const Vector3& vertex : range.local_vertices) {
                    value.local_vertices.push_back(vertex.x());
                    value.local_vertices.push_back(vertex.y());
                    value.local_vertices.push_back(vertex.z());
                }
                value.local_center_of_mass[0] =
                        range.local_center_of_mass.x();
                value.local_center_of_mass[1] =
                        range.local_center_of_mass.y();
                value.local_center_of_mass[2] =
                        range.local_center_of_mass.z();
                affine_ranges.push_back(std::move(value));
            }
        }

        std::vector<DeviceAttachmentVertex> attachment_vertices;
        attachment_vertices.reserve(deformable_attachment_vertices_.size());
        for (const DeformableAttachmentVertex& attachment :
             deformable_attachment_vertices_) {
            DeviceAttachmentVertex value;
            value.deformable_output_offset =
                    attachment.deformable_output_offset;
            value.affine_output_offset = attachment.affine_output_offset;
            value.link_local_position[0] = attachment.link_local_position.x();
            value.link_local_position[1] = attachment.link_local_position.y();
            value.link_local_position[2] = attachment.link_local_position.z();
            value.link_local_center_of_mass[0] =
                    attachment.link_local_center_of_mass.x();
            value.link_local_center_of_mass[1] =
                    attachment.link_local_center_of_mass.y();
            value.link_local_center_of_mass[2] =
                    attachment.link_local_center_of_mass.z();
            value.vertex_mass = attachment.vertex_mass;
            value.strength_rate = attachment.strength_rate;
            attachment_vertices.push_back(value);
        }

        device_workspace_ = std::make_unique<DeviceCouplingWorkspace>(
                device_index_, initial_fem_positions_.size(),
                initial_affine_transforms_.size(),
                std::move(deformable_ranges), std::move(affine_ranges),
                std::move(attachment_vertices));
    }

    void InitializeContactForceExport(double fixed_time_step) {
        contact_forces_.assign(initial_fem_positions_.size() * 3, 0.0);
        affine_contact_wrenches_.assign(initial_affine_transforms_.size() * 6,
                                        0.0);
        inverse_time_step_squared_ =
                1.0 / (fixed_time_step * fixed_time_step);

        // The current libuipc AL pipeline does not expose the simplex contact
        // gradient used by the exact wrench path. The composite provider uses
        // the affine proxy-constraint reaction for AL-IPC instead.
        if (!exact_contact_wrench_) {
            return;
        }

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
        for (AffineContactRange& range : affine_contact_ranges_) {
            auto global_offset = range.geometry->geometry().meta().find<IndexT>(
                    uipc::builtin::global_vertex_offset);
            if (global_offset == nullptr) {
                throw std::runtime_error(
                        "libuipc affine geometry has no global vertex offset");
            }
            const auto offsets = view(*global_offset);
            if (offsets.size() != 1 || offsets[0] < 0) {
                throw std::runtime_error(
                        "libuipc affine geometry has an invalid global vertex offset");
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
        device_contact_gradient_views_.reserve(
            contact_gradient_buffers_.size());
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

    std::optional<std::pair<std::size_t, std::size_t>> FindAffineVertex(
            IndexT global_vertex) const {
        for (std::size_t range_index = 0;
             range_index < affine_contact_ranges_.size(); ++range_index) {
            const AffineContactRange& range = affine_contact_ranges_[range_index];
            if (global_vertex < range.global_vertex_offset) {
                continue;
            }
            const IndexT relative = global_vertex - range.global_vertex_offset;
            if (static_cast<std::size_t>(relative) < range.vertex_count) {
                return std::pair{range_index, static_cast<std::size_t>(relative)};
            }
        }
        return std::nullopt;
    }

    void RefreshContactForces() {
        std::ranges::fill(contact_forces_, 0.0);
        std::ranges::fill(affine_contact_wrenches_, 0.0);
        std::span<const Matrix4x4> affine_transforms;
        if (!affine_contact_ranges_.empty()) {
            auto transforms = affine_state_->instances().find<Matrix4x4>(
                    uipc::builtin::transform);
            if (transforms == nullptr) {
                throw std::runtime_error(
                        "libuipc affine state has no transforms for contact wrench export");
            }
            affine_transforms = view(*transforms);
        }
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
                // libuipc assembles dt^2 times the physical IPC potential.
                const Vector3 force =
                        -gradient_values[index] * inverse_time_step_squared_;
                if (!force.allFinite()) {
                    throw std::runtime_error(
                            "libuipc contact-gradient exporter returned a non-finite force");
                }
                if (const auto output_vertex =
                            FindDeformableVertex(index_values[index]);
                    output_vertex.has_value()) {
                    for (std::size_t axis = 0; axis < 3; ++axis) {
                        double& value = contact_forces_[*output_vertex * 3 + axis];
                        value += force[static_cast<Eigen::Index>(axis)];
                        if (!std::isfinite(value)) {
                            throw std::runtime_error(
                                    "libuipc accumulated a non-finite deformable contact force");
                        }
                    }
                    continue;
                }
                const auto affine_vertex = FindAffineVertex(index_values[index]);
                if (!affine_vertex.has_value()) {
                    continue;
                }
                const AffineContactRange& range =
                        affine_contact_ranges_[affine_vertex->first];
                if (range.output_offset >= affine_transforms.size() ||
                    affine_vertex->second >= range.local_vertices.size()) {
                    throw std::runtime_error(
                            "libuipc affine contact vertex mapping is out of range");
                }
                const Matrix4x4& transform = affine_transforms[range.output_offset];
                const Vector3 world_point = TransformPoint(
                        transform, range.local_vertices[affine_vertex->second]);
                const Vector3 world_center = TransformPoint(
                        transform, range.local_center_of_mass);
                const Vector3 torque = (world_point - world_center).cross(force);
                if (!torque.allFinite()) {
                    throw std::runtime_error(
                            "libuipc accumulated a non-finite affine contact torque");
                }
                for (std::size_t axis = 0; axis < 3; ++axis) {
                    double& force_value = affine_contact_wrenches_[
                            range.output_offset * 6 + axis];
                    double& torque_value = affine_contact_wrenches_[
                            range.output_offset * 6 + 3 + axis];
                    force_value += force[static_cast<Eigen::Index>(axis)];
                    torque_value += torque[static_cast<Eigen::Index>(axis)];
                    if (!std::isfinite(force_value) ||
                        !std::isfinite(torque_value)) {
                        throw std::runtime_error(
                                "libuipc accumulated a non-finite affine contact wrench");
                    }
                }
            }
        }
        AccumulateDeformableAttachmentWrenches();
    }

    void AccumulateDeformableAttachmentWrenches() {
        if (deformable_attachment_vertices_.empty()) {
            return;
        }
        auto positions = fem_state_->vertices().find<Vector3>(
                uipc::builtin::position);
        if (positions == nullptr) {
            throw std::runtime_error(
                    "libuipc FEM state has no positions for attachment wrench export");
        }
        const auto position_values = view(*positions);
        for (const DeformableAttachmentVertex& attachment :
             deformable_attachment_vertices_) {
            if (attachment.deformable_output_offset >= position_values.size() ||
                attachment.affine_output_offset >=
                        attachment_target_transforms_.size()) {
                throw std::runtime_error(
                        "libuipc deformable attachment wrench mapping is out of range");
            }
            const Matrix4x4& transform =
                    attachment_target_transforms_[attachment.affine_output_offset];
            const Vector3 target = TransformPoint(
                    transform, attachment.link_local_position);
            // SoftPositionConstraint contributes s*m*(x-aim) directly to the
            // incremental potential. Dividing by dt^2 recovers physical force.
            const Vector3 force_on_link =
                    attachment.strength_rate * attachment.vertex_mass *
                    (position_values[attachment.deformable_output_offset] - target) *
                    inverse_time_step_squared_;
            const Vector3 world_center = TransformPoint(
                    transform, attachment.link_local_center_of_mass);
            const Vector3 torque_on_link =
                    (target - world_center).cross(force_on_link);
            if (!force_on_link.allFinite() || !torque_on_link.allFinite()) {
                throw std::runtime_error(
                        "libuipc deformable attachment produced a non-finite wrench");
            }
            for (std::size_t axis = 0; axis < 3; ++axis) {
                affine_contact_wrenches_[attachment.affine_output_offset * 6 + axis] +=
                        force_on_link[static_cast<Eigen::Index>(axis)];
                affine_contact_wrenches_[attachment.affine_output_offset * 6 + 3 + axis] +=
                        torque_on_link[static_cast<Eigen::Index>(axis)];
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
            if (affine_accessor_ != nullptr) {
                affine_accessor_->copy_to(*affine_state_);
            }
            RefreshContactForces();
        } else {
            std::ranges::fill(contact_forces_, 0.0);
            std::ranges::fill(affine_contact_wrenches_, 0.0);
        }
    }

    std::unordered_map<std::string, IpcSolverArtifactBlobView> blobs_;
    std::unique_ptr<Engine> engine_;
    std::unique_ptr<World> world_;
    std::unique_ptr<Scene> scene_;
    std::shared_ptr<FiniteElementStateAccessorFeature> fem_accessor_;
    std::shared_ptr<FiniteElementExternalForceAccessorFeature>
            external_force_accessor_;
    std::shared_ptr<AffineBodyStateAccessorFeature> affine_accessor_;
    std::shared_ptr<ContactSystemFeature> contact_system_;
    std::shared_ptr<SoftPositionConstraintAccessorFeature>
        soft_position_accessor_;
    std::shared_ptr<SoftTransformConstraintAccessorFeature>
        soft_transform_accessor_;
    std::shared_ptr<SolverDiagnosticsFeature> solver_diagnostics_;
    std::shared_ptr<SolverControlFeature> solver_control_;
    std::unique_ptr<SimplicialComplex> fem_state_;
    std::unique_ptr<SimplicialComplex> affine_state_;
    std::vector<BodyRecord> deformable_bodies_;
    std::vector<DeformableContactRange> deformable_contact_ranges_;
    std::vector<std::vector<DeformableAttachmentAim>> attachment_aims_by_geometry_;
    std::vector<DeformableAttachmentVertex> deformable_attachment_vertices_;
    std::vector<AffineContactRange> affine_contact_ranges_;
    std::vector<BodyRecord> affine_bodies_;
    std::vector<std::vector<ContactElementRecord>>
            proxy_contact_elements_by_environment_;
    std::vector<ContactGradientBuffer> contact_gradient_buffers_;
    std::vector<DeviceContactGradientView> device_contact_gradient_views_;
    std::unordered_map<std::string, std::shared_ptr<AffineTarget>> affine_targets_;
    std::vector<std::shared_ptr<AffineTarget>> external_affine_targets_;
    std::unordered_map<std::string, std::shared_ptr<JointTarget>> joint_targets_;
    std::unordered_map<std::string, std::size_t> affine_output_offsets_;
    std::vector<Vector3> initial_fem_positions_;
    std::vector<Matrix4x4> initial_affine_transforms_;
    std::vector<Matrix4x4> attachment_target_transforms_;
    std::vector<double> positions_;
    std::vector<double> velocities_;
    std::vector<double> contact_forces_;
    std::vector<double> affine_contact_wrenches_;
    std::vector<double> affine_transforms_;
    PinnedDoubleBuffer target_row_major_;
    PinnedDoubleBuffer target_twists_;
    std::unique_ptr<DeviceCouplingWorkspace> device_workspace_;
    std::optional<std::uint64_t> checkpoint_frame_;
    std::vector<Matrix4x4> checkpoint_affine_targets_;
    std::vector<Matrix4x4> checkpoint_attachment_targets_;
    IpcBatchSolverModuleBuffers device_buffers_{};
    double inverse_time_step_squared_{0.0};
    std::uint64_t frame_{0};
    std::uint64_t contact_force_host_frame_{0};
    std::uint64_t deformable_contact_force_frame_{0};
    double last_step_latency_ms_{0.0};
    double last_target_staging_latency_ms_{0.0};
    double last_ipc_advance_latency_ms_{0.0};
    double last_reaction_export_latency_ms_{0.0};
    double last_state_sync_latency_ms_{0.0};
    std::size_t environment_count_{1};
    std::size_t static_collider_count_per_environment_{0};
    std::uint32_t device_index_{0};
    bool external_affine_proxies_{false};
    std::string contact_constitution_{"ipc"};
    bool exact_contact_wrench_{true};
    bool device_native_coupling_enabled_{true};
    std::uint32_t output_flags_{IpcBatchSolverOutputAll};
    bool contact_forces_current_{true};
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
          device_index_(config.solver.device_index),
          contact_constitution_(
                  config.contact_constitution != nullptr
                          ? config.contact_constitution
                          : ""),
          output_flags_(config.output_flags),
          workspace_suffix_(MakeBatchWorkspaceSuffix()) {
        if (environment_count_ == 0 || environments_per_shard_ == 0 ||
            environment_count_ % environments_per_shard_ != 0) {
            throw std::runtime_error(
                    "libuipc batch environment count must be a positive "
                    "multiple of environments_per_shard");
        }
        if (!config.external_affine_proxies) {
            throw std::runtime_error(
                    "libuipc batch v5 requires external affine proxies");
        }
        ValidateOutputFlags(output_flags_);
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
                            .string(),
                    contact_constitution_, config.al_ipc_mu_scale_fem,
                    config.al_ipc_mu_scale_abd,
                    config.al_ipc_toi_threshold,
                    config.al_ipc_alpha_lower_bound,
                    config.al_ipc_decay_factor,
                    config.newton_max_iterations,
                    config.line_search_max_iterations,
                    config.linear_system_tolerance_rate,
                    config.strict_convergence,
                    output_flags_));
        }
        const auto& first_deformables = shards_.front()->DeformableBodies();
        const auto& first_affines = shards_.front()->AffineBodies();
        static_collider_count_per_environment_ =
                shards_.front()->StaticColliderCountPerEnvironment();
        deformable_bodies_ = first_deformables;
        affine_bodies_ = first_affines;
        for (const auto& shard : shards_) {
            if (shard->DeformableBodies().size() !=
                        deformable_bodies_.size() ||
                shard->AffineBodies().size() != affine_bodies_.size() ||
                shard->StaticColliderCountPerEnvironment() !=
                        static_collider_count_per_environment_) {
                throw std::runtime_error(
                        "libuipc batch shards produced inconsistent layouts");
            }
        }
        for (const BodyRecord& body : deformable_bodies_) {
            deformable_vertex_count_per_environment_ += body.count;
        }
        UpdateSolverDiagnostics();
    }

    ~BatchSession() {
        if (external_input_event_ != nullptr) {
            cudaSetDevice(static_cast<int>(device_index_));
            cudaEventDestroy(external_input_event_);
        }
    }

    void SetExecutionContext(const IpcBatchSolverExecutionContext& context) {
        RequireCuda(cudaSetDevice(static_cast<int>(device_index_)),
                    "selecting the libuipc batch CUDA device");
        if (context.cuda_stream != 0) {
            if (external_input_event_ == nullptr) {
                RequireCuda(cudaEventCreateWithFlags(
                                    &external_input_event_,
                                    cudaEventDisableTiming),
                            "creating the libuipc input-ready event");
            }
            auto stream = reinterpret_cast<cudaStream_t>(context.cuda_stream);
            RequireCuda(cudaEventRecord(external_input_event_, stream),
                        "recording the libuipc input-ready event");
            RequireCuda(cudaStreamWaitEvent(nullptr, external_input_event_, 0),
                        "waiting for libuipc device inputs");
        }
        cuda_stream_interop_ = true;
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
            local.deformable_external_forces = OffsetDeviceBuffer(
                    buffers.deformable_external_forces,
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
            local.affine_target_twists = OffsetDeviceBuffer(
                    buffers.affine_target_twists,
                    environment_offset * affines * 6,
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
        const auto start = SteadyClock::now();
        last_target_staging_latency_ms_ = 0.0;
        last_ipc_advance_latency_ms_ = 0.0;
        last_reaction_export_latency_ms_ = 0.0;
        last_state_sync_latency_ms_ = 0.0;
        failing_shard_index_ = static_cast<std::size_t>(-1);
        for (std::size_t index = 0; index < shards_.size(); ++index) {
            const auto& shard = shards_[index];
            try {
                shard->StepDevice(steps);
            } catch (...) {
                failing_shard_index_ = index;
                UpdateSolverDiagnostics();
                throw;
            }
            last_target_staging_latency_ms_ +=
                    shard->LastTargetStagingLatencyMs();
            last_ipc_advance_latency_ms_ +=
                    shard->LastIpcAdvanceLatencyMs();
            last_reaction_export_latency_ms_ +=
                    shard->LastReactionExportLatencyMs();
            last_state_sync_latency_ms_ +=
                    shard->LastStateSyncLatencyMs();
        }
        frame_ += steps;
        last_step_latency_ms_ = ElapsedMilliseconds(start);
        UpdateSolverDiagnostics();
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
        checkpoint_frame_.reset();
        failing_shard_index_ = static_cast<std::size_t>(-1);
        UpdateSolverDiagnostics();
    }

    void CaptureCheckpoint() {
        if (!buffers_bound_) {
            throw std::runtime_error(
                    "libuipc batch device buffers are not bound");
        }
        if (checkpoint_frame_.has_value()) {
            throw std::runtime_error(
                    "libuipc batch checkpoint slot is already active");
        }
        const auto start = SteadyClock::now();
        for (const auto& shard : shards_) {
            shard->CaptureDeviceCheckpoint();
        }
        checkpoint_frame_ = frame_;
        last_checkpoint_latency_ms_ = ElapsedMilliseconds(start);
    }

    void RewindCheckpoint() {
        if (!checkpoint_frame_.has_value()) {
            throw std::runtime_error(
                    "libuipc batch checkpoint slot is not active");
        }
        const auto start = SteadyClock::now();
        for (const auto& shard : shards_) {
            shard->RewindDeviceCheckpoint();
        }
        frame_ = *checkpoint_frame_;
        last_checkpoint_latency_ms_ = ElapsedMilliseconds(start);
        UpdateSolverDiagnostics();
    }

    void CommitCheckpoint() {
        if (!checkpoint_frame_.has_value()) {
            throw std::runtime_error(
                    "libuipc batch checkpoint slot is not active");
        }
        const auto start = SteadyClock::now();
        for (const auto& shard : shards_) {
            shard->CommitDeviceCheckpoint();
        }
        checkpoint_frame_.reset();
        last_checkpoint_latency_ms_ = ElapsedMilliseconds(start);
    }

    void Synchronize() {
        for (const auto& shard : shards_) {
            shard->SynchronizeDevice();
        }
        RequireCuda(cudaDeviceSynchronize(),
                    "synchronizing libuipc batch shards");
    }

    void SetOutputFlags(std::uint32_t output_flags) {
        ValidateOutputFlags(output_flags);
        for (const auto& shard : shards_) {
            shard->SetOutputFlags(output_flags);
        }
        output_flags_ = output_flags;
    }

    void RefreshOutputs(std::uint32_t output_flags) {
        ValidateOutputFlags(output_flags);
        if (output_flags == IpcBatchSolverOutputNone) {
            throw std::runtime_error(
                    "libuipc batch refresh output flags are empty");
        }
        last_reaction_export_latency_ms_ = 0.0;
        last_state_sync_latency_ms_ = 0.0;
        for (const auto& shard : shards_) {
            shard->RefreshOutputs(output_flags);
            last_reaction_export_latency_ms_ +=
                    shard->LastReactionExportLatencyMs();
            last_state_sync_latency_ms_ +=
                    shard->LastStateSyncLatencyMs();
        }
    }

    void SetRuntimeOptions(const IpcBatchSolverRuntimeOptions& options) {
        if (options.newton_max_iterations == 0 ||
            options.line_search_max_iterations == 0 ||
            !std::isfinite(options.linear_system_tolerance_rate) ||
            options.linear_system_tolerance_rate <= 0.0) {
            throw std::runtime_error(
                    "libuipc runtime solver options are invalid");
        }
        for (const auto& shard : shards_) {
            shard->SetRuntimeOptions(options);
        }
        UpdateSolverDiagnostics();
    }

    const SolverDiagnostics& LastSolverDiagnostics() const {
        return solver_diagnostics_;
    }
    std::size_t FailingShardIndex() const { return failing_shard_index_; }

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
    std::size_t StaticColliderCountPerEnvironment() const {
        return static_collider_count_per_environment_;
    }
    std::uint64_t Frame() const { return frame_; }
    double LastStepLatencyMs() const { return last_step_latency_ms_; }
    double LastCheckpointLatencyMs() const {
        return last_checkpoint_latency_ms_;
    }
    double LastTargetStagingLatencyMs() const {
        return last_target_staging_latency_ms_;
    }
    double LastIpcAdvanceLatencyMs() const {
        return last_ipc_advance_latency_ms_;
    }
    double LastReactionExportLatencyMs() const {
        return last_reaction_export_latency_ms_;
    }
    double LastStateSyncLatencyMs() const {
        return last_state_sync_latency_ms_;
    }
    std::uint32_t OutputFlags() const { return output_flags_; }
    std::uint64_t DeformableContactForceFrame() const {
        if (shards_.empty()) {
            return 0;
        }
        return std::ranges::min(
                shards_ | std::views::transform([](const auto& shard) {
                    return shard->DeformableContactForceFrame();
                }));
    }
    bool DeviceNativeCoupling() const {
        return std::ranges::all_of(shards_, [](const auto& shard) {
            return shard->DeviceNativeCoupling();
        });
    }
    bool CudaStreamInterop() const { return cuda_stream_interop_; }
    std::size_t DeviceWorkspaceAllocationCount() const {
        std::size_t result = 0;
        for (const auto& shard : shards_) {
            result += shard->DeviceWorkspaceAllocationCount();
        }
        return result;
    }
    const std::string& ContactConstitution() const {
        return contact_constitution_;
    }
    bool ExactContactWrench() const {
        return !shards_.empty() && shards_.front()->ExactContactWrench();
    }
    bool CheckpointActive() const { return checkpoint_frame_.has_value(); }
    bool IsValid() const {
        return std::ranges::all_of(
                shards_, [](const auto& shard) { return shard->IsValid(); });
    }

private:
    void UpdateSolverDiagnostics() {
        if (shards_.empty()) {
            solver_diagnostics_ = {};
            return;
        }
        const std::size_t selected =
                failing_shard_index_ < shards_.size()
                        ? failing_shard_index_
                        : 0;
        solver_diagnostics_ = shards_[selected]->GetSolverDiagnostics();
        if (failing_shard_index_ < shards_.size()) {
            return;
        }
        for (std::size_t index = 1; index < shards_.size(); ++index) {
            const SolverDiagnostics current =
                    shards_[index]->GetSolverDiagnostics();
            if (solver_diagnostics_.failure_kind ==
                        uipc::core::SolverFailureKind::None &&
                current.failure_kind !=
                        uipc::core::SolverFailureKind::None) {
                solver_diagnostics_.stage = current.stage;
                solver_diagnostics_.failure_kind = current.failure_kind;
                solver_diagnostics_.failure_message = current.failure_message;
            }
            solver_diagnostics_.newton_iterations = std::max(
                    solver_diagnostics_.newton_iterations,
                    current.newton_iterations);
            solver_diagnostics_.line_search_iterations_total = std::max(
                    solver_diagnostics_.line_search_iterations_total,
                    current.line_search_iterations_total);
            solver_diagnostics_.line_search_iterations_max = std::max(
                    solver_diagnostics_.line_search_iterations_max,
                    current.line_search_iterations_max);
            solver_diagnostics_.pcg_iterations_total = std::max(
                    solver_diagnostics_.pcg_iterations_total,
                    current.pcg_iterations_total);
            solver_diagnostics_.pcg_iterations_max = std::max(
                    solver_diagnostics_.pcg_iterations_max,
                    current.pcg_iterations_max);
            solver_diagnostics_.pcg_iterations_last = std::max(
                    solver_diagnostics_.pcg_iterations_last,
                    current.pcg_iterations_last);
            solver_diagnostics_.pcg_relative_residual = std::max(
                    solver_diagnostics_.pcg_relative_residual,
                    current.pcg_relative_residual);
            solver_diagnostics_.minimum_step_length = std::min(
                    solver_diagnostics_.minimum_step_length,
                    current.minimum_step_length);
            solver_diagnostics_.newton_converged =
                    solver_diagnostics_.newton_converged &&
                    current.newton_converged;
            solver_diagnostics_.linear_system_converged =
                    solver_diagnostics_.linear_system_converged &&
                    current.linear_system_converged;
            solver_diagnostics_.strict_mode =
                    solver_diagnostics_.strict_mode && current.strict_mode;
            solver_diagnostics_.recovered =
                    solver_diagnostics_.recovered && current.recovered;
        }
    }

    static void RequireCuda(cudaError_t result, std::string_view operation) {
        if (result != cudaSuccess) {
            throw std::runtime_error(
                    std::string(operation) + ": " +
                    cudaGetErrorString(result));
        }
    }

    std::size_t environment_count_{0};
    std::size_t environments_per_shard_{0};
    std::uint32_t device_index_{0};
    std::size_t deformable_vertex_count_per_environment_{0};
    std::size_t static_collider_count_per_environment_{0};
    std::string contact_constitution_;
    std::uint32_t output_flags_{IpcBatchSolverOutputAll};
    std::string workspace_suffix_;
    WorkspaceLease workspace_lease_;
    std::vector<std::unique_ptr<Session>> shards_;
    std::vector<BodyRecord> deformable_bodies_;
    std::vector<BodyRecord> affine_bodies_;
    std::uint64_t frame_{0};
    double last_step_latency_ms_{0.0};
    double last_checkpoint_latency_ms_{0.0};
    double last_target_staging_latency_ms_{0.0};
    double last_ipc_advance_latency_ms_{0.0};
    double last_reaction_export_latency_ms_{0.0};
    double last_state_sync_latency_ms_{0.0};
    std::optional<std::uint64_t> checkpoint_frame_;
    SolverDiagnostics solver_diagnostics_;
    std::size_t failing_shard_index_{static_cast<std::size_t>(-1)};
    bool buffers_bound_{false};
    bool cuda_stream_interop_{false};
    cudaEvent_t external_input_event_{nullptr};
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

bool BatchSetExecutionContext(
        void* session,
        const IpcBatchSolverExecutionContext* context,
        char* error,
        std::size_t error_size) {
    return Guard(error, error_size, [&] {
        if (context == nullptr) {
            throw std::runtime_error(
                    "libuipc batch execution context is null");
        }
        CastBatch(session)->SetExecutionContext(*context);
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

bool BatchCaptureCheckpoint(void* session,
                            char* error,
                            std::size_t error_size) {
    return Guard(error, error_size,
                 [&] { CastBatch(session)->CaptureCheckpoint(); });
}

bool BatchRewindCheckpoint(void* session,
                           char* error,
                           std::size_t error_size) {
    return Guard(error, error_size,
                 [&] { CastBatch(session)->RewindCheckpoint(); });
}

bool BatchCommitCheckpoint(void* session,
                           char* error,
                           std::size_t error_size) {
    return Guard(error, error_size,
                 [&] { CastBatch(session)->CommitCheckpoint(); });
}

bool BatchSynchronize(void* session, char* error, std::size_t error_size) {
    return Guard(error, error_size,
                 [&] { CastBatch(session)->Synchronize(); });
}

bool BatchSetOutputFlags(void* session,
                         std::uint32_t output_flags,
                         char* error,
                         std::size_t error_size) {
    return Guard(error, error_size, [&] {
        CastBatch(session)->SetOutputFlags(output_flags);
    });
}

bool BatchRefreshOutputs(void* session,
                         std::uint32_t output_flags,
                         char* error,
                         std::size_t error_size) {
    return Guard(error, error_size, [&] {
        CastBatch(session)->RefreshOutputs(output_flags);
    });
}

bool BatchSetRuntimeOptions(
        void* session,
        const IpcBatchSolverRuntimeOptions* options,
        char* error,
        std::size_t error_size) {
    return Guard(error, error_size, [&] {
        if (options == nullptr) {
            throw std::runtime_error(
                    "libuipc runtime solver options are null");
        }
        CastBatch(session)->SetRuntimeOptions(*options);
    });
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
        const SolverDiagnostics& solver = value->LastSolverDiagnostics();
        IpcBatchSolverModuleDiagnostics result;
        result.frame = value->Frame();
        result.environment_count = value->EnvironmentCount();
        result.shard_count = value->ShardCount();
        result.deformable_body_count_per_environment =
                value->DeformableBodies().size();
        result.deformable_vertex_count_per_environment =
                value->DeformableVertexCountPerEnvironment();
        result.affine_body_count_per_environment =
                value->AffineBodies().size();
        result.static_collider_count_per_environment =
                value->StaticColliderCountPerEnvironment();
        result.last_step_latency_ms = value->LastStepLatencyMs();
        result.last_checkpoint_latency_ms = value->LastCheckpointLatencyMs();
        result.last_target_staging_latency_ms =
                value->LastTargetStagingLatencyMs();
        result.last_ipc_advance_latency_ms =
                value->LastIpcAdvanceLatencyMs();
        result.last_reaction_export_latency_ms =
                value->LastReactionExportLatencyMs();
        result.last_state_sync_latency_ms = value->LastStateSyncLatencyMs();
        result.output_flags = value->OutputFlags();
        result.deformable_contact_force_frame =
                value->DeformableContactForceFrame();
        result.contact_constitution = value->ContactConstitution().c_str();
        result.exact_contact_wrench = value->ExactContactWrench();
        result.checkpoint_active = value->CheckpointActive();
        result.device_native_coupling = value->DeviceNativeCoupling();
        result.cuda_stream_interop = value->CudaStreamInterop();
        result.device_workspace_allocation_count =
                value->DeviceWorkspaceAllocationCount();
        result.solver_stage = static_cast<IpcSolverPipelineStage>(solver.stage);
        result.solver_failure =
                static_cast<IpcSolverFailureKind>(solver.failure_kind);
        result.newton_iterations = static_cast<std::uint32_t>(
                solver.newton_iterations);
        result.line_search_iterations_total = static_cast<std::uint32_t>(
                solver.line_search_iterations_total);
        result.line_search_iterations_max = static_cast<std::uint32_t>(
                solver.line_search_iterations_max);
        result.pcg_iterations_total = static_cast<std::uint32_t>(
                solver.pcg_iterations_total);
        result.pcg_iterations_max = static_cast<std::uint32_t>(
                solver.pcg_iterations_max);
        result.pcg_iterations_last = static_cast<std::uint32_t>(
                solver.pcg_iterations_last);
        result.pcg_relative_residual = solver.pcg_relative_residual;
        result.minimum_step_length = solver.minimum_step_length;
        result.solver_failure_message = solver.failure_message.c_str();
        result.failing_shard_index = value->FailingShardIndex();
        result.newton_converged = solver.newton_converged;
        result.linear_system_converged = solver.linear_system_converged;
        result.strict_convergence = solver.strict_mode;
        result.recovered = solver.recovered;
        result.valid = value->IsValid();
        *diagnostics = result;
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
        &BatchSetExecutionContext,
        &BatchStep,
        &BatchResetFull,
        &BatchCaptureCheckpoint,
        &BatchRewindCheckpoint,
        &BatchCommitCheckpoint,
        &BatchSynchronize,
        &BatchSetOutputFlags,
        &BatchRefreshOutputs,
        &BatchSetRuntimeOptions,
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
