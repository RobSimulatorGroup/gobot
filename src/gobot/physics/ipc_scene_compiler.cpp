/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/physics/ipc_scene_compiler.hpp"

#include <algorithm>
#include <bit>
#include <array>
#include <cmath>
#include <map>
#include <limits>
#include <set>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "gobot/core/sha256.hpp"
#include "gobot/core/math/mesh_validation.hpp"
#include "gobot/core/types.hpp"
#include "gobot/physics/physics_types.hpp"

namespace gobot {
namespace {

constexpr std::string_view kMeshEncoding = "gobot.tetrahedral-mesh.le.v1";
constexpr std::string_view kSurfaceMeshEncoding = "gobot.triangle-mesh.le.v1";

struct TetrahedralMeshView {
    std::span<const Vector3> vertices;
    std::span<const std::uint32_t> tetrahedra;
    std::span<const std::uint32_t> surface;
};

struct SurfaceMeshData {
    std::span<const Vector3> vertices;
    std::span<const std::uint32_t> triangles;
};

constexpr std::string_view ProducerVersion() {
#ifdef GOBOT_VERSION
    return GOBOT_VERSION;
#else
    return "unknown";
#endif
}

bool SetCompileError(std::string* error, std::string message) {
    if (error != nullptr) {
        *error = std::move(message);
    }
    return false;
}

void AppendU32(std::vector<std::uint8_t>* output, std::uint32_t value) {
    for (std::size_t index = 0; index < 4; ++index) {
        output->push_back(static_cast<std::uint8_t>((value >> (index * 8U)) & 0xffU));
    }
}

void AppendU64(std::vector<std::uint8_t>* output, std::uint64_t value) {
    for (std::size_t index = 0; index < 8; ++index) {
        output->push_back(static_cast<std::uint8_t>((value >> (index * 8U)) & 0xffU));
    }
}

void AppendF64(std::vector<std::uint8_t>* output, double value) {
    AppendU64(output, std::bit_cast<std::uint64_t>(value));
}

std::vector<std::uint8_t> EncodeMesh(const TetrahedralMeshView& mesh) {
    const auto surface = mesh.surface;
    std::vector<std::uint8_t> data;
    data.reserve(24 + mesh.vertices.size() * 24 +
                 mesh.tetrahedra.size() * 4 + surface.size() * 4);
    constexpr std::array<std::uint8_t, 8> magic{'G', 'O', 'B', 'T', 'I', 'P', 'C', '1'};
    data.insert(data.end(), magic.begin(), magic.end());
    AppendU32(&data, 1);
    AppendU32(&data, static_cast<std::uint32_t>(mesh.vertices.size()));
    AppendU32(&data, static_cast<std::uint32_t>(mesh.tetrahedra.size() / 4));
    AppendU32(&data, static_cast<std::uint32_t>(surface.size() / 3));
    for (const Vector3& vertex : mesh.vertices) {
        AppendF64(&data, static_cast<double>(vertex.x()));
        AppendF64(&data, static_cast<double>(vertex.y()));
        AppendF64(&data, static_cast<double>(vertex.z()));
    }
    for (const std::uint32_t index : mesh.tetrahedra) {
        AppendU32(&data, index);
    }
    for (const std::uint32_t index : surface) {
        AppendU32(&data, index);
    }
    return data;
}

std::vector<std::uint8_t> EncodeSurfaceMesh(const SurfaceMeshData& mesh) {
    std::vector<std::uint8_t> data;
    data.reserve(20 + mesh.vertices.size() * 24 + mesh.triangles.size() * 4);
    constexpr std::array<std::uint8_t, 8> magic{'G', 'O', 'B', 'T', 'T', 'R', 'I', '1'};
    data.insert(data.end(), magic.begin(), magic.end());
    AppendU32(&data, 1);
    AppendU32(&data, static_cast<std::uint32_t>(mesh.vertices.size()));
    AppendU32(&data, static_cast<std::uint32_t>(mesh.triangles.size() / 3));
    for (const Vector3& vertex : mesh.vertices) {
        AppendF64(&data, static_cast<double>(vertex.x()));
        AppendF64(&data, static_cast<double>(vertex.y()));
        AppendF64(&data, static_cast<double>(vertex.z()));
    }
    for (const std::uint32_t index : mesh.triangles) {
        AppendU32(&data, index);
    }
    return data;
}

std::string MeshTopologyDigest(const TetrahedralMeshView& mesh) {
    const auto surface = mesh.surface;
    std::vector<std::uint8_t> data;
    constexpr std::array<std::uint8_t, 12> magic{
            'G', 'O', 'B', 'T', 'I', 'P', 'C', 'T', 'O', 'P', '1', 0};
    data.insert(data.end(), magic.begin(), magic.end());
    AppendU32(&data, static_cast<std::uint32_t>(mesh.vertices.size()));
    AppendU32(&data, static_cast<std::uint32_t>(mesh.tetrahedra.size() / 4));
    AppendU32(&data, static_cast<std::uint32_t>(surface.size() / 3));
    for (const std::uint32_t index : mesh.tetrahedra) {
        AppendU32(&data, index);
    }
    for (const std::uint32_t index : surface) {
        AppendU32(&data, index);
    }
    return Sha256Digest(std::span<const std::uint8_t>(data));
}

Json Vector2Json(const Vector2& value) {
    return Json::array({value.x(), value.y()});
}

Json Vector3Json(const Vector3& value) {
    return Json::array({value.x(), value.y(), value.z()});
}

Json Vector4Json(const Vector4& value) {
    return Json::array({value.x(), value.y(), value.z(), value.w()});
}

Json QuaternionWxyzJson(const Quaternion& value) {
    return Json::array({value.w(), value.x(), value.y(), value.z()});
}

Json TransformJson(const Affine3& transform) {
    Json matrix = Json::array();
    for (int row = 0; row < 4; ++row) {
        for (int column = 0; column < 4; ++column) {
            matrix.push_back(transform.matrix()(row, column));
        }
    }
    return Json{{"matrix_row_major", std::move(matrix)}};
}

bool ValidateMaterial(const PhysicsDeformableSnapshot& body, std::string* error) {
    if (!std::isfinite(body.density) || body.density <= 0.0) {
        return SetCompileError(error, "deformable body density must be finite and positive");
    }
    if (!std::isfinite(body.young_modulus) || body.young_modulus <= 0.0) {
        return SetCompileError(error, "deformable body Young modulus must be finite and positive");
    }
    if (!std::isfinite(body.poisson_ratio) || body.poisson_ratio <= -1.0 ||
        body.poisson_ratio >= 0.5) {
        return SetCompileError(error, "deformable body Poisson ratio must be in (-1, 0.5)");
    }
    if (!std::isfinite(body.damping) || body.damping < 0.0) {
        return SetCompileError(error, "deformable body damping must be finite and non-negative");
    }
    if (body.model == static_cast<int>(DeformableBodyModel::ThinShell)) {
        if (!std::isfinite(body.thickness) || body.thickness <= 0.0) {
            return SetCompileError(
                    error, "thin-shell deformable thickness must be finite and positive");
        }
        if (!std::isfinite(body.bending_stiffness) ||
            body.bending_stiffness < 0.0) {
            return SetCompileError(
                    error,
                    "thin-shell deformable bending stiffness must be finite and non-negative");
        }
    }
    return true;
}

bool ValidateDeformableTransform(
        const Affine3& transform, std::string_view description, std::string* error) {
    if (!transform.matrix().allFinite()) {
        return SetCompileError(
                error, std::string(description) + " has a non-finite transform");
    }
    const RealType determinant = transform.linear().determinant();
    const RealType column_scale = transform.linear().col(0).norm() *
                                  transform.linear().col(1).norm() *
                                  transform.linear().col(2).norm();
    const RealType relative_tolerance =
            std::numeric_limits<RealType>::epsilon() * 128.0 * column_scale;
    if (!std::isfinite(determinant) || !std::isfinite(column_scale) ||
        column_scale <= 0.0 || determinant <= relative_tolerance) {
        return SetCompileError(
                error,
                std::string(description) +
                        " requires a finite, non-singular, orientation-preserving transform");
    }
    return true;
}

class CompilerState {
public:
    explicit CompilerState(const PhysicsSceneSnapshot& snapshot) : snapshot_(snapshot) {}

    bool Compile(std::string* error) {
        if (!snapshot_.terrains.empty()) {
            return SetCompileError(error, "Terrain3D is not supported by the IPC scene compiler; use loose static CollisionShape3D nodes instead");
        }
        for (const auto& body : snapshot_.deformables) {
            if (!AddDeformable(body, error)) return false;
        }
        for (const auto& robot : snapshot_.robots) {
            if (!AddRobot(robot, error)) return false;
        }
        // Preserve Scene preorder across sensors captured in different owners.
        std::vector<const PhysicsSensorSnapshot*> tactile;
        for (const auto& sensor : snapshot_.loose_sensors) {
            if (sensor.tactile) tactile.push_back(&sensor);
        }
        for (const auto& robot : snapshot_.robots) {
            for (const auto& sensor : robot.sensors) {
                if (sensor.tactile) tactile.push_back(&sensor);
            }
        }
        std::ranges::sort(tactile, {}, [](const auto* sensor) { return sensor->scene_order; });
        for (const auto* sensor : tactile) {
            if (!AddTactile(*sensor, error)) return false;
        }
        for (const auto& shape : snapshot_.loose_collision_shapes) {
            if (!AddCollisionShape(shape, nullptr, &static_colliders_, &static_collider_paths_, error)) return false;
        }
        return FinalizeCouplings(error) && FinalizeExternalFloatingBases(error) && FinalizeAttachments(error);
    }

    bool FinalizeAttachments(std::string* error) {
        struct PendingAttachment {
            std::string attachment_path;
            std::string deformable_body_path;
            std::string rigid_link_path;
            std::size_t proxy_index;
            RealType strength_rate;
            std::vector<std::uint32_t> vertex_indices;
        };

        std::unordered_map<std::string, std::size_t> proxy_indices;
        for (const Json& coupling : couplings_) {
            proxy_indices.emplace(
                    coupling.at("link_path").get<std::string>(),
                    coupling.at("proxy_index").get<std::size_t>());
        }
        std::unordered_set<std::string> deformable_paths;
        for (const Json& body : deformable_bodies_) {
            deformable_paths.insert(body.at("path").get<std::string>());
        }

        std::vector<PendingAttachment> pending;
        std::set<std::pair<std::string, std::uint32_t>> attached_vertices;
        for (const auto& attachment : snapshot_.deformable_attachments) {
            if (!attachment.enabled) {
                continue;
            }
            const std::string attachment_path = attachment.scene_path;
            const auto body_it = std::ranges::find(snapshot_.deformables, attachment.deformable_body_path, &PhysicsDeformableSnapshot::scene_path);
            if (body_it == snapshot_.deformables.end()) {
                return SetCompileError(error, "DeformableAttachment3D '" + attachment_path + "' deformable_body_path must resolve to a DeformableBody3D in the compiled scene");
            }
            const auto& body = *body_it;
            const auto& body_path = body.scene_path;
            if (!deformable_paths.contains(body_path) || body.kinematic || body.tetrahedra.empty()) {
                return SetCompileError(error, "DeformableAttachment3D '" + attachment_path + "' requires a compiled dynamic deformable body");
            }
            const auto& link_path = attachment.rigid_link_path;
            if (FindLink(link_path) == nullptr) {
                return SetCompileError(error, "DeformableAttachment3D '" + attachment_path + "' rigid_link_path must resolve to a Link3D in the compiled scene");
            }
            const auto proxy = proxy_indices.find(link_path);
            if (proxy == proxy_indices.end()) {
                return SetCompileError(
                        error, "DeformableAttachment3D '" + attachment_path +
                                       "' rigid Link3D requires an enabled PhysicsCoupling");
            }
            if (!std::isfinite(attachment.strength_rate) ||
                attachment.strength_rate <= 0.0) {
                return SetCompileError(
                        error, "DeformableAttachment3D '" + attachment_path +
                                       "' strength_rate must be finite and positive");
            }

            std::vector<std::uint32_t> indices = attachment.vertex_indices;
            std::ranges::sort(indices);
            if (indices.empty() ||
                std::ranges::adjacent_find(indices) != indices.end()) {
                return SetCompileError(
                        error, "DeformableAttachment3D '" + attachment_path +
                                       "' vertex_indices must be non-empty and unique");
            }
            const std::size_t vertex_count = body.vertices.size();
            for (const std::uint32_t vertex : indices) {
                if (vertex >= vertex_count) {
                    return SetCompileError(
                            error, "DeformableAttachment3D '" + attachment_path +
                                           "' contains an out-of-range vertex index");
                }
                if (!attached_vertices.emplace(body_path, vertex).second) {
                    return SetCompileError(
                            error, "multiple DeformableAttachment3D nodes target the same deformable vertex");
                }
            }
            pending.push_back(PendingAttachment{
                    attachment_path, body_path, link_path, proxy->second,
                    attachment.strength_rate, std::move(indices)});
        }

        std::ranges::sort(pending, {}, &PendingAttachment::attachment_path);
        for (const PendingAttachment& attachment : pending) {
            deformable_attachments_.push_back({
                    {"attachment_path", attachment.attachment_path},
                    {"deformable_body_path", attachment.deformable_body_path},
                    {"proxy_index", attachment.proxy_index},
                    {"rigid_link_path", attachment.rigid_link_path},
                    {"strength_rate", attachment.strength_rate},
                    {"vertex_indices", attachment.vertex_indices}});
        }
        return true;
    }

    bool FinalizeCouplings(std::string* error) {
        struct PendingCoupling {
            std::string coupling_path;
            std::string link_path;
            std::string robot_name;
            std::string link_name;
            int mode;
            RealType force_scale;
            RealType torque_scale;
        };

        std::vector<PendingCoupling> pending;
        std::unordered_set<std::string> linked_paths;
        for (const auto& coupling : snapshot_.couplings) {
            if (!coupling.enabled) {
                continue;
            }
            const std::string coupling_path = coupling.scene_path;
            const auto& link_path = coupling.rigid_link_path;
            const Json* compiled_link = nullptr;
            std::string robot_name;
            for (const Json& robot : robots_) {
                for (const Json& link : robot.at("links")) {
                    if (link.at("path") == link_path) {
                        compiled_link = &link;
                        robot_name = robot.at("name").get<std::string>();
                    }
                }
            }
            if (compiled_link == nullptr) {
                return SetCompileError(error, "PhysicsCoupling '" + coupling_path + "' target_body_path must resolve to a RigidBody3D or Link3D in the compiled scene");
            }
            const bool has_enabled_collision = std::ranges::any_of(
                    compiled_link->at("collision_shapes"), [](const Json& shape) {
                        return !shape.value("disabled", false);
                    });
            if (!has_enabled_collision) {
                return SetCompileError(
                        error, "PhysicsCoupling '" + coupling_path +
                                       "' target body has no enabled CollisionShape3D");
            }

            const int mode = static_cast<int>(coupling.mode);
            if (mode < static_cast<int>(PhysicsCouplingMode::OneWay) ||
                mode > static_cast<int>(PhysicsCouplingMode::TwoWay)) {
                return SetCompileError(
                        error, "PhysicsCoupling '" + coupling_path +
                                       "' has an invalid coupling mode");
            }
            if (!std::isfinite(coupling.force_scale) ||
                coupling.force_scale < 0.0 ||
                !std::isfinite(coupling.torque_scale) ||
                coupling.torque_scale < 0.0) {
                return SetCompileError(
                        error, "PhysicsCoupling '" + coupling_path +
                                       "' force and torque scales must be finite and non-negative");
            }
            if (!linked_paths.insert(link_path).second) {
                return SetCompileError(
                        error, "multiple enabled PhysicsCoupling nodes target rigid body '" +
                                       link_path + "'");
            }
            pending.push_back(PendingCoupling{
                    coupling_path,
                    link_path,
                    robot_name,
                    compiled_link->at("name").get<std::string>(),
                    coupling.mode,
                    coupling.force_scale,
                    coupling.torque_scale});
        }

        std::ranges::sort(pending, {}, &PendingCoupling::coupling_path);
        for (std::size_t proxy_index = 0; proxy_index < pending.size(); ++proxy_index) {
            const PendingCoupling& coupling = pending[proxy_index];
            couplings_.push_back({
                    {"coupling_path", coupling.coupling_path},
                    {"force_scale", coupling.force_scale},
                    {"link_name", coupling.link_name},
                    {"link_path", coupling.link_path},
                    {"mode", coupling.mode == static_cast<int>(PhysicsCouplingMode::OneWay)
                                     ? "OneWay"
                                     : "TwoWay"},
                    {"proxy_index", proxy_index},
                    {"robot_name", coupling.robot_name},
                    {"torque_scale", coupling.torque_scale}});
        }
        return true;
    }

    bool FinalizeExternalFloatingBases(std::string* error) const {
        std::unordered_set<std::string> coupled_link_paths;
        for (const Json& coupling : couplings_) {
            coupled_link_paths.insert(
                    coupling.at("link_path").get<std::string>());
        }
        for (const ExternalFloatingBase& base : external_floating_bases_) {
            if (!coupled_link_paths.contains(base.link_path)) {
                return SetCompileError(
                        error, "root floating joint '" + base.joint_path +
                                       "' requires an enabled PhysicsCoupling on its child Link3D");
            }
        }
        return true;
    }

    Json BuildManifest() const {
        Json blob_table = Json::array();
        for (const auto& [id, blob] : blobs_) {
            blob_table.push_back({
                    {"byte_length", blob.data.size()},
                    {"encoding", blob.encoding},
                    {"id", id},
                    {"sha256", blob.sha256}});
        }
        std::vector<Json> sorted_static_colliders;
        sorted_static_colliders.reserve(static_colliders_.size());
        for (const Json& collider : static_colliders_) {
            sorted_static_colliders.push_back(collider);
        }
        std::ranges::sort(sorted_static_colliders, {}, [](const Json& collider) {
            return collider.at("path").get<std::string>();
        });
        Json static_colliders = Json::array();
        for (Json& collider : sorted_static_colliders) {
            static_colliders.push_back(std::move(collider));
        }
        return Json{
                {"blobs", std::move(blob_table)},
                {"couplings", couplings_},
                {"deformable_attachments", deformable_attachments_},
                {"deformable_bodies", deformable_bodies_},
                {"format", "gobot-ipc"},
                {"producer", "gobot"},
                {"producer_version", ProducerVersion()},
                {"robots", robots_},
                {"scene_name", snapshot_.scene_name},
                {"schema_version", 5},
                {"static_colliders", std::move(static_colliders)},
                {"tactile_sensors", tactile_sensors_}};
    }

    std::vector<IpcSceneArtifactBlob> TakeBlobs() {
        std::vector<IpcSceneArtifactBlob> values;
        values.reserve(blobs_.size());
        for (auto& [id, blob] : blobs_) {
            GOB_UNUSED(id);
            values.push_back(std::move(blob));
        }
        return values;
    }

private:
    struct ExternalFloatingBase {
        std::string joint_path;
        std::string link_path;
    };

    const PhysicsLinkSnapshot* FindLink(const std::string& path) const {
        for (const auto& robot : snapshot_.robots) {
            for (const auto& link : robot.links) {
                if (link.scene_path == path) return &link;
            }
        }
        return nullptr;
    }

    std::string AddBlob(std::vector<std::uint8_t> data, std::string_view encoding) {
        const auto digest = Sha256Digest(std::span<const std::uint8_t>(data));
        if (!blobs_.contains(digest)) {
            blobs_.emplace(digest, IpcSceneArtifactBlob{digest, std::string(encoding), digest, std::move(data)});
        }
        return digest;
    }

    std::string AddMesh(const TetrahedralMeshView& mesh) {
        return AddBlob(EncodeMesh(mesh), kMeshEncoding);
    }

    std::string AddSurfaceMesh(const SurfaceMeshData& mesh) {
        return AddBlob(EncodeSurfaceMesh(mesh), kSurfaceMeshEncoding);
    }

    bool AddDeformable(const PhysicsDeformableSnapshot& body, std::string* error) {
        const std::string path = body.scene_path;
        if (body.name.empty()) {
            return SetCompileError(error, "IPC deformable bodies require a non-empty name");
        }
        if (!ValidateDeformableTransform(
                    body.global_transform, "deformable body '" + path + "'", error)) {
            return false;
        }
        std::string validation_error;
        if (!ValidateMaterial(body, &validation_error)) {
            return SetCompileError(
                    error, "deformable body '" + path + "' is invalid: " + validation_error);
        }

        std::string blob_id;
        std::size_t vertex_count = 0;
        std::size_t tetrahedron_count = 0;
        std::size_t surface_triangle_count = 0;
        std::string model_name;
        vertex_count = body.vertices.size();
        if (body.model == static_cast<int>(DeformableBodyModel::ThinShell)) {
            if (!ValidateTriangleMesh(body.vertices, body.surface_triangles, &validation_error)) {
                return SetCompileError(error, "thin-shell deformable body '" + path + "' has invalid mesh: " + validation_error);
            }
            blob_id = AddSurfaceMesh({body.vertices, body.surface_triangles});
            surface_triangle_count = body.surface_triangles.size() / 3;
            model_name = "thin_shell";
        } else {
            if (!ValidateTetrahedralMesh(body.vertices, body.tetrahedra, body.surface_triangles, &validation_error)) {
                return SetCompileError(error, "deformable body '" + path + "' has invalid mesh: " + validation_error);
            }
            const auto surface = ResolveTetrahedralSurface(body.tetrahedra, body.surface_triangles);
            blob_id = AddMesh({body.vertices, body.tetrahedra, surface});
            tetrahedron_count = body.tetrahedra.size() / 4;
            surface_triangle_count = surface.size() / 3;
            model_name = "volumetric";
        }
        deformable_bodies_.push_back({
                {"bending_stiffness", body.bending_stiffness},
                {"collision_layer", body.collision_layer},
                {"collision_mask", body.collision_mask},
                {"damping", body.damping},
                {"density", body.density},
                {"kinematic", body.kinematic},
                {"mesh_blob", blob_id},
                {"model", model_name},
                {"name", body.name},
                {"path", path},
                {"poisson_ratio", body.poisson_ratio},
                {"self_collision", body.self_collision_enabled},
                {"surface_triangle_count", surface_triangle_count},
                {"tetrahedron_count", tetrahedron_count},
                {"thickness", body.thickness},
                {"transform", TransformJson(body.global_transform)},
                {"vertex_count", vertex_count},
                {"young_modulus", body.young_modulus}});
        return true;
    }

    bool AddTactile(const PhysicsSensorSnapshot& sensor, std::string* error) {
        const std::string path = sensor.scene_path;
        if (sensor.name.empty()) {
            return SetCompileError(error, "IPC tactile sensors require a non-empty name");
        }
        if (!ValidateDeformableTransform(
                    sensor.global_transform, "tactile sensor '" + path + "'", error)) {
            return false;
        }
        const auto& data = *sensor.tactile;
        if (!data.parameters) {
            return SetCompileError(error, "tactile sensor '" + path + "' has no config");
        }
        const auto& config = *data.parameters;
        std::string validation_error;
        if (!ValidateTetrahedralMesh(data.gel_vertices, data.gel_tetrahedra, data.gel_surface_triangles, &validation_error)) {
            return SetCompileError(error, "tactile sensor '" + path + "' has invalid gel mesh: " + validation_error);
        }
        if (!config.Validate(data.gel_vertices.size(), data.gel_tetrahedra.size() / 4, &validation_error)) {
            return SetCompileError(error, "tactile sensor '" + path + "' has invalid config: " + validation_error);
        }
        const auto surface = ResolveTetrahedralSurface(data.gel_tetrahedra, data.gel_surface_triangles);
        const TetrahedralMeshView gel_mesh{data.gel_vertices, data.gel_tetrahedra, surface};
        Json marker_positions = Json::array();
        for (const Vector2& position : config.marker_positions) {
            marker_positions.push_back(Vector2Json(position));
        }
        Json marker_barycentric = Json::array();
        for (const Vector4& weights : config.marker_barycentric) {
            marker_barycentric.push_back(Vector4Json(weights));
        }
        Json attachment = nullptr;
        if (!data.attachment_link_path.empty()) {
            if (!FindLink(data.attachment_link_path)) {
                return SetCompileError(error, "tactile sensor '" + path + "' attachment link is outside the compiled scene");
            }
            attachment = {{"link_path", data.attachment_link_path}, {"transform", TransformJson(data.attachment_transform)}};
        }
        tactile_sensors_.push_back({
                {"attachment", std::move(attachment)},
                {"collision_layer", data.collision_layer},
                {"collision_mask", data.collision_mask},
                {"coat_vertex_indices", config.coat_vertex_indices},
                {"damping", config.damping},
                {"density", config.density},
                {"enabled", sensor.enabled},
                {"far_plane", config.far_plane},
                {"friction_coefficient", config.friction_coefficient},
                {"gel_mesh_blob", AddMesh(gel_mesh)},
                {"gel_topology_sha256", MeshTopologyDigest(gel_mesh)},
                {"gel_tetrahedron_count", data.gel_tetrahedra.size() / 4},
                {"gel_vertex_count", data.gel_vertices.size()},
                {"marker_barycentric", std::move(marker_barycentric)},
                {"marker_positions", std::move(marker_positions)},
                {"marker_tetrahedra", config.marker_tetrahedra},
                {"name", sensor.name},
                {"near_plane", config.near_plane},
                {"path", path},
                {"pixel_size", config.pixel_size},
                {"poisson_ratio", config.poisson_ratio},
                {"resolution", Json::array({config.image_height, config.image_width})},
                {"rgb_model", config.rgb_model},
                {"stick_vertex_indices", config.stick_vertex_indices},
                {"transform", TransformJson(sensor.global_transform)},
                {"young_modulus", config.young_modulus}});
        return true;
    }

    bool AddCollisionShape(const PhysicsShapeSnapshot& collision,
                           const PhysicsLinkSnapshot* link,
                           Json* collision_shapes,
                           std::unordered_set<std::string>* collision_paths,
                           std::string* error) {
        const std::string path = collision.scene_path;
        const std::string kind = link != nullptr ? "robot" : "static";
        if (collision.name.empty() || !collision_paths->insert(path).second) {
            return SetCompileError(
                    error, kind +
                                   " collision shape paths and names must be non-empty and unique");
        }
        if (!ValidateDeformableTransform(
                    collision.global_transform, kind + " collision shape '" + path + "'", error)) {
            return false;
        }
        const auto& material = collision.material;
        const std::array<RealType, 6> material_values{
                material.sliding_friction,
                material.torsional_friction,
                material.rolling_friction,
                material.restitution,
                material.contact_compliance,
                material.contact_damping};
        if (!std::ranges::all_of(material_values, [](RealType value) {
                return std::isfinite(value) && value >= 0.0;
            }) || material.restitution > 1.0 ||
            !std::isfinite(collision.contact_offset) || collision.contact_offset < 0.0 ||
            !std::isfinite(collision.rest_offset) ||
            collision.rest_offset > collision.contact_offset) {
            return SetCompileError(
                    error, kind + " collision shape '" + path +
                                   "' has invalid contact parameters");
        }

        Json compiled = {
                {"collision_layer", collision.collision_layer},
                {"collision_mask", collision.collision_mask},
                {"contact_offset", collision.contact_offset},
                {"disabled", collision.disabled},
                {"material", {
                        {"contact_compliance", material.contact_compliance},
                        {"contact_damping", material.contact_damping},
                        {"restitution", material.restitution},
                        {"rolling_friction", material.rolling_friction},
                        {"sliding_friction", material.sliding_friction},
                        {"torsional_friction", material.torsional_friction}}},
                {"name", collision.name},
                {"path", path},
                {"rest_offset", collision.rest_offset},
                {"transform", TransformJson(collision.global_transform)}};
        if (link != nullptr) {
            compiled["link_transform"] = TransformJson(
                    link->global_transform.inverse() *
                    collision.global_transform);
        }

        if (collision.type == PhysicsShapeType::Box) {
            const Vector3 size = collision.box_size;
            if (!size.allFinite() || (size.array() <= 0.0).any()) {
                return SetCompileError(
                        error, kind + " box collision shape '" + path + "' has invalid size");
            }
            compiled["shape_type"] = "box";
            compiled["size"] = Vector3Json(size);
        } else if (collision.type == PhysicsShapeType::Sphere) {
            const double radius = collision.radius;
            if (!std::isfinite(radius) || radius <= 0.0) {
                return SetCompileError(
                        error, kind + " sphere collision shape '" + path + "' has invalid radius");
            }
            compiled["radius"] = radius;
            compiled["shape_type"] = "sphere";
        } else if (collision.type == PhysicsShapeType::Capsule) {
            const double radius = collision.radius;
            const double height = collision.height;
            if (!std::isfinite(radius) || radius <= 0.0 ||
                !std::isfinite(height) || height <= 0.0) {
                return SetCompileError(
                        error, kind + " capsule collision shape '" + path +
                                       "' has invalid dimensions");
            }
            compiled["height"] = height;
            compiled["radius"] = radius;
            compiled["shape_type"] = "capsule";
        } else if (collision.type == PhysicsShapeType::Cylinder) {
            const double radius = collision.radius;
            const double height = collision.height;
            if (!std::isfinite(radius) || radius <= 0.0 ||
                !std::isfinite(height) || height <= 0.0) {
                return SetCompileError(
                        error, kind + " cylinder collision shape '" + path +
                                       "' has invalid dimensions");
            }
            compiled["height"] = height;
            compiled["radius"] = radius;
            compiled["shape_type"] = "cylinder";
        } else if (collision.type == PhysicsShapeType::Mesh) {
            std::string mesh_error;
            if (!ValidateTriangleMesh(collision.vertices, collision.indices, &mesh_error, false)) {
                return SetCompileError(error, kind + " triangle collision shape '" + path + "' is invalid: " + mesh_error);
            }
            const auto blob_id = AddSurfaceMesh({collision.vertices, collision.indices});
            const auto vertex_count = collision.vertices.size();
            const auto triangle_count = collision.indices.size() / 3;
            compiled["mesh_blob"] = blob_id;
            compiled["shape_type"] = "triangle_mesh";
            compiled["triangle_count"] = triangle_count;
            compiled["vertex_count"] = vertex_count;
        } else {
            return SetCompileError(
                    error, kind + " collision shape '" + path + "' uses an unsupported shape type");
        }

        collision_shapes->push_back(std::move(compiled));
        return true;
    }

    bool AddRobot(const PhysicsRobotSnapshot& robot, std::string* error) {
        const std::string robot_path = robot.scene_path;
        if (robot.name.empty()) {
            return SetCompileError(error, "IPC robots require a non-empty name");
        }
        if (!ValidateDeformableTransform(
                    robot.global_transform, "robot '" + robot_path + "'", error)) {
            return false;
        }
        Json links = Json::array();
        Json joints = Json::array();
        std::unordered_set<std::string> link_names;
        std::unordered_set<std::string> joint_names;
        if (!CollectRobotNodes(
                    robot, &links, &joints, &link_names, &joint_names, error)) {
            return false;
        }
        if (robot.standalone_rigid_body) {
            if (links.size() != 1 || !joints.empty() || links.at(0).at("path") != robot_path) {
                return SetCompileError(error, "RigidBody3D '" + robot_path + "' must contain collision, visual, or sensor children only");
            }
            robots_.push_back({{"joints", Json::array()}, {"kind", "rigid_body"},
                {"links", std::move(links)}, {"name", robot.name}, {"path", robot_path},
                {"root_link_paths", Json::array({robot_path})}, {"transform", TransformJson(robot.global_transform)}});
            return true;
        }
        if (links.empty()) {
            return SetCompileError(error, "robot '" + robot_path + "' has no links");
        }

        std::unordered_map<std::string, std::string> link_paths;
        for (const Json& link : links) {
            link_paths.emplace(link.at("name").get<std::string>(),
                               link.at("path").get<std::string>());
        }
        std::unordered_set<std::string> child_links;
        for (Json& joint : joints) {
            const std::string parent_link = joint.at("parent_link").get<std::string>();
            const std::string child_link = joint.at("child_link").get<std::string>();
            if (!link_paths.contains(parent_link) || !link_paths.contains(child_link)) {
                return SetCompileError(
                        error,
                        "robot joint '" + joint.at("path").get<std::string>() +
                                "' references a link outside robot '" + robot_path + "'");
            }
            if (!child_links.insert(child_link).second) {
                return SetCompileError(
                        error, "robot '" + robot_path +
                                       "' has multiple joints for child link '" + child_link + "'");
            }
            joint["parent_link_path"] = link_paths.at(parent_link);
            joint["child_link_path"] = link_paths.at(child_link);
        }

        Json root_link_paths = Json::array();
        for (const auto& [name, path] : link_paths) {
            if (!child_links.contains(name)) {
                root_link_paths.push_back(path);
            }
        }
        std::sort(root_link_paths.begin(), root_link_paths.end());
        robots_.push_back({
                {"joints", std::move(joints)},
                {"kind", "articulation"},
                {"links", std::move(links)},
                {"name", robot.name},
                {"path", robot_path},
                {"root_link_paths", std::move(root_link_paths)},
                {"transform", TransformJson(robot.global_transform)}});
        return true;
    }

    bool CollectRobotNodes(const PhysicsRobotSnapshot& robot,
                           Json* links, Json* joints,
                           std::unordered_set<std::string>* link_names,
                           std::unordered_set<std::string>* joint_names,
                           std::string* error) {
        for (const auto& link : robot.links) {
            const std::string path = link.scene_path;
            if (link.name.empty() || !link_names->insert(link.name).second) {
                return SetCompileError(
                        error, "robot link names must be non-empty and unique in '" +
                                       robot.scene_path + "'");
            }
            if (!ValidateDeformableTransform(link.global_transform, "robot link '" + path + "'", error)) {
                return false;
            }
            Json collision_shapes = Json::array();
            std::unordered_set<std::string> collision_paths;
            for (const auto& shape : link.collision_shapes) {
                if (!AddCollisionShape(shape, &link, &collision_shapes, &collision_paths, error)) return false;
            }
            const Quaternion& inertia_orientation = link.inertia_orientation;
            if (!std::isfinite(link.mass) || link.mass < 0.0 ||
                !link.center_of_mass.allFinite() ||
                !inertia_orientation.coeffs().allFinite() ||
                inertia_orientation.squaredNorm() <=
                        std::numeric_limits<RealType>::epsilon() ||
                !link.inertia_diagonal.allFinite() ||
                (link.inertia_diagonal.array() < 0.0).any() ||
                !link.inertia_off_diagonal.allFinite() ||
                static_cast<int>(link.authored_role) <
                        static_cast<int>(PhysicsLinkRole::Physical) ||
                static_cast<int>(link.authored_role) >
                        static_cast<int>(PhysicsLinkRole::VirtualRoot)) {
                return SetCompileError(
                        error, "robot link '" + path + "' has invalid inertial properties");
            }
            links->push_back({
                    {"center_of_mass", Vector3Json(link.center_of_mass)},
                    {"collision_shapes", std::move(collision_shapes)},
                    {"has_inertial", link.has_inertial},
                    {"inertia_diagonal", Vector3Json(link.inertia_diagonal)},
                    {"inertia_off_diagonal", Vector3Json(link.inertia_off_diagonal)},
                    {"inertia_orientation_wxyz", QuaternionWxyzJson(inertia_orientation)},
                    {"local_transform", TransformJson(link.local_transform)},
                    {"mass", link.mass},
                    {"name", link.name},
                    {"path", path},
                    {"role", static_cast<int>(link.authored_role)},
                    {"transform", TransformJson(link.global_transform)}});
        }
        for (const auto& joint : robot.joints) {
            const std::string path = joint.scene_path;
            if (joint.name.empty() || !joint_names->insert(joint.name).second) {
                return SetCompileError(
                        error, "robot joint names must be non-empty and unique in '" +
                                       robot.scene_path + "'");
            }
            if (!ValidateDeformableTransform(joint.global_transform, "robot joint '" + path + "'", error)) {
                return false;
            }
            const auto find_robot_link = [&](const std::string& link_path) -> const PhysicsLinkSnapshot* {
                const auto found = std::ranges::find(robot.links, link_path, &PhysicsLinkSnapshot::scene_path);
                return found == robot.links.end() ? nullptr : &*found;
            };
            const auto* parent_link = find_robot_link(joint.structural_parent_link_path);
            if (!joint.structural_parent_link_path.empty() && parent_link == nullptr) {
                return SetCompileError(error, "robot joint '" + path + "' references a link outside its robot");
            }
            if (joint.structural_child_link_paths.size() > 1) {
                return SetCompileError(error, "robot joint '" + path + "' has more than one direct child link");
            }
            const auto* child_link = joint.structural_child_link_paths.empty() ? nullptr : find_robot_link(joint.structural_child_link_paths.front());
            const std::array<RealType, 20> parameters{
                    joint.lower_limit, joint.upper_limit,
                    joint.effort_limit, joint.velocity_limit,
                    joint.damping, joint.armature, joint.friction_loss,
                    joint.joint_position, joint.initial_position,
                    joint.drive_stiffness, joint.drive_damping,
                    joint.control_lower_limit, joint.control_upper_limit,
                    joint.force_lower_limit, joint.force_upper_limit,
                    joint.affine_actuator_control_gain,
                    joint.affine_actuator_force_offset,
                    joint.affine_actuator_position_gain,
                    joint.affine_actuator_velocity_gain,
                    joint.affine_actuator_inherit_range};
            if (joint.joint_type == static_cast<int>(JointType::Floating) &&
                parent_link == nullptr && child_link != nullptr) {
                // MuJoCo owns floating-base dynamics. In the IPC artifact the
                // coupled child is an externally driven affine proxy, so this
                // root joint is intentionally omitted from IPC articulation.
                external_floating_bases_.push_back(
                        {path, child_link->scene_path});
            } else if (parent_link == nullptr || child_link == nullptr ||
                !joint.axis.allFinite() ||
                joint.axis.squaredNorm() <=
                        std::numeric_limits<RealType>::epsilon() ||
                !std::all_of(parameters.begin(), parameters.end(), [](RealType value) {
                    return std::isfinite(value);
                }) ||
                joint.effort_limit < 0.0 || joint.velocity_limit < 0.0 ||
                joint.damping < 0.0 || joint.armature < 0.0 ||
                joint.friction_loss < 0.0 || joint.drive_stiffness < 0.0 ||
                joint.drive_damping < 0.0 ||
                static_cast<int>(joint.joint_type) <
                        static_cast<int>(JointType::Fixed) ||
                static_cast<int>(joint.joint_type) >
                        static_cast<int>(JointType::Planar) ||
                static_cast<int>(joint.drive_mode) <
                        static_cast<int>(JointDriveMode::Passive) ||
                static_cast<int>(joint.drive_mode) >
                        static_cast<int>(JointDriveMode::Velocity) ||
                !std::all_of(
                        joint.gear.begin(), joint.gear.end(),
                        [](RealType value) { return std::isfinite(value); })) {
                return SetCompileError(
                        error, "robot joint '" + path + "' has invalid topology or parameters");
            } else {
                joints->push_back({
                    {"affine_actuator_control_gain", joint.affine_actuator_control_gain},
                    {"affine_actuator_enabled", joint.affine_actuator_enabled},
                    {"affine_actuator_force_offset", joint.affine_actuator_force_offset},
                    {"affine_actuator_inherit_range", joint.affine_actuator_inherit_range},
                    {"affine_actuator_position_gain", joint.affine_actuator_position_gain},
                    {"affine_actuator_velocity_gain", joint.affine_actuator_velocity_gain},
                    {"authored_child_link", joint.child_link},
                    {"authored_parent_link", joint.parent_link},
                    {"axis", Vector3Json(joint.axis)},
                    {"armature", joint.armature},
                    {"child_link", child_link->name},
                    {"control_lower_limit", joint.control_lower_limit},
                    {"control_upper_limit", joint.control_upper_limit},
                    {"damping", joint.damping},
                    {"drive_damping", joint.drive_damping},
                    {"drive_mode", static_cast<int>(joint.drive_mode)},
                    {"drive_stiffness", joint.drive_stiffness},
                    {"effort_limit", joint.effort_limit},
                    {"force_lower_limit", joint.force_lower_limit},
                    {"force_upper_limit", joint.force_upper_limit},
                    {"friction_loss", joint.friction_loss},
                    {"gear", joint.gear},
                    {"initial_position", joint.initial_position},
                    {"joint_type", static_cast<int>(joint.joint_type)},
                    {"joint_position", joint.joint_position},
                    {"local_transform", TransformJson(joint.local_transform)},
                    {"lower_limit", joint.lower_limit},
                    {"name", joint.name},
                    {"parent_link", parent_link->name},
                    {"path", path},
                    {"transform", TransformJson(joint.global_transform)},
                    {"velocity_limit", joint.velocity_limit},
                    {"upper_limit", joint.upper_limit}});
            }
        }
        return true;
    }

    const PhysicsSceneSnapshot& snapshot_;
    Json deformable_bodies_ = Json::array();
    Json tactile_sensors_ = Json::array();
    Json robots_ = Json::array();
    Json couplings_ = Json::array();
    Json deformable_attachments_ = Json::array();
    Json static_colliders_ = Json::array();
    std::unordered_set<std::string> static_collider_paths_;
    std::vector<ExternalFloatingBase> external_floating_bases_;
    std::map<std::string, IpcSceneArtifactBlob> blobs_;
};

} // namespace

bool IpcSceneCompiler::Compile(
        const PhysicsSceneSnapshot& snapshot, IpcSceneArtifact* artifact, std::string* error) {
    if (artifact == nullptr) {
        return SetCompileError(error, "Cannot compile an IPC artifact into a null output.");
    }
    if (snapshot.scene_name.empty()) {
        return SetCompileError(error, "Cannot compile an IPC artifact with an unnamed scene root.");
    }

    CompilerState compiler(snapshot);
    if (!compiler.Compile(error)) return false;
    const Json manifest_json = compiler.BuildManifest();
    IpcSceneArtifact result;
    result.schema_version = 5;
    result.producer = "gobot";
    result.producer_version = ProducerVersion();
    result.format = "gobot-ipc";
    result.manifest = manifest_json.dump();
    result.manifest_sha256 = Sha256Digest(result.manifest);
    result.blobs = compiler.TakeBlobs();
    *artifact = std::move(result);
    if (error != nullptr) {
        error->clear();
    }
    return true;
}

} // namespace gobot
