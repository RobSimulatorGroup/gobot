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
#include "gobot/core/types.hpp"
#include "gobot/physics/physics_types.hpp"
#include "gobot/scene/collision_shape_3d.hpp"
#include "gobot/scene/deformable_body_3d.hpp"
#include "gobot/scene/joint_3d.hpp"
#include "gobot/scene/link_3d.hpp"
#include "gobot/scene/node.hpp"
#include "gobot/scene/node_3d.hpp"
#include "gobot/scene/physics_coupling.hpp"
#include "gobot/scene/resources/box_shape_3d.hpp"
#include "gobot/scene/resources/capsule_shape_3d.hpp"
#include "gobot/scene/resources/convex_mesh_shape_3d.hpp"
#include "gobot/scene/resources/cylinder_shape_3d.hpp"
#include "gobot/scene/resources/mesh.hpp"
#include "gobot/scene/resources/sphere_shape_3d.hpp"
#include "gobot/scene/robot_3d.hpp"
#include "gobot/scene/tactile_sensor_3d.hpp"

namespace gobot {
namespace {

constexpr std::string_view kMeshEncoding = "gobot.tetrahedral-mesh.le.v1";
constexpr std::string_view kSurfaceMeshEncoding = "gobot.triangle-mesh.le.v1";

struct SurfaceMeshData {
    std::vector<Vector3> vertices;
    std::vector<std::uint32_t> triangles;
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

std::vector<std::uint8_t> EncodeMesh(const TetrahedralMesh& mesh) {
    const std::vector<std::uint32_t> surface = mesh.GetResolvedSurfaceTriangles();
    std::vector<std::uint8_t> data;
    data.reserve(24 + mesh.GetVertexCount() * 24 +
                 mesh.GetTetrahedra().size() * 4 + surface.size() * 4);
    constexpr std::array<std::uint8_t, 8> magic{'G', 'O', 'B', 'T', 'I', 'P', 'C', '1'};
    data.insert(data.end(), magic.begin(), magic.end());
    AppendU32(&data, 1);
    AppendU32(&data, static_cast<std::uint32_t>(mesh.GetVertexCount()));
    AppendU32(&data, static_cast<std::uint32_t>(mesh.GetTetrahedronCount()));
    AppendU32(&data, static_cast<std::uint32_t>(surface.size() / 3));
    for (const Vector3& vertex : mesh.GetVertices()) {
        AppendF64(&data, static_cast<double>(vertex.x()));
        AppendF64(&data, static_cast<double>(vertex.y()));
        AppendF64(&data, static_cast<double>(vertex.z()));
    }
    for (const std::uint32_t index : mesh.GetTetrahedra()) {
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

bool CollectSurfaceMesh(const Mesh& mesh, SurfaceMeshData* output, std::string* error) {
    const std::shared_ptr<const MeshSurfaceList> surfaces = mesh.GetSurfaceData();
    if (!surfaces || surfaces->empty()) {
        return SetCompileError(error, "convex collision mesh has no surface data");
    }
    SurfaceMeshData result;
    for (const MeshSurfaceData& surface : *surfaces) {
        if (surface.vertices.empty()) {
            continue;
        }
        if (result.vertices.size() + surface.vertices.size() >
            std::numeric_limits<std::uint32_t>::max()) {
            return SetCompileError(error, "convex collision mesh has too many vertices");
        }
        const std::uint32_t vertex_offset =
                static_cast<std::uint32_t>(result.vertices.size());
        result.vertices.insert(
                result.vertices.end(), surface.vertices.begin(), surface.vertices.end());
        if (surface.indices.empty()) {
            if (surface.vertices.size() % 3 != 0) {
                return SetCompileError(
                        error, "unindexed convex collision mesh is not a triangle list");
            }
            for (std::uint32_t index = 0; index < surface.vertices.size(); ++index) {
                result.triangles.push_back(vertex_offset + index);
            }
        } else {
            if (surface.indices.size() % 3 != 0) {
                return SetCompileError(
                        error, "indexed convex collision mesh is not a triangle list");
            }
            for (const std::uint32_t index : surface.indices) {
                if (index >= surface.vertices.size()) {
                    return SetCompileError(
                            error, "convex collision mesh has an out-of-range index");
                }
                result.triangles.push_back(vertex_offset + index);
            }
        }
    }
    if (result.vertices.size() < 3 || result.triangles.empty()) {
        return SetCompileError(error, "convex collision mesh has no triangles");
    }
    for (const Vector3& vertex : result.vertices) {
        if (!vertex.allFinite()) {
            return SetCompileError(error, "convex collision mesh has a non-finite vertex");
        }
    }
    std::set<std::array<std::uint32_t, 3>> unique_triangles;
    for (std::size_t offset = 0; offset < result.triangles.size(); offset += 3) {
        const std::uint32_t ia = result.triangles[offset];
        const std::uint32_t ib = result.triangles[offset + 1];
        const std::uint32_t ic = result.triangles[offset + 2];
        if (ia == ib || ib == ic || ic == ia) {
            return SetCompileError(
                    error, "convex collision mesh has a repeated triangle vertex");
        }
        std::array<std::uint32_t, 3> triangle_key{ia, ib, ic};
        std::sort(triangle_key.begin(), triangle_key.end());
        if (!unique_triangles.insert(triangle_key).second) {
            return SetCompileError(
                    error, "convex collision mesh has a duplicate triangle");
        }
        const Vector3 edge_ab = result.vertices[ib] - result.vertices[ia];
        const Vector3 edge_ac = result.vertices[ic] - result.vertices[ia];
        const RealType edge_scale = std::max(edge_ab.norm(), edge_ac.norm());
        const RealType area_scale = edge_scale * edge_scale;
        const RealType tolerance =
                std::numeric_limits<RealType>::epsilon() * 128.0 * area_scale;
        const RealType double_area = edge_ab.cross(edge_ac).norm();
        if (!std::isfinite(double_area) || !std::isfinite(tolerance) ||
            edge_scale <= 0.0 || double_area <= tolerance) {
            return SetCompileError(
                    error, "convex collision mesh has a degenerate triangle");
        }
    }
    *output = std::move(result);
    return true;
}

std::string MeshTopologyDigest(const TetrahedralMesh& mesh) {
    const std::vector<std::uint32_t> surface = mesh.GetResolvedSurfaceTriangles();
    std::vector<std::uint8_t> data;
    constexpr std::array<std::uint8_t, 12> magic{
            'G', 'O', 'B', 'T', 'I', 'P', 'C', 'T', 'O', 'P', '1', 0};
    data.insert(data.end(), magic.begin(), magic.end());
    AppendU32(&data, static_cast<std::uint32_t>(mesh.GetVertexCount()));
    AppendU32(&data, static_cast<std::uint32_t>(mesh.GetTetrahedronCount()));
    AppendU32(&data, static_cast<std::uint32_t>(surface.size() / 3));
    for (const std::uint32_t index : mesh.GetTetrahedra()) {
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

Json GlobalTransformJson(const Node3D& node) {
    return TransformJson(node.GetGlobalTransform());
}

Json LocalTransformJson(const Node3D& node) {
    return TransformJson(node.GetTransform());
}

std::string NodePathString(const Node& node) {
    return static_cast<std::string>(node.GetPath());
}

const Node* ResolveNodePath(const Node& source, const NodePath& path) {
    if (path.IsEmpty() || path.GetSubNameCount() != 0) {
        return nullptr;
    }
    const Node* current = &source;
    std::size_t name_index = 0;
    const std::vector<std::string> names = path.GetNames();
    if (path.IsAbsolute()) {
        while (current->GetParent() != nullptr) {
            current = current->GetParent();
        }
        if (!names.empty() && names.front() == current->GetName()) {
            name_index = 1;
        }
    }
    for (; name_index < names.size(); ++name_index) {
        const std::string& name = names[name_index];
        if (name == ".") {
            continue;
        }
        if (name == "..") {
            current = current->GetParent();
            if (current == nullptr) {
                return nullptr;
            }
            continue;
        }
        const Node* child = nullptr;
        for (std::size_t index = 0; index < current->GetChildCount(); ++index) {
            const Node* candidate = current->GetChild(static_cast<int>(index));
            if (candidate->GetName() == name) {
                child = candidate;
                break;
            }
        }
        if (child == nullptr) {
            return nullptr;
        }
        current = child;
    }
    return current;
}

const Link3D* FindAncestorLink(const Node& node) {
    const Node* ancestor = node.GetParent();
    while (ancestor != nullptr) {
        if (const auto* link = Object::PointerCastTo<Link3D>(ancestor)) {
            return link;
        }
        ancestor = ancestor->GetParent();
    }
    return nullptr;
}

bool ValidateMaterial(const DeformableBody3D& body, std::string* error) {
    if (!std::isfinite(body.GetDensity()) || body.GetDensity() <= 0.0) {
        return SetCompileError(error, "deformable body density must be finite and positive");
    }
    if (!std::isfinite(body.GetYoungModulus()) || body.GetYoungModulus() <= 0.0) {
        return SetCompileError(error, "deformable body Young modulus must be finite and positive");
    }
    if (!std::isfinite(body.GetPoissonRatio()) || body.GetPoissonRatio() <= -1.0 ||
        body.GetPoissonRatio() >= 0.5) {
        return SetCompileError(error, "deformable body Poisson ratio must be in (-1, 0.5)");
    }
    if (!std::isfinite(body.GetDamping()) || body.GetDamping() < 0.0) {
        return SetCompileError(error, "deformable body damping must be finite and non-negative");
    }
    return true;
}

bool ValidateDeformableTransform(
        const Node3D& node, std::string_view description, std::string* error) {
    const Affine3 transform = node.GetGlobalTransform();
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
    bool Visit(const Node* node, std::string* error) {
        if (const auto* body = Object::PointerCastTo<DeformableBody3D>(node)) {
            if (!AddDeformable(*body, error)) {
                return false;
            }
        }
        if (const auto* sensor = Object::PointerCastTo<TactileSensor3D>(node)) {
            if (!AddTactile(*sensor, error)) {
                return false;
            }
        }
        if (const auto* robot = Object::PointerCastTo<Robot3D>(node)) {
            if (!AddRobot(*robot, error)) {
                return false;
            }
        }
        if (const auto* coupling = Object::PointerCastTo<PhysicsCoupling>(node)) {
            coupling_nodes_.push_back(coupling);
        }
        for (std::size_t index = 0; index < node->GetChildCount(); ++index) {
            if (!Visit(node->GetChild(static_cast<int>(index)), error)) {
                return false;
            }
        }
        return true;
    }

    bool FinalizeCouplings(const Node& scene_root, std::string* error) {
        struct PendingCoupling {
            std::string coupling_path;
            std::string link_path;
            std::string robot_name;
            std::string link_name;
            PhysicsCouplingMode mode;
            RealType force_scale;
            RealType torque_scale;
        };

        std::vector<PendingCoupling> pending;
        std::unordered_set<std::string> linked_paths;
        for (const PhysicsCoupling* coupling : coupling_nodes_) {
            if (!coupling->IsEnabled()) {
                continue;
            }
            const std::string coupling_path = NodePathString(*coupling);
            const NodePath& target_path = coupling->GetRigidLinkPath();
            if (target_path.IsEmpty()) {
                return SetCompileError(
                        error, "PhysicsCoupling '" + coupling_path +
                                       "' requires a rigid_link_path");
            }
            const Node* target = ResolveNodePath(*coupling, target_path);
            const auto* link = Object::PointerCastTo<Link3D>(target);
            if (link == nullptr) {
                return SetCompileError(
                        error, "PhysicsCoupling '" + coupling_path +
                                       "' rigid_link_path does not resolve to a Link3D");
            }
            if (link != &scene_root && !scene_root.IsAncestorOf(link)) {
                return SetCompileError(
                        error, "PhysicsCoupling '" + coupling_path +
                                       "' must target a Link3D in the compiled scene");
            }

            const Robot3D* robot = nullptr;
            for (const Node* ancestor = link->GetParent(); ancestor != nullptr;
                 ancestor = ancestor->GetParent()) {
                if (const auto* candidate = Object::PointerCastTo<Robot3D>(ancestor)) {
                    robot = candidate;
                    break;
                }
            }
            if (robot == nullptr ||
                (robot != &scene_root && !scene_root.IsAncestorOf(robot))) {
                return SetCompileError(
                        error, "PhysicsCoupling '" + coupling_path +
                                       "' target must belong to a Robot3D in the compiled scene");
            }

            const std::string link_path = NodePathString(*link);
            const Json* compiled_link = nullptr;
            for (const Json& compiled_robot : robots_) {
                if (compiled_robot.at("path").get<std::string>() !=
                    NodePathString(*robot)) {
                    continue;
                }
                for (const Json& candidate : compiled_robot.at("links")) {
                    if (candidate.at("path").get<std::string>() == link_path) {
                        compiled_link = &candidate;
                        break;
                    }
                }
                break;
            }
            if (compiled_link == nullptr) {
                return SetCompileError(
                        error, "PhysicsCoupling '" + coupling_path +
                                       "' target is not a compiled Robot3D Link3D");
            }
            const bool has_enabled_collision = std::ranges::any_of(
                    compiled_link->at("collision_shapes"), [](const Json& shape) {
                        return !shape.value("disabled", false);
                    });
            if (!has_enabled_collision) {
                return SetCompileError(
                        error, "PhysicsCoupling '" + coupling_path +
                                       "' target Link3D has no enabled CollisionShape3D");
            }

            const int mode = static_cast<int>(coupling->GetMode());
            if (mode < static_cast<int>(PhysicsCouplingMode::OneWay) ||
                mode > static_cast<int>(PhysicsCouplingMode::TwoWay)) {
                return SetCompileError(
                        error, "PhysicsCoupling '" + coupling_path +
                                       "' has an invalid coupling mode");
            }
            if (!std::isfinite(coupling->GetForceScale()) ||
                coupling->GetForceScale() < 0.0 ||
                !std::isfinite(coupling->GetTorqueScale()) ||
                coupling->GetTorqueScale() < 0.0) {
                return SetCompileError(
                        error, "PhysicsCoupling '" + coupling_path +
                                       "' force and torque scales must be finite and non-negative");
            }
            if (!linked_paths.insert(link_path).second) {
                return SetCompileError(
                        error, "multiple enabled PhysicsCoupling nodes target Link3D '" +
                                       link_path + "'");
            }
            pending.push_back(PendingCoupling{
                    coupling_path,
                    link_path,
                    robot->GetName(),
                    link->GetName(),
                    coupling->GetMode(),
                    coupling->GetForceScale(),
                    coupling->GetTorqueScale()});
        }

        std::ranges::sort(pending, {}, &PendingCoupling::coupling_path);
        for (std::size_t proxy_index = 0; proxy_index < pending.size(); ++proxy_index) {
            const PendingCoupling& coupling = pending[proxy_index];
            couplings_.push_back({
                    {"coupling_path", coupling.coupling_path},
                    {"force_scale", coupling.force_scale},
                    {"link_name", coupling.link_name},
                    {"link_path", coupling.link_path},
                    {"mode", coupling.mode == PhysicsCouplingMode::OneWay
                                     ? "OneWay"
                                     : "TwoWay"},
                    {"proxy_index", proxy_index},
                    {"robot_name", coupling.robot_name},
                    {"torque_scale", coupling.torque_scale}});
        }
        return true;
    }

    Json BuildManifest(const Node& scene_root) const {
        Json blob_table = Json::array();
        for (const auto& [id, blob] : blobs_) {
            blob_table.push_back({
                    {"byte_length", blob.data.size()},
                    {"encoding", blob.encoding},
                    {"id", id},
                    {"sha256", blob.sha256}});
        }
        return Json{
                {"blobs", std::move(blob_table)},
                {"couplings", couplings_},
                {"deformable_bodies", deformable_bodies_},
                {"format", "gobot-ipc"},
                {"producer", "gobot"},
                {"producer_version", ProducerVersion()},
                {"robots", robots_},
                {"scene_name", scene_root.GetName()},
                {"schema_version", 3},
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
    std::string AddMesh(const TetrahedralMesh& mesh) {
        std::vector<std::uint8_t> data = EncodeMesh(mesh);
        const std::string digest = Sha256Digest(std::span<const std::uint8_t>(data));
        if (!blobs_.contains(digest)) {
            blobs_.emplace(digest, IpcSceneArtifactBlob{
                    digest, std::string(kMeshEncoding), digest, std::move(data)});
        }
        return digest;
    }

    bool AddSurfaceMesh(const Mesh& mesh,
                        std::string* blob_id,
                        std::size_t* vertex_count,
                        std::size_t* triangle_count,
                        std::string* error) {
        SurfaceMeshData surface_mesh;
        if (!CollectSurfaceMesh(mesh, &surface_mesh, error)) {
            return false;
        }
        *vertex_count = surface_mesh.vertices.size();
        *triangle_count = surface_mesh.triangles.size() / 3;
        std::vector<std::uint8_t> data = EncodeSurfaceMesh(surface_mesh);
        const std::string digest = Sha256Digest(std::span<const std::uint8_t>(data));
        if (!blobs_.contains(digest)) {
            blobs_.emplace(digest, IpcSceneArtifactBlob{
                    digest, std::string(kSurfaceMeshEncoding), digest, std::move(data)});
        }
        *blob_id = digest;
        return true;
    }

    bool AddDeformable(const DeformableBody3D& body, std::string* error) {
        const std::string path = NodePathString(body);
        if (body.GetName().empty()) {
            return SetCompileError(error, "IPC deformable bodies require a non-empty name");
        }
        if (!ValidateDeformableTransform(
                    body, "deformable body '" + path + "'", error)) {
            return false;
        }
        const Ref<TetrahedralMesh>& mesh = body.GetMesh();
        if (!mesh.IsValid()) {
            return SetCompileError(error, "deformable body '" + path + "' has no tetrahedral mesh");
        }
        std::string validation_error;
        if (!mesh->Validate(&validation_error)) {
            return SetCompileError(
                    error, "deformable body '" + path + "' has invalid mesh: " + validation_error);
        }
        if (!ValidateMaterial(body, &validation_error)) {
            return SetCompileError(
                    error, "deformable body '" + path + "' is invalid: " + validation_error);
        }
        const std::string blob_id = AddMesh(*mesh.Get());
        deformable_bodies_.push_back({
                {"collision_layer", body.GetCollisionLayer()},
                {"collision_mask", body.GetCollisionMask()},
                {"damping", body.GetDamping()},
                {"density", body.GetDensity()},
                {"kinematic", body.IsKinematic()},
                {"mesh_blob", blob_id},
                {"name", body.GetName()},
                {"path", path},
                {"poisson_ratio", body.GetPoissonRatio()},
                {"self_collision", body.IsSelfCollisionEnabled()},
                {"surface_triangle_count", mesh->GetResolvedSurfaceTriangles().size() / 3},
                {"tetrahedron_count", mesh->GetTetrahedronCount()},
                {"transform", GlobalTransformJson(body)},
                {"vertex_count", mesh->GetVertexCount()},
                {"young_modulus", body.GetYoungModulus()}});
        return true;
    }

    bool AddTactile(const TactileSensor3D& sensor, std::string* error) {
        const std::string path = NodePathString(sensor);
        if (sensor.GetName().empty()) {
            return SetCompileError(error, "IPC tactile sensors require a non-empty name");
        }
        if (!ValidateDeformableTransform(
                    sensor, "tactile sensor '" + path + "'", error)) {
            return false;
        }
        const Ref<TetrahedralMesh>& gel_mesh = sensor.GetGelMesh();
        const Ref<TactileSensorConfig>& config = sensor.GetConfig();
        if (!gel_mesh.IsValid()) {
            return SetCompileError(error, "tactile sensor '" + path + "' has no gel mesh");
        }
        if (!config.IsValid()) {
            return SetCompileError(error, "tactile sensor '" + path + "' has no config");
        }
        std::string validation_error;
        if (!gel_mesh->Validate(&validation_error)) {
            return SetCompileError(
                    error, "tactile sensor '" + path + "' has invalid gel mesh: " + validation_error);
        }
        if (!config->Validate(*gel_mesh.Get(), &validation_error)) {
            return SetCompileError(
                    error, "tactile sensor '" + path + "' has invalid config: " + validation_error);
        }

        Json marker_positions = Json::array();
        for (const Vector2& position : config->GetMarkerPositions()) {
            marker_positions.push_back(Vector2Json(position));
        }
        Json marker_barycentric = Json::array();
        for (const Vector4& weights : config->GetMarkerBarycentric()) {
            marker_barycentric.push_back(Vector4Json(weights));
        }
        Json attachment = nullptr;
        if (const Link3D* link = FindAncestorLink(sensor)) {
            attachment = {
                    {"link_path", NodePathString(*link)},
                    {"transform", TransformJson(
                            link->GetGlobalTransform().inverse() *
                            sensor.GetGlobalTransform())}};
        }
        tactile_sensors_.push_back({
                {"attachment", std::move(attachment)},
                {"collision_layer", sensor.GetCollisionLayer()},
                {"collision_mask", sensor.GetCollisionMask()},
                {"coat_vertex_indices", config->GetCoatVertexIndices()},
                {"damping", config->GetDamping()},
                {"density", config->GetDensity()},
                {"enabled", sensor.IsEnabled()},
                {"far_plane", config->GetFarPlane()},
                {"friction_coefficient", config->GetFrictionCoefficient()},
                {"gel_mesh_blob", AddMesh(*gel_mesh.Get())},
                {"gel_topology_sha256", MeshTopologyDigest(*gel_mesh.Get())},
                {"gel_tetrahedron_count", gel_mesh->GetTetrahedronCount()},
                {"gel_vertex_count", gel_mesh->GetVertexCount()},
                {"marker_barycentric", std::move(marker_barycentric)},
                {"marker_positions", std::move(marker_positions)},
                {"marker_tetrahedra", config->GetMarkerTetrahedra()},
                {"name", sensor.GetName()},
                {"near_plane", config->GetNearPlane()},
                {"path", path},
                {"pixel_size", config->GetPixelSize()},
                {"poisson_ratio", config->GetPoissonRatio()},
                {"resolution", Json::array({config->GetImageHeight(), config->GetImageWidth()})},
                {"rgb_model", config->GetRgbModel()},
                {"stick_vertex_indices", config->GetStickVertexIndices()},
                {"transform", GlobalTransformJson(sensor)},
                {"young_modulus", config->GetYoungModulus()}});
        return true;
    }

    bool AddCollisionShape(const CollisionShape3D& collision,
                           const Link3D& link,
                           Json* collision_shapes,
                           std::unordered_set<std::string>* collision_paths,
                           std::string* error) {
        const std::string path = NodePathString(collision);
        if (collision.GetName().empty() || !collision_paths->insert(path).second) {
            return SetCompileError(
                    error, "robot collision shape paths and names must be non-empty and unique");
        }
        if (!ValidateDeformableTransform(
                    collision, "robot collision shape '" + path + "'", error)) {
            return false;
        }
        const Ref<Shape3D>& shape = collision.GetShape();
        if (!shape.IsValid()) {
            return SetCompileError(
                    error, "robot collision shape '" + path + "' has no shape resource");
        }
        PhysicsMaterialSnapshot material;
        if (const Ref<PhysicsMaterial3D>& authored = collision.GetPhysicsMaterial(); authored.IsValid()) {
            material.sliding_friction = authored->GetSlidingFriction();
            material.torsional_friction = authored->GetTorsionalFriction();
            material.rolling_friction = authored->GetRollingFriction();
            material.restitution = authored->GetRestitution();
            material.contact_compliance = authored->GetContactCompliance();
            material.contact_damping = authored->GetContactDamping();
        }
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
            !std::isfinite(collision.GetContactOffset()) || collision.GetContactOffset() < 0.0 ||
            !std::isfinite(collision.GetRestOffset()) ||
            collision.GetRestOffset() > collision.GetContactOffset()) {
            return SetCompileError(
                    error, "robot collision shape '" + path + "' has invalid contact parameters");
        }

        Json compiled = {
                {"collision_layer", collision.GetCollisionLayer()},
                {"collision_mask", collision.GetCollisionMask()},
                {"contact_offset", collision.GetContactOffset()},
                {"disabled", collision.IsDisabled()},
                {"link_transform", TransformJson(
                        link.GetGlobalTransform().inverse() *
                        collision.GetGlobalTransform())},
                {"material", {
                        {"contact_compliance", material.contact_compliance},
                        {"contact_damping", material.contact_damping},
                        {"restitution", material.restitution},
                        {"rolling_friction", material.rolling_friction},
                        {"sliding_friction", material.sliding_friction},
                        {"torsional_friction", material.torsional_friction}}},
                {"name", collision.GetName()},
                {"path", path},
                {"rest_offset", collision.GetRestOffset()},
                {"transform", GlobalTransformJson(collision)}};

        if (const auto box = dynamic_pointer_cast<BoxShape3D>(shape)) {
            const Vector3 size = box->GetSize();
            if (!size.allFinite() || (size.array() <= 0.0).any()) {
                return SetCompileError(
                        error, "robot box collision shape '" + path + "' has invalid size");
            }
            compiled["shape_type"] = "box";
            compiled["size"] = Vector3Json(size);
        } else if (const auto sphere = dynamic_pointer_cast<SphereShape3D>(shape)) {
            const double radius = sphere->GetRadius();
            if (!std::isfinite(radius) || radius <= 0.0) {
                return SetCompileError(
                        error, "robot sphere collision shape '" + path + "' has invalid radius");
            }
            compiled["radius"] = radius;
            compiled["shape_type"] = "sphere";
        } else if (const auto capsule = dynamic_pointer_cast<CapsuleShape3D>(shape)) {
            const double radius = capsule->GetRadius();
            const double height = capsule->GetHeight();
            if (!std::isfinite(radius) || radius <= 0.0 ||
                !std::isfinite(height) || height <= 0.0) {
                return SetCompileError(
                        error, "robot capsule collision shape '" + path +
                                       "' has invalid dimensions");
            }
            compiled["height"] = height;
            compiled["radius"] = radius;
            compiled["shape_type"] = "capsule";
        } else if (const auto cylinder = dynamic_pointer_cast<CylinderShape3D>(shape)) {
            const double radius = cylinder->GetRadius();
            const double height = cylinder->GetHeight();
            if (!std::isfinite(radius) || radius <= 0.0 ||
                !std::isfinite(height) || height <= 0.0) {
                return SetCompileError(
                        error, "robot cylinder collision shape '" + path +
                                       "' has invalid dimensions");
            }
            compiled["height"] = height;
            compiled["radius"] = radius;
            compiled["shape_type"] = "cylinder";
        } else if (const auto convex_mesh = dynamic_pointer_cast<ConvexMeshShape3D>(shape)) {
            const Ref<Mesh>& mesh = convex_mesh->GetMesh();
            if (!mesh.IsValid()) {
                return SetCompileError(
                        error, "robot triangle collision shape '" + path + "' has no mesh");
            }
            std::string blob_id;
            std::size_t vertex_count = 0;
            std::size_t triangle_count = 0;
            std::string mesh_error;
            if (!AddSurfaceMesh(
                        *mesh.Get(), &blob_id, &vertex_count, &triangle_count, &mesh_error)) {
                return SetCompileError(
                        error, "robot triangle collision shape '" + path +
                                       "' is invalid: " + mesh_error);
            }
            compiled["mesh_blob"] = blob_id;
            compiled["shape_type"] = "triangle_mesh";
            compiled["triangle_count"] = triangle_count;
            compiled["vertex_count"] = vertex_count;
        } else {
            return SetCompileError(
                    error, "robot collision shape '" + path + "' uses an unsupported shape type");
        }

        collision_shapes->push_back(std::move(compiled));
        return true;
    }

    bool CollectLinkCollisionShapes(
            const Node* node,
            const Link3D* owner,
            Json* collision_shapes,
            std::unordered_set<std::string>* collision_paths,
            std::string* error) {
        if (node != owner &&
            (Object::PointerCastTo<Link3D>(node) != nullptr ||
             Object::PointerCastTo<Robot3D>(node) != nullptr)) {
            return true;
        }
        if (const auto* collision = Object::PointerCastTo<CollisionShape3D>(node)) {
            if (!AddCollisionShape(
                        *collision, *owner, collision_shapes, collision_paths, error)) {
                return false;
            }
        }
        for (std::size_t index = 0; index < node->GetChildCount(); ++index) {
            if (!CollectLinkCollisionShapes(
                        node->GetChild(static_cast<int>(index)), owner,
                        collision_shapes, collision_paths, error)) {
                return false;
            }
        }
        return true;
    }

    bool AddRobot(const Robot3D& robot, std::string* error) {
        const std::string robot_path = NodePathString(robot);
        if (robot.GetName().empty()) {
            return SetCompileError(error, "IPC robots require a non-empty name");
        }
        if (!ValidateDeformableTransform(
                    robot, "robot '" + robot_path + "'", error)) {
            return false;
        }
        Json links = Json::array();
        Json joints = Json::array();
        std::unordered_set<std::string> link_names;
        std::unordered_set<std::string> joint_names;
        if (!CollectRobotNodes(
                    &robot, &robot, &links, &joints, &link_names, &joint_names, error)) {
            return false;
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
                {"links", std::move(links)},
                {"name", robot.GetName()},
                {"path", robot_path},
                {"root_link_paths", std::move(root_link_paths)},
                {"transform", GlobalTransformJson(robot)}});
        return true;
    }

    bool CollectRobotNodes(const Node* node,
                           const Robot3D* root,
                           Json* links,
                           Json* joints,
                           std::unordered_set<std::string>* link_names,
                           std::unordered_set<std::string>* joint_names,
                           std::string* error) {
        if (node != root && Object::PointerCastTo<Robot3D>(node) != nullptr) {
            return true;
        }
        if (const auto* link = Object::PointerCastTo<Link3D>(node)) {
            const std::string path = NodePathString(*link);
            if (link->GetName().empty() || !link_names->insert(link->GetName()).second) {
                return SetCompileError(
                        error, "robot link names must be non-empty and unique in '" +
                                       NodePathString(*root) + "'");
            }
            if (!ValidateDeformableTransform(*link, "robot link '" + path + "'", error)) {
                return false;
            }
            Json collision_shapes = Json::array();
            std::unordered_set<std::string> collision_paths;
            if (!CollectLinkCollisionShapes(
                        link, link, &collision_shapes, &collision_paths, error)) {
                return false;
            }
            const Quaternion& inertia_orientation = link->GetInertiaOrientation();
            if (!std::isfinite(link->GetMass()) || link->GetMass() < 0.0 ||
                !link->GetCenterOfMass().allFinite() ||
                !inertia_orientation.coeffs().allFinite() ||
                inertia_orientation.squaredNorm() <=
                        std::numeric_limits<RealType>::epsilon() ||
                !link->GetInertiaDiagonal().allFinite() ||
                (link->GetInertiaDiagonal().array() < 0.0).any() ||
                !link->GetInertiaOffDiagonal().allFinite() ||
                static_cast<int>(link->GetRole()) <
                        static_cast<int>(LinkRole::Physical) ||
                static_cast<int>(link->GetRole()) >
                        static_cast<int>(LinkRole::VirtualRoot)) {
                return SetCompileError(
                        error, "robot link '" + path + "' has invalid inertial properties");
            }
            links->push_back({
                    {"center_of_mass", Vector3Json(link->GetCenterOfMass())},
                    {"collision_shapes", std::move(collision_shapes)},
                    {"has_inertial", link->HasInertial()},
                    {"inertia_diagonal", Vector3Json(link->GetInertiaDiagonal())},
                    {"inertia_off_diagonal", Vector3Json(link->GetInertiaOffDiagonal())},
                    {"inertia_orientation_wxyz", QuaternionWxyzJson(inertia_orientation)},
                    {"local_transform", LocalTransformJson(*link)},
                    {"mass", link->GetMass()},
                    {"name", link->GetName()},
                    {"path", path},
                    {"role", static_cast<int>(link->GetRole())},
                    {"transform", GlobalTransformJson(*link)}});
        }
        if (const auto* joint = Object::PointerCastTo<Joint3D>(node)) {
            const std::string path = NodePathString(*joint);
            if (joint->GetName().empty() || !joint_names->insert(joint->GetName()).second) {
                return SetCompileError(
                        error, "robot joint names must be non-empty and unique in '" +
                                       NodePathString(*root) + "'");
            }
            if (!ValidateDeformableTransform(*joint, "robot joint '" + path + "'", error)) {
                return false;
            }
            const auto* parent_link = Object::PointerCastTo<Link3D>(joint->GetParent());
            const Link3D* child_link = nullptr;
            for (std::size_t index = 0; index < joint->GetChildCount(); ++index) {
                if (const auto* candidate =
                            Object::PointerCastTo<Link3D>(joint->GetChild(static_cast<int>(index)))) {
                    if (child_link != nullptr) {
                        return SetCompileError(
                                error, "robot joint '" + path +
                                               "' has more than one direct child link");
                    }
                    child_link = candidate;
                }
            }
            const std::array<RealType, 20> parameters{
                    joint->GetLowerLimit(), joint->GetUpperLimit(),
                    joint->GetEffortLimit(), joint->GetVelocityLimit(),
                    joint->GetDamping(), joint->GetArmature(), joint->GetFrictionLoss(),
                    joint->GetJointPosition(), joint->GetInitialPosition(),
                    joint->GetDriveStiffness(), joint->GetDriveDamping(),
                    joint->GetControlLowerLimit(), joint->GetControlUpperLimit(),
                    joint->GetForceLowerLimit(), joint->GetForceUpperLimit(),
                    joint->GetAffineActuatorControlGain(),
                    joint->GetAffineActuatorForceOffset(),
                    joint->GetAffineActuatorPositionGain(),
                    joint->GetAffineActuatorVelocityGain(),
                    joint->GetAffineActuatorInheritRange()};
            if (parent_link == nullptr || child_link == nullptr ||
                !joint->GetAxis().allFinite() ||
                joint->GetAxis().squaredNorm() <=
                        std::numeric_limits<RealType>::epsilon() ||
                !std::all_of(parameters.begin(), parameters.end(), [](RealType value) {
                    return std::isfinite(value);
                }) ||
                joint->GetEffortLimit() < 0.0 || joint->GetVelocityLimit() < 0.0 ||
                joint->GetDamping() < 0.0 || joint->GetArmature() < 0.0 ||
                joint->GetFrictionLoss() < 0.0 || joint->GetDriveStiffness() < 0.0 ||
                joint->GetDriveDamping() < 0.0 ||
                static_cast<int>(joint->GetJointType()) <
                        static_cast<int>(JointType::Fixed) ||
                static_cast<int>(joint->GetJointType()) >
                        static_cast<int>(JointType::Planar) ||
                static_cast<int>(joint->GetDriveMode()) <
                        static_cast<int>(JointDriveMode::Passive) ||
                static_cast<int>(joint->GetDriveMode()) >
                        static_cast<int>(JointDriveMode::Velocity) ||
                !std::all_of(
                        joint->GetGear().begin(), joint->GetGear().end(),
                        [](RealType value) { return std::isfinite(value); })) {
                return SetCompileError(
                        error, "robot joint '" + path + "' has invalid topology or parameters");
            }
            joints->push_back({
                    {"affine_actuator_control_gain", joint->GetAffineActuatorControlGain()},
                    {"affine_actuator_enabled", joint->IsAffineActuatorEnabled()},
                    {"affine_actuator_force_offset", joint->GetAffineActuatorForceOffset()},
                    {"affine_actuator_inherit_range", joint->GetAffineActuatorInheritRange()},
                    {"affine_actuator_position_gain", joint->GetAffineActuatorPositionGain()},
                    {"affine_actuator_velocity_gain", joint->GetAffineActuatorVelocityGain()},
                    {"authored_child_link", joint->GetChildLink()},
                    {"authored_parent_link", joint->GetParentLink()},
                    {"axis", Vector3Json(joint->GetAxis())},
                    {"armature", joint->GetArmature()},
                    {"child_link", child_link->GetName()},
                    {"control_lower_limit", joint->GetControlLowerLimit()},
                    {"control_upper_limit", joint->GetControlUpperLimit()},
                    {"damping", joint->GetDamping()},
                    {"drive_damping", joint->GetDriveDamping()},
                    {"drive_mode", static_cast<int>(joint->GetDriveMode())},
                    {"drive_stiffness", joint->GetDriveStiffness()},
                    {"effort_limit", joint->GetEffortLimit()},
                    {"force_lower_limit", joint->GetForceLowerLimit()},
                    {"force_upper_limit", joint->GetForceUpperLimit()},
                    {"friction_loss", joint->GetFrictionLoss()},
                    {"gear", joint->GetGear()},
                    {"initial_position", joint->GetInitialPosition()},
                    {"joint_type", static_cast<int>(joint->GetJointType())},
                    {"joint_position", joint->GetJointPosition()},
                    {"local_transform", LocalTransformJson(*joint)},
                    {"lower_limit", joint->GetLowerLimit()},
                    {"name", joint->GetName()},
                    {"parent_link", parent_link->GetName()},
                    {"path", path},
                    {"transform", GlobalTransformJson(*joint)},
                    {"velocity_limit", joint->GetVelocityLimit()},
                    {"upper_limit", joint->GetUpperLimit()}});
        }
        for (std::size_t index = 0; index < node->GetChildCount(); ++index) {
            if (!CollectRobotNodes(
                        node->GetChild(static_cast<int>(index)), root, links, joints,
                        link_names, joint_names, error)) {
                return false;
            }
        }
        return true;
    }

    Json deformable_bodies_ = Json::array();
    Json tactile_sensors_ = Json::array();
    Json robots_ = Json::array();
    Json couplings_ = Json::array();
    std::vector<const PhysicsCoupling*> coupling_nodes_;
    std::map<std::string, IpcSceneArtifactBlob> blobs_;
};

} // namespace

bool IpcSceneCompiler::Compile(
        const Node* scene_root, IpcSceneArtifact* artifact, std::string* error) {
    if (scene_root == nullptr) {
        return SetCompileError(error, "Cannot compile an IPC artifact without a scene root.");
    }
    if (artifact == nullptr) {
        return SetCompileError(error, "Cannot compile an IPC artifact into a null output.");
    }
    if (scene_root->GetName().empty()) {
        return SetCompileError(error, "Cannot compile an IPC artifact with an unnamed scene root.");
    }

    CompilerState compiler;
    if (!compiler.Visit(scene_root, error)) {
        return false;
    }
    if (!compiler.FinalizeCouplings(*scene_root, error)) {
        return false;
    }
    const Json manifest_json = compiler.BuildManifest(*scene_root);
    IpcSceneArtifact result;
    result.schema_version = 3;
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
