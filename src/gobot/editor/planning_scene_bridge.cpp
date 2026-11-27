/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/editor/planning_scene_bridge.hpp"

#include "gobot/core/object.hpp"
#include "gobot/physics/physics_scene_compiler.hpp"
#include "gobot/scene/joint_3d.hpp"
#include "gobot/scene/link_3d.hpp"
#include "gobot/scene/mesh_instance_3d.hpp"
#include "gobot/scene/node.hpp"
#include "gobot/scene/node_3d.hpp"
#include "gobot/scene/resources/material.hpp"
#include "gobot/scene/resources/mesh.hpp"
#include "gobot/scene/rigid_body_3d.hpp"
#include "gobot/scene/robot_3d.hpp"

#if defined(GOBOT_HAS_PLANNING)
#include "planning_shapes.hpp"
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <queue>
#include <set>
#include <sstream>
#include <unordered_set>

namespace gobot {

namespace {

std::string CanonicalScenePath(const Node* node) {
    std::vector<std::string> names;
    for (const Node* current = node; current != nullptr; current = current->GetParent()) {
        names.push_back(current->GetName());
    }
    std::string path;
    for (auto it = names.rbegin(); it != names.rend(); ++it) {
        path.push_back('/');
        path += *it;
    }
    return path.empty() ? std::string{"/"} : path;
}

void CollectNodePathIdsRecursive(const Node* node,
                                 std::unordered_map<std::string, std::uint64_t>& path_to_id) {
    if (node == nullptr) {
        return;
    }
    const std::uint64_t id = static_cast<std::uint64_t>(node->GetInstanceId());
    if (id != 0) {
        path_to_id[CanonicalScenePath(node)] = id;
    }
    for (std::size_t i = 0; i < node->GetChildCount(); ++i) {
        CollectNodePathIdsRecursive(node->GetChild(static_cast<int>(i)), path_to_id);
    }
}

std::uint64_t HashCombine(std::uint64_t seed, std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL + (seed << 6U) + (seed >> 2U);
    return seed ^ value;
}

std::uint64_t HashString(std::uint64_t seed, const std::string& value) {
    std::uint64_t h = 1469598103934665603ULL;
    for (unsigned char c : value) {
        h ^= c;
        h *= 1099511628211ULL;
    }
    return HashCombine(seed, h);
}

std::uint64_t HashDouble(std::uint64_t seed, double value) {
    if (std::isnan(value)) {
        return HashCombine(seed, 0x7ff8000000000000ULL);
    }
    if (value == 0.0) {
        value = 0.0;  // normalize -0.0
    }
    // Quantize finite floats gently (1e-9) so tiny FP noise doesn't trigger unnecessary rebuilds.
    if (std::isfinite(value)) {
        value = std::round(value * 1e9) * 1e-9;
    }
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return HashCombine(seed, bits);
}

std::uint64_t HashShapeGeometry(const PhysicsShapeSnapshot& sh) {
    std::uint64_t h = 1469598103934665603ULL;
    h = HashCombine(h, static_cast<std::uint64_t>(sh.type));
    h = HashCombine(h, sh.convex ? 1ULL : 0ULL);
    switch (sh.type) {
        case PhysicsShapeType::Box:
            h = HashDouble(h, static_cast<double>(sh.box_size.x()));
            h = HashDouble(h, static_cast<double>(sh.box_size.y()));
            h = HashDouble(h, static_cast<double>(sh.box_size.z()));
            break;
        case PhysicsShapeType::Sphere:
            h = HashDouble(h, static_cast<double>(sh.radius));
            break;
        case PhysicsShapeType::Cylinder:
        case PhysicsShapeType::Capsule:
            h = HashDouble(h, static_cast<double>(sh.radius));
            h = HashDouble(h, static_cast<double>(sh.height));
            break;
        case PhysicsShapeType::Mesh:
            h = HashCombine(h, sh.vertices.size());
            h = HashCombine(h, sh.indices.size());
            for (const Vector3& v : sh.vertices) {
                h = HashDouble(h, static_cast<double>(v.x()));
                h = HashDouble(h, static_cast<double>(v.y()));
                h = HashDouble(h, static_cast<double>(v.z()));
            }
            for (std::uint32_t idx : sh.indices) {
                h = HashCombine(h, idx);
            }
            break;
        case PhysicsShapeType::Plane:
        default:
            break;
    }
    return h;
}

std::uint64_t HashTerrainBoxGeometry(const PhysicsTerrainBoxSnapshot& box) {
    std::uint64_t h = HashCombine(1469598103934665603ULL, static_cast<std::uint64_t>(PhysicsShapeType::Box));
    h = HashDouble(h, static_cast<double>(box.size.x()));
    h = HashDouble(h, static_cast<double>(box.size.y()));
    h = HashDouble(h, static_cast<double>(box.size.z()));
    return h;
}

std::uint64_t HashTerrainMeshGeometry(const PhysicsTerrainMeshPatchSnapshot& patch) {
    std::uint64_t h = HashCombine(1469598103934665603ULL, static_cast<std::uint64_t>(PhysicsShapeType::Mesh));
    h = HashCombine(h, patch.vertices.size());
    h = HashCombine(h, patch.indices.size());
    for (const Vector3& v : patch.vertices) {
        h = HashDouble(h, static_cast<double>(v.x()));
        h = HashDouble(h, static_cast<double>(v.y()));
        h = HashDouble(h, static_cast<double>(v.z()));
    }
    for (std::uint32_t idx : patch.indices) {
        h = HashCombine(h, idx);
    }
    return h;
}

Affine3 ResolveNodeGlobal(const Node3D* node, const Affine3& parent_global) {
    if (node == nullptr) {
        return parent_global;
    }
    return node->IsInsideTree() ? node->GetGlobalTransform() : (parent_global * node->GetTransform());
}

void CollectLinkVisualsRecursive(const Node* node,
                                 const Affine3& parent_global,
                                 int current_body_id,
                                 const std::string& current_link_name,
                                 const Affine3& current_link_global_inv,
                                 const std::unordered_map<std::string, int>& link_name_to_body_id,
                                 const ObjectID& owner_robot_id,
                                 std::vector<LinkVisualTemplate>& out) {
    if (node == nullptr) {
        return;
    }
    if (node->GetInstanceId() != owner_robot_id &&
        (Object::PointerCastTo<Robot3D>(node) != nullptr ||
         Object::PointerCastTo<RigidBody3D>(node) != nullptr)) {
        return;
    }

    const auto* node_3d = Object::PointerCastTo<Node3D>(node);
    const Affine3 global = ResolveNodeGlobal(node_3d, parent_global);

    int body_id = current_body_id;
    std::string link_name = current_link_name;
    Affine3 link_global_inv = current_link_global_inv;

    if (const auto* link = Object::PointerCastTo<Link3D>(node)) {
        if (auto it = link_name_to_body_id.find(link->GetName()); it != link_name_to_body_id.end()) {
            body_id = it->second;
            link_name = link->GetName();
            link_global_inv = global.inverse();
        }
    }

    if (body_id >= 0) {
        if (const auto* mesh_instance = Object::PointerCastTo<MeshInstance3D>(node)) {
            const bool visible = mesh_instance->IsInsideTree()
                                         ? mesh_instance->IsVisibleInTree()
                                         : mesh_instance->IsVisible();
            const Ref<Mesh>& mesh = mesh_instance->GetMesh();
            if (visible && mesh.IsValid()) {
                const std::shared_ptr<const MeshSurfaceList> surfaces = mesh->GetSurfaceData();
                if (surfaces != nullptr) {
                    const Affine3 link_to_mesh = link_global_inv * global;
                    for (std::size_t si = 0; si < surfaces->size(); ++si) {
                        const MeshSurfaceData& surf = (*surfaces)[si];
                        if (surf.vertices.empty() || surf.indices.empty()) {
                            continue;
                        }
                        VisualMeshRenderItem item;
                        item.object_id = mesh_instance->GetInstanceId();
                        item.mesh_id = mesh->GetInstanceId();
                        item.mesh_revision = mesh->GetGeometryRevision();
                        item.mesh_topology_revision = mesh->GetTopologyRevision();
                        item.surface_index = si;
                        item.surfaces = surfaces;
                        item.model = global.matrix();
                        item.world_bounds = surf.bounds.Transformed(item.model);
                        item.visible_in_rgb = true;
                        item.cast_shadow = false;
                        out.push_back(LinkVisualTemplate{body_id, link_name, link_to_mesh, std::move(item)});
                    }
                }
            }
        }
    }

    for (std::size_t i = 0; i < node->GetChildCount(); ++i) {
        CollectLinkVisualsRecursive(node->GetChild(static_cast<int>(i)),
                                    global,
                                    body_id,
                                    link_name,
                                    link_global_inv,
                                    link_name_to_body_id,
                                    owner_robot_id,
                                    out);
    }
}

#if defined(GOBOT_HAS_PLANNING)

bool ExtractRigidTransform(const Affine3& tf, planning::Vec3& out_pos, planning::Mat3& out_rot) {
    const Eigen::Matrix4d mat = tf.matrix().template cast<double>();
    if (!mat.allFinite()) {
        return false;
    }
    const Eigen::RowVector4d bottom(0.0, 0.0, 0.0, 1.0);
    if (!(mat.row(3) - bottom).isZero(1e-5)) {
        return false;
    }
    const planning::Mat3 r = mat.topLeftCorner<3, 3>();
    if (!(r.transpose() * r - planning::Mat3::Identity()).isZero(1e-4) ||
        std::abs(r.determinant() - 1.0) > 1e-4) {
        return false;
    }
    out_pos = mat.topRightCorner<3, 1>();
    out_rot = r;
    return true;
}

planning::Frame FrameFromRT(const planning::Vec3& p, const planning::Mat3& r) {
    planning::Frame f;
    f.leftCols<3>() = r;
    f.col(3) = p;
    return f;
}

bool FrameApproxEqual(const planning::Frame& a, const planning::Frame& b, double tol = 1e-7) {
    return (a - b).cwiseAbs().maxCoeff() <= tol;
}

Eigen::Matrix4d MakeJointMotionMatrix(int joint_type, const planning::Vec3& axis_unit, double q) {
    Eigen::Matrix4d m = Eigen::Matrix4d::Identity();
    if (joint_type == 1 || joint_type == 2) {
        const Eigen::AngleAxisd aa(q, axis_unit);
        m.topLeftCorner<3, 3>() = aa.toRotationMatrix();
    } else if (joint_type == 3) {
        m.topRightCorner<3, 1>() = axis_unit * q;
    }
    return m;
}

std::shared_ptr<planning::CollisionShape> BuildNativeShape(const PhysicsShapeSnapshot& sh,
                                                           const std::string& label) {
    switch (sh.type) {
        case PhysicsShapeType::Box:
            return planning::make_box_shape(label, sh.box_size.template cast<double>());
        case PhysicsShapeType::Sphere:
            return planning::make_sphere_shape(label, static_cast<double>(sh.radius));
        case PhysicsShapeType::Cylinder:
            return planning::make_cylinder_shape(label,
                                                 static_cast<double>(sh.radius),
                                                 static_cast<double>(sh.height));
        case PhysicsShapeType::Capsule:
            return planning::make_capsule_shape(label,
                                                static_cast<double>(sh.radius),
                                                static_cast<double>(sh.height));
        case PhysicsShapeType::Plane:
            return planning::make_plane_shape(label);
        case PhysicsShapeType::Mesh: {
            std::vector<planning::Vec3> verts;
            verts.reserve(sh.vertices.size());
            for (const Vector3& v : sh.vertices) {
                verts.push_back(v.template cast<double>());
            }
            return planning::make_mesh_shape(label, verts, sh.indices);
        }
        default:
            throw std::invalid_argument("unsupported collision shape type on " + label);
    }
}

std::string MakeUniqueKey(std::unordered_map<std::string, int>& counts, const std::string& base) {
    const int occ = counts[base]++;
    return occ == 0 ? base : (base + "#" + std::to_string(occ));
}

struct IncomingObstacle {
    std::string sync_key;
    std::string node_key;
    std::string stable_key;
    std::string path_key;
    std::string default_name;
    std::string link_key;
    planning::Frame world_frame = planning::Frame::Zero();
    int layer = 1;
    int mask = 1;
    std::uint64_t geom_hash = 0;
    std::function<std::shared_ptr<planning::CollisionShape>(const std::string&)> build_shape;
};

#endif

} // namespace

PlanningSceneBridge::PlanningSceneBridge() = default;
PlanningSceneBridge::~PlanningSceneBridge() = default;

bool PlanningSceneBridge::IsPlanningAvailable() noexcept {
#if defined(GOBOT_HAS_PLANNING)
    return true;
#else
    return false;
#endif
}

void PlanningSceneBridge::Clear() {
    robot_id_ = ObjectID();
    robot_name_.clear();
    base_link_name_.clear();
    default_ee_link_id_ = -1;
    robot_topology_hash_ = 0;
    body_names_.clear();
    link_names_.clear();
    joint_names_.clear();
    link_name_to_body_id_.clear();
    movable_joint_ids_.clear();
    current_q_.resize(0);
    initial_q_.resize(0);
    tracked_obstacles_.clear();
    used_shape_names_.clear();
    visual_templates_.clear();
    last_sync_stats_ = {};
    has_sync_cache_ = false;
    last_sync_error_.clear();
#if defined(GOBOT_HAS_PLANNING)
    world_.reset();
#endif
}

void PlanningSceneBridge::SetExtraIgnoredPairs(std::vector<std::pair<std::string, std::string>> pairs) {
    if (pairs == extra_ignored_pairs_) return;
    extra_ignored_pairs_ = std::move(pairs);
    has_sync_cache_ = false;
}

PlanningSnapshotIdentity PlanningSceneBridge::GetCurrentSnapshotIdentity() const noexcept {
    PlanningSnapshotIdentity id;
#if defined(GOBOT_HAS_PLANNING)
    if (world_ != nullptr && robot_id_.IsValid()) {
        const auto snap = world_->snapshot();
        if (snap) {
            id.scene_epoch = synced_scene_epoch_;
            id.robot_id = robot_id_;
            id.world_generation = world_generation_;
            id.world_version = snap->version;
            id.snapshot_ptr = snap.get();
        }
    }
#endif
    return id;
}

bool PlanningSceneBridge::EnsureSynced(
        const Node* scene_root,
        const Robot3D* target_robot,
        std::uint64_t scene_epoch,
        std::uint64_t scene_change_version) {
    if (scene_root == nullptr || target_robot == nullptr) {
        Clear();
        synced_scene_epoch_ = scene_epoch;
        synced_robot_id_ = ObjectID{};
        synced_change_version_ = scene_change_version;
        return false;
    }

    const ObjectID robot_id = target_robot->GetInstanceId();
    if (has_sync_cache_ &&
        synced_scene_epoch_ == scene_epoch &&
        synced_robot_id_ == robot_id &&
        synced_change_version_ == scene_change_version) {
        return IsReady() && last_sync_error_.empty();
    }

    if (synced_scene_epoch_ != scene_epoch || synced_robot_id_ != robot_id) {
        Clear();
    }

    std::string sync_error;
    const bool ok = SyncFromScene(scene_root, target_robot, "", extra_ignored_pairs_, &sync_error);
    if (!ok) {
        Clear();
        has_sync_cache_ = true;
        synced_scene_epoch_ = scene_epoch;
        synced_robot_id_ = robot_id;
        synced_change_version_ = scene_change_version;
        last_sync_error_ = sync_error.empty() ? "Planning scene synchronization failed." : sync_error;
        return false;
    }

    has_sync_cache_ = true;
    synced_scene_epoch_ = scene_epoch;
    synced_robot_id_ = robot_id;
    synced_change_version_ = scene_change_version;
    last_sync_error_.clear();
    return IsReady();
}

bool PlanningSceneBridge::IsReady() const noexcept {
#if defined(GOBOT_HAS_PLANNING)
    return world_ != nullptr && default_ee_link_id_ >= 0 && current_q_.size() > 0;
#else
    return false;
#endif
}

int PlanningSceneBridge::FindLinkId(const std::string& link_name) const {
    if (auto it = link_name_to_body_id_.find(link_name); it != link_name_to_body_id_.end()) {
        return it->second;
    }
    return -1;
}

std::uint64_t PlanningSceneBridge::GetWorldVersion() const noexcept {
#if defined(GOBOT_HAS_PLANNING)
    return world_ != nullptr ? world_->version() : 0;
#else
    return 0;
#endif
}

void PlanningSceneBridge::CollectRobotVisualTemplates(const Robot3D* target_robot) {
    visual_templates_.clear();
    if (target_robot == nullptr || link_name_to_body_id_.empty()) {
        return;
    }
    Affine3 parent_global = Affine3::Identity();
    if (const Node* parent = target_robot->GetParent()) {
        std::vector<const Node3D*> chain;
        for (const Node* cur = parent; cur != nullptr; cur = cur->GetParent()) {
            if (const auto* n3 = Object::PointerCastTo<Node3D>(cur)) {
                if (n3->IsInsideTree()) {
                    parent_global = n3->GetGlobalTransform();
                    break;
                }
                chain.push_back(n3);
            }
        }
        for (auto it = chain.rbegin(); it != chain.rend(); ++it) {
            parent_global = parent_global * (*it)->GetTransform();
        }
    }
    CollectLinkVisualsRecursive(target_robot,
                                parent_global,
                                -1,
                                "",
                                Affine3::Identity(),
                                link_name_to_body_id_,
                                target_robot->GetInstanceId(),
                                visual_templates_);
}

bool PlanningSceneBridge::SyncFromScene(
        const Node* scene_root,
        const Robot3D* target_robot,
        const std::string& preferred_base_link,
        const std::vector<std::pair<std::string, std::string>>& extra_ignored_pairs,
        std::string* error) {
#if !defined(GOBOT_HAS_PLANNING)
    (void)scene_root;
    (void)target_robot;
    (void)preferred_base_link;
    (void)extra_ignored_pairs;
    if (error != nullptr) {
        *error = "Motion planning requires a Gobot build with GOB_BUILD_COAL=ON.";
    }
    return false;
#else
    if (scene_root == nullptr || target_robot == nullptr) {
        if (error != nullptr) {
            *error = "Scene root and target Robot3D are required.";
        }
        return false;
    }

    ++sync_count_;

    CompiledPhysicsScene compiled;
    std::string compile_error;
    if (!PhysicsSceneCompiler::Compile(scene_root, &compiled, &compile_error)) {
        if (error != nullptr) {
            *error = compile_error.empty() ? "Failed to compile physics scene snapshot." : compile_error;
        }
        return false;
    }
    if (!compiled.snapshot.deformables.empty()) {
        if (error != nullptr) {
            *error = "Planning collision does not support deformable bodies.";
        }
        return false;
    }

    // Build canonical path -> runtime ObjectID map on the main thread.
    std::unordered_map<std::string, std::uint64_t> path_to_node_id;
    CollectNodePathIdsRecursive(scene_root, path_to_node_id);

    // Locate target robot by runtime ObjectID.
    const ObjectID target_robot_id = target_robot->GetInstanceId();
    int robot_idx = -1;
    for (std::size_t i = 0; i < compiled.bindings.robots.size(); ++i) {
        if (compiled.bindings.robots[i].robot_id == target_robot_id) {
            robot_idx = static_cast<int>(i);
            break;
        }
    }
    if (robot_idx < 0 || static_cast<std::size_t>(robot_idx) >= compiled.snapshot.robots.size()) {
        if (error != nullptr) {
            *error = "Target Robot3D was not found in compiled physics scene.";
        }
        return false;
    }

    const PhysicsRobotSnapshot& robot_snap = compiled.snapshot.robots[static_cast<std::size_t>(robot_idx)];
    const PhysicsRobotSceneBinding& robot_bind = compiled.bindings.robots[static_cast<std::size_t>(robot_idx)];
    const std::string& robot_name = robot_snap.name;

    std::unordered_map<std::string, const PhysicsLinkSnapshot*> links_by_name;
    for (const PhysicsLinkSnapshot& lk : robot_snap.links) {
        links_by_name[lk.name] = &lk;
    }

    std::unordered_map<std::string, std::vector<std::string>> children_by_link;
    std::unordered_map<std::string, std::size_t> parent_joint_index;
    std::set<std::pair<std::string, std::string>> adjacent_link_pairs;
    for (std::size_t ji = 0; ji < robot_snap.joints.size(); ++ji) {
        const PhysicsJointSnapshot& j = robot_snap.joints[ji];
        children_by_link[j.parent_link].push_back(j.child_link);
        if (parent_joint_index.contains(j.child_link)) {
            if (error != nullptr) {
                *error = "Multiple parent joints on link '" + j.child_link + "'.";
            }
            return false;
        }
        parent_joint_index[j.child_link] = ji;
        adjacent_link_pairs.insert({robot_name + "::" + j.parent_link,
                                    robot_name + "::" + j.child_link});
    }

    // Resolve base link: preferred if valid, otherwise first non-virtual root link (or first root link).
    std::string base_link = preferred_base_link;
    if (base_link.empty() || !links_by_name.contains(base_link)) {
        base_link.clear();
        for (const PhysicsLinkSnapshot& lk : robot_snap.links) {
            if (lk.role == PhysicsLinkRole::VirtualRoot) continue;
            auto pit = parent_joint_index.find(lk.name);
            if (pit == parent_joint_index.end()) {
                base_link = lk.name;
                break;
            }
            const std::string& p_name = robot_snap.joints[pit->second].parent_link;
            auto p_lk = links_by_name.find(p_name);
            if (p_lk == links_by_name.end() || p_lk->second->role == PhysicsLinkRole::VirtualRoot) {
                base_link = lk.name;
                break;
            }
        }
        if (base_link.empty() && !robot_snap.links.empty()) {
            base_link = robot_snap.links.front().name;
        }
    }
    if (base_link.empty() || !links_by_name.contains(base_link)) {
        if (error != nullptr) {
            *error = "Robot '" + robot_name + "' has no valid base link.";
        }
        return false;
    }

    // BFS link traversal starting from base_link (matches world.py:_extract_robot_model).
    std::vector<std::string> link_order;
    std::unordered_set<std::string> seen_links;
    link_order.push_back(base_link);
    seen_links.insert(base_link);
    for (std::size_t head = 0; head < link_order.size(); ++head) {
        const std::string cur = link_order[head];
        if (auto it = children_by_link.find(cur); it != children_by_link.end()) {
            for (const std::string& ch : it->second) {
                if (links_by_name.contains(ch) && !seen_links.contains(ch)) {
                    seen_links.insert(ch);
                    link_order.push_back(ch);
                }
            }
        }
    }

    // Extract movable joints in robot_snap.joints order (matching world.py).
    std::vector<std::size_t> movable_joint_snap_indices;
    std::unordered_map<std::string, int> movable_idx_map;
    std::vector<std::string> joint_names_canon;
    std::vector<std::string> joint_names_short;
    std::vector<ObjectID> movable_joint_ids;
    std::vector<double> lower_list, upper_list, vel_list, acc_list, jerk_list, init_list, cur_list;

    for (std::size_t ji = 0; ji < robot_snap.joints.size(); ++ji) {
        const PhysicsJointSnapshot& j = robot_snap.joints[ji];
        if (!seen_links.contains(j.child_link)) continue;
        if (j.joint_type != 1 && j.joint_type != 2 && j.joint_type != 3) continue;
        const int mj_idx = static_cast<int>(movable_joint_snap_indices.size());
        movable_joint_snap_indices.push_back(ji);
        movable_idx_map[j.name] = mj_idx;
        joint_names_canon.push_back(robot_name + "::" + j.name);
        joint_names_short.push_back(j.name);
        movable_joint_ids.push_back(ji < robot_bind.joint_ids.size() ? robot_bind.joint_ids[ji] : ObjectID{});

        if (j.joint_type == 2) {
            lower_list.push_back(-std::numeric_limits<double>::infinity());
            upper_list.push_back(std::numeric_limits<double>::infinity());
        } else if (j.upper_limit > j.lower_limit) {
            lower_list.push_back(static_cast<double>(j.lower_limit));
            upper_list.push_back(static_cast<double>(j.upper_limit));
        } else {
            lower_list.push_back(-std::numeric_limits<double>::infinity());
            upper_list.push_back(std::numeric_limits<double>::infinity());
        }
        const double speed = static_cast<double>(j.velocity_limit);
        vel_list.push_back(speed > 0.0 ? speed : std::numeric_limits<double>::infinity());
        acc_list.push_back(default_acceleration_limit_);
        jerk_list.push_back(std::numeric_limits<double>::infinity());
        init_list.push_back(std::clamp(static_cast<double>(j.initial_position),
                                       lower_list.back(), upper_list.back()));
        cur_list.push_back(std::clamp(static_cast<double>(j.joint_position),
                                      lower_list.back(), upper_list.back()));
    }

    if (movable_joint_snap_indices.empty()) {
        if (error != nullptr) {
            *error = "Robot '" + robot_name + "' has no movable joints.";
        }
        return false;
    }

    // Build kinematics arrays and compute robot topology hash.
    const int B = static_cast<int>(link_order.size());
    std::vector<int> parent(static_cast<std::size_t>(B), -1);
    std::vector<int> kinds(static_cast<std::size_t>(B), 4);
    std::vector<int> joint_idx(static_cast<std::size_t>(B), -1);
    std::vector<planning::Mat3> a_rot(static_cast<std::size_t>(B), planning::Mat3::Identity());
    std::vector<planning::Vec3> a_pos(static_cast<std::size_t>(B), planning::Vec3::Zero());
    std::vector<planning::Mat3> b_rot(static_cast<std::size_t>(B), planning::Mat3::Identity());
    std::vector<planning::Vec3> b_pos(static_cast<std::size_t>(B), planning::Vec3::Zero());
    std::vector<planning::Vec3> axes(static_cast<std::size_t>(B), planning::Vec3::UnitX());
    std::vector<std::string> body_names;
    body_names.reserve(static_cast<std::size_t>(B));
    std::unordered_map<std::string, int> local_body_id;
    std::unordered_map<std::string, int> link_index_str;
    std::vector<int> link_depth(static_cast<std::size_t>(B), 0);

    planning::Vec3 root_pos = planning::Vec3::Zero();
    planning::Mat3 root_rot = planning::Mat3::Identity();
    if (!ExtractRigidTransform(links_by_name[base_link]->global_transform, root_pos, root_rot)) {
        if (error != nullptr) {
            *error = "Invalid non-rigid base link transform on '" + base_link + "'.";
        }
        return false;
    }

    struct RobotShapeEntry {
        const PhysicsShapeSnapshot* snap = nullptr;
        std::string link_name;
        int body_id = -1;
        planning::Frame local_frame = planning::Frame::Zero();
        std::uint64_t geom_hash = 0;
    };
    std::vector<RobotShapeEntry> robot_shapes;

    std::uint64_t topo_hash = 1469598103934665603ULL;
    topo_hash = HashCombine(topo_hash, static_cast<std::uint64_t>(target_robot_id));
    topo_hash = HashString(topo_hash, robot_name);
    topo_hash = HashString(topo_hash, base_link);

    for (int bid = 0; bid < B; ++bid) {
        const std::string& lk_name = link_order[static_cast<std::size_t>(bid)];
        const PhysicsLinkSnapshot& lk_snap = *links_by_name[lk_name];
        local_body_id[lk_name] = bid;
        const std::string canon_lk = robot_name + "::" + lk_name;
        body_names.push_back(canon_lk);
        link_index_str[canon_lk] = bid;

        planning::Vec3 lk_p = planning::Vec3::Zero();
        planning::Mat3 lk_r = planning::Mat3::Identity();
        if (!ExtractRigidTransform(lk_snap.global_transform, lk_p, lk_r)) {
            if (error != nullptr) {
                *error = "Invalid non-rigid transform on link '" + lk_name + "'.";
            }
            return false;
        }

        if (lk_snap.role != PhysicsLinkRole::VirtualRoot) {
            for (const PhysicsShapeSnapshot& sh : lk_snap.collision_shapes) {
                if (sh.disabled) continue;
                planning::Vec3 sp = planning::Vec3::Zero();
                planning::Mat3 sr = planning::Mat3::Identity();
                if (!ExtractRigidTransform(sh.global_transform, sp, sr)) {
                    if (error != nullptr) {
                        *error = "Invalid non-rigid shape transform on '" + sh.scene_path + "'.";
                    }
                    return false;
                }
                const planning::Vec3 lp = lk_r.transpose() * (sp - lk_p);
                const planning::Mat3 lr = lk_r.transpose() * sr;
                const planning::Frame lf = FrameFromRT(lp, lr);
                const std::uint64_t gh = HashShapeGeometry(sh);
                robot_shapes.push_back({&sh, lk_name, bid, lf, gh});
                topo_hash = HashCombine(topo_hash, gh);
                topo_hash = HashCombine(topo_hash, static_cast<std::uint64_t>(sh.collision_layer));
                topo_hash = HashCombine(topo_hash, static_cast<std::uint64_t>(sh.collision_mask));
                for (int k = 0; k < 12; ++k) topo_hash = HashDouble(topo_hash, lf.data()[k]);
            }
        }

        if (lk_name == base_link) {
            parent[static_cast<std::size_t>(bid)] = -1;
            kinds[static_cast<std::size_t>(bid)] = 4;
            joint_idx[static_cast<std::size_t>(bid)] = -1;
            a_rot[static_cast<std::size_t>(bid)] = planning::Mat3::Identity();
            a_pos[static_cast<std::size_t>(bid)] = planning::Vec3::Zero();
            b_rot[static_cast<std::size_t>(bid)] = planning::Mat3::Identity();
            b_pos[static_cast<std::size_t>(bid)] = planning::Vec3::Zero();
            axes[static_cast<std::size_t>(bid)] = planning::Vec3::UnitX();
            link_depth[static_cast<std::size_t>(bid)] = 0;
        } else {
            const std::size_t ji = parent_joint_index.at(lk_name);
            const PhysicsJointSnapshot& j = robot_snap.joints[ji];
            if (j.joint_type < 0 || j.joint_type > 3) {
                if (error != nullptr) {
                    *error = "Unsupported joint type on '" + j.name + "'.";
                }
                return false;
            }
            const int p_bid = local_body_id.at(j.parent_link);
            parent[static_cast<std::size_t>(bid)] = p_bid;
            kinds[static_cast<std::size_t>(bid)] = j.joint_type;
            joint_idx[static_cast<std::size_t>(bid)] =
                    (j.joint_type == 0) ? -1 : movable_idx_map.at(j.name);
            link_depth[static_cast<std::size_t>(bid)] = link_depth[static_cast<std::size_t>(p_bid)] + 1;

            const Eigen::Matrix4d p_tf = links_by_name[j.parent_link]->global_transform.matrix().template cast<double>();
            const Eigen::Matrix4d j_tf = j.global_transform.matrix().template cast<double>();
            const Eigen::Matrix4d c_tf = lk_snap.global_transform.matrix().template cast<double>();

            planning::Vec3 axis = j.axis.template cast<double>();
            const double axis_norm = axis.norm();
            if (!axis.allFinite() || axis_norm < 1e-10) {
                if (error != nullptr) {
                    *error = "Invalid joint axis on '" + j.name + "'.";
                }
                return false;
            }
            axis /= axis_norm;

            const Eigen::Matrix4d pj = p_tf.inverse() * j_tf;
            Eigen::Matrix4d jc = j_tf.inverse() * c_tf;

            // If Joint3D is currently in Motion mode with non-zero joint_position, un-bake the
            // active joint motion from child_link so PlanningWorld's q=0 stays at true zero angle.
            if (j.joint_type != 0 && ji < robot_bind.joint_ids.size()) {
                if (const auto* joint_node = Object::PointerCastTo<Joint3D>(
                            ObjectDB::GetInstance(robot_bind.joint_ids[ji]));
                    joint_node != nullptr && joint_node->IsMotionModeEnabled()) {
                    const double q_cur = static_cast<double>(joint_node->GetJointPosition());
                    if (std::abs(q_cur) > 1e-12) {
                        jc = MakeJointMotionMatrix(j.joint_type, axis, -q_cur) * jc;
                    }
                }
            }

            a_rot[static_cast<std::size_t>(bid)] = pj.topLeftCorner<3, 3>();
            a_pos[static_cast<std::size_t>(bid)] = pj.topRightCorner<3, 1>();
            b_rot[static_cast<std::size_t>(bid)] = jc.topLeftCorner<3, 3>();
            b_pos[static_cast<std::size_t>(bid)] = jc.topRightCorner<3, 1>();
            axes[static_cast<std::size_t>(bid)] = axis;
        }

        topo_hash = HashString(topo_hash, canon_lk);
        topo_hash = HashCombine(topo_hash, static_cast<std::uint64_t>(parent[static_cast<std::size_t>(bid)] + 2));
        topo_hash = HashCombine(topo_hash, static_cast<std::uint64_t>(kinds[static_cast<std::size_t>(bid)]));
        topo_hash = HashCombine(topo_hash, static_cast<std::uint64_t>(joint_idx[static_cast<std::size_t>(bid)] + 2));
        for (int k = 0; k < 9; ++k) {
            topo_hash = HashDouble(topo_hash, a_rot[static_cast<std::size_t>(bid)].data()[k]);
            topo_hash = HashDouble(topo_hash, b_rot[static_cast<std::size_t>(bid)].data()[k]);
        }
        for (int k = 0; k < 3; ++k) {
            topo_hash = HashDouble(topo_hash, a_pos[static_cast<std::size_t>(bid)](k));
            topo_hash = HashDouble(topo_hash, b_pos[static_cast<std::size_t>(bid)](k));
            topo_hash = HashDouble(topo_hash, axes[static_cast<std::size_t>(bid)](k));
        }
    }

    for (std::size_t j = 0; j < joint_names_canon.size(); ++j) {
        topo_hash = HashString(topo_hash, joint_names_canon[j]);
        topo_hash = HashDouble(topo_hash, lower_list[j]);
        topo_hash = HashDouble(topo_hash, upper_list[j]);
        topo_hash = HashDouble(topo_hash, vel_list[j]);
        topo_hash = HashDouble(topo_hash, acc_list[j]);
    }
    for (const auto& pair : robot_snap.collision_exclusions) {
        topo_hash = HashString(topo_hash, pair[0]);
        topo_hash = HashString(topo_hash, pair[1]);
    }
    for (const auto& pair : extra_ignored_pairs) {
        topo_hash = HashString(topo_hash, pair.first);
        topo_hash = HashString(topo_hash, pair.second);
    }

    // Choose default end-effector link: deepest link driven by a movable joint, descending along
    // any unique fixed-joint chain (e.g., panda_link7 -> panda_link8 -> panda_hand, stopping before twin fingers).
    int default_ee = B - 1;
    {
        int best_movable_bid = -1;
        int best_depth = -1;
        for (int bid = 0; bid < B; ++bid) {
            if (joint_idx[static_cast<std::size_t>(bid)] >= 0 &&
                link_depth[static_cast<std::size_t>(bid)] >= best_depth) {
                best_depth = link_depth[static_cast<std::size_t>(bid)];
                best_movable_bid = bid;
            }
        }
        if (best_movable_bid >= 0) {
            int cur = best_movable_bid;
            while (true) {
                std::vector<int> fixed_children;
                for (int bid = 0; bid < B; ++bid) {
                    if (parent[static_cast<std::size_t>(bid)] == cur &&
                        kinds[static_cast<std::size_t>(bid)] == 0) {
                        fixed_children.push_back(bid);
                    }
                }
                if (fixed_children.size() == 1) {
                    cur = fixed_children.front();
                } else {
                    break;
                }
            }
            default_ee = cur;
        }
    }

    // Extract static world obstacles (matching world.py:_extract_world_obstacles).
    std::vector<IncomingObstacle> incoming_obstacles;
    std::unordered_map<std::string, int> node_key_counts;
    std::unordered_map<std::string, int> stable_key_counts;
    std::unordered_map<std::string, int> path_key_counts;

    auto push_shape_obstacle = [&](const PhysicsShapeSnapshot& sh,
                                   const std::string& owner_robot,
                                   const std::string& owner_link) -> bool {
        planning::Vec3 sp = planning::Vec3::Zero();
        planning::Mat3 sr = planning::Mat3::Identity();
        if (!ExtractRigidTransform(sh.global_transform, sp, sr)) {
            if (error != nullptr) {
                *error = "Invalid non-rigid transform on obstacle '" + sh.scene_path + "'.";
            }
            return false;
        }
        const std::string raw_name = !sh.scene_path.empty() ? sh.scene_path : sh.name;
        const std::string default_name = owner_robot.empty()
                                                 ? raw_name
                                                 : (owner_robot + "/" + owner_link + "/" + raw_name);
        const std::string link_key = owner_robot + "::" + owner_link;

        std::string node_key;
        if (auto it = path_to_node_id.find(sh.scene_path); it != path_to_node_id.end() && it->second != 0) {
            node_key = MakeUniqueKey(node_key_counts, "node:" + std::to_string(it->second));
        }
        std::string stable_key;
        if (sh.stable_id != 0) {
            stable_key = MakeUniqueKey(stable_key_counts, "shape:" + std::to_string(sh.stable_id));
        }
        const std::string path_key = MakeUniqueKey(path_key_counts, "path:" + default_name);
        const std::string sync_key = !node_key.empty() ? node_key
                                                       : (!stable_key.empty() ? stable_key : path_key);

        IncomingObstacle inc;
        inc.sync_key = sync_key;
        inc.node_key = node_key;
        inc.stable_key = stable_key;
        inc.path_key = path_key;
        inc.default_name = default_name;
        inc.link_key = link_key;
        inc.world_frame = FrameFromRT(sp, sr);
        inc.layer = static_cast<int>(sh.collision_layer);
        inc.mask = static_cast<int>(sh.collision_mask);
        inc.geom_hash = HashShapeGeometry(sh);
        const PhysicsShapeSnapshot sh_copy = sh;
        inc.build_shape = [sh_copy](const std::string& label) {
            return BuildNativeShape(sh_copy, label);
        };
        incoming_obstacles.push_back(std::move(inc));
        return true;
    };

    for (std::size_t ri = 0; ri < compiled.snapshot.robots.size(); ++ri) {
        const PhysicsRobotSnapshot& r = compiled.snapshot.robots[ri];
        for (const PhysicsLinkSnapshot& lk : r.links) {
            if (lk.role == PhysicsLinkRole::VirtualRoot) continue;
            const std::string lk_canon = r.name + "::" + lk.name;
            if (static_cast<int>(ri) == robot_idx && link_index_str.contains(lk_canon)) {
                continue;
            }
            for (const PhysicsShapeSnapshot& sh : lk.collision_shapes) {
                if (sh.disabled) continue;
                if (!push_shape_obstacle(sh, r.name, lk.name)) return false;
            }
        }
    }
    for (const PhysicsShapeSnapshot& sh : compiled.snapshot.loose_collision_shapes) {
        if (sh.disabled) continue;
        const std::string owner_link = !sh.scene_path.empty() ? sh.scene_path : sh.name;
        if (!push_shape_obstacle(sh, "", owner_link)) return false;
    }
    for (const PhysicsTerrainSnapshot& terrain : compiled.snapshot.terrains) {
        if (!terrain.heightfields.empty()) {
            if (error != nullptr) {
                *error = "Planning collision does not support terrain heightfields.";
            }
            return false;
        }
        std::uint64_t t_nid = 0;
        if (auto it = path_to_node_id.find(terrain.scene_path); it != path_to_node_id.end()) {
            t_nid = it->second;
        }
        const std::string t_path = !terrain.scene_path.empty() ? terrain.scene_path : terrain.name;
        const std::string link_key = "::" + terrain.name;

        for (std::size_t idx = 0; idx < terrain.boxes.size(); ++idx) {
            const PhysicsTerrainBoxSnapshot& box = terrain.boxes[idx];
            planning::Vec3 sp = planning::Vec3::Zero();
            planning::Mat3 sr = planning::Mat3::Identity();
            if (!ExtractRigidTransform(box.global_transform, sp, sr)) {
                if (error != nullptr) *error = "Invalid terrain box transform on '" + terrain.name + "'.";
                return false;
            }
            IncomingObstacle inc;
            if (t_nid != 0) {
                inc.node_key = MakeUniqueKey(node_key_counts, "tbox_node:" + std::to_string(t_nid) + ":" + std::to_string(idx));
            }
            if (terrain.stable_id != 0) {
                inc.stable_key = MakeUniqueKey(stable_key_counts, "tbox:" + std::to_string(terrain.stable_id) + ":" + std::to_string(idx));
            }
            inc.path_key = MakeUniqueKey(path_key_counts, "tbox_path:" + t_path + ":" + std::to_string(idx));
            inc.sync_key = !inc.node_key.empty() ? inc.node_key
                                                 : (!inc.stable_key.empty() ? inc.stable_key : inc.path_key);
            inc.default_name = t_path + "#box_" + std::to_string(idx);
            inc.link_key = link_key;
            inc.world_frame = FrameFromRT(sp, sr);
            inc.layer = static_cast<int>(terrain.collision_layer);
            inc.mask = static_cast<int>(terrain.collision_mask);
            inc.geom_hash = HashTerrainBoxGeometry(box);
            const planning::Vec3 sz = box.size.template cast<double>();
            inc.build_shape = [sz](const std::string& label) {
                return planning::make_box_shape(label, sz);
            };
            incoming_obstacles.push_back(std::move(inc));
        }

        for (std::size_t idx = 0; idx < terrain.mesh_patches.size(); ++idx) {
            const PhysicsTerrainMeshPatchSnapshot& patch = terrain.mesh_patches[idx];
            planning::Vec3 sp = planning::Vec3::Zero();
            planning::Mat3 sr = planning::Mat3::Identity();
            if (!ExtractRigidTransform(patch.global_transform, sp, sr)) {
                if (error != nullptr) *error = "Invalid terrain mesh transform on '" + terrain.name + "'.";
                return false;
            }
            IncomingObstacle inc;
            if (t_nid != 0) {
                inc.node_key = MakeUniqueKey(node_key_counts, "tmesh_node:" + std::to_string(t_nid) + ":" + std::to_string(idx));
            }
            if (terrain.stable_id != 0) {
                inc.stable_key = MakeUniqueKey(stable_key_counts, "tmesh:" + std::to_string(terrain.stable_id) + ":" + std::to_string(idx));
            }
            inc.path_key = MakeUniqueKey(path_key_counts, "tmesh_path:" + t_path + ":" + std::to_string(idx));
            inc.sync_key = !inc.node_key.empty() ? inc.node_key
                                                 : (!inc.stable_key.empty() ? inc.stable_key : inc.path_key);
            inc.default_name = t_path + "#mesh_" + std::to_string(idx);
            inc.link_key = link_key;
            inc.world_frame = FrameFromRT(sp, sr);
            inc.layer = static_cast<int>(terrain.collision_layer);
            inc.mask = static_cast<int>(terrain.collision_mask);
            inc.geom_hash = HashTerrainMeshGeometry(patch);
            std::vector<planning::Vec3> verts;
            verts.reserve(patch.vertices.size());
            for (const Vector3& v : patch.vertices) verts.push_back(v.template cast<double>());
            std::vector<uint32_t> idxs = patch.indices;
            inc.build_shape = [verts = std::move(verts), idxs = std::move(idxs)](const std::string& label) {
                return planning::make_mesh_shape(label, verts, idxs);
            };
            incoming_obstacles.push_back(std::move(inc));
        }
    }

    auto alloc_unique_shape_name = [&](const std::string& base) -> std::string {
        const int occ = used_shape_names_[base]++;
        return occ == 0 ? base : (base + "#" + std::to_string(occ));
    };

    PlanningSceneSyncStats stats;
    const bool need_rebuild = (world_ == nullptr) || (robot_topology_hash_ != topo_hash);

    try {
        if (need_rebuild) {
            auto new_world = std::make_unique<planning::PlanningWorld>();
            std::vector<planning::KinematicsModel::Branch> branches;
            branches.push_back(planning::KinematicsModel::Branch{robot_name, 0, root_pos, root_rot});

            Eigen::VectorXd lo_vec = Eigen::Map<const Eigen::VectorXd>(lower_list.data(), static_cast<Eigen::Index>(lower_list.size()));
            Eigen::VectorXd hi_vec = Eigen::Map<const Eigen::VectorXd>(upper_list.data(), static_cast<Eigen::Index>(upper_list.size()));
            Eigen::VectorXd vel_vec = Eigen::Map<const Eigen::VectorXd>(vel_list.data(), static_cast<Eigen::Index>(vel_list.size()));
            Eigen::VectorXd acc_vec = Eigen::Map<const Eigen::VectorXd>(acc_list.data(), static_cast<Eigen::Index>(acc_list.size()));
            Eigen::VectorXd init_vec = Eigen::Map<const Eigen::VectorXd>(init_list.data(), static_cast<Eigen::Index>(init_list.size()));

            new_world->init_kinematics(std::move(parent), std::move(kinds), std::move(joint_idx),
                                       std::move(a_rot), std::move(a_pos),
                                       std::move(b_rot), std::move(b_pos),
                                       std::move(axes), std::move(branches),
                                       body_names, joint_names_canon,
                                       lo_vec, hi_vec, vel_vec, acc_vec, init_vec, link_index_str);

            for (const auto& [la, lb] : adjacent_link_pairs) {
                new_world->set_auto_collision_disabled(la, lb);
            }
            for (const auto& pair : robot_snap.collision_exclusions) {
                new_world->set_auto_collision_disabled(robot_name + "::" + pair[0],
                                                       robot_name + "::" + pair[1]);
            }
            for (const auto& [a, b] : extra_ignored_pairs) {
                const std::string ca = (a.find("::") == std::string::npos) ? (robot_name + "::" + a) : a;
                const std::string cb = (b.find("::") == std::string::npos) ? (robot_name + "::" + b) : b;
                new_world->set_auto_collision_disabled(ca, cb);
            }

            used_shape_names_.clear();
            for (const RobotShapeEntry& rse : robot_shapes) {
                const std::string raw = !rse.snap->scene_path.empty() ? rse.snap->scene_path : rse.snap->name;
                const std::string full_name = alloc_unique_shape_name(robot_name + "/" + rse.link_name + "/" + raw);
                planning::WorldShape ws;
                ws.name = full_name;
                ws.link_key = robot_name + "::" + rse.link_name;
                ws.body_id = rse.body_id;
                ws.local_frame = rse.local_frame;
                ws.layer = static_cast<int>(rse.snap->collision_layer);
                ws.mask = static_cast<int>(rse.snap->collision_mask);
                ws.shape = BuildNativeShape(*rse.snap, full_name);
                new_world->add_shape(std::move(ws));
            }

            std::unordered_map<std::string, TrackedObstacle> new_tracked;
            for (const IncomingObstacle& obs : incoming_obstacles) {
                const std::string full_name = alloc_unique_shape_name(obs.default_name);
                planning::WorldShape ws;
                ws.name = full_name;
                ws.link_key = obs.link_key;
                ws.body_id = -1;
                ws.local_frame = obs.world_frame;
                ws.layer = obs.layer;
                ws.mask = obs.mask;
                ws.shape = obs.build_shape(full_name);
                new_world->add_shape(std::move(ws));
                new_tracked[obs.sync_key] = TrackedObstacle{
                        obs.sync_key, obs.node_key, obs.stable_key, obs.path_key,
                        full_name, obs.link_key, obs.world_frame, obs.layer, obs.mask, obs.geom_hash};
            }

            world_ = std::move(new_world);
            ++world_generation_;
            tracked_obstacles_ = std::move(new_tracked);
            robot_id_ = target_robot_id;
            robot_name_ = robot_name;
            base_link_name_ = base_link;
            default_ee_link_id_ = default_ee;
            robot_topology_hash_ = topo_hash;
            body_names_ = std::move(body_names);
            link_names_ = std::move(link_order);
            joint_names_ = std::move(joint_names_short);
            link_name_to_body_id_ = std::move(local_body_id);
            movable_joint_ids_ = std::move(movable_joint_ids);
            initial_q_ = init_vec;
            last_base_pos_ = root_pos;
            last_base_rot_ = root_rot;
            stats.rebuilt_world = true;
            stats.world_generation = world_generation_;
            stats.added_obstacles = tracked_obstacles_.size();
            CollectRobotVisualTemplates(target_robot);
        } else {
            // Incremental sync: match existing obstacles by runtime ObjectID -> stable_id -> path_key.
            planning::SyncBatch batch;
            if ((root_pos - last_base_pos_).cwiseAbs().maxCoeff() > 1e-7 ||
                (root_rot - last_base_rot_).cwiseAbs().maxCoeff() > 1e-7) {
                batch.base_updates.push_back(planning::SyncBaseUpdate{0, root_pos, root_rot});
                last_base_pos_ = root_pos;
                last_base_rot_ = root_rot;
            }

            std::unordered_map<std::string, std::string> by_node, by_stable, by_path;
            for (const auto& [k, tr] : tracked_obstacles_) {
                if (!tr.node_key.empty()) by_node[tr.node_key] = k;
                if (!tr.stable_key.empty()) by_stable[tr.stable_key] = k;
                if (!tr.path_key.empty()) by_path[tr.path_key] = k;
            }

            std::unordered_set<std::string> matched_old_keys;
            std::unordered_map<std::string, TrackedObstacle> next_tracked;

            for (const IncomingObstacle& inc : incoming_obstacles) {
                std::string old_key;
                if (!inc.node_key.empty()) {
                    if (auto it = by_node.find(inc.node_key);
                        it != by_node.end() && !matched_old_keys.contains(it->second)) {
                        old_key = it->second;
                    }
                }
                if (old_key.empty() && !inc.stable_key.empty()) {
                    if (auto it = by_stable.find(inc.stable_key);
                        it != by_stable.end() && !matched_old_keys.contains(it->second)) {
                        old_key = it->second;
                    }
                }
                if (old_key.empty() && !inc.path_key.empty()) {
                    if (auto it = by_path.find(inc.path_key);
                        it != by_path.end() && !matched_old_keys.contains(it->second)) {
                        old_key = it->second;
                    }
                }

                if (!old_key.empty()) {
                    matched_old_keys.insert(old_key);
                    const TrackedObstacle& prev = tracked_obstacles_.at(old_key);
                    const bool geom_changed = (prev.geom_hash != inc.geom_hash);
                    const bool meta_changed = (prev.layer != inc.layer || prev.mask != inc.mask ||
                                               prev.link_key != inc.link_key);
                    const bool pose_changed = !FrameApproxEqual(prev.world_frame, inc.world_frame);

                    if (geom_changed || meta_changed) {
                        planning::SyncShapeReplacement rep;
                        rep.name = prev.shape_name;
                        rep.link_key = inc.link_key;
                        rep.world_frame = inc.world_frame;
                        rep.layer = inc.layer;
                        rep.mask = inc.mask;
                        rep.shape = geom_changed ? inc.build_shape(prev.shape_name) : nullptr;
                        batch.replace_shapes.push_back(std::move(rep));
                    } else if (pose_changed) {
                        batch.pose_updates.push_back(planning::SyncPoseUpdate{prev.shape_name, inc.world_frame});
                    }
                    next_tracked[inc.sync_key] = TrackedObstacle{
                            inc.sync_key, inc.node_key, inc.stable_key, inc.path_key,
                            prev.shape_name, inc.link_key, inc.world_frame, inc.layer, inc.mask, inc.geom_hash};
                } else {
                    const std::string full_name = alloc_unique_shape_name(inc.default_name);
                    planning::WorldShape ws;
                    ws.name = full_name;
                    ws.link_key = inc.link_key;
                    ws.body_id = -1;
                    ws.local_frame = inc.world_frame;
                    ws.layer = inc.layer;
                    ws.mask = inc.mask;
                    ws.shape = inc.build_shape(full_name);
                    ws.update_fixed_bounds();
                    batch.add_shapes.push_back(std::move(ws));
                    next_tracked[inc.sync_key] = TrackedObstacle{
                            inc.sync_key, inc.node_key, inc.stable_key, inc.path_key,
                            full_name, inc.link_key, inc.world_frame, inc.layer, inc.mask, inc.geom_hash};
                }
            }

            for (const auto& [k, tr] : tracked_obstacles_) {
                if (!matched_old_keys.contains(k)) {
                    batch.remove_names.push_back(tr.shape_name);
                }
            }

            const bool any_change = !batch.base_updates.empty() || !batch.pose_updates.empty() ||
                                    !batch.remove_names.empty() || !batch.replace_shapes.empty() ||
                                    !batch.add_shapes.empty();
            if (any_change) {
                const planning::SyncResultNative res = world_->apply_sync_batch(batch);
                stats.base_updates = static_cast<std::size_t>(res.robot_bases_updated);
                stats.pose_updates = static_cast<std::size_t>(res.poses_updated);
                stats.added_obstacles = static_cast<std::size_t>(res.shapes_added);
                stats.replaced_obstacles = static_cast<std::size_t>(res.shapes_replaced);
                stats.removed_obstacles = static_cast<std::size_t>(res.shapes_removed);
            }
            tracked_obstacles_ = std::move(next_tracked);
            // Refresh visual templates only if empty (mesh changes are tracked by mesh_revision).
            if (visual_templates_.empty()) {
                CollectRobotVisualTemplates(target_robot);
            }
        }
    } catch (const std::exception& ex) {
        if (error != nullptr) {
            *error = ex.what();
        }
        return false;
    }

    current_q_ = Eigen::Map<const Eigen::VectorXd>(cur_list.data(), static_cast<Eigen::Index>(cur_list.size()));
    stats.world_generation = world_generation_;
    stats.world_version = world_->version();
    last_sync_stats_ = stats;
    return true;
#endif
}

bool PlanningSceneBridge::ComputeLinkWorldTransform(const Eigen::VectorXd& q,
                                                    int link_id,
                                                    Affine3* out_transform) const {
#if !defined(GOBOT_HAS_PLANNING)
    (void)q;
    (void)link_id;
    (void)out_transform;
    return false;
#else
    if (world_ == nullptr || out_transform == nullptr) {
        return false;
    }
    const auto snap = world_->snapshot();
    if (!snap || link_id < 0 || link_id >= snap->kin.num_bodies() ||
        q.size() != snap->kin.num_joints()) {
        return false;
    }
    const int B = snap->kin.num_bodies();
    std::vector<planning::Vec3> bpos(static_cast<std::size_t>(B));
    std::vector<planning::Mat3> brot(static_cast<std::size_t>(B));
    snap->kin.fk_single(q.data(), bpos.data(), brot.data());

    Affine3 tf = Affine3::Identity();
    tf.linear() = brot[static_cast<std::size_t>(link_id)].template cast<RealType>();
    tf.translation() = bpos[static_cast<std::size_t>(link_id)].template cast<RealType>();
    *out_transform = tf;
    return true;
#endif
}

#if defined(GOBOT_HAS_PLANNING)
std::shared_ptr<const planning::WorldSnapshot> PlanningSceneBridge::CaptureSnapshot() const {
    return world_ != nullptr ? world_->snapshot() : nullptr;
}

std::vector<VisualMeshRenderItem> PlanningSceneBridge::BuildGhostOverlayMeshes(
        const planning::WorldSnapshot& snapshot,
        const Eigen::VectorXd& q,
        const Color& tint_color,
        const Color& emissive_color) const {
    std::vector<VisualMeshRenderItem> result;
    const int B = snapshot.kin.num_bodies();
    if (visual_templates_.empty() || B <= 0 || q.size() != snapshot.kin.num_joints()) {
        return result;
    }

    std::vector<planning::Vec3> bpos(static_cast<std::size_t>(B));
    std::vector<planning::Mat3> brot(static_cast<std::size_t>(B));
    snapshot.kin.fk_single(q.data(), bpos.data(), brot.data());

    result.reserve(visual_templates_.size());
    for (const LinkVisualTemplate& tpl : visual_templates_) {
        if (tpl.body_id < 0 || tpl.body_id >= B) continue;
        Affine3 link_world = Affine3::Identity();
        link_world.linear() = brot[static_cast<std::size_t>(tpl.body_id)].template cast<RealType>();
        link_world.translation() = bpos[static_cast<std::size_t>(tpl.body_id)].template cast<RealType>();

        VisualMeshRenderItem item = tpl.item_template;
        const Affine3 mesh_world = link_world * tpl.link_to_mesh;
        item.model = mesh_world.matrix();
        if (const MeshSurfaceData* surf = item.GetSurface(); surf != nullptr) {
            item.world_bounds = surf->bounds.Transformed(item.model);
        } else {
            item.world_bounds = {};
        }

        // Override material to clean translucent tinted PBR material (no textures, no shadow).
        item.material = RenderMaterialSnapshot{};
        item.material.albedo = tint_color;
        item.material.emissive = emissive_color;
        item.material.metallic = 0.05;
        item.material.roughness = 0.45;
        item.material.alpha_mode = AlphaMode::Blend;
        item.material.double_sided = false;
        item.visible_in_rgb = true;
        item.cast_shadow = false;
        item.instance_id = 0;
        item.semantic_id = 0;
        result.push_back(std::move(item));
    }
    return result;
}

std::vector<LinkSphereOverlay> PlanningSceneBridge::CollectLinkSpheres(
        const planning::WorldSnapshot& snapshot,
        const Eigen::VectorXd& q) const {
    std::vector<LinkSphereOverlay> spheres;
    const int B = snapshot.kin.num_bodies();
    if (B <= 0 || q.size() != snapshot.kin.num_joints()) {
        return spheres;
    }

    std::vector<planning::Vec3> bpos(static_cast<std::size_t>(B));
    std::vector<planning::Mat3> brot(static_cast<std::size_t>(B));
    snapshot.kin.fk_single(q.data(), bpos.data(), brot.data());

    spheres.reserve(snapshot.link_spheres.size());
    for (const auto& sp : snapshot.link_spheres) {
        if (sp.body_id < 0 || sp.body_id >= B) continue;
        const planning::Vec3 center = brot[sp.body_id] * sp.local_center + bpos[sp.body_id];
        spheres.push_back(LinkSphereOverlay{center.template cast<RealType>(), static_cast<float>(sp.radius), sp.body_id});
    }
    return spheres;
}
#endif

std::vector<LinkSphereOverlay> PlanningSceneBridge::CollectLinkSpheres(const Eigen::VectorXd& q) const {
#if defined(GOBOT_HAS_PLANNING)
    if (world_ == nullptr) {
        return {};
    }
    const auto snap = world_->snapshot();
    if (!snap) {
        return {};
    }
    return CollectLinkSpheres(*snap, q);
#else
    (void)q;
    return {};
#endif
}

std::vector<VisualMeshRenderItem> PlanningSceneBridge::BuildGhostOverlayMeshes(
        const Eigen::VectorXd& q,
        const Color& tint_color,
        const Color& emissive_color) const {
#if defined(GOBOT_HAS_PLANNING)
    if (world_ == nullptr) {
        return {};
    }
    const auto snap = world_->snapshot();
    if (!snap) {
        return {};
    }
    return BuildGhostOverlayMeshes(*snap, q, tint_color, emissive_color);
#else
    (void)q;
    (void)tint_color;
    (void)emissive_color;
    return {};
#endif
}

} // namespace gobot
