/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "gobot/core/color.hpp"
#include "gobot/core/math/geometry.hpp"
#include "gobot/core/object_id.hpp"
#include "gobot/rendering/scene_render_items.hpp"
#include "gobot_export.h"

#include <Eigen/Core>

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(GOBOT_HAS_PLANNING)
#include "planning_world.hpp"
#endif

namespace gobot {

class Node;
class Robot3D;

struct PlanningSceneSyncStats {
    bool rebuilt_world = false;
    std::uint64_t world_version = 0;
    std::size_t base_updates = 0;
    std::size_t pose_updates = 0;
    std::size_t added_obstacles = 0;
    std::size_t replaced_obstacles = 0;
    std::size_t removed_obstacles = 0;
};

struct LinkVisualTemplate {
    int body_id = -1;
    std::string link_name;
    Affine3 link_to_mesh = Affine3::Identity();
    VisualMeshRenderItem item_template;
};

class GOBOT_EXPORT PlanningSceneBridge {
public:
    PlanningSceneBridge();
    ~PlanningSceneBridge();

    PlanningSceneBridge(const PlanningSceneBridge&) = delete;
    PlanningSceneBridge& operator=(const PlanningSceneBridge&) = delete;

    [[nodiscard]] static bool IsPlanningAvailable() noexcept;

    void Clear();

    // Synchronize the planning world from the live scene root for the given Robot3D.
    // Rebuilds PlanningWorld when robot identity, kinematic topology, or robot link collision shapes change;
    // otherwise applies incremental SyncBatch updates keyed by runtime ObjectID -> stable_id -> scene_path.
    bool SyncFromScene(const Node* scene_root,
                       const Robot3D* target_robot,
                       const std::string& preferred_base_link = "",
                       const std::vector<std::pair<std::string, std::string>>& extra_ignored_pairs = {},
                       std::string* error = nullptr);

    [[nodiscard]] bool IsReady() const noexcept;
    [[nodiscard]] ObjectID GetRobotId() const noexcept { return robot_id_; }
    [[nodiscard]] const std::string& GetRobotName() const noexcept { return robot_name_; }
    [[nodiscard]] const std::string& GetBaseLinkName() const noexcept { return base_link_name_; }
    [[nodiscard]] int GetDefaultEndEffectorLinkId() const noexcept { return default_ee_link_id_; }
    [[nodiscard]] int FindLinkId(const std::string& link_name) const;
    [[nodiscard]] const std::vector<std::string>& GetBodyNames() const noexcept { return body_names_; }
    [[nodiscard]] const std::vector<std::string>& GetLinkNames() const noexcept { return link_names_; }
    [[nodiscard]] const std::vector<std::string>& GetJointNames() const noexcept { return joint_names_; }
    [[nodiscard]] const Eigen::VectorXd& GetCurrentJointPositions() const noexcept { return current_q_; }
    [[nodiscard]] const Eigen::VectorXd& GetInitialJointPositions() const noexcept { return initial_q_; }
    [[nodiscard]] const PlanningSceneSyncStats& GetLastSyncStats() const noexcept { return last_sync_stats_; }
    [[nodiscard]] std::uint64_t GetWorldVersion() const noexcept;
    [[nodiscard]] const std::vector<LinkVisualTemplate>& GetVisualTemplates() const noexcept { return visual_templates_; }

    // Build tinted translucent VisualMeshRenderItems for the ghost robot at configuration q.
    // Updates both item.model = T_world_link(q) * T_link_visual and item.world_bounds.
    [[nodiscard]] std::vector<VisualMeshRenderItem> BuildGhostOverlayMeshes(
            const Eigen::VectorXd& q,
            const Color& tint_color,
            const Color& emissive_color) const;

    // Evaluate link world transform at configuration q using the active world snapshot.
    bool ComputeLinkWorldTransform(const Eigen::VectorXd& q, int link_id, Affine3* out_transform) const;

#if defined(GOBOT_HAS_PLANNING)
    [[nodiscard]] std::shared_ptr<const planning::WorldSnapshot> CaptureSnapshot() const;
    [[nodiscard]] planning::PlanningWorld* GetPlanningWorld() noexcept { return world_.get(); }
    [[nodiscard]] const planning::PlanningWorld* GetPlanningWorld() const noexcept { return world_.get(); }

    // Build ghost overlay meshes directly from an immutable WorldSnapshot (safe across updates).
    [[nodiscard]] std::vector<VisualMeshRenderItem> BuildGhostOverlayMeshes(
            const planning::WorldSnapshot& snapshot,
            const Eigen::VectorXd& q,
            const Color& tint_color,
            const Color& emissive_color) const;
#endif

private:
    struct TrackedObstacle {
        std::string sync_key;
        std::string node_key;
        std::string stable_key;
        std::string path_key;
        std::string shape_name;
        std::string link_key;
        Eigen::Matrix<double, 3, 4, Eigen::RowMajor> world_frame =
                Eigen::Matrix<double, 3, 4, Eigen::RowMajor>::Zero();
        int layer = 1;
        int mask = 1;
        std::uint64_t geom_hash = 0;
    };

    void CollectRobotVisualTemplates(const Robot3D* target_robot);

    ObjectID robot_id_{};
    std::string robot_name_;
    std::string base_link_name_;
    int default_ee_link_id_ = -1;
    std::uint64_t robot_topology_hash_ = 0;
    std::vector<std::string> body_names_;
    std::vector<std::string> link_names_;
    std::vector<std::string> joint_names_;
    std::unordered_map<std::string, int> link_name_to_body_id_;
    std::vector<ObjectID> movable_joint_ids_;
    Eigen::VectorXd current_q_;
    Eigen::VectorXd initial_q_;
    Eigen::Vector3d last_base_pos_ = Eigen::Vector3d::Zero();
    Eigen::Matrix<double, 3, 3, Eigen::RowMajor> last_base_rot_ =
            Eigen::Matrix<double, 3, 3, Eigen::RowMajor>::Identity();
    std::unordered_map<std::string, TrackedObstacle> tracked_obstacles_;
    std::unordered_map<std::string, int> used_shape_names_;
    std::vector<LinkVisualTemplate> visual_templates_;
    PlanningSceneSyncStats last_sync_stats_;
    double default_acceleration_limit_ = 10.0;

#if defined(GOBOT_HAS_PLANNING)
    std::unique_ptr<planning::PlanningWorld> world_;
#endif
};

} // namespace gobot
