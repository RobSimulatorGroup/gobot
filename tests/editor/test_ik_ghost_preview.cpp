#include <gtest/gtest.h>

#if defined(GOBOT_HAS_PLANNING)
#include "planning_cartesian.hpp"
#endif

#include "gobot/core/config/project_setting.hpp"
#include "gobot/core/io/resource_loader.hpp"
#include "gobot/editor/editor_viewport_renderer.hpp"
#include "gobot/editor/ik_ghost_session.hpp"
#include "gobot/editor/planning_scene_bridge.hpp"
#include "gobot/main/engine_context.hpp"
#include "gobot/rendering/headless_render_context.hpp"
#include "gobot/rendering/render_server.hpp"
#include "gobot/scene/camera_3d.hpp"
#include "gobot/scene/collision_shape_3d.hpp"
#include "gobot/scene/joint_3d.hpp"
#include "gobot/scene/link_3d.hpp"
#include "gobot/scene/mesh_instance_3d.hpp"
#include "gobot/scene/node_3d.hpp"
#include "gobot/scene/resources/box_shape_3d.hpp"
#include "gobot/scene/resources/material.hpp"
#include "gobot/scene/resources/packed_scene.hpp"
#include "gobot/scene/resources/primitive_mesh.hpp"
#include "gobot/scene/rigid_body_3d.hpp"
#include "gobot/scene/robot_3d.hpp"
#include "gobot/scene/scene_initializer.hpp"
#include "gobot/simulation/simulation_server.hpp"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace gobot {
namespace {

std::size_t CountNodesRecursive(const Node* node) {
    if (node == nullptr) return 0;
    std::size_t count = 1;
    for (std::size_t i = 0; i < node->GetChildCount(); ++i) {
        count += CountNodesRecursive(node->GetChild(static_cast<int>(i)));
    }
    return count;
}

Robot3D* FindFirstRobot(Node* node) {
    if (node == nullptr) return nullptr;
    if (auto* robot = Object::PointerCastTo<Robot3D>(node)) return robot;
    for (std::size_t i = 0; i < node->GetChildCount(); ++i) {
        if (auto* r = FindFirstRobot(node->GetChild(static_cast<int>(i)))) return r;
    }
    return nullptr;
}

void EnsureSceneInitialized() {
    static const bool initialized = [] {
        SceneInitializer::Init();
        return true;
    }();
    (void)initialized;
}

Node* LoadPandaWorld(ProjectSettings* settings) {
    EnsureSceneInitialized();
    const std::filesystem::path proj_dir = std::filesystem::path(GOBOT_ROOT_DIR) / "examples" / "panda_trajopt";
    if (!std::filesystem::exists(proj_dir / "world.jscn")) {
        return nullptr;
    }
    if (!settings->SetProjectPath(proj_dir.string())) {
        return nullptr;
    }
    ProjectSettings::Scope scope(settings);
    Ref<Resource> res = ResourceLoader::Load("res://world.jscn", "PackedScene",
                                             ResourceFormatLoader::CacheMode::Ignore);
    Ref<PackedScene> packed = dynamic_pointer_cast<PackedScene>(res);
    if (!packed.IsValid()) {
        return nullptr;
    }
    return packed->Instantiate();
}

} // namespace

TEST(IKGhostPreview, ReportsAvailabilityMatchingBuildConfig) {
#if defined(GOBOT_HAS_PLANNING)
    EXPECT_TRUE(PlanningSceneBridge::IsPlanningAvailable());
    EXPECT_TRUE(InteractiveIKGhostSession::IsAvailable());
#else
    EXPECT_FALSE(PlanningSceneBridge::IsPlanningAvailable());
    EXPECT_FALSE(InteractiveIKGhostSession::IsAvailable());
    InteractiveIKGhostSession session;
    session.SetEnabled(true);
    EXPECT_FALSE(session.IsEnabled());
    EXPECT_EQ(session.GetDiagnostics().status, IKGhostStatus::Unavailable);
#endif
}

#if defined(GOBOT_HAS_PLANNING)

TEST(IKGhostPreview, SceneBridgeObjectIdIdentityAndIncrementalSync) {
    ProjectSettings settings;
    Node* root = LoadPandaWorld(&settings);
    ASSERT_NE(root, nullptr) << "Failed to load examples/panda_trajopt/world.jscn";
    Robot3D* panda = FindFirstRobot(root);
    ASSERT_NE(panda, nullptr);

    // Add a standalone obstacle box under root.
    auto* box_holder = Object::New<Node3D>();
    box_holder->SetName("DynamicBox");
    box_holder->SetPosition({0.85, 0.0, 0.5});
    auto* col = Object::New<CollisionShape3D>();
    col->SetName("BoxCollider");
    auto box_shape = MakeRef<BoxShape3D>();
    box_shape->SetSize({0.16, 0.16, 0.16});
    col->SetShape(box_shape);
    box_holder->AddChild(col);
    root->AddChild(box_holder);

    PlanningSceneBridge bridge;
    std::string error;
    ASSERT_TRUE(bridge.SyncFromScene(root, panda, "panda_link0",
                                     {{"panda_link0", "panda_link2"},
                                      {"panda_link4", "panda_link6"},
                                      {"panda_link7", "panda_hand"}},
                                     &error)) << error;
    EXPECT_TRUE(bridge.GetLastSyncStats().rebuilt_world);
    EXPECT_EQ(bridge.GetJointNames().size(), 7u);
    EXPECT_GE(bridge.GetDefaultEndEffectorLinkId(), 0);
    EXPECT_FALSE(bridge.GetVisualTemplates().empty());
    const std::uint64_t ver0 = bridge.GetWorldVersion();

    // 1. Rename AND move obstacle: runtime ObjectID preserves identity -> pose_updates only, no rebuild!
    box_holder->SetName("RenamedDynamicBox");
    box_holder->SetPosition({0.75, 0.10, 0.52});
    ASSERT_TRUE(bridge.SyncFromScene(root, panda, "panda_link0",
                                     {{"panda_link0", "panda_link2"},
                                      {"panda_link4", "panda_link6"},
                                      {"panda_link7", "panda_hand"}},
                                     &error)) << error;
    EXPECT_FALSE(bridge.GetLastSyncStats().rebuilt_world);
    // link_key changes because loose shape link_key uses scene_path; verify it did NOT rebuild world or add/remove.
    EXPECT_EQ(bridge.GetLastSyncStats().added_obstacles, 0u);
    EXPECT_EQ(bridge.GetLastSyncStats().removed_obstacles, 0u);
    EXPECT_GT(bridge.GetWorldVersion(), ver0);

    // Pure move without rename -> pure pose_update (zero shape replacement).
    const std::uint64_t ver1 = bridge.GetWorldVersion();
    box_holder->SetPosition({0.72, 0.12, 0.55});
    ASSERT_TRUE(bridge.SyncFromScene(root, panda, "panda_link0",
                                     {{"panda_link0", "panda_link2"},
                                      {"panda_link4", "panda_link6"},
                                      {"panda_link7", "panda_hand"}},
                                     &error)) << error;
    EXPECT_FALSE(bridge.GetLastSyncStats().rebuilt_world);
    EXPECT_EQ(bridge.GetLastSyncStats().pose_updates, 1u);
    EXPECT_EQ(bridge.GetLastSyncStats().replaced_obstacles, 0u);
    EXPECT_GT(bridge.GetWorldVersion(), ver1);

    // 2. Geometry resize -> replaced_obstacles == 1, no world rebuild.
    box_shape->SetSize({0.24, 0.24, 0.24});
    ASSERT_TRUE(bridge.SyncFromScene(root, panda, "panda_link0",
                                     {{"panda_link0", "panda_link2"},
                                      {"panda_link4", "panda_link6"},
                                      {"panda_link7", "panda_hand"}},
                                     &error)) << error;
    EXPECT_FALSE(bridge.GetLastSyncStats().rebuilt_world);
    EXPECT_EQ(bridge.GetLastSyncStats().replaced_obstacles, 1u);

    // 3. Remove obstacle -> removed_obstacles == 1, no world rebuild.
    root->RemoveChild(box_holder);
    Object::Delete(box_holder);
    ASSERT_TRUE(bridge.SyncFromScene(root, panda, "panda_link0",
                                     {{"panda_link0", "panda_link2"},
                                      {"panda_link4", "panda_link6"},
                                      {"panda_link7", "panda_hand"}},
                                     &error)) << error;
    EXPECT_FALSE(bridge.GetLastSyncStats().rebuilt_world);
    EXPECT_EQ(bridge.GetLastSyncStats().removed_obstacles, 1u);

    Object::Delete(root);
}

TEST(IKGhostPreview, PandaDragIntoBoxAndOutEndToEnd) {
    ProjectSettings settings;
    SimulationServer sim(PhysicsBackendType::Null, false);
    EngineContext ctx(&settings, &sim);

    Node* root = LoadPandaWorld(&settings);
    ASSERT_NE(root, nullptr) << "Failed to load examples/panda_trajopt/world.jscn";
    ctx.SetSceneRoot(root, true, "res://world.jscn");

    Robot3D* panda = FindFirstRobot(root);
    ASSERT_NE(panda, nullptr);

    // Place a solid test box inside Panda's reachable workspace where we will drag the hand into.
    auto* test_box = Object::New<Node3D>();
    test_box->SetName("AcceptanceSolidBox");
    test_box->SetPosition({0.42, 0.22, 0.50});
    auto* box_col = Object::New<CollisionShape3D>();
    box_col->SetName("AcceptanceBoxShape");
    auto shape = MakeRef<BoxShape3D>();
    shape->SetSize({0.18, 0.18, 0.18});
    box_col->SetShape(shape);
    test_box->AddChild(box_col);
    root->AddChild(test_box);

    const std::size_t initial_node_count = CountNodesRecursive(root);
    const std::size_t initial_cmd_version = ctx.GetSceneCommandVersion();
    const bool initial_dirty = ctx.IsSceneDirty();

    InteractiveIKGhostSession session;
    // Note: relies on Robot3D.collision_exclusions authored directly in examples/panda_trajopt/panda.jscn.
    IKGhostPreviewConfig cfg;
    cfg.margin = 0.002;
    cfg.world_margin = 0.010;
    cfg.pos_tol = 2e-3;
    cfg.rot_tol = 2e-2;
    session.SetPreviewConfig(cfg);
    session.SetEnabled(true);

    // Initial sync & reset target.
    session.Update(root, panda, ctx.GetSceneEpoch(), 1, false);
    ASSERT_TRUE(session.WaitForIdle());
    session.Update(root, panda, ctx.GetSceneEpoch(), 1, false);
    ASSERT_TRUE(session.HasActiveTarget());
    ASSERT_TRUE(session.SetEndEffectorLink("panda_hand"));
    ASSERT_TRUE(session.WaitForIdle());
    session.Update(root, panda, ctx.GetSceneEpoch(), 1, false);

    // Orientation pointing downward (matches Panda plan.py DOWN).
    Matrix3 down_rot = Matrix3::Identity();
    down_rot << -1.0, 0.0, 0.0,
                 0.0, 1.0, 0.0,
                 0.0, 0.0, -1.0;

    auto drag_and_wait = [&](const Vector3& pos) {
        Affine3 pose = Affine3::Identity();
        pose.linear() = down_rot;
        pose.translation() = pos;
        session.SetTargetPose(pose);
        EXPECT_TRUE(session.WaitForIdle());
        session.Update(root, panda, ctx.GetSceneEpoch(), 1, false);
        return session.GetDiagnostics();
    };

    // 1. Free space -> Green (Ok)
    const IKGhostDiagnostics free_diag = drag_and_wait({0.42, -0.22, 0.56});
    EXPECT_EQ(free_diag.status, IKGhostStatus::Ok) << free_diag.status_message;
    EXPECT_FALSE(session.GetOverlayMeshes().empty());
    EXPECT_GT(session.GetOverlayMeshes().front().material.albedo.green(), 0.7f);
    // Verify world_bounds were updated to match the ghost's new model transform.
    for (const VisualMeshRenderItem& item : session.GetOverlayMeshes()) {
        ASSERT_NE(item.GetSurface(), nullptr);
        const AABB expected_bounds = item.GetSurface()->bounds.Transformed(item.model);
        EXPECT_TRUE(item.world_bounds.GetMin().isApprox(expected_bounds.GetMin(), 1e-4));
        EXPECT_TRUE(item.world_bounds.GetMax().isApprox(expected_bounds.GetMax(), 1e-4));
        EXPECT_EQ(item.material.alpha_mode, AlphaMode::Blend);
        EXPECT_FALSE(item.cast_shadow);
    }

    // 2. Drag hand INTO AcceptanceSolidBox -> Orange (Collision), preserving colliding pose!
    const IKGhostDiagnostics col_diag = drag_and_wait({0.42, 0.22, 0.50});
    EXPECT_EQ(col_diag.status, IKGhostStatus::Collision) << col_diag.status_message;
    EXPECT_LT(col_diag.pos_err, 5e-3);
    EXPECT_LT(col_diag.min_distance, cfg.world_margin);
    EXPECT_FALSE(col_diag.collision_pair.empty());
    ASSERT_FALSE(session.GetOverlayMeshes().empty());
    // Orange tint: high red, medium green.
    EXPECT_GT(session.GetOverlayMeshes().front().material.albedo.red(), 0.8f);
    EXPECT_GT(session.GetOverlayMeshes().front().material.albedo.green(), 0.4f);

    // 3. Drag back OUT to free space -> Green (Ok) again!
    const IKGhostDiagnostics out_diag = drag_and_wait({0.42, -0.22, 0.56});
    EXPECT_EQ(out_diag.status, IKGhostStatus::Ok) << out_diag.status_message;
    EXPECT_GT(session.GetOverlayMeshes().front().material.albedo.green(), 0.7f);

    // 4. Drag far outside workspace -> Red (PoseNotSolved)
    const IKGhostDiagnostics far_diag = drag_and_wait({2.50, 0.0, 2.00});
    EXPECT_EQ(far_diag.status, IKGhostStatus::PoseNotSolved);
    EXPECT_GT(far_diag.pos_err, 0.5);
    ASSERT_FALSE(session.GetOverlayMeshes().empty());
    EXPECT_GT(session.GetOverlayMeshes().front().material.albedo.red(), 0.8f);
    EXPECT_LT(session.GetOverlayMeshes().front().material.albedo.green(), 0.35f);

    // 5. Measure interactive drag throughput & latency over 30 sequential poses.
    double max_main_update_ms = 0.0;
    for (int i = 0; i < 30; ++i) {
        const double frac = static_cast<double>(i) / 29.0;
        Affine3 pose = Affine3::Identity();
        pose.linear() = down_rot;
        pose.translation() = Vector3{
                static_cast<RealType>(0.40 + 0.08 * std::sin(frac * 3.14159)),
                static_cast<RealType>(-0.25 + 0.50 * frac),
                static_cast<RealType>(0.52)};
        const auto t0 = std::chrono::steady_clock::now();
        session.SetTargetPose(pose);
        session.Update(root, panda, ctx.GetSceneEpoch(), 1, false);
        const auto t1 = std::chrono::steady_clock::now();
        max_main_update_ms = std::max(
                max_main_update_ms,
                std::chrono::duration<double, std::milli>(t1 - t0).count());
        ASSERT_TRUE(session.WaitForIdle());
        session.Update(root, panda, ctx.GetSceneEpoch(), 1, false);
    }
    const IKGhostDiagnostics perf_diag = session.GetDiagnostics();
    std::cout << "[Panda IK Drag Telemetry] max_main_update_ms=" << max_main_update_ms
              << " last_solve_ms=" << perf_diag.last_solve_ms
              << " p95_input_to_display_ms=" << perf_diag.p95_input_to_display_ms << std::endl;
    EXPECT_LT(max_main_update_ms, 5.0);

    // 6. Zero scene-tree mutation & zero command-stack edits throughout dragging.
    EXPECT_EQ(CountNodesRecursive(root), initial_node_count);
    EXPECT_EQ(ctx.GetSceneCommandVersion(), initial_cmd_version);
    EXPECT_EQ(ctx.IsSceneDirty(), initial_dirty);
}

TEST(IKGhostPreview, AsyncStaleEpochAndPlayModeGuard) {
    ProjectSettings settings;
    Node* root = LoadPandaWorld(&settings);
    ASSERT_NE(root, nullptr);
    Robot3D* panda = FindFirstRobot(root);
    ASSERT_NE(panda, nullptr);

    InteractiveIKGhostSession session;
    session.SetEnabled(true);
    session.Update(root, panda, 1, 1, false);
    ASSERT_TRUE(session.WaitForIdle());
    session.Update(root, panda, 1, 1, false);
    ASSERT_FALSE(session.GetOverlayMeshes().empty());

    // Enter Play Mode -> overlay meshes immediately cleared and stale results suppressed.
    session.Update(root, panda, 1, 1, true);
    EXPECT_TRUE(session.GetOverlayMeshes().empty());

    // Switch scene_epoch -> previous epoch results cannot flash back.
    session.Update(root, panda, 2, 1, false);
    ASSERT_TRUE(session.WaitForIdle());
    session.Update(root, panda, 2, 1, false);
    EXPECT_EQ(session.GetDiagnostics().scene_epoch, 2u);

    Object::Delete(root);
}

TEST(IKGhostPreview, DragProgressAndContextInvalidation) {
    ProjectSettings settings;
    Node* root = LoadPandaWorld(&settings);
    ASSERT_NE(root, nullptr);
    Robot3D* panda = FindFirstRobot(root);
    ASSERT_NE(panda, nullptr);

    InteractiveIKGhostSession session;
    session.SetEnabled(true);
    session.Update(root, panda, 1, 1, false);
    ASSERT_TRUE(session.WaitForIdle());
    session.Update(root, panda, 1, 1, false);
    ASSERT_TRUE(session.HasActiveTarget());
    const Affine3 initial_pose = session.GetTargetPose();

    // Leave a finished result unread while submitting a different goal.
    Affine3 moved_pose = initial_pose;
    moved_pose.translation().x() += 0.02f;
    session.SetTargetPose(moved_pose);
    ASSERT_TRUE(session.WaitForIdle());
    const auto unread = session.GetDiagnostics();
    session.SetTargetPose(initial_pose);
    EXPECT_EQ(session.GetDiagnostics().status, unread.status);
    EXPECT_EQ(session.GetDiagnostics().collision_pair, unread.collision_pair);
    session.Update(root, panda, 1, 1, false);
    // A newer goal may already be ready. In either case the displayed pose must advance.
    EXPECT_GE(session.GetDiagnostics().displayed_seq, unread.target_seq);
    ASSERT_TRUE(session.WaitForIdle());
    session.Update(root, panda, 1, 1, false);
    EXPECT_EQ(session.GetDiagnostics().displayed_seq, session.GetDiagnostics().target_seq);

    // Changing collision margins invalidates even an unread solve for the same goal and world.
    session.SetTargetPose(moved_pose);
    ASSERT_TRUE(session.WaitForIdle());
    auto config = session.GetPreviewConfig();
    config.world_margin += 0.01;
    session.SetPreviewConfig(config);
    const auto first_new_policy = session.GetDiagnostics().target_seq;
    EXPECT_EQ(session.GetDiagnostics().status, IKGhostStatus::Solving);
    EXPECT_TRUE(session.GetOverlayMeshes().empty());
    session.SetTargetPose(initial_pose);
    session.Update(root, panda, 1, 1, false);
    const auto shown = session.GetDiagnostics().displayed_seq;
    EXPECT_TRUE(shown == 0 || shown >= first_new_policy);
    ASSERT_TRUE(session.WaitForIdle());
    session.Update(root, panda, 1, 1, false);
    EXPECT_EQ(session.GetDiagnostics().displayed_seq, session.GetDiagnostics().target_seq);

    // Synchronize an obstacle edit before consuming the old world's result.
    session.SetTargetPose(moved_pose);
    ASSERT_TRUE(session.WaitForIdle());
    const auto old_world = session.GetDiagnostics();
    auto* obstacle = Object::New<CollisionShape3D>();
    obstacle->SetShape(MakeRef<BoxShape3D>());
    root->AddChild(obstacle);
    session.Update(root, panda, 1, 2, false);
    EXPECT_GT(session.GetDiagnostics().world_version, old_world.world_version);
    EXPECT_NE(session.GetDiagnostics().displayed_seq, old_world.target_seq);

    session.SetEnabled(false);
    session.SetEnabled(true);
    session.Update(root, panda, 1, 2, false);
    ASSERT_TRUE(session.HasActiveTarget());
    EXPECT_TRUE(session.GetTargetPose().matrix().isApprox(initial_pose.matrix()));
    ASSERT_TRUE(session.WaitForIdle());
    session.Update(root, panda, 1, 2, false);
    EXPECT_FALSE(session.GetOverlayMeshes().empty());

    session.SetTargetPose(moved_pose);
    session.Update(root, nullptr, 2, 1, false);
    ASSERT_TRUE(session.WaitForIdle());
    session.Update(root, nullptr, 2, 1, false);
    EXPECT_FALSE(session.HasActiveTarget());
    EXPECT_TRUE(session.GetOverlayMeshes().empty());
    EXPECT_EQ(session.GetDiagnostics().displayed_seq, 0u);

    session.Update(root, panda, 3, 1, false);
    session.Update(root, panda, 3, 1, true);
    EXPECT_FALSE(session.HasActiveTarget());
    session.Update(root, panda, 3, 1, false);
    ASSERT_TRUE(session.WaitForIdle());
    session.Update(root, panda, 3, 1, false);
    EXPECT_TRUE(session.HasActiveTarget());
    EXPECT_FALSE(session.GetOverlayMeshes().empty());
    EXPECT_EQ(session.GetDiagnostics().displayed_seq, session.GetDiagnostics().target_seq);
    Object::Delete(root);
}

TEST(IKGhostPreview, PandaBridgeMatchesPythonPlanningWorldParity) {
    ProjectSettings settings;
    Node* root = LoadPandaWorld(&settings);
    ASSERT_NE(root, nullptr);
    Robot3D* panda = FindFirstRobot(root);
    ASSERT_NE(panda, nullptr);

    PlanningSceneBridge bridge;
    std::string error;
    ASSERT_TRUE(bridge.SyncFromScene(root, panda, "panda_link0",
                                     {{"panda_link0", "panda_link2"},
                                      {"panda_link4", "panda_link6"},
                                      {"panda_link7", "panda_hand"}},
                                     &error)) << error;

    const auto snap = bridge.CaptureSnapshot();
    ASSERT_NE(snap, nullptr);
    // Reference values from Python PlanningWorld.from_scene() on examples/panda_trajopt/world.jscn:
    EXPECT_EQ(snap->pairs.size(), 182u);
    EXPECT_EQ(snap->self_pair_count, 39);

    const int hand_id = bridge.FindLinkId("panda_hand");
    ASSERT_GE(hand_id, 0);

    auto eval_min_dist = [&](const Eigen::VectorXd& q, bool verify_orig) {
        std::vector<double> frames(snap->shapes.size() * 12);
        snap->compute_frames(q.data(), 1, frames.data());
        double min_d = std::numeric_limits<double>::infinity();
        for (const auto& pr : snap->pairs) {
            double out[10] = {};
            planning::eval_coal_distance(*snap->shapes[static_cast<std::size_t>(pr.shape_a)].shape,
                                         *snap->shapes[static_cast<std::size_t>(pr.shape_b)].shape,
                                         frames.data() + pr.shape_a * 12,
                                         frames.data() + pr.shape_b * 12,
                                         verify_orig,
                                         std::numeric_limits<double>::infinity(),
                                         out);
            min_d = std::min(min_d, out[0]);
        }
        return min_d;
    };

    // 1. Seed pose touching UpperJaw (matches Python scipy ConvexHull min_cvx to < 1e-5 m).
    Eigen::VectorXd q_seed(7);
    q_seed << 0.0, -0.5, 0.0, -2.0, 0.0, 1.5, -2.3;
    Affine3 hand_tf = Affine3::Identity();
    ASSERT_TRUE(bridge.ComputeLinkWorldTransform(q_seed, hand_id, &hand_tf));
    EXPECT_NEAR(static_cast<double>(hand_tf.translation().x()), 0.386104199, 1e-6);
    EXPECT_NEAR(static_cast<double>(hand_tf.translation().y()), 3.8e-08, 1e-6);
    EXPECT_NEAR(static_cast<double>(hand_tf.translation().z()), 0.652325158, 1e-6);
    EXPECT_NEAR(static_cast<double>(hand_tf.linear()(0, 0)), -0.998421293, 1e-5);
    EXPECT_NEAR(static_cast<double>(hand_tf.linear()(1, 1)), 0.998421506, 1e-5);
    EXPECT_NEAR(static_cast<double>(hand_tf.linear()(2, 2)), -0.999999854, 1e-5);
    EXPECT_NEAR(eval_min_dist(q_seed, false), -0.025174842, 1e-5);

    // 2. Collision-free pose (matches Python convex & original mesh distance to < 1e-5 m).
    Eigen::VectorXd q_free(7);
    q_free << 0.5, -0.5, 0.0, -2.0, 0.0, 1.5, -2.3;
    ASSERT_TRUE(bridge.ComputeLinkWorldTransform(q_free, hand_id, &hand_tf));
    EXPECT_NEAR(static_cast<double>(hand_tf.translation().x()), 0.338838294, 1e-6);
    EXPECT_NEAR(static_cast<double>(hand_tf.translation().y()), 0.185108247, 1e-6);
    EXPECT_NEAR(static_cast<double>(hand_tf.translation().z()), 0.652325158, 1e-6);
    EXPECT_NEAR(eval_min_dist(q_free, false), 0.039795654, 1e-5);
    EXPECT_NEAR(eval_min_dist(q_free, true), 0.039795654, 1e-5);

    // Two wrist meshes intersect UpperJaw here. link7 has the larger convex
    // penetration estimate. Neither original mesh query provides a penetration depth.
    Eigen::VectorXd q_inside(7);
    q_inside << 0.0, -0.3, 0.0, -2.0, 0.0, 1.5, -2.3;
    ASSERT_TRUE(bridge.ComputeLinkWorldTransform(q_inside, hand_id, &hand_tf));
    planning::PreviewIKConfigNative preview;
    const auto collision = planning::solve_ik_preview_on_snapshot(
            *snap, hand_id, planning::Vec3::Zero(), planning::Mat3::Identity(),
            hand_tf.translation().cast<double>(), hand_tf.linear().cast<double>(),
            q_inside, std::vector<uint8_t>(7, 0), preview);
    EXPECT_EQ(collision.status, planning::PreviewIKStatus::Collision);
    EXPECT_NE(collision.collision_pair.find("panda/panda_link7/"), std::string::npos);
    EXPECT_NE(collision.collision_pair.find("/UpperJaw/Collision"), std::string::npos);
    EXPECT_EQ(collision.min_distance, -std::numeric_limits<double>::infinity());
    EXPECT_FALSE(collision.has_witness);

    Object::Delete(root);
}

#endif // GOBOT_HAS_PLANNING

TEST(OverlayMeshPass, DepthOcclusionWorksInRasterFxaaAndLuisa) {
    if (std::getenv("GOBOT_RUN_RENDER_GPU_TEST") == nullptr) {
        GTEST_SKIP() << "Set GOBOT_RUN_RENDER_GPU_TEST=1 on an EGL runner.";
    }

    EnsureSceneInitialized();
    HeadlessRenderContext context;
    ASSERT_TRUE(context.Initialize()) << context.GetLastError();
    auto* rs = RenderServer::GetInstance();
    ASSERT_NE(rs, nullptr);

    // Scene: solid red opaque box centered at (0, 0, 1.5), well above the z=0 editor grid.
    auto* root = Object::New<Node3D>();
    root->SetName("OcclusionScene");
    auto* solid_box = Object::New<MeshInstance3D>();
    solid_box->SetName("SolidBox");
    solid_box->SetPosition({0.0, 0.0, 1.5});
    auto box_mesh = MakeRef<BoxMesh>();
    box_mesh->SetSize({1.0, 1.0, 1.0});
    auto red_mat = MakeRef<PBRMaterial3D>();
    red_mat->SetAlbedo(Color(0.95f, 0.08f, 0.08f, 1.0f));
    red_mat->SetEmissive(Color(0.8f, 0.04f, 0.04f, 1.0f));
    solid_box->SetMesh(box_mesh);
    solid_box->SetMaterial(red_mat);
    root->AddChild(solid_box);

    // Camera at (0, -3.5, 1.5) looking at (0, 0, 1.5) along +Y.
    Camera3D camera;
    camera.SetAspect(1.0);
    camera.SetFovy(45.0);
    camera.SetViewMatrix({0.0, -3.5, 1.5}, {0.0, 0.0, 1.5}, {0.0, 0.0, 1.0});

    // Build a green translucent overlay box item (no node added to SceneTree!).
    auto make_ghost_item = [&](RealType y_pos) {
        VisualMeshRenderItem item;
        item.mesh_id = box_mesh->GetInstanceId();
        item.mesh_revision = box_mesh->GetGeometryRevision();
        item.mesh_topology_revision = box_mesh->GetTopologyRevision();
        item.surface_index = 0;
        item.surfaces = box_mesh->GetSurfaceData();
        Affine3 tf = Affine3::Identity();
        tf.translation() = Vector3{0.0, y_pos, 1.5};
        tf.scale(Vector3{0.7, 0.7, 0.7});
        item.model = tf.matrix();
        item.world_bounds = item.GetSurface()->bounds.Transformed(item.model);
        item.material.albedo = Color(0.08f, 0.96f, 0.18f, 0.70f);
        item.material.emissive = Color(0.04f, 0.85f, 0.12f, 1.0f);
        item.material.alpha_mode = AlphaMode::Blend;
        item.visible_in_rgb = true;
        item.cast_shadow = false;
        return item;
    };

    const RID vp = rs->ViewportCreate();
    rs->ViewportSetSize(vp, 64, 64);

    auto center_pixel = [&](const std::vector<std::uint8_t>& rgb) {
        const std::size_t idx = (32u * 64u + 32u) * 3u;
        return std::array<int, 3>{rgb[idx], rgb[idx + 1], rgb[idx + 2]};
    };

    auto test_mode = [&](const char* label, const SceneRendererSettings& settings) {
        rs->SetSceneRendererSettings(settings);
        EditorViewportRenderer evr;

        // 1. Baseline: solid red box only.
        evr.Render(vp, root, &camera, nullptr, nullptr, {}, false, {});
        const auto st = rs->GetSceneRendererStats();
        const auto base_px = center_pixel(rs->ReadViewportRgbPixels(vp));
        std::cout << "[" << label << "] visible=" << st.visible_items << " draws=" << st.draw_calls
                  << " base=(" << base_px[0] << "," << base_px[1] << "," << base_px[2] << ")";
        EXPECT_GT(base_px[0], base_px[1] + 40) << label << ": baseline center should be red";

        // 2. Ghost BEHIND solid box (y = +1.2, farther from camera at y = -3.5) -> occluded by scene depth!
        const std::vector<VisualMeshRenderItem> behind_ghost = {make_ghost_item(1.2)};
        evr.Render(vp, root, &camera, nullptr, nullptr, {}, false, behind_ghost);
        const auto occluded_px = center_pixel(rs->ReadViewportRgbPixels(vp));
        std::cout << " behind=(" << occluded_px[0] << "," << occluded_px[1] << "," << occluded_px[2] << ")";
        EXPECT_NEAR(occluded_px[0], base_px[0], 6) << label << ": occluded ghost must not overwrite solid box R";
        EXPECT_NEAR(occluded_px[1], base_px[1], 6) << label << ": occluded ghost must not overwrite solid box G";

        // 3. Ghost IN FRONT of solid box (y = -1.2, closer to camera) -> visible green blend!
        const std::vector<VisualMeshRenderItem> front_ghost = {make_ghost_item(-1.2)};
        evr.Render(vp, root, &camera, nullptr, nullptr, {}, false, front_ghost);
        const auto visible_px = center_pixel(rs->ReadViewportRgbPixels(vp));
        std::cout << " front=(" << visible_px[0] << "," << visible_px[1] << "," << visible_px[2] << ")" << std::endl;
        EXPECT_GT(visible_px[1], base_px[1] + 40) << label << ": front ghost must blend green over solid box";
    };

    // Mode A: OpenGL Raster (FXAA Disabled)
    {
        SceneRendererSettings s;
        s.mode = SceneRendererMode::Raster;
        s.raster.anti_aliasing = RasterAntiAliasingMode::Disabled;
        test_mode("Raster-NoFXAA", s);
    }
    // Mode B: OpenGL Raster + FXAA
    {
        SceneRendererSettings s;
        s.mode = SceneRendererMode::Raster;
        s.raster.anti_aliasing = RasterAntiAliasingMode::Fxaa;
        test_mode("Raster-FXAA", s);
    }
    // Mode C: Luisa Realtime Ray Tracing (if module available on this runner)
    const auto luisa_caps = rs->GetSceneRendererCapabilities();
    if (luisa_caps.realtime) {
        SceneRendererSettings s;
        s.mode = SceneRendererMode::RealtimeRayTracing;
        s.samples_per_frame = 2;
        test_mode("Luisa-RealtimeRT", s);
    } else {
        std::cout << "[Luisa-RealtimeRT] skipped: " << luisa_caps.status << std::endl;
    }

    rs->Free(vp);
    Object::Delete(root);
}

} // namespace gobot
