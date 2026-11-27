#include <gtest/gtest.h>

#include "gobot/core/config/project_setting.hpp"
#include "gobot/core/io/resource_loader.hpp"
#include "gobot/editor/drag_transaction_controller.hpp"
#include "gobot/editor/ik_ghost_session.hpp"
#include "gobot/editor/motion_planning_session.hpp"
#include "gobot/editor/planning_scene_bridge.hpp"
#include "gobot/scene/collision_shape_3d.hpp"
#include "gobot/scene/link_3d.hpp"
#include "gobot/scene/node_3d.hpp"
#include "gobot/scene/resources/box_shape_3d.hpp"
#include "gobot/scene/resources/packed_scene.hpp"
#include "gobot/scene/resources/sphere_shape_3d.hpp"
#include "gobot/scene/robot_3d.hpp"
#include "gobot/scene/scene_command.hpp"
#include "gobot/scene/scene_initializer.hpp"
#include "gobot/scene/terrain_3d.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <future>
#include <mutex>
#include <string>

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

Link3D* FindFirstLink(Node* node) {
    if (node == nullptr) return nullptr;
    if (auto* link = Object::PointerCastTo<Link3D>(node)) return link;
    for (std::size_t i = 0; i < node->GetChildCount(); ++i) {
        if (auto* l = FindFirstLink(node->GetChild(static_cast<int>(i)))) return l;
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

Eigen::VectorXd MakeSafePandaStart(const PlanningSceneBridge& bridge) {
    Eigen::VectorXd q = bridge.GetCurrentJointPositions();
    const double ready_vals[] = {0.0, -0.5, 0.0, -1.8, 0.0, 1.5, 0.785};
    for (int i = 0; i < 7 && i < q.size(); ++i) {
        q(i) = ready_vals[i];
    }
    return q;
}

} // namespace

TEST(MotionPlanningSession, ReportsAvailabilityMatchingBuildConfig) {
#if defined(GOBOT_HAS_PLANNING)
    EXPECT_TRUE(MotionPlanningSession::IsAvailable());
#else
    EXPECT_FALSE(MotionPlanningSession::IsAvailable());
    MotionPlanningSession session;
    EXPECT_FALSE(session.StartPlan());
    EXPECT_EQ(session.GetStatus(), MotionPlanningStatus::Unavailable);
#endif
}

#if defined(GOBOT_HAS_PLANNING)

TEST(MotionPlanningSession, SharedBridgeCoalescesSyncAndPlanStaleRePlanLifecycle) {
    ProjectSettings settings;
    Node* root = LoadPandaWorld(&settings);
    ASSERT_NE(root, nullptr);
    Robot3D* panda = FindFirstRobot(root);
    ASSERT_NE(panda, nullptr);

    // Add a movable obstacle box far from the arm initially.
    auto* obstacle = Object::New<Node3D>();
    obstacle->SetName("MovableObstacle");
    obstacle->SetPosition({1.2, 0.0, 0.5});
    auto* col = Object::New<CollisionShape3D>();
    col->SetName("ObstacleShape");
    auto box = MakeRef<BoxShape3D>();
    box->SetSize({0.14, 0.14, 0.14});
    col->SetShape(box);
    obstacle->AddChild(col);
    root->AddChild(obstacle);

    const std::size_t initial_node_count = CountNodesRecursive(root);

    PlanningSceneBridge shared_bridge;
    InteractiveIKGhostSession ghost(&shared_bridge);
    MotionPlanningSession planner(&shared_bridge);
    ghost.SetEnabled(true);

    // Multiple mutations in one frame before Update() must coalesce into a single SyncFromScene().
    SceneCommandStack cmd_stack;
    const Affine3 tf0 = obstacle->GetTransform();
    Affine3 tf1 = tf0;
    tf1.translation() = Vector3(1.22, 0.0, 0.5);
    Affine3 tf2 = tf0;
    tf2.translation() = Vector3(1.25, 0.0, 0.5);
    ASSERT_TRUE(cmd_stack.Execute(std::make_unique<SetNode3DTransformCommand>(obstacle->GetInstanceId(), tf1, false)));
    ASSERT_TRUE(cmd_stack.Execute(std::make_unique<SetNode3DTransformCommand>(obstacle->GetInstanceId(), tf2, false)));

    const std::uint64_t sync_before = shared_bridge.GetSyncCount();
    const std::uint64_t ver1 = cmd_stack.GetVersion();
    planner.Update(root, panda, /*scene_epoch=*/1, ver1, /*is_play_mode=*/false, 0.0);
    ghost.Update(root, panda, /*scene_epoch=*/1, ver1, /*is_play_mode=*/false);
    EXPECT_EQ(shared_bridge.GetSyncCount(), sync_before + 1)
        << "Shared bridge should coalesce multiple Do() calls and multiple consumers into one sync per frame";

    // Configure a collision-free start/goal pair on Panda.
    const Eigen::VectorXd start_q = MakeSafePandaStart(shared_bridge);
    ASSERT_GE(start_q.size(), 7);
    Eigen::VectorXd goal_q = start_q;
    goal_q(0) += 0.35;
    planner.SetStartJointPositions(start_q);
    planner.SetGoalJointPositions(goal_q);

    MotionPlanningConfig cfg = planner.GetConfig();
    cfg.strict_ccd = true;
    cfg.waypoints = 12;
    cfg.timeout_s = 10.0;
    planner.SetConfig(cfg);

    ASSERT_TRUE(planner.StartPlan());
    EXPECT_EQ(planner.GetStatus(), MotionPlanningStatus::Planning);
    ASSERT_TRUE(planner.WaitForIdle(std::chrono::seconds(10)));
    planner.Update(root, panda, 1, ver1, false, 0.0);

    const MotionPlanningDiagnostics ready_diag = planner.GetDiagnostics();
    EXPECT_EQ(ready_diag.status, MotionPlanningStatus::Ready) << ready_diag.status_message;
    EXPECT_EQ(ready_diag.strict_ccd_status, StrictCCDStatus::Passed);
    EXPECT_TRUE(planner.HasTrajectory());
    EXPECT_GT(ready_diag.trajectory_duration_s, 0.0);
    EXPECT_FALSE(planner.GetDebugOverlayData().ee_path_points.empty());
    EXPECT_FALSE(planner.GetDebugOverlayData().is_stale);
    EXPECT_TRUE(ready_diag.has_sampled_clearance);
    EXPECT_FALSE(ready_diag.sampled_shape_pair.empty());

    // Read-only preview emits overlay meshes without mutating SceneTree.
    ASSERT_TRUE(planner.StartPreview());
    EXPECT_FALSE(planner.GetPreviewOverlayMeshes().empty());
    planner.Update(root, panda, 1, ver1, false, 0.1);
    EXPECT_EQ(CountNodesRecursive(root), initial_node_count);

    // 1. Move obstacle -> existing trajectory becomes Stale automatically (keeps old EE path for orange display).
    Affine3 tf3 = tf2;
    tf3.translation() = Vector3(1.30, 0.15, 0.5);
    ASSERT_TRUE(cmd_stack.Execute(std::make_unique<SetNode3DTransformCommand>(obstacle->GetInstanceId(), tf3, false)));
    const std::uint64_t ver2 = cmd_stack.GetVersion();
    planner.Update(root, panda, 1, ver2, false, 0.0);
    EXPECT_EQ(planner.GetStatus(), MotionPlanningStatus::Stale);
    EXPECT_TRUE(planner.HasTrajectory());
    EXPECT_TRUE(planner.GetDebugOverlayData().is_stale);

    // 2. Re-Plan -> recovers to Ready on new snapshot.
    ASSERT_TRUE(planner.RePlan());
    ASSERT_TRUE(planner.WaitForIdle(std::chrono::seconds(10)));
    planner.Update(root, panda, 1, ver2, false, 0.0);
    EXPECT_EQ(planner.GetStatus(), MotionPlanningStatus::Ready);
    EXPECT_FALSE(planner.GetDebugOverlayData().is_stale);

    // A retained preview must change color even when a newer request owns the status banner.
    ASSERT_TRUE(planner.StartPlan());
    cfg.world_margin += 0.001;
    planner.SetConfig(cfg);
    EXPECT_TRUE(planner.GetDebugOverlayData().is_stale);
    planner.Stop();
    EXPECT_EQ(planner.GetStatus(), MotionPlanningStatus::Cancelled);
    EXPECT_TRUE(planner.StartPreview());
    EXPECT_TRUE(planner.GetDebugOverlayData().is_stale);
    ASSERT_TRUE(planner.RePlan());
    ASSERT_TRUE(planner.WaitForIdle());
    planner.Update(root, panda, 1, ver2, false);
    EXPECT_EQ(planner.GetStatus(), MotionPlanningStatus::Ready);

    // 3. Mutate solver parameter (margin) even with unchanged world -> Ready becomes Stale immediately.
    cfg.margin += 0.002;
    planner.SetConfig(cfg);
    EXPECT_EQ(planner.GetStatus(), MotionPlanningStatus::Stale);
    EXPECT_TRUE(planner.GetDebugOverlayData().is_stale);

    Object::Delete(root);
}

TEST(MotionPlanningSession, SameRobotTopologyRebuildWithIdenticalNumericVersionInvalidatesOldPlan) {
    ProjectSettings settings;
    Node* root = LoadPandaWorld(&settings);
    ASSERT_NE(root, nullptr);
    Robot3D* panda = FindFirstRobot(root);
    ASSERT_NE(panda, nullptr);
    Link3D* link = FindFirstLink(panda);
    ASSERT_NE(link, nullptr);

    PlanningSceneBridge shared_bridge;
    MotionPlanningSession planner(&shared_bridge);

    std::uint64_t change_ver = 1;
    planner.Update(root, panda, /*scene_epoch=*/1, change_ver, false, 0.0);
    ASSERT_TRUE(shared_bridge.IsReady());
    const std::uint64_t initial_world_ver = shared_bridge.GetWorldVersion();
    const std::uint64_t initial_world_gen = shared_bridge.GetWorldGeneration();

    const Eigen::VectorXd start_q = MakeSafePandaStart(shared_bridge);
    Eigen::VectorXd goal_q = start_q;
    goal_q(0) += 0.25;
    planner.SetStartJointPositions(start_q);
    planner.SetGoalJointPositions(goal_q);
    ASSERT_TRUE(planner.StartPlan());
    ASSERT_TRUE(planner.WaitForIdle(std::chrono::seconds(10)));
    planner.Update(root, panda, 1, change_ver, false, 0.0);
    ASSERT_EQ(planner.GetStatus(), MotionPlanningStatus::Ready) << planner.GetDiagnostics().status_message;

    // Mutate an existing robot link CollisionShape3D transform on the SAME Robot3D (keeping shape count
    // identical!). PlanningSceneBridge rebuilds a brand-new PlanningWorld whose numeric world_version
    // matches initial_world_ver identically, while world_generation advances. Session MUST NOT remain Ready!
    CollisionShape3D* link_col = nullptr;
    for (std::size_t i = 0; i < link->GetChildCount(); ++i) {
        if (auto* cs = Object::PointerCastTo<CollisionShape3D>(link->GetChild(static_cast<int>(i)))) {
            link_col = cs;
            break;
        }
    }
    ASSERT_NE(link_col, nullptr);
    link_col->SetPosition(link_col->GetPosition() + Vector3(0.0, 0.0, 0.002));

    ++change_ver;
    planner.Update(root, panda, 1, change_ver, false, 0.0);
    EXPECT_GT(shared_bridge.GetWorldGeneration(), initial_world_gen);
    // Confirm the rebuilt PlanningWorld has the EXACT same numeric version as before,
    // proving world_generation / snapshot pointer identity is essential.
    EXPECT_EQ(shared_bridge.GetWorldVersion(), initial_world_ver);
    EXPECT_NE(planner.GetStatus(), MotionPlanningStatus::Ready);
    EXPECT_FALSE(planner.HasTrajectory());
    EXPECT_TRUE(planner.GetPreviewOverlayMeshes().empty());

    Object::Delete(root);
}

TEST(MotionPlanningSession, StopImmediatelyDisqualifiesLateSuccessAndAllowsImmediateSecondPlan) {
    ProjectSettings settings;
    Node* root = LoadPandaWorld(&settings);
    ASSERT_NE(root, nullptr);
    Robot3D* panda = FindFirstRobot(root);
    ASSERT_NE(panda, nullptr);

    PlanningSceneBridge shared_bridge;
    MotionPlanningSession planner(&shared_bridge);

    const std::uint64_t change_ver = 1;
    planner.Update(root, panda, 1, change_ver, false, 0.0);
    const Eigen::VectorXd start_q = MakeSafePandaStart(shared_bridge);
    Eigen::VectorXd goal_q = start_q;
    goal_q(0) += 0.25;
    planner.SetStartJointPositions(start_q);
    planner.SetGoalJointPositions(goal_q);

    // Synchronization barrier holding Request #1 AFTER solve succeeds, before the UI polls its queued result.
    std::mutex hook_mu;
    std::condition_variable hook_cv;
    bool req1_reached_after = false;
    bool release_req1 = false;

    planner.SetTestWorkerHooks(
            {},
            [&](std::uint64_t seq, const std::atomic<bool>&) {
                if (seq == 1) {
                    std::unique_lock<std::mutex> lk(hook_mu);
                    req1_reached_after = true;
                    hook_cv.notify_all();
                    hook_cv.wait(lk, [&] { return release_req1; });
                }
            });

    // Launch Request #1 and wait until it has ALREADY produced an Ok plan inside worker.
    ASSERT_TRUE(planner.StartPlan());
    {
        std::unique_lock<std::mutex> lk(hook_mu);
        ASSERT_TRUE(hook_cv.wait_for(lk, std::chrono::seconds(10), [&] { return req1_reached_after; }));
    }

    // Click Stop on UI thread -> immediate Cancelled status + request #1 disqualified.
    planner.Stop();
    EXPECT_EQ(planner.GetStatus(), MotionPlanningStatus::Cancelled);

    // Release Request #1 so the worker can finish the request.
    {
        std::lock_guard<std::mutex> lk(hook_mu);
        release_req1 = true;
    }
    hook_cv.notify_all();
    ASSERT_TRUE(planner.WaitForIdle(std::chrono::seconds(10)));

    // Poll on UI thread -> late Ok result from Request #1 MUST NOT overwrite Cancelled back to Ready!
    planner.Update(root, panda, 1, change_ver, false, 0.0);
    EXPECT_EQ(planner.GetStatus(), MotionPlanningStatus::Cancelled);
    EXPECT_FALSE(planner.HasTrajectory());

    // Now test Stop() followed IMMEDIATELY by a 2nd Plan while the cancelled request #2 is still holding its Ok result.
    bool req2_reached_after = false;
    bool release_req2 = false;
    planner.SetTestWorkerHooks(
            {},
            [&](std::uint64_t seq, const std::atomic<bool>&) {
                if (seq == 2) {
                    std::unique_lock<std::mutex> lk(hook_mu);
                    req2_reached_after = true;
                    hook_cv.notify_all();
                    hook_cv.wait(lk, [&] { return release_req2; });
                }
            });

    ASSERT_TRUE(planner.StartPlan()); // seq == 2
    {
        std::unique_lock<std::mutex> lk(hook_mu);
        ASSERT_TRUE(hook_cv.wait_for(lk, std::chrono::seconds(10), [&] { return req2_reached_after; }));
    }

    planner.Stop();
    EXPECT_EQ(planner.GetStatus(), MotionPlanningStatus::Cancelled);

    // Immediately launch Request #3 (seq == 3) while Request #2 is still blocked in worker.
    goal_q(0) += 0.10;
    planner.SetGoalJointPositions(goal_q);
    ASSERT_TRUE(planner.StartPlan()); // seq == 3
    EXPECT_EQ(planner.GetStatus(), MotionPlanningStatus::Planning);

    // Release Request #2 -> its late Ok result lands before Request #3 finishes.
    {
        std::lock_guard<std::mutex> lk(hook_mu);
        release_req2 = true;
    }
    hook_cv.notify_all();
    ASSERT_TRUE(planner.WaitForIdle(std::chrono::seconds(10)));
    planner.Update(root, panda, 1, change_ver, false, 0.0);

    const MotionPlanningDiagnostics diag = planner.GetDiagnostics();
    EXPECT_EQ(diag.status, MotionPlanningStatus::Ready) << diag.status_message;
    EXPECT_EQ(diag.last_completed_request_seq, 3u)
        << "Late Ok result from stopped Request #2 must not be accepted in place of Request #3";

    Object::Delete(root);
}

TEST(MotionPlanningSession, UseIKGhostRequiresExactSolvedSequenceAndMatchingSnapshot) {
    ProjectSettings settings;
    Node* root = LoadPandaWorld(&settings);
    ASSERT_NE(root, nullptr);
    Robot3D* panda = FindFirstRobot(root);
    ASSERT_NE(panda, nullptr);

    PlanningSceneBridge shared_bridge;
    InteractiveIKGhostSession ghost(&shared_bridge);
    MotionPlanningSession planner(&shared_bridge);
    ghost.SetEnabled(true);

    const std::uint64_t change_ver = 1;
    MotionPlanningConfig pcfg = planner.GetConfig();
    pcfg.end_effector_link = "panda_hand";
    planner.SetConfig(pcfg);
    IKGhostPreviewConfig gcfg;
    gcfg.margin = 0.002;
    gcfg.world_margin = 0.010;
    gcfg.pos_tol = 2e-3;
    gcfg.rot_tol = 2e-2;
    ghost.SetPreviewConfig(gcfg);

    planner.Update(root, panda, 1, change_ver, false, 0.0);
    ghost.Update(root, panda, 1, change_ver, false);
    ASSERT_TRUE(ghost.WaitForIdle(std::chrono::seconds(5)));
    ghost.Update(root, panda, 1, change_ver, false);
    ASSERT_TRUE(ghost.SetEndEffectorLink("panda_hand"));
    ASSERT_TRUE(ghost.WaitForIdle(std::chrono::seconds(5)));
    ghost.Update(root, panda, 1, change_ver, false);

    // Submit a brand new reachable free-space pose without polling its completion yet -> displayed_seq < target_seq!
    Matrix3 down_rot = Matrix3::Identity();
    down_rot << -1.0, 0.0, 0.0,
                 0.0, 1.0, 0.0,
                 0.0, 0.0, -1.0;
    Affine3 new_pose = Affine3::Identity();
    new_pose.linear() = down_rot;
    new_pose.translation() = Vector3(0.42, -0.22, 0.56);
    ghost.SetTargetPose(new_pose);

    std::string reject_reason;
    EXPECT_FALSE(planner.CaptureGoalFromIKGhost(ghost, &reject_reason))
        << "Must reject Use IK Ghost while latest target_seq has not yet been displayed";
    EXPECT_FALSE(reject_reason.empty());

    // Wait for IK Ghost to finish & publish the exact target_seq -> now Use IK Ghost succeeds.
    ASSERT_TRUE(ghost.WaitForIdle(std::chrono::seconds(5)));
    ghost.Update(root, panda, 1, change_ver, false);
    ASSERT_EQ(ghost.GetDiagnostics().status, IKGhostStatus::Ok);
    ASSERT_EQ(ghost.GetDiagnostics().displayed_seq, ghost.GetDiagnostics().target_seq);

    EXPECT_TRUE(planner.CaptureGoalFromIKGhost(ghost, &reject_reason)) << reject_reason;
    EXPECT_TRUE(planner.HasGoalJointPositions());

    Object::Delete(root);
}

TEST(MotionPlanningSession, PublishesVerifiedPlanBeforeClearanceAndHonorsQueuedDeadline) {
    ProjectSettings settings;
    Node* root = LoadPandaWorld(&settings);
    ASSERT_NE(root, nullptr);
    Robot3D* robot = FindFirstRobot(root);
    PlanningSceneBridge bridge;
    MotionPlanningSession planner(&bridge);
    planner.Update(root, robot, 1, 1, false);
    auto q = MakeSafePandaStart(bridge);
    planner.SetStartJointPositions(q);
    q(0) += 0.2;
    planner.SetGoalJointPositions(q);

    std::promise<void> solved, resume;
    auto release = resume.get_future();
    planner.SetTestWorkerHooks({}, [&](std::uint64_t, const std::atomic<bool>&) {
        solved.set_value();
        release.wait();
    });
    ASSERT_TRUE(planner.StartPlan());
    const auto published = solved.get_future().wait_for(std::chrono::seconds(10));
    planner.Update(root, robot, 1, 1, false);
    EXPECT_EQ(planner.GetStatus(), MotionPlanningStatus::Ready);
    EXPECT_FALSE(planner.GetDiagnostics().has_sampled_clearance);
    resume.set_value();
    ASSERT_EQ(published, std::future_status::ready);
    ASSERT_TRUE(planner.WaitForIdle());
    planner.Update(root, robot, 1, 1, false);
    EXPECT_TRUE(planner.GetDiagnostics().has_sampled_clearance);

    auto cfg = planner.GetConfig();
    cfg.timeout_s = 0.1;
    cfg.strict_ccd = true;
    planner.SetConfig(cfg);
    planner.SetTestWorkerHooks([](std::uint64_t, const std::atomic<bool>&) {
        std::this_thread::sleep_for(std::chrono::milliseconds(120));
    }, {});
    ASSERT_TRUE(planner.StartPlan());
    ASSERT_TRUE(planner.WaitForIdle());
    planner.Update(root, robot, 1, 1, false);
    EXPECT_EQ(planner.GetStatus(), MotionPlanningStatus::TimedOut);
    EXPECT_EQ(planner.GetDiagnostics().strict_ccd_status, StrictCCDStatus::NotRun);
    EXPECT_TRUE(planner.GetDebugOverlayData().is_stale);
    Object::Delete(root);
}

TEST(MotionPlanningSession, RejectsEmptyGroupAndReleasesGoalOnSceneSwitch) {
    ProjectSettings settings;
    Node* root = LoadPandaWorld(&settings);
    ASSERT_NE(root, nullptr);
    Robot3D* robot = FindFirstRobot(root);
    PlanningSceneBridge bridge;
    MotionPlanningSession session(&bridge);
    session.Update(root, robot, 1, 1, false);
    const auto q = MakeSafePandaStart(bridge);
    session.SetStartJointPositions(q);
    Eigen::VectorXd goal = q;
    goal(0) += 0.35;
    session.SetGoalJointPositions(goal);
    auto cfg = session.GetConfig();
    cfg.use_all_joints = false;
    cfg.strict_ccd = true;
    session.SetConfig(cfg);
    EXPECT_FALSE(session.StartPlan());
    EXPECT_EQ(session.GetDiagnostics().strict_ccd_status, StrictCCDStatus::NotRun);
    cfg.selected_joint_names = {"missing_joint"};
    session.SetConfig(cfg);
    EXPECT_FALSE(session.RePlan());

    cfg.use_all_joints = true;
    session.SetConfig(cfg);
    goal(0) = 100.0;
    session.SetGoalJointPositions(goal);
    ASSERT_TRUE(session.StartPlan());
    ASSERT_TRUE(session.WaitForIdle());
    session.Update(root, robot, 1, 1, false);
    EXPECT_EQ(session.GetStatus(), MotionPlanningStatus::Failed);
    EXPECT_EQ(session.GetDiagnostics().strict_ccd_status, StrictCCDStatus::NotRun);

    session.Update(root, robot, 2, 1, false);
    EXPECT_FALSE(session.HasGoalJointPositions());
    EXPECT_FALSE(session.StartPlan());
    Object::Delete(root);
}

TEST(MotionPlanningSession, ColdGhostGoalSharedFiltersAndNativeSphereCover) {
    ProjectSettings settings;
    Node* root = LoadPandaWorld(&settings);
    ASSERT_NE(root, nullptr);
    Robot3D* robot = FindFirstRobot(root);
    PlanningSceneBridge bridge;
    InteractiveIKGhostSession ghost(&bridge);
    MotionPlanningSession planner(&bridge);
    planner.Update(root, robot, 1, 1, false);
    auto q = MakeSafePandaStart(bridge);
    q(0) += 0.35;
    planner.SetGoalJointPositions(q);
    Affine3 goal;
    ASSERT_TRUE(bridge.ComputeLinkWorldTransform(q, bridge.GetDefaultEndEffectorLinkId(), &goal));
    ASSERT_FALSE(ghost.IsEnabled());
    ASSERT_TRUE(planner.PushGoalToIKGhost(&ghost));
    ghost.Update(root, robot, 1, 1, false);
    EXPECT_TRUE(ghost.GetTargetPose().matrix().isApprox(goal.matrix()));

    const auto& names = bridge.GetBodyNames();
    ghost.SetExtraIgnoredPairs({{names.front(), names.back()}});
    const auto previous_count = bridge.GetSyncCount();
    for (int frame = 0; frame < 3; ++frame) {
        planner.Update(root, robot, 1, 1, false);
        ghost.Update(root, robot, 1, 1, false);
    }
    EXPECT_EQ(bridge.GetSyncCount(), previous_count + 1);
    const auto snap = bridge.CaptureSnapshot();
    const auto spheres = bridge.CollectLinkSpheres(q);
    ASSERT_GT(spheres.size(), 0u);
    ASSERT_EQ(spheres.size(), snap->link_spheres.size());
    for (size_t i = 0; i < spheres.size(); ++i) {
        const auto& source = snap->link_spheres[i];
        Affine3 transform;
        ASSERT_TRUE(bridge.ComputeLinkWorldTransform(q, source.body_id, &transform));
        EXPECT_EQ(spheres[i].body_id, source.body_id);
        EXPECT_FLOAT_EQ(spheres[i].radius, static_cast<float>(source.radius));
        EXPECT_TRUE(spheres[i].center.isApprox(transform * source.local_center.cast<RealType>(), 1e-5));
    }
    Object::Delete(root);
}

TEST(MotionPlanningSession, SyncErrorExitsReadyAndClearsOldTrajectory) {
    ProjectSettings settings;
    Node* root = LoadPandaWorld(&settings);
    ASSERT_NE(root, nullptr);
    Robot3D* panda = FindFirstRobot(root);
    ASSERT_NE(panda, nullptr);

    PlanningSceneBridge shared_bridge;
    MotionPlanningSession planner(&shared_bridge);

    std::uint64_t change_ver = 1;
    planner.Update(root, panda, 1, change_ver, false, 0.0);
    const Eigen::VectorXd start_q = MakeSafePandaStart(shared_bridge);
    Eigen::VectorXd goal_q = start_q;
    goal_q(0) += 0.20;
    planner.SetStartJointPositions(start_q);
    planner.SetGoalJointPositions(goal_q);
    ASSERT_TRUE(planner.StartPlan());
    ASSERT_TRUE(planner.WaitForIdle(std::chrono::seconds(10)));
    planner.Update(root, panda, 1, change_ver, false, 0.0);
    ASSERT_EQ(planner.GetStatus(), MotionPlanningStatus::Ready) << planner.GetDiagnostics().status_message;

    // Add a Terrain3D heightfield (unsupported by planning collision) to trigger a clean SyncFromScene error.
    auto* terrain = Object::New<Terrain3D>();
    terrain->SetName("UnsupportedHeightfield");
    TerrainHeightField hf;
    hf.rows = 2;
    hf.cols = 2;
    hf.heights = {0.0, 0.1, 0.2, 0.0};
    terrain->AddHeightField(hf);
    root->AddChild(terrain);

    ++change_ver;
    planner.Update(root, panda, 1, change_ver, false, 0.0);
    EXPECT_EQ(planner.GetStatus(), MotionPlanningStatus::SyncError);
    EXPECT_FALSE(planner.HasTrajectory());
    EXPECT_FALSE(planner.GetDiagnostics().status_message.empty());

    Object::Delete(root);
}

TEST(MotionPlanningSession, DragTransactionUndoRedoAndEscWhileMouseHeldKeepsPlanningWorldSynced) {
    ProjectSettings settings;
    Node* root = LoadPandaWorld(&settings);
    ASSERT_NE(root, nullptr);
    Robot3D* panda = FindFirstRobot(root);
    ASSERT_NE(panda, nullptr);

    auto* obstacle = Object::New<Node3D>();
    obstacle->SetName("DragBox");
    obstacle->SetPosition({0.90, 0.10, 0.45});
    auto* col = Object::New<CollisionShape3D>();
    col->SetName("DragBoxShape");
    auto box = MakeRef<BoxShape3D>();
    box->SetSize({0.14, 0.14, 0.14});
    col->SetShape(box);
    obstacle->AddChild(col);
    root->AddChild(obstacle);

    PlanningSceneBridge bridge;
    SceneCommandStack stack;
    EditorDragTransactionController drag_tx;

    auto find_obstacle_pos_in_world = [&]() -> Eigen::Vector3d {
        const auto snap = bridge.CaptureSnapshot();
        EXPECT_NE(snap, nullptr);
        for (const auto& ws : snap->shapes) {
            if (ws.body_id == -1 && ws.name.find("DragBox") != std::string::npos) {
                return ws.local_frame.col(3);
            }
        }
        return Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
    };

    ASSERT_TRUE(bridge.EnsureSynced(root, panda, 1, stack.GetVersion()));
    const Eigen::Vector3d orig_pos(0.90, 0.10, 0.45);
    EXPECT_TRUE(find_obstacle_pos_in_world().isApprox(orig_pos, 1e-5));

    // 1. Multi-frame drag -> Commit -> single Undo restores orig_pos, Redo restores final_pos.
    Affine3 tf_mid = obstacle->GetTransform();
    tf_mid.translation() = Vector3(1.05, 0.10, 0.45);
    Affine3 tf_end = obstacle->GetTransform();
    tf_end.translation() = Vector3(1.20, 0.25, 0.45);

    ASSERT_TRUE(drag_tx.ExecuteMutation(
            &stack, "Transform Node",
            std::make_unique<SetNode3DTransformCommand>(obstacle->GetInstanceId(), tf_mid, false)));
    ASSERT_TRUE(drag_tx.ExecuteMutation(
            &stack, "Transform Node",
            std::make_unique<SetNode3DTransformCommand>(obstacle->GetInstanceId(), tf_end, false)));
    ASSERT_TRUE(drag_tx.CommitAndReset(&stack));

    ASSERT_TRUE(bridge.EnsureSynced(root, panda, 1, stack.GetVersion()));
    EXPECT_TRUE(find_obstacle_pos_in_world().isApprox(Eigen::Vector3d(1.20, 0.25, 0.45), 1e-5));

    // Single Undo reverses the entire multi-frame drag transaction.
    ASSERT_TRUE(stack.Undo());
    EXPECT_TRUE(obstacle->GetPosition().isApprox(Vector3(0.90, 0.10, 0.45), 1e-5));
    ASSERT_TRUE(bridge.EnsureSynced(root, panda, 1, stack.GetVersion()));
    EXPECT_TRUE(find_obstacle_pos_in_world().isApprox(orig_pos, 1e-5));

    // Redo restores the committed final position.
    ASSERT_TRUE(stack.Redo());
    EXPECT_TRUE(obstacle->GetPosition().isApprox(Vector3(1.20, 0.25, 0.45), 1e-5));
    ASSERT_TRUE(bridge.EnsureSynced(root, panda, 1, stack.GetVersion()));
    EXPECT_TRUE(find_obstacle_pos_in_world().isApprox(Eigen::Vector3d(1.20, 0.25, 0.45), 1e-5));

    // 2. Start a second drag, press Esc mid-drag, then keep sending drag frames while mouse remains held!
    Affine3 tf_drag1 = obstacle->GetTransform();
    tf_drag1.translation() = Vector3(1.40, 0.40, 0.45);
    Affine3 tf_drag_after_esc = obstacle->GetTransform();
    tf_drag_after_esc.translation() = Vector3(1.60, 0.60, 0.45);

    ASSERT_TRUE(drag_tx.ExecuteMutation(
            &stack, "Transform Node",
            std::make_unique<SetNode3DTransformCommand>(obstacle->GetInstanceId(), tf_drag1, false)));
    ASSERT_TRUE(bridge.EnsureSynced(root, panda, 1, stack.GetVersion()));
    EXPECT_TRUE(find_obstacle_pos_in_world().isApprox(Eigen::Vector3d(1.40, 0.40, 0.45), 1e-5));

    // Press Esc -> transaction rolls back to (1.20, 0.25, 0.45) and latches gesture_cancelled_.
    ASSERT_TRUE(drag_tx.CancelOnEscape(&stack));
    EXPECT_TRUE(drag_tx.IsGestureCancelled());
    EXPECT_TRUE(obstacle->GetPosition().isApprox(Vector3(1.20, 0.25, 0.45), 1e-5));

    // Next frame: mouse is STILL held down and sends another drag delta -> MUST BE IGNORED!
    EXPECT_FALSE(drag_tx.ExecuteMutation(
            &stack, "Transform Node",
            std::make_unique<SetNode3DTransformCommand>(obstacle->GetInstanceId(), tf_drag_after_esc, false)));
    EXPECT_TRUE(obstacle->GetPosition().isApprox(Vector3(1.20, 0.25, 0.45), 1e-5));
    ASSERT_TRUE(bridge.EnsureSynced(root, panda, 1, stack.GetVersion()));
    EXPECT_TRUE(find_obstacle_pos_in_world().isApprox(Eigen::Vector3d(1.20, 0.25, 0.45), 1e-5));

    // Mouse finally releases -> latch clears cleanly, pose remains restored.
    ASSERT_TRUE(drag_tx.CommitAndReset(&stack));
    EXPECT_FALSE(drag_tx.IsGestureCancelled());
    EXPECT_TRUE(obstacle->GetPosition().isApprox(Vector3(1.20, 0.25, 0.45), 1e-5));

    Object::Delete(root);
}

#endif

} // namespace gobot
