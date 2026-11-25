#include "planning_planner.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <cmath>
#include <memory>
#include <thread>
#include <vector>

namespace gp = gobot::planning;

namespace {

// Build a simple 2-DOF revolute planar arm (L1 = 0.5 m, L2 = 0.5 m) with a sphere tip
// and an optional world box obstacle — 100% pure C++, zero Python dependency.
std::shared_ptr<gp::PlanningWorld> make_planar_arm_world() {
    auto world = std::make_shared<gp::PlanningWorld>();

    // Bodies: 0=base, 1=link1 (rot Z at base), 2=link2 (rot Z at x=0.5), 3=tip (fixed at x=0.5)
    std::vector<int> parent = {-1, 0, 1, 2};
    std::vector<int> kinds = {4, 1, 1, 0};
    std::vector<int> joint_idx = {-1, 0, 1, -1};

    const gp::Mat3 I3 = gp::Mat3::Identity();
    std::vector<gp::Mat3> a_rot(4, I3), b_rot(4, I3);
    std::vector<gp::Vec3> a_pos = {
        gp::Vec3::Zero(),
        gp::Vec3::Zero(),
        gp::Vec3(0.5, 0.0, 0.0),
        gp::Vec3(0.5, 0.0, 0.0),
    };
    std::vector<gp::Vec3> b_pos(4, gp::Vec3::Zero());
    std::vector<gp::Vec3> axes = {
        gp::Vec3::UnitX(),
        gp::Vec3::UnitZ(),
        gp::Vec3::UnitZ(),
        gp::Vec3::UnitX(),
    };
    std::vector<gp::KinematicsModel::Branch> branches = {
        {"arm", 0, gp::Vec3::Zero(), I3},
    };
    std::vector<std::string> body_names = {"arm::base", "arm::link1", "arm::link2", "arm::tip"};
    std::vector<std::string> joint_names = {"arm::j1", "arm::j2"};
    Eigen::VectorXd lower(2), upper(2), vel(2), acc(2), init_q(2);
    lower << -M_PI, -M_PI;
    upper << M_PI, M_PI;
    vel << 2.0, 2.0;
    acc << 6.0, 6.0;
    init_q << 0.0, 0.0;

    std::unordered_map<std::string, int> link_index = {
        {"arm::base", 0}, {"arm::link1", 1}, {"arm::link2", 2}, {"arm::tip", 3},
    };
    world->init_kinematics(parent, kinds, joint_idx, a_rot, a_pos, b_rot, b_pos, axes,
                           branches, body_names, joint_names, lower, upper, vel, acc, init_q, link_index);

    auto tip_shape = std::make_shared<gp::CollisionShape>();
    tip_shape->label = "tip_sphere";
    tip_shape->kind = 2;
    tip_shape->radius = 0.04;
    tip_shape->convex = std::make_shared<coal::Sphere>(0.04);
    tip_shape->finalize_bounds();

    gp::WorldShape tip_ws;
    tip_ws.name = "arm/tip/sphere";
    tip_ws.link_key = "arm::tip";
    tip_ws.body_id = 3;
    tip_ws.local_frame.setZero();
    tip_ws.local_frame.leftCols<3>() = I3;
    tip_ws.shape = tip_shape;
    world->add_shape(tip_ws);

    return world;
}

void add_box_obstacle(gp::PlanningWorld& world, const std::string& name,
                      const gp::Vec3& size, const gp::Vec3& pos) {
    auto sh = std::make_shared<gp::CollisionShape>();
    sh->label = name;
    sh->kind = 1;
    sh->box_half_size = size * 0.5;
    sh->convex = std::make_shared<coal::Box>(size);
    sh->finalize_bounds();

    gp::WorldShape ws;
    ws.name = name;
    ws.link_key = "::" + name;
    ws.body_id = -1;
    ws.local_frame.setZero();
    ws.local_frame.leftCols<3>() = gp::Mat3::Identity();
    ws.local_frame.col(3) = pos;
    ws.shape = sh;
    world.add_shape(ws);
}

}  // namespace

TEST(MotionPlanningCpp, StandalonePlanAndHermiteBounds) {
    auto world = make_planar_arm_world();
    add_box_obstacle(*world, "pillar", gp::Vec3(0.16, 0.16, 0.30), gp::Vec3(0.70, 0.45, 0.0));

    gp::MotionPlannerNative planner(world);
    gp::MotionPlanRequestNative req;
    req.start_q = (Eigen::VectorXd(2) << 0.0, 0.0).finished();
    req.goal_qs = {(Eigen::VectorXd(2) << 1.4, 0.0).finished()};
    req.config.waypoints = 12;
    req.config.margin = 0.005;
    req.config.has_world_margin = true;
    req.config.world_margin = 0.015;
    req.config.collision_substeps = 6;
    req.config.stop_at_endpoints = true;

    const auto res = planner.plan(req);
    ASSERT_TRUE(res.success) << res.reason;
    EXPECT_GT(res.trajectory.duration(), 0.0);

    auto snap = world->snapshot();
    const auto bchk = gp::check_trajectory_bounds(
        res.trajectory, snap->kin.velocity_limit(), snap->kin.acceleration_limit(), true, true, 1e-5);
    EXPECT_TRUE(bchk.ok) << bchk.reason;

    // Verify continuous Hermite CCD passes on the returned trajectory.
    std::vector<double> samples, frames;
    res.trajectory.sample_segments(12, samples);
    const int64_t S = static_cast<int64_t>(samples.size() / 2);
    frames.resize(static_cast<size_t>(S * snap->shapes.size() * 12));
    snap->compute_frames(samples.data(), S, frames.data());
    const Eigen::VectorXd margins = snap->pair_margins(0.005, 0.015);
    auto hit = snap->first_violation_continuous(
        res.trajectory.positions.data(), res.trajectory.num_waypoints(), 12,
        frames.data(), margins.data(), false, 5e-4,
        res.trajectory.times.data(), res.trajectory.velocities.data());
    EXPECT_FALSE(hit.has_value());
}

TEST(MotionPlanningCpp, KinematicChainCCDCatchesCurvedArcObstacle) {
    // 180-degree revolute sweep of tip at radius 1.0 m:
    // Start q0 = (0, 0) -> tip at (+1, 0), Goal q1 = (pi, 0) -> tip at (-1, 0).
    // Straight-line chord displacement of tip in 3D is 2.0 m, while curved arc passes through (0, +1.0)!
    // Place a small obstacle at (0.0, 1.0, 0.0): endpoints are > 1.3 m away, yet the revolute arc hits it dead-on!
    auto world = make_planar_arm_world();
    add_box_obstacle(*world, "arc_post", gp::Vec3(0.08, 0.08, 0.30), gp::Vec3(0.0, 1.0, 0.0));
    auto snap = world->snapshot();

    gp::RowMatrixXd q_path(2, 2);
    q_path << 0.0, 0.0,
              M_PI, 0.0;
    std::vector<double> frames(static_cast<size_t>(2 * snap->shapes.size() * 12));
    snap->compute_frames(q_path.data(), 2, frames.data());
    const Eigen::VectorXd margins = snap->pair_margins(0.005, 0.01);

    // At substeps=1, discrete endpoint check sees zero collision.
    std::vector<int64_t> rows, pids;
    snap->screen_aabbs(frames.data(), 2, margins.data(), false, 1, rows, pids);
    EXPECT_FALSE(snap->first_violation_indexed(frames.data(), 2, rows.data(), pids.data(),
                                               static_cast<int64_t>(rows.size()), margins.data(), false).has_value());

    // Kinematic-chain Lipschitz CCD must catch the curved-arc collision at u ~ 0.5!
    auto ccd = snap->first_violation_continuous(q_path.data(), 2, 1, frames.data(), margins.data(), false, 5e-4);
    ASSERT_TRUE(ccd.has_value());
    EXPECT_EQ(ccd->seg, 0);
    EXPECT_GT(ccd->u, 0.35);
    EXPECT_LT(ccd->u, 0.65);
    EXPECT_LT(ccd->distance, 0.01);
}

TEST(MotionPlanningCpp, ConcurrentWorldMutationDuringPlanningIsThreadSafe) {
    auto world = make_planar_arm_world();
    // Place moving box well inside the 1.0 m reach envelope so all oscillation phases are cleanly feasible.
    add_box_obstacle(*world, "moving_box", gp::Vec3(0.14, 0.14, 0.30), gp::Vec3(0.50, 0.50, 0.0));
    gp::MotionPlannerNative planner(world);

    std::atomic<bool> stop{false};
    std::atomic<int> mutations{0};
    std::thread mutator([&] {
        int step = 0;
        while (!stop.load(std::memory_order_relaxed)) {
            gp::Frame f = gp::Frame::Zero();
            f.leftCols<3>() = gp::Mat3::Identity();
            f.col(3) = gp::Vec3(0.50, 0.45 + 0.15 * std::sin(0.1 * step), 0.0);
            world->update_obstacle_pose("moving_box", f);
            if ((step % 5) == 0) {
                const std::string tmp = "temp_obs";
                add_box_obstacle(*world, tmp, gp::Vec3(0.05, 0.05, 0.05), gp::Vec3(-0.8, -0.8, 0.0));
                world->remove_obstacle(tmp);
            }
            ++step;
            ++mutations;
        }
    });

    gp::MotionPlanRequestNative req;
    req.start_q = (Eigen::VectorXd(2) << 0.0, 0.35).finished();
    req.goal_qs = {(Eigen::VectorXd(2) << 1.2, 0.35).finished()};
    req.config.waypoints = 12;
    req.config.margin = 0.005;
    req.config.has_world_margin = true;
    req.config.world_margin = 0.01;
    req.config.collision_substeps = 6;

    std::vector<gp::MotionPlanRequestNative> batch(8, req);
    for (int iter = 0; iter < 5; ++iter) {
        auto out = planner.plan_batch(batch, 8);
        ASSERT_EQ(out.size(), batch.size());
        for (const auto& r : out) {
            EXPECT_TRUE(r.success) << r.reason;
        }
    }
    stop.store(true, std::memory_order_relaxed);
    mutator.join();
    EXPECT_GT(mutations.load(), 0);
}

TEST(MotionPlanningCpp, RevolutePlusUnboundedPrismaticHermiteCCDCatchesMidSegmentExtension) {
    // 2-DOF robot: Joint 0 = revolute Z (unbounded), Joint 1 = prismatic X (unbounded).
    // Link 2 carries a 5 cm sphere at local origin.
    auto world = std::make_shared<gp::PlanningWorld>();
    std::vector<int> parent = {-1, 0, 1};
    std::vector<int> kinds = {4, 1, 3};  // root base, revolute, prismatic
    std::vector<int> joint_idx = {-1, 0, 1};
    std::vector<gp::Mat3> a_rot(3, gp::Mat3::Identity()), b_rot(3, gp::Mat3::Identity());
    std::vector<gp::Vec3> a_pos(3, gp::Vec3::Zero()), b_pos(3, gp::Vec3::Zero());
    std::vector<gp::Vec3> axes = {gp::Vec3::UnitX(), gp::Vec3::UnitZ(), gp::Vec3::UnitX()};
    std::vector<gp::KinematicsModel::Branch> branches = {{"arm", 0, gp::Vec3::Zero(), gp::Mat3::Identity()}};
    std::vector<std::string> body_names = {"base", "link1", "tip"};
    std::vector<std::string> joint_names = {"j_rev", "j_pris"};
    const double inf = std::numeric_limits<double>::infinity();
    Eigen::VectorXd lower(2), upper(2), vel(2), acc(2), init_q(2);
    lower << -inf, -inf;
    upper << inf, inf;
    vel << 10.0, 10.0;
    acc << 50.0, 50.0;
    init_q << 0.0, 0.0;
    std::unordered_map<std::string, int> link_index = {{"arm::base", 0}, {"arm::link1", 1}, {"arm::tip", 2}};
    world->init_kinematics(parent, kinds, joint_idx, a_rot, a_pos, b_rot, b_pos, axes, branches,
                           body_names, joint_names, lower, upper, vel, acc, init_q, link_index);

    auto tip_shape = std::make_shared<gp::CollisionShape>();
    tip_shape->label = "tip_ball";
    tip_shape->kind = 2;
    tip_shape->radius = 0.05;
    tip_shape->convex = std::make_shared<coal::Sphere>(0.05);
    tip_shape->finalize_bounds();

    gp::WorldShape tip;
    tip.name = "tip_ball";
    tip.link_key = "arm::tip";
    tip.body_id = 2;
    tip.local_frame.setZero();
    tip.local_frame.leftCols<3>() = gp::Mat3::Identity();
    tip.shape = tip_shape;
    world->add_shape(std::move(tip));

    // At u = 0.5: q_rev = pi/2, q_pris = 1.0 m -> tip center at (0, 1.0, 0), radius 0.05.
    // Box centered at (0, 1.06, 0) with size (0.04, 0.04, 0.20) -> y in [1.04, 1.08] -> 10 mm penetration!
    add_box_obstacle(*world, "post", gp::Vec3(0.04, 0.04, 0.20), gp::Vec3(0.0, 1.06, 0.0));
    auto snap = world->snapshot();

    gp::RowMatrixXd q_wp(2, 2), v_wp(2, 2);
    q_wp << 0.0, 0.0,
            M_PI, 0.0;
    v_wp << 0.0, 4.0,
            0.0, -4.0;
    const double times[2] = {0.0, 1.0};

    // Must catch the 10 mm mid-segment Hermite penetration for every collision_substeps value!
    for (int substeps : {1, 2, 4, 8, 16, 24}) {
        const int64_t S = substeps + 1;
        gp::RowMatrixXd samples(S, 2);
        for (int s = 0; s < S; ++s) {
            const double u = static_cast<double>(s) / static_cast<double>(substeps);
            const double u2 = u * u, u3 = u2 * u;
            const double h00 = 2.0 * u3 - 3.0 * u2 + 1.0;
            const double h10 = u3 - 2.0 * u2 + u;
            const double h01 = -2.0 * u3 + 3.0 * u2;
            const double h11 = u3 - u2;
            for (int j = 0; j < 2; ++j) {
                samples(s, j) = h00 * q_wp(0, j) + h10 * v_wp(0, j) + h01 * q_wp(1, j) + h11 * v_wp(1, j);
            }
        }
        auto val = snap->validate_trajectory_collision(
            q_wp.data(), 2, samples.data(), S, substeps, 0.0, 0.0, true, 5e-4, times, v_wp.data(), false);
        EXPECT_FALSE(val.valid) << "Missed 10 mm Hermite prismatic+revolute collision at substeps=" << substeps;
    }
}

TEST(MotionPlanningCpp, AtomicSyncBatchPublishesSingleSnapshotAndIsExceptionSafe) {
    auto world = make_planar_arm_world();
    constexpr int N_OBS = 32;
    for (int i = 0; i < N_OBS; ++i) {
        add_box_obstacle(*world, "box_" + std::to_string(i), gp::Vec3(0.05, 0.05, 0.10),
                         gp::Vec3(0.2 + 0.05 * i, 1.0, 0.0));
    }

    const uint64_t v0 = world->version();
    const uint64_t tv0 = world->topology_version();

    // 1. No-op sync must not publish a new version.
    gp::SyncBatch empty_batch;
    const auto noop = world->apply_sync_batch(empty_batch);
    EXPECT_FALSE(noop.changed);
    EXPECT_EQ(noop.version, v0);
    EXPECT_EQ(noop.topology_version, tv0);
    EXPECT_EQ(world->version(), v0);

    // 2. Move all 32 obstacles + robot base in one atomic batch -> version +1, topology_version +0.
    gp::SyncBatch move_batch;
    move_batch.base_updates.push_back({0, gp::Vec3(0.10, -0.05, 0.0), gp::Mat3::Identity()});
    for (int i = 0; i < N_OBS; ++i) {
        gp::Frame f = gp::Frame::Zero();
        f.leftCols<3>() = gp::Mat3::Identity();
        f.col(3) = gp::Vec3(0.2 + 0.05 * i, -1.0, 0.0);
        move_batch.pose_updates.push_back({"box_" + std::to_string(i), f});
    }
    const auto moved = world->apply_sync_batch(move_batch);
    EXPECT_TRUE(moved.changed);
    EXPECT_EQ(moved.version, v0 + 1);
    EXPECT_EQ(moved.topology_version, tv0);
    EXPECT_EQ(moved.poses_updated, N_OBS);
    EXPECT_EQ(moved.robot_bases_updated, 1);

    // Verify PlanResultNative records the exact world_version used during planning.
    gp::MotionPlannerNative planner(world);
    gp::MotionPlanRequestNative req;
    req.start_q = (Eigen::VectorXd(2) << 0.0, 0.0).finished();
    req.goal_qs = {(Eigen::VectorXd(2) << 0.5, 0.2).finished()};
    req.config.waypoints = 8;
    const auto plan_res = planner.plan(req);
    ASSERT_TRUE(plan_res.success) << plan_res.reason;
    EXPECT_EQ(plan_res.world_version, moved.version);

    // 3. Atomic failure: valid base + pose update combined with invalid remove must throw and leave state untouched.
    gp::SyncBatch bad_batch;
    bad_batch.base_updates.push_back({0, gp::Vec3(9.0, 9.0, 0.0), gp::Mat3::Identity()});
    gp::Frame bad_f = gp::Frame::Zero();
    bad_f.leftCols<3>() = gp::Mat3::Identity();
    bad_f.col(3) = gp::Vec3(5.0, 5.0, 0.0);
    bad_batch.pose_updates.push_back({"box_0", bad_f});
    bad_batch.remove_names.push_back("nonexistent_obstacle");
    EXPECT_THROW(world->apply_sync_batch(bad_batch), std::invalid_argument);
    EXPECT_EQ(world->version(), moved.version);
    EXPECT_EQ(world->topology_version(), moved.topology_version);
    auto snap_after_fail = world->snapshot();
    EXPECT_NEAR(snap_after_fail->shapes[snap_after_fail->shape_index.at("box_0")].local_frame(1, 3), -1.0, 1e-12);

    // 4. Concurrent atomic observation: all 32 obstacles toggle between y=+1.0 and y=-1.0 in lockstep.
    // Readers must NEVER observe a half-updated state where some boxes are at +1 and others at -1!
    std::atomic<bool> stop{false};
    std::thread writer([&] {
        bool sign = false;
        while (!stop.load(std::memory_order_relaxed)) {
            sign = !sign;
            const double y = sign ? 1.0 : -1.0;
            gp::SyncBatch b;
            for (int i = 0; i < N_OBS; ++i) {
                gp::Frame f = gp::Frame::Zero();
                f.leftCols<3>() = gp::Mat3::Identity();
                f.col(3) = gp::Vec3(0.2 + 0.05 * i, y, 0.0);
                b.pose_updates.push_back({"box_" + std::to_string(i), f});
            }
            world->apply_sync_batch(b);
        }
    });
    for (int iter = 0; iter < 500; ++iter) {
        auto s = world->snapshot();
        const double y0 = s->shapes[s->shape_index.at("box_0")].local_frame(1, 3);
        for (int i = 1; i < N_OBS; ++i) {
            const double yi = s->shapes[s->shape_index.at("box_" + std::to_string(i))].local_frame(1, 3);
            ASSERT_DOUBLE_EQ(yi, y0) << "Observed torn half-updated snapshot across obstacles!";
        }
    }
    stop.store(true, std::memory_order_relaxed);
    writer.join();
}

TEST(MotionPlanningCpp, AcmLayeringPreservesUserOverridesAcrossSync) {
    auto world = make_planar_arm_world();
    add_box_obstacle(*world, "obs_a", gp::Vec3(0.10, 0.10, 0.10), gp::Vec3(0.5, 0.0, 0.0));
    ASSERT_EQ(world->num_pairs(), 1);

    // Auto-disable (arm::tip, ::obs_a) -> 0 active pairs.
    world->set_auto_collision_disabled("arm::tip", "::obs_a");
    EXPECT_EQ(world->num_pairs(), 0);

    // User explicitly re-enables (arm::tip, ::obs_a) -> overrides auto ACM -> 1 active pair.
    world->set_collision_enabled("arm::tip", "::obs_a", true);
    EXPECT_EQ(world->num_pairs(), 1);

    // Sync batch adds another obstacle -> user override on obs_a must remain active.
    auto sh_b = std::make_shared<gp::CollisionShape>();
    sh_b->label = "obs_b";
    sh_b->kind = 1;
    sh_b->box_half_size = gp::Vec3(0.05, 0.05, 0.05);
    sh_b->convex = std::make_shared<coal::Box>(gp::Vec3(0.10, 0.10, 0.10));
    sh_b->finalize_bounds();
    gp::WorldShape ws_b;
    ws_b.name = "obs_b";
    ws_b.link_key = "::obs_b";
    ws_b.body_id = -1;
    ws_b.local_frame.setZero();
    ws_b.local_frame.leftCols<3>() = gp::Mat3::Identity();
    ws_b.local_frame.col(3) = gp::Vec3(-0.5, 0.0, 0.0);
    ws_b.shape = sh_b;
    gp::SyncBatch add_b;
    add_b.add_shapes.push_back(ws_b);
    world->apply_sync_batch(add_b);
    EXPECT_EQ(world->num_pairs(), 2);

    // User disables obs_b -> survives subsequent pose sync.
    world->set_collision_enabled("arm::tip", "::obs_b", false);
    EXPECT_EQ(world->num_pairs(), 1);
}

TEST(MotionPlanningCpp, QuinticSplineC2ContinuityGlobalScalingAndBounds) {
    auto world = make_planar_arm_world();
    add_box_obstacle(*world, "pillar", gp::Vec3(0.16, 0.16, 0.30), gp::Vec3(0.70, 0.45, 0.0));
    world->set_jerk_limits((Eigen::VectorXd(2) << 12.0, 12.0).finished());

    gp::MotionPlannerNative planner(world);
    gp::MotionPlanRequestNative req;
    req.start_q = (Eigen::VectorXd(2) << 0.0, 0.0).finished();
    req.goal_qs = {(Eigen::VectorXd(2) << 1.4, 0.0).finished()};
    req.config.waypoints = 12;
    req.config.margin = 0.005;
    req.config.has_world_margin = true;
    req.config.world_margin = 0.015;
    req.config.collision_substeps = 6;
    req.config.stop_at_endpoints = true;
    req.config.spline_order = 5;
    req.config.strict_ccd = true;

    const auto res = planner.plan(req);
    ASSERT_TRUE(res.success) << res.reason;
    ASSERT_EQ(res.trajectory.spline_order, 5);
    EXPECT_GT(res.trajectory.duration(), 0.0);

    // Endpoint rest conditions: v(0) = a(0) = v(T) = a(T) = 0.
    Eigen::VectorXd p, v, a, jk;
    res.trajectory.sample_single(res.trajectory.times(0), p, &v, &a, &jk);
    EXPECT_NEAR(v.norm(), 0.0, 1e-10);
    EXPECT_NEAR(a.norm(), 0.0, 1e-10);
    res.trajectory.sample_single(res.trajectory.times(res.trajectory.num_waypoints() - 1), p, &v, &a, &jk);
    EXPECT_NEAR(v.norm(), 0.0, 1e-10);
    EXPECT_NEAR(a.norm(), 0.0, 1e-10);

    // Verify C^2 continuity (pos, vel, acc) across every interior knot, and both left/right jerk <= limit.
    const double eps = 1e-7;
    for (int k = 1; k + 1 < res.trajectory.num_waypoints(); ++k) {
        const double tk = res.trajectory.times(k);
        if (tk - res.trajectory.times(k - 1) <= 1e-5 || res.trajectory.times(k + 1) - tk <= 1e-5) continue;
        Eigen::VectorXd pl, vl, al, jl, pr, vr, ar, jr;
        res.trajectory.sample_single(tk - eps, pl, &vl, &al, &jl);
        res.trajectory.sample_single(tk + eps, pr, &vr, &ar, &jr);
        EXPECT_NEAR((pr - pl).norm(), 0.0, 1e-4);
        EXPECT_NEAR((vr - vl).norm(), 0.0, 1e-4);
        EXPECT_NEAR((ar - al).norm(), 0.0, 1e-3);
        EXPECT_LE(jl.cwiseAbs().maxCoeff(), 12.0 * (1.0 + 1e-4));
        EXPECT_LE(jr.cwiseAbs().maxCoeff(), 12.0 * (1.0 + 1e-4));
    }

    // Dense check over 1000 samples confirms |v| <= 2, |a| <= 6, |j| <= 12 everywhere.
    auto snap = world->snapshot();
    const auto bchk = gp::check_trajectory_bounds(
        res.trajectory, snap->kin.velocity_limit(), snap->kin.acceleration_limit(),
        true, true, 1e-5, &snap->kin.lower(), &snap->kin.upper(), &snap->kin.jerk_limit(), true);
    EXPECT_TRUE(bchk.ok) << bchk.reason;
    const double dur = res.trajectory.duration();
    for (int i = 0; i <= 1000; ++i) {
        res.trajectory.sample_single(dur * static_cast<double>(i) / 1000.0, p, &v, &a, &jk);
        EXPECT_LE(v.cwiseAbs().maxCoeff(), 2.0 * (1.0 + 1e-5));
        EXPECT_LE(a.cwiseAbs().maxCoeff(), 6.0 * (1.0 + 1e-5));
        EXPECT_LE(jk.cwiseAbs().maxCoeff(), 12.0 * (1.0 + 1e-5));
    }
}

TEST(MotionPlanningCpp, QuinticMidSegmentPositionExcursionAndFixedDtJerkOnlyFailure) {
    // Counterexample 1: both knots strictly inside [-1.0, 1.0], yet mid-segment quintic exceeds +1.0!
    gp::TrajectoryNative traj;
    traj.spline_order = 5;
    traj.times = (Eigen::VectorXd(2) << 0.0, 1.0).finished();
    traj.positions.resize(2, 1);
    traj.positions << 0.90, 0.90;
    traj.velocities.resize(2, 1);
    traj.velocities << 1.0, -1.0;
    traj.accelerations.resize(2, 1);
    traj.accelerations << 0.0, 0.0;

    const Eigen::VectorXd lo = Eigen::VectorXd::Constant(1, -1.0);
    const Eigen::VectorXd hi = Eigen::VectorXd::Constant(1, 1.0);
    const Eigen::VectorXd vmax = Eigen::VectorXd::Constant(1, 10.0);
    const Eigen::VectorXd amax = Eigen::VectorXd::Constant(1, 50.0);
    const Eigen::VectorXd jmax = Eigen::VectorXd::Constant(1, 500.0);

    // At u = 0.5: p(0.5) = 0.9 + (5/16)*1.0 = 1.2125 > 1.0!
    const auto pos_chk = gp::check_trajectory_bounds(traj, vmax, amax, true, true, 1e-5, &lo, &hi, &jmax, true);
    EXPECT_FALSE(pos_chk.ok);
    EXPECT_NE(pos_chk.reason.find("position"), std::string::npos);
    EXPECT_NE(pos_chk.reason.find("segment 0"), std::string::npos);
    EXPECT_NE(pos_chk.reason.find("joint 0"), std::string::npos);

    // Counterexample 2: fixed dt where velocity and acceleration are well inside limits, ONLY jerk exceeds limit!
    gp::RowMatrixXd wp(2, 1);
    wp << 0.0, 0.2;
    const Eigen::VectorXd vmax2 = Eigen::VectorXd::Constant(1, 5.0);
    const Eigen::VectorXd amax2 = Eigen::VectorXd::Constant(1, 20.0);
    const Eigen::VectorXd jmax2 = Eigen::VectorXd::Constant(1, 10.0);  // 60*0.2 / 0.5^3 = 96 > 10!
    const auto fixed_traj = gp::retime_trajectory(wp, vmax2, amax2, true, true, 0.5, nullptr, &jmax2, 5);
    EXPECT_DOUBLE_EQ(fixed_traj.times(1) - fixed_traj.times(0), 0.5);  // Fixed dt preserved!
    const auto jerk_chk = gp::check_trajectory_bounds(
        fixed_traj, vmax2, amax2, true, true, 1e-5, &lo, &hi, &jmax2, true);
    EXPECT_FALSE(jerk_chk.ok);
    EXPECT_NE(jerk_chk.reason.find("jerk"), std::string::npos);
    EXPECT_NE(jerk_chk.reason.find("segment 0"), std::string::npos);
    EXPECT_NE(jerk_chk.reason.find("joint 0"), std::string::npos);
}

TEST(MotionPlanningCpp, QuinticUnboundedPrismaticCCDCatchesMidSegmentExtension) {
    // 2-DOF revolute + unbounded prismatic arm on a quintic spline that extends to ~1.25 m at u=0.5!
    auto world = std::make_shared<gp::PlanningWorld>();
    std::vector<int> parent = {-1, 0, 1};
    std::vector<int> kinds = {4, 1, 3};
    std::vector<int> joint_idx = {-1, 0, 1};
    std::vector<gp::Mat3> a_rot(3, gp::Mat3::Identity()), b_rot(3, gp::Mat3::Identity());
    std::vector<gp::Vec3> a_pos(3, gp::Vec3::Zero()), b_pos(3, gp::Vec3::Zero());
    std::vector<gp::Vec3> axes = {gp::Vec3::UnitX(), gp::Vec3::UnitZ(), gp::Vec3::UnitX()};
    std::vector<gp::KinematicsModel::Branch> branches = {{"arm", 0, gp::Vec3::Zero(), gp::Mat3::Identity()}};
    std::vector<std::string> body_names = {"base", "link1", "tip"};
    std::vector<std::string> joint_names = {"j_rev", "j_pris"};
    const double inf = std::numeric_limits<double>::infinity();
    Eigen::VectorXd lower(2), upper(2), vel(2), acc(2), init_q(2);
    lower << -inf, -inf;
    upper << inf, inf;
    vel << 10.0, 10.0;
    acc << 50.0, 50.0;
    init_q << 0.0, 0.0;
    std::unordered_map<std::string, int> link_index = {{"arm::base", 0}, {"arm::link1", 1}, {"arm::tip", 2}};
    world->init_kinematics(parent, kinds, joint_idx, a_rot, a_pos, b_rot, b_pos, axes, branches,
                           body_names, joint_names, lower, upper, vel, acc, init_q, link_index);

    auto tip_shape = std::make_shared<gp::CollisionShape>();
    tip_shape->label = "tip_ball";
    tip_shape->kind = 2;
    tip_shape->radius = 0.05;
    tip_shape->convex = std::make_shared<coal::Sphere>(0.05);
    tip_shape->finalize_bounds();
    gp::WorldShape tip;
    tip.name = "tip_ball";
    tip.link_key = "arm::tip";
    tip.body_id = 2;
    tip.local_frame.setZero();
    tip.local_frame.leftCols<3>() = gp::Mat3::Identity();
    tip.shape = tip_shape;
    world->add_shape(std::move(tip));

    // Quintic: p_pris(0)=p_pris(1)=0, v_pris(0)=+3.2, v_pris(1)=-3.2, a=0, h=1 -> p_pris(0.5) = (5/16)*3.2 = 1.0 m!
    // At u = 0.5: q_rev = pi/2, tip center = (0, 1.0, 0), radius 0.05. Box at (0, 1.06, 0) size 0.04 -> 10 mm penetration!
    add_box_obstacle(*world, "post", gp::Vec3(0.04, 0.04, 0.20), gp::Vec3(0.0, 1.06, 0.0));
    auto snap = world->snapshot();

    gp::TrajectoryNative traj;
    traj.spline_order = 5;
    traj.times = (Eigen::VectorXd(2) << 0.0, 1.0).finished();
    traj.positions.resize(2, 2);
    traj.positions << 0.0, 0.0,
                      M_PI, 0.0;
    traj.velocities.resize(2, 2);
    traj.velocities << 0.0, 3.2,
                       0.0, -3.2;
    traj.accelerations = gp::RowMatrixXd::Zero(2, 2);

    for (int substeps : {1, 2, 4, 8, 16}) {
        std::vector<double> samples;
        traj.sample_segments(substeps, samples);
        const int64_t S = static_cast<int64_t>(samples.size() / 2);
        auto val = snap->validate_trajectory_collision(
            traj.positions.data(), 2, samples.data(), S, substeps,
            0.0, 0.0, true, 5e-4, traj.times.data(), traj.velocities.data(), false,
            traj.accelerations.data(), 5);
        EXPECT_FALSE(val.valid) << "Missed 10 mm quintic prismatic+revolute collision at substeps=" << substeps;
    }

    // Time-scaling invariance regression (cubic & quintic): scaling h = 1.0 s -> 5e-13 s with
    // v -> v / h and a -> a / h^2 preserves the exact geometric curve; at substeps=1 (only collision-free
    // endpoints sampled), CCD must catch the identical mid-segment penetration for both orders.
    for (int order : {3, 5}) {
        // Choose endpoint velocity so u=0.5 reaches q_pris = 1.0 m:
        // cubic:   p(0.5) = (h/8)*(v0 - v1) = 1.0  -> v0 = +4.0/h, v1 = -4.0/h
        // quintic: p(0.5) = (5*h/32)*(v0 - v1) = 1.0 -> v0 = +3.2/h, v1 = -3.2/h
        const double v_base = (order == 5) ? 3.2 : 4.0;
        Eigen::VectorXd mid_ref;
        for (double h_scale : {1.0, 5e-13}) {
            gp::TrajectoryNative t_scaled;
            t_scaled.spline_order = order;
            t_scaled.times = (Eigen::VectorXd(2) << 0.0, h_scale).finished();
            t_scaled.positions.resize(2, 2);
            t_scaled.positions << 0.0, 0.0,
                                  M_PI, 0.0;
            t_scaled.velocities.resize(2, 2);
            t_scaled.velocities << 0.0, v_base / h_scale,
                                   0.0, -v_base / h_scale;
            t_scaled.accelerations = gp::RowMatrixXd::Zero(2, 2);

            Eigen::VectorXd mid_q;
            t_scaled.sample_single(0.5 * h_scale, mid_q);
            if (h_scale == 1.0) {
                mid_ref = mid_q;
            } else {
                EXPECT_NEAR(mid_q(0), mid_ref(0), 1e-12);
                EXPECT_NEAR(mid_q(1), mid_ref(1), 1e-12);
            }

            std::vector<double> samples;
            t_scaled.sample_segments(1, samples);  // only safe endpoints sampled!
            auto val = snap->validate_trajectory_collision(
                t_scaled.positions.data(), 2, samples.data(), 2, 1,
                0.0, 0.0, true, 5e-4,
                t_scaled.times.data(), t_scaled.velocities.data(), true,
                (order == 5) ? t_scaled.accelerations.data() : nullptr, order);
            EXPECT_FALSE(val.valid)
                << "Tiny-duration CCD missed mid-segment collision for order=" << order << ", h=" << h_scale;
        }
    }
}

TEST(MotionPlanningCpp, QuinticNearZeroC5AccelerationPeakLimitLengthsZeroDtAndSplineOrder) {
    // 1. P1: Near-zero c5 (qa ~ 1e-15 vs qb = 2, qc = -1) where naive quadratic formula suffers cancellation
    //    and drops the interior acceleration peak at u = 0.5!
    //    p0=p1=0, v0=v1=0, a0=1.0, a1=1.0 + 4e-15, h=1.0:
    //    At u=0.5: a(0.5) = 1 - 6*(0.5) + 6*(0.25) = -0.5 ... wait, let's make |a(0.5)| > max(|a0|, |a1|):
    //    p0=0, p1=0, v0=0, v1=0, a0=-1.0, a1=-1.0 - 4e-15, h=1.0 -> c2=-0.5, c3=+1.0, c4=-0.5, c5 ~ -2e-15;
    //    Even stronger: p0=0, p1=0, v0=+1.0, v1=-1.0, a0=0.0, a1=4e-15, h=1.0:
    //    c3 = -2.0, c4 = +1.0, c5 = 2e-15 -> qa = 2e-14, qb = 4.0, qc = -2.0 -> u* = 0.5:
    //    a(0) = 0, a(1) = 4e-15, a(0.5) = 6*(-2)*(0.5) + 12*(1)*(0.25) = -3.0!!
    {
        const double p0 = 0.0, p1 = 0.0, v0 = 1.0, v1 = -1.0, a0 = 0.0, a1 = 4e-15, h = 1.0;
        const auto pk = gp::quintic_hermite_scalar_peaks(p0, v0, a0, p1, v1, a1, h);
        EXPECT_GE(pk.max_acc, 3.0 - 1e-9) << "Underestimated interior quintic acceleration peak with tiny c5!";

        gp::TrajectoryNative traj;
        traj.spline_order = 5;
        traj.times = (Eigen::VectorXd(2) << 0.0, 1.0).finished();
        traj.positions.resize(2, 1); traj.positions << p0, p1;
        traj.velocities.resize(2, 1); traj.velocities << v0, v1;
        traj.accelerations.resize(2, 1); traj.accelerations << a0, a1;
        const Eigen::VectorXd vmax = Eigen::VectorXd::Constant(1, 10.0);
        const Eigen::VectorXd amax = Eigen::VectorXd::Constant(1, 2.0);  // True peak is 3.0 > 2.0!
        const auto chk = gp::check_trajectory_bounds(traj, vmax, amax, true, true, 1e-5);
        EXPECT_FALSE(chk.ok);
        EXPECT_NE(chk.reason.find("acceleration"), std::string::npos);
    }

    // 2. P1: Wrong-length limit vectors must throw std::invalid_argument (never skip or read OOB).
    {
        gp::TrajectoryNative traj;
        traj.spline_order = 5;
        traj.times = (Eigen::VectorXd(2) << 0.0, 1.0).finished();
        traj.positions = gp::RowMatrixXd::Zero(2, 2);
        traj.velocities = gp::RowMatrixXd::Zero(2, 2);
        traj.accelerations = gp::RowMatrixXd::Zero(2, 2);
        const Eigen::VectorXd ok2 = Eigen::VectorXd::Constant(2, 5.0);
        const Eigen::VectorXd bad1 = Eigen::VectorXd::Constant(1, 5.0);
        EXPECT_THROW(gp::check_trajectory_bounds(traj, bad1, ok2), std::invalid_argument);
        EXPECT_THROW(gp::check_trajectory_bounds(traj, ok2, bad1), std::invalid_argument);
        EXPECT_THROW(gp::check_trajectory_bounds(traj, ok2, ok2, true, true, 1e-5, &bad1, &ok2), std::invalid_argument);
        EXPECT_THROW(gp::check_trajectory_bounds(traj, ok2, ok2, true, true, 1e-5, nullptr, nullptr, &bad1), std::invalid_argument);
        EXPECT_THROW(gp::retime_trajectory(traj.positions, bad1, ok2), std::invalid_argument);
        EXPECT_THROW(gp::retime_trajectory(traj.positions, ok2, ok2, true, false, 0.0, nullptr, &bad1, 5), std::invalid_argument);
    }

    // 3. P2: Zero-duration moving segment & non-monotonic timestamps must fail check_trajectory_bounds.
    {
        gp::TrajectoryNative traj;
        traj.spline_order = 5;
        traj.times = (Eigen::VectorXd(2) << 0.0, 0.0).finished();
        traj.positions.resize(2, 1); traj.positions << 0.0, 0.5;
        traj.velocities = gp::RowMatrixXd::Zero(2, 1);
        traj.accelerations = gp::RowMatrixXd::Zero(2, 1);
        const Eigen::VectorXd lim = Eigen::VectorXd::Constant(1, 10.0);
        const auto chk = gp::check_trajectory_bounds(traj, lim, lim, true, true, 1e-5);
        EXPECT_FALSE(chk.ok);
        EXPECT_NE(chk.reason.find("Zero-duration"), std::string::npos);

        traj.times << 1.0, 0.5;
        traj.positions << 0.0, 0.0;
        const auto chk_neg = gp::check_trajectory_bounds(traj, lim, lim, true, true, 1e-5);
        EXPECT_FALSE(chk_neg.ok);
        EXPECT_NE(chk_neg.reason.find("non-monotonic"), std::string::npos);
    }

    // 4. P2: Invalid spline_order=4 and invalid jerk_scale<=0 must throw in MotionPlannerNative::plan.
    {
        auto world = make_planar_arm_world();
        gp::MotionPlannerNative planner(world);
        gp::MotionPlanRequestNative req;
        req.start_q = (Eigen::VectorXd(2) << 0.0, 0.0).finished();
        req.goal_qs = {(Eigen::VectorXd(2) << 0.2, 0.1).finished()};
        req.config.spline_order = 4;
        EXPECT_THROW(planner.plan(req), std::invalid_argument);
        req.config.spline_order = 5;
        req.config.jerk_scale = 0.0;
        EXPECT_THROW(planner.plan(req), std::invalid_argument);
        req.config.jerk_scale = -1.0;
        EXPECT_THROW(planner.plan(req), std::invalid_argument);
    }

    // 5. P1: Quintic CCD missing all/some spline params or empty/mismatched samples must throw.
    {
        auto world = make_planar_arm_world();
        auto snap = world->snapshot();
        gp::RowMatrixXd q_wp = gp::RowMatrixXd::Zero(2, 2);
        gp::RowMatrixXd v_wp = gp::RowMatrixXd::Zero(2, 2);
        const double ts[2] = {0.0, 1.0};
        // Missing accelerations with times+velocities:
        EXPECT_THROW(
            snap->validate_trajectory_collision(
                q_wp.data(), 2, q_wp.data(), 2, 1, 0.0, 0.0, false, 5e-4, ts, v_wp.data(), false, nullptr, 5),
            std::invalid_argument);
        // Missing ALL spline params with spline_order=5:
        EXPECT_THROW(
            snap->validate_trajectory_collision(
                q_wp.data(), 2, q_wp.data(), 2, 1, 0.0, 0.0, false, 5e-4, nullptr, nullptr, false, nullptr, 5),
            std::invalid_argument);
        // Empty samples (S=0) must throw (never falsely pass!):
        EXPECT_THROW(
            snap->validate_trajectory_collision(
                q_wp.data(), 2, q_wp.data(), 0, 1, 0.0, 0.0, false, 5e-4, nullptr, nullptr, false, nullptr, 3),
            std::invalid_argument);
    }

    // 6. P2: Tiny positive duration playback must reach the exact endpoint; detach restores original link_key.
    {
        gp::TrajectoryNative tiny_play;
        tiny_play.spline_order = 5;
        tiny_play.times = (Eigen::VectorXd(2) << 0.0, 5e-13).finished();
        tiny_play.positions.resize(2, 1); tiny_play.positions << 0.0, 1.0;
        tiny_play.velocities = gp::RowMatrixXd::Zero(2, 1);
        tiny_play.accelerations = gp::RowMatrixXd::Zero(2, 1);
        Eigen::VectorXd p_end;
        tiny_play.sample_single(5e-13, p_end);
        EXPECT_NEAR(p_end(0), 1.0, 1e-12);
        double p_batch[1] = {0.0};
        const double t_q[1] = {5e-13};
        tiny_play.sample_batch(t_q, 1, p_batch);
        EXPECT_NEAR(p_batch[0], 1.0, 1e-12);

        // Detaching a payload originally from a non-selected body link ("wp::body") must restore "wp::body" ACM filtering!
        auto world = make_planar_arm_world();
        auto sh = std::make_shared<gp::CollisionShape>();
        sh->label = "wp_box";
        sh->kind = 1;
        sh->box_half_size = gp::Vec3(0.05, 0.05, 0.05);
        sh->convex = std::make_shared<coal::Box>(gp::Vec3(0.10, 0.10, 0.10));
        sh->finalize_bounds();
        gp::WorldShape ws;
        ws.name = "wp/body/box";
        ws.link_key = "wp::body";
        ws.body_id = -1;
        ws.local_frame.setZero();
        ws.local_frame.leftCols<3>() = gp::Mat3::Identity();
        ws.local_frame.col(3) = gp::Vec3(0.8, 0.0, 0.0);
        ws.shape = sh;
        world->add_shape(ws);
        world->set_collision_enabled("arm::tip", "wp::body", false);
        ASSERT_EQ(world->num_pairs(), 0);
        gp::Frame loc = gp::Frame::Zero();
        loc.leftCols<3>() = gp::Mat3::Identity();
        world->attach_object("wp/body/box", "arm::tip", loc, {});
        world->detach_object("wp/body/box", (Eigen::VectorXd(2) << 0.0, 0.0).finished());
        // Original link_key "wp::body" restored -> ACM rule ("arm::tip", "wp::body") = disabled still applies!
        EXPECT_EQ(world->num_pairs(), 0);
    }

    // 6. P2: Tiny non-zero displacement (5e-13) must NOT skip jerk retiming; tiny positive dt (5e-13) must NOT bypass bounds.
    {
        gp::RowMatrixXd wp(2, 1);
        wp << 0.0, 5e-13;
        const Eigen::VectorXd vmax = Eigen::VectorXd::Constant(1, 2.0);
        const Eigen::VectorXd amax = Eigen::VectorXd::Constant(1, 5.0);
        const Eigen::VectorXd jmax = Eigen::VectorXd::Constant(1, 5.0);
        const auto retimed = gp::retime_trajectory(wp, vmax, amax, true, false, 0.0, nullptr, &jmax, 5);
        const auto chk_tiny = gp::check_trajectory_bounds(retimed, vmax, amax, true, true, 1e-5, nullptr, nullptr, &jmax, true);
        EXPECT_TRUE(chk_tiny.ok) << chk_tiny.reason;

        gp::TrajectoryNative tiny_h;
        tiny_h.spline_order = 5;
        tiny_h.times = (Eigen::VectorXd(2) << 0.0, 5e-13).finished();
        tiny_h.positions.resize(2, 1); tiny_h.positions << 0.0, 5e-13;
        tiny_h.velocities = gp::RowMatrixXd::Zero(2, 1);
        tiny_h.accelerations = gp::RowMatrixXd::Zero(2, 1);
        const auto chk_short = gp::check_trajectory_bounds(tiny_h, vmax, amax, true, true, 1e-5, nullptr, nullptr, &jmax, true);
        EXPECT_FALSE(chk_short.ok);

        // One-sided lower_limits only must be enforced (not ignored).
        gp::TrajectoryNative one_side;
        one_side.spline_order = 5;
        one_side.times = (Eigen::VectorXd(2) << 0.0, 1.0).finished();
        one_side.positions.resize(2, 1); one_side.positions << -0.5, -0.5;
        one_side.velocities = gp::RowMatrixXd::Zero(2, 1);
        one_side.accelerations = gp::RowMatrixXd::Zero(2, 1);
        const Eigen::VectorXd lo_only = Eigen::VectorXd::Constant(1, 0.0);
        const auto chk_lo = gp::check_trajectory_bounds(one_side, vmax, amax, true, true, 1e-5, &lo_only, nullptr);
        EXPECT_FALSE(chk_lo.ok);
    }
}
