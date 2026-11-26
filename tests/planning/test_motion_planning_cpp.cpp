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

TEST(MotionPlanningCpp, CartesianLinearCircularArcAndTcpOffset) {
    auto world = make_planar_arm_world();
    auto snap = world->snapshot();
    gp::MotionPlannerNative planner(world);

    // Non-trivial TCP offset on arm::tip (body 3)
    const gp::Vec3 tcp_pos(0.05, 0.02, 0.0);
    const double ang_off = 0.35;
    gp::Mat3 tcp_rot = gp::Mat3::Identity();
    tcp_rot << std::cos(ang_off), -std::sin(ang_off), 0.0,
               std::sin(ang_off),  std::cos(ang_off), 0.0,
               0.0,                0.0,               1.0;

    auto eval_tcp = [&](const Eigen::VectorXd& q) {
        std::vector<gp::Vec3> bp(static_cast<size_t>(snap->kin.num_bodies()));
        std::vector<gp::Mat3> br(static_cast<size_t>(snap->kin.num_bodies()));
        snap->kin.fk_single(q.data(), bp.data(), br.data());
        return std::make_pair(gp::Vec3(bp[3] + br[3] * tcp_pos), gp::Mat3(br[3] * tcp_rot));
    };

    const Eigen::VectorXd q_start = (Eigen::VectorXd(2) << 0.4, 0.9).finished();
    const auto [p0, r0] = eval_tcp(q_start);

    // 1. Linear MoveL + Directed Circular Arc MoveC for both cubic (order=3) and quintic (order=5).
    const gp::Vec3 p1 = p0 + gp::Vec3(-0.18, 0.14, 0.0);
    // Directed circular arc from p1 through p_via to p2 (radius ~0.12 m in XY plane).
    const gp::Vec3 arc_c = p1 + gp::Vec3(-0.12, 0.0, 0.0);
    const gp::Vec3 p_via = arc_c + gp::Vec3(0.0, 0.12, 0.0);
    const gp::Vec3 p2 = arc_c + gp::Vec3(-0.12, 0.0, 0.0);

    for (int order : {3, 5}) {
        gp::CartesianPlanRequestNative req;
        req.start_q = q_start;
        req.link_id = 3;
        req.tcp_offset_position = tcp_pos;
        req.tcp_offset_rotation = tcp_rot;
        req.config.max_step = 0.02;
        req.config.max_pos_error = 1e-3;
        req.config.spline_order = order;
        req.config.avoid_collisions = false;

        gp::CartesianWaypointNative wp0;
        wp0.motion_type = gp::CartesianMotionType::Linear;
        wp0.position = p1;
        wp0.blend_radius = 0.0;

        gp::CartesianWaypointNative wp1;
        wp1.motion_type = gp::CartesianMotionType::Circular;
        wp1.via_position = p_via;
        wp1.has_via_position = true;
        wp1.position = p2;
        wp1.blend_radius = 0.0;

        req.waypoints = {wp0, wp1};
        const auto res = planner.plan_cartesian_path(req);
        ASSERT_TRUE(res.success) << "order=" << order << " reason=" << res.reason;
        ASSERT_TRUE(res.has_trajectory);
        EXPECT_DOUBLE_EQ(res.fraction, 1.0);
        EXPECT_EQ(res.completed_waypoints, 2);
        EXPECT_EQ(res.world_version, snap->version);
        EXPECT_LE(res.max_pos_error, req.config.max_pos_error + 1e-6);

        // Dense independent verification along the actual played spline: every point lies on line p0->p1 or arc p1->p_via->p2.
        const int S = 160;
        bool passed_near_via = false;
        const double T_dur = res.trajectory.duration();
        for (int i = 0; i <= S; ++i) {
            const double t = T_dur * static_cast<double>(i) / static_cast<double>(S);
            Eigen::VectorXd q_s;
            res.trajectory.sample_single(t, q_s);
            const auto [pt, rt] = eval_tcp(q_s);
            (void)rt;
            // Distance to line segment [p0, p1]
            const gp::Vec3 d_lin = p1 - p0;
            const double u_lin = std::clamp((pt - p0).dot(d_lin) / d_lin.squaredNorm(), 0.0, 1.0);
            const double err_lin = (pt - (p0 + u_lin * d_lin)).norm();
            // Distance to upper semicircle arc around arc_c with radius 0.12 (y >= arc_c.y)
            const gp::Vec3 rel = pt - arc_c;
            double err_arc = std::abs(rel.head<2>().norm() - 0.12) + std::abs(rel.z());
            if (rel.y() < -1e-3) err_arc += 1.0;  // Must be on the directed upper arc passing p_via!
            if ((pt - p_via).norm() < 0.015) passed_near_via = true;
            EXPECT_LE(std::min(err_lin, err_arc), req.config.max_pos_error + 1e-4)
                << "order=" << order << " sample=" << i;
        }
        EXPECT_TRUE(passed_near_via);
        const auto bchk = gp::check_trajectory_bounds(
            res.trajectory, snap->kin.velocity_limit(), snap->kin.acceleration_limit(),
            true, true, 1e-5, &snap->kin.lower(), &snap->kin.upper(), &snap->kin.jerk_limit(), order == 5);
        EXPECT_TRUE(bchk.ok) << bchk.reason;
    }

    // 2. Degenerate / collinear / duplicate circular arcs must throw std::invalid_argument.
    {
        gp::CartesianPlanRequestNative bad;
        bad.start_q = q_start;
        bad.link_id = 3;
        bad.tcp_offset_position = tcp_pos;
        bad.tcp_offset_rotation = tcp_rot;
        gp::CartesianWaypointNative wp;
        wp.motion_type = gp::CartesianMotionType::Circular;
        wp.has_via_position = true;
        // Collinear triplet
        wp.via_position = p0 + gp::Vec3(0.05, 0.0, 0.0);
        wp.position = p0 + gp::Vec3(0.10, 0.0, 0.0);
        bad.waypoints = {wp};
        EXPECT_THROW(planner.plan_cartesian_path(bad), std::invalid_argument);
        // Duplicate via == start
        wp.via_position = p0;
        wp.position = p0 + gp::Vec3(0.05, 0.05, 0.0);
        bad.waypoints = {wp};
        EXPECT_THROW(planner.plan_cartesian_path(bad), std::invalid_argument);
    }
}

TEST(MotionPlanningCpp, CartesianCornerHandlingAndSplineDeviationGuard) {
    auto world = make_planar_arm_world();
    auto snap = world->snapshot();
    gp::MotionPlannerNative planner(world);

    auto eval_tip = [&](const Eigen::VectorXd& q) {
        std::vector<gp::Vec3> bp(static_cast<size_t>(snap->kin.num_bodies()));
        std::vector<gp::Mat3> br(static_cast<size_t>(snap->kin.num_bodies()));
        snap->kin.fk_single(q.data(), bp.data(), br.data());
        return bp[3];
    };

    const Eigen::VectorXd q_start = (Eigen::VectorXd(2) << 0.25, 1.1).finished();
    const gp::Vec3 p0 = eval_tip(q_start);
    const gp::Vec3 p_corner = p0 + gp::Vec3(-0.18, 0.0, 0.0);
    const gp::Vec3 p_end = p_corner + gp::Vec3(0.0, 0.18, 0.0);

    // 1. Sharp corner (blend_radius == 0): must come to exact zero velocity at corner and never cut it.
    gp::CartesianPlanRequestNative req_sharp;
    req_sharp.start_q = q_start;
    req_sharp.link_id = 3;
    req_sharp.config.max_step = 0.03;
    req_sharp.config.max_pos_error = 1e-3;
    req_sharp.config.avoid_collisions = false;
    req_sharp.config.spline_order = 5;
    req_sharp.waypoints = {
        {gp::CartesianMotionType::Linear, p_corner, false, gp::Mat3::Identity(), false, gp::Vec3::Zero(), 0.0},
        {gp::CartesianMotionType::Linear, p_end,    false, gp::Mat3::Identity(), false, gp::Vec3::Zero(), 0.0},
    };
    const auto res_sharp = planner.plan_cartesian_path(req_sharp);
    ASSERT_TRUE(res_sharp.success) << res_sharp.reason;

    int corner_knot = -1;
    double best_corner_d = 1e9;
    for (int k = 0; k < res_sharp.trajectory.num_waypoints(); ++k) {
        const double d = (eval_tip(res_sharp.trajectory.positions.row(k).transpose()) - p_corner).norm();
        if (d < best_corner_d) { best_corner_d = d; corner_knot = k; }
    }
    ASSERT_GT(corner_knot, 0);
    ASSERT_LT(corner_knot, res_sharp.trajectory.num_waypoints() - 1);
    EXPECT_LT(best_corner_d, 1e-3);
    EXPECT_LT(res_sharp.trajectory.velocities.row(corner_knot).norm(), 1e-12);
    EXPECT_LT(res_sharp.trajectory.accelerations.row(corner_knot).norm(), 1e-12);

    // 2. Blended corner (blend_radius = 0.05): non-zero speed through blend and bounded corner distance.
    gp::CartesianPlanRequestNative req_blend = req_sharp;
    req_blend.waypoints[0].blend_radius = 0.05;
    const auto res_blend = planner.plan_cartesian_path(req_blend);
    ASSERT_TRUE(res_blend.success) << res_blend.reason;
    double min_dist_to_corner = 1e9;
    double speed_at_closest = 0.0;
    for (int i = 0; i <= 100; ++i) {
        const double t = res_blend.trajectory.duration() * static_cast<double>(i) / 100.0;
        Eigen::VectorXd qs, vs;
        res_blend.trajectory.sample_single(t, qs, &vs);
        const double d = (eval_tip(qs) - p_corner).norm();
        if (d < min_dist_to_corner) {
            min_dist_to_corner = d;
            speed_at_closest = vs.norm();
        }
    }
    EXPECT_GT(min_dist_to_corner, 0.005);   // Actually rounded the corner
    EXPECT_LE(min_dist_to_corner, 0.05);    // Strictly inside blend_radius
    EXPECT_GT(speed_at_closest, 1e-3);      // Smooth motion through blend (no full stop)

    // 3. Direct counterexample: all IK knots lie EXACTLY on the line (error == 0 at knots),
    // yet mid-segment spline bows off the line by > 10 mm -> verify_spline_tcp_deviation MUST reject it,
    // and plan_cartesian_path with coarse max_step MUST automatically refine it!
    {
        const Eigen::VectorXd q_a = q_start;
        const gp::Vec3 p_b = p0 + gp::Vec3(-0.32, 0.10, 0.0);
        // Solve exact IK for endpoint p_b.
        gp::PoseConstraintSpec target;
        target.link_id = 3;
        target.has_position = true;
        target.position = p_b;
        target.pos_tol = gp::Vec3::Constant(1e-6);
        const auto ik_b = planner.solve_ik({target}, &q_a, nullptr, 0.0, 0.0, 40, 1, false, 0);
        ASSERT_TRUE(ik_b.success);

        gp::TrajectoryNative coarse_traj;
        coarse_traj.spline_order = 3;
        coarse_traj.times = (Eigen::VectorXd(2) << 0.0, 1.0).finished();
        coarse_traj.positions.resize(2, 2);
        coarse_traj.positions.row(0) = q_a.transpose();
        coarse_traj.positions.row(1) = ik_b.q.transpose();
        coarse_traj.velocities = gp::RowMatrixXd::Zero(2, 2);
        coarse_traj.accelerations = gp::RowMatrixXd::Zero(2, 2);

        gp::cartesian_detail::CartesianPrimitive lin_prim;
        lin_prim.kind = gp::cartesian_detail::CartesianPrimitive::Kind::Linear;
        lin_prim.p0 = p0;
        lin_prim.p1 = p_b;
        lin_prim.arc_length = (p_b - p0).norm();
        const auto chk = gp::cartesian_detail::verify_spline_tcp_deviation(
            *snap, 3, gp::Vec3::Zero(), gp::Mat3::Identity(),
            coarse_traj, {lin_prim}, {0}, {0.0}, {1.0}, 1e-3, 1e-2);
        EXPECT_FALSE(chk.ok);
        EXPECT_GT(chk.max_pos_err, 5e-3);

        // Now ask plan_cartesian_path with huge max_step=1.0 (initially 1 segment) and tight max_pos_error=1e-3:
        gp::CartesianPlanRequestNative req_coarse;
        req_coarse.start_q = q_a;
        req_coarse.link_id = 3;
        req_coarse.config.max_step = 1.0;
        req_coarse.config.max_pos_error = 1e-3;
        req_coarse.config.max_subdivision_depth = 6;
        req_coarse.config.avoid_collisions = false;
        req_coarse.waypoints = {
            {gp::CartesianMotionType::Linear, p_b, false, gp::Mat3::Identity(), false, gp::Vec3::Zero(), 0.0},
        };
        const auto res_ref = planner.plan_cartesian_path(req_coarse);
        ASSERT_TRUE(res_ref.success) << res_ref.reason;
        EXPECT_GT(res_ref.trajectory.num_waypoints(), 2);
        EXPECT_LE(res_ref.max_pos_error, 1e-3 + 1e-6);

        // Drive post-retiming midpoint insertion directly, bypassing initial path subdivision.
        std::vector<gp::cartesian_detail::CartesianKnot> knots(2);
        knots[0].q = q_a;
        knots[1].q = ik_b.q;
        knots[1].tau = 1.0;
        auto finalize = [&](double max_step) {
            auto cfg = req_coarse.config;
            cfg.max_step = max_step;
            cfg.jump_threshold_revolute = 3.0;
            return gp::cartesian_detail::finalize_and_verify_cartesian_prefix(
                *snap, 3, gp::Vec3::Zero(), gp::Mat3::Identity(), {1, 1}, cfg,
                snap->pair_margins(0.0, 0.0), {lin_prim}, knots);
        };
        const auto refined = finalize(1.0);
        ASSERT_TRUE(refined.ok) << refined.reason;
        EXPECT_GT(refined.trajectory.num_waypoints(), 2);
        EXPECT_FALSE(finalize(0.01).ok);
    }
}

TEST(MotionPlanningCpp, CartesianObstaclePartialPrefixJumpAndUnwrap) {
    // 1. Obstacle blocking middle of linear path -> monotone verified partial prefix (0 < fraction < 1).
    auto world = make_planar_arm_world();
    auto eval_tip = [&](const gp::WorldSnapshot& s, const Eigen::VectorXd& q) {
        std::vector<gp::Vec3> bp(static_cast<size_t>(s.kin.num_bodies()));
        std::vector<gp::Mat3> br(static_cast<size_t>(s.kin.num_bodies()));
        s.kin.fk_single(q.data(), bp.data(), br.data());
        return bp[3];
    };
    const Eigen::VectorXd q_start = (Eigen::VectorXd(2) << 0.25, 1.1).finished();
    const gp::Vec3 p0 = eval_tip(*world->snapshot(), q_start);
    const gp::Vec3 p_end = p0 + gp::Vec3(-0.30, 0.0, 0.0);
    // Put a blocking box halfway along the line p0 -> p_end.
    add_box_obstacle(*world, "blocker", gp::Vec3(0.08, 0.20, 0.20), p0 + gp::Vec3(-0.18, 0.0, 0.0));

    gp::MotionPlannerNative planner(world);
    gp::CartesianPlanRequestNative req;
    req.start_q = q_start;
    req.link_id = 3;
    req.config.max_step = 0.02;
    req.config.max_pos_error = 1e-3;
    req.config.avoid_collisions = true;
    req.config.margin = 0.002;
    req.config.has_world_margin = true;
    req.config.world_margin = 0.005;
    req.config.strict_ccd = true;
    req.waypoints = {
        {gp::CartesianMotionType::Linear, p_end, false, gp::Mat3::Identity(), false, gp::Vec3::Zero(), 0.0},
    };
    const auto res = planner.plan_cartesian_path(req);
    EXPECT_FALSE(res.success);
    ASSERT_TRUE(res.has_trajectory);
    EXPECT_GT(res.fraction, 0.1);
    EXPECT_LT(res.fraction, 0.9);
    EXPECT_EQ(res.completed_waypoints, 0);
    EXPECT_FALSE(res.reason.empty());

    // Verify the returned partial trajectory passes strict continuous CCD!
    auto snap = world->snapshot();
    std::vector<double> samples;
    res.trajectory.sample_segments(8, samples);
    const auto vres = snap->validate_trajectory_collision(
        res.trajectory.positions.data(), res.trajectory.num_waypoints(),
        samples.data(), static_cast<int64_t>(samples.size() / 2), 8,
        req.config.margin, req.config.effective_world_margin(), true, 5e-4,
        res.trajectory.times.data(), res.trajectory.velocities.data(), true,
        res.trajectory.accelerations.data(), res.trajectory.spline_order);
    EXPECT_TRUE(vres.valid) << vres.reason;

    // 2. Tight jump_threshold_revolute fails when max_subdivision_depth=0, succeeds when adaptive subdivision enabled.
    auto clear_world = make_planar_arm_world();
    gp::MotionPlannerNative clear_planner(clear_world);
    gp::CartesianPlanRequestNative jreq;
    jreq.start_q = q_start;
    jreq.link_id = 3;
    jreq.config.max_step = 0.25;  // coarse step producing > 0.2 rad jump
    jreq.config.jump_threshold_revolute = 0.15;
    jreq.config.max_subdivision_depth = 0;
    jreq.config.avoid_collisions = false;
    jreq.waypoints = {
        {gp::CartesianMotionType::Linear, p_end, false, gp::Mat3::Identity(), false, gp::Vec3::Zero(), 0.0},
    };
    const auto j_fail = clear_planner.plan_cartesian_path(jreq);
    EXPECT_FALSE(j_fail.success);
    EXPECT_LT(j_fail.fraction, 1.0);

    jreq.config.max_subdivision_depth = 5;
    const auto j_ok = clear_planner.plan_cartesian_path(jreq);
    EXPECT_TRUE(j_ok.success) << j_ok.reason;
    EXPECT_DOUBLE_EQ(j_ok.fraction, 1.0);

    // 3. Continuous revolute unwrapping across +pi without a 2*pi flip.
    {
        Eigen::VectorXd q_prev(2), q_wrapped(2), lo(2), hi(2);
        q_prev << 3.10, 0.5;
        q_wrapped << -3.08, 0.5;  // equivalent to +3.2032 rad (only +0.1032 rad away from 3.10!)
        lo << -2.0 * M_PI, -M_PI;
        hi <<  2.0 * M_PI,  M_PI;
        std::vector<uint8_t> is_pris = {0, 0};
        std::vector<uint8_t> mask = {1, 1};
        gp::cartesian_detail::unwrap_revolute_near(q_wrapped, q_prev, lo, hi, is_pris, mask);
        EXPECT_NEAR(q_wrapped(0), -3.08 + 2.0 * M_PI, 1e-12);
    }
}

TEST(MotionPlanningCpp, CartesianReviewCounterexamplesMoveCViaBlendRotPureRotAndFirstSegPrefix) {
    // 1. P1: Small near-full YZ circular arc (R=1.5 mm) on an XY-only arm MUST NOT skip via_position and report success!
    {
        auto world = make_planar_arm_world();
        auto snap = world->snapshot();
        gp::MotionPlannerNative planner(world);
        const Eigen::VectorXd q0 = (Eigen::VectorXd(2) << 0.4, 0.9).finished();
        std::vector<gp::Vec3> bp(static_cast<size_t>(snap->kin.num_bodies()));
        std::vector<gp::Mat3> br(static_cast<size_t>(snap->kin.num_bodies()));
        snap->kin.fk_single(q0.data(), bp.data(), br.data());
        const gp::Vec3 p0 = bp[3];

        // Circle in YZ plane centered at p0 + (0, 0, 1.5mm) with radius 1.5 mm.
        // p0 is at bottom (0, 0, 0); via is at top (0, 0, +3mm); end is near bottom (0, -0.3mm, +0.03mm) -> ~348 deg arc!
        const double R = 0.0015;
        const gp::Vec3 c = p0 + gp::Vec3(0.0, 0.0, R);
        const gp::Vec3 p_via = c + gp::Vec3(0.0, 0.0, R);  // +3.0 mm in unreachable Z
        const double ang_end = -0.5 * M_PI + 2.0 * M_PI - 0.15;
        const gp::Vec3 p_end = c + gp::Vec3(0.0, R * std::cos(ang_end), R * std::sin(ang_end));

        gp::CartesianPlanRequestNative req;
        req.start_q = q0;
        req.link_id = 3;
        req.config.max_step = 0.01;
        req.config.max_pos_error = 1e-3;
        req.config.avoid_collisions = false;
        req.waypoints = {
            {gp::CartesianMotionType::Circular, p_end, false, gp::Mat3::Identity(), true, p_via, 0.0},
        };
        const auto res = planner.plan_cartesian_path(req);
        EXPECT_FALSE(res.success);
        EXPECT_LT(res.fraction, 1.0);
        EXPECT_EQ(res.completed_waypoints, 0);
    }

    // 2. P1: Blended transition from constrained orientation to free orientation MUST reach the specified orientation!
    // Build a 3-DOF X-Y-Yaw planar robot (independent slide X, slide Y, revolute Z).
    {
        auto world = std::make_shared<gp::PlanningWorld>();
        std::vector<int> parent = {-1, 0, 1, 2};
        std::vector<int> kinds = {4, 3, 3, 1};
        std::vector<int> joint_idx = {-1, 0, 1, 2};
        const gp::Mat3 I3 = gp::Mat3::Identity();
        std::vector<gp::Mat3> a_rot(4, I3), b_rot(4, I3);
        std::vector<gp::Vec3> a_pos(4, gp::Vec3::Zero()), b_pos(4, gp::Vec3::Zero());
        std::vector<gp::Vec3> axes = {gp::Vec3::UnitX(), gp::Vec3::UnitX(), gp::Vec3::UnitY(), gp::Vec3::UnitZ()};
        std::vector<gp::KinematicsModel::Branch> branches = {{"xyyaw", 0, gp::Vec3::Zero(), I3}};
        std::vector<std::string> bnames = {"xyyaw::base", "xyyaw::bx", "xyyaw::by", "xyyaw::tip"};
        std::vector<std::string> jnames = {"xyyaw::jx", "xyyaw::jy", "xyyaw::jyaw"};
        Eigen::VectorXd lo(3), hi(3), vel(3), acc(3), qinit(3);
        lo << -2.0, -2.0, -3.14;
        hi <<  2.0,  2.0,  3.14;
        vel << 2.0, 2.0, 3.0;
        acc << 6.0, 6.0, 10.0;
        qinit << 0.0, 0.0, 0.0;
        std::unordered_map<std::string, int> lidx = {
            {"xyyaw::base", 0}, {"xyyaw::bx", 1}, {"xyyaw::by", 2}, {"xyyaw::tip", 3},
        };
        world->init_kinematics(parent, kinds, joint_idx, a_rot, a_pos, b_rot, b_pos, axes,
                               branches, bnames, jnames, lo, hi, vel, acc, qinit, lidx);

        gp::MotionPlannerNative planner(world);
        const double yaw_target = 1.0;
        gp::Mat3 r_target = gp::Mat3::Identity();
        r_target << std::cos(yaw_target), -std::sin(yaw_target), 0.0,
                    std::sin(yaw_target),  std::cos(yaw_target), 0.0,
                    0.0,                   0.0,                  1.0;

        gp::CartesianPlanRequestNative req;
        req.start_q = qinit;
        req.link_id = 3;
        req.config.max_step = 0.02;
        req.config.max_rot_step = 0.05;
        req.config.max_pos_error = 1e-3;
        req.config.max_rot_error = 1e-2;
        req.config.avoid_collisions = false;
        req.waypoints = {
            {gp::CartesianMotionType::Linear, gp::Vec3(0.4, 0.0, 0.0), true, r_target, false, gp::Vec3::Zero(), 0.15},
            {gp::CartesianMotionType::Linear, gp::Vec3(0.4, 0.4, 0.0), false, gp::Mat3::Identity(), false, gp::Vec3::Zero(), 0.0},
        };
        const auto res = planner.plan_cartesian_path(req);
        ASSERT_TRUE(res.success) << res.reason;

        // Verify the played trajectory actually reaches yaw_target within max_rot_error = 0.01 rad!
        double min_rot_err = 1e9;
        for (int i = 0; i <= 200; ++i) {
            const double t = res.trajectory.duration() * static_cast<double>(i) / 200.0;
            Eigen::VectorXd qs;
            res.trajectory.sample_single(t, qs);
            min_rot_err = std::min(min_rot_err, std::abs(qs(2) - yaw_target));
        }
        EXPECT_LE(min_rot_err, req.config.max_rot_error);
    }

    // 3. P2: Pure-rotation second waypoint after a free-orientation arc failing IK must NOT report fraction=1 or completed_waypoints=2!
    {
        auto world = make_planar_arm_world();
        auto snap = world->snapshot();
        gp::MotionPlannerNative planner(world);
        const Eigen::VectorXd q0 = (Eigen::VectorXd(2) << 0.4, 0.9).finished();
        std::vector<gp::Vec3> bp(static_cast<size_t>(snap->kin.num_bodies()));
        std::vector<gp::Mat3> br(static_cast<size_t>(snap->kin.num_bodies()));
        snap->kin.fk_single(q0.data(), bp.data(), br.data());
        const gp::Vec3 p0 = bp[3];
        const gp::Mat3 r0 = br[3];

        // Segment 0: free-orientation circular arc in XY plane (succeeds, changes tip yaw away from r0).
        const gp::Vec3 arc_c = p0 + gp::Vec3(-0.10, 0.0, 0.0);
        const gp::Vec3 p_via = arc_c + gp::Vec3(0.0, 0.10, 0.0);
        const gp::Vec3 p1 = arc_c + gp::Vec3(-0.10, 0.0, 0.0);
        // Segment 1: pure rotation in place back to r0 (impossible for 2-DOF arm while holding position p1!).
        gp::CartesianPlanRequestNative req;
        req.start_q = q0;
        req.link_id = 3;
        req.config.max_step = 0.02;
        req.config.max_pos_error = 1e-3;
        req.config.max_rot_error = 1e-2;
        req.config.avoid_collisions = false;
        req.waypoints = {
            {gp::CartesianMotionType::Circular, p1, false, gp::Mat3::Identity(), true, p_via, 0.0},
            {gp::CartesianMotionType::Linear,   p1, true,  r0,                   false, gp::Vec3::Zero(), 0.0},
        };
        const auto res = planner.plan_cartesian_path(req);
        EXPECT_FALSE(res.success);
        ASSERT_TRUE(res.has_trajectory);
        EXPECT_EQ(res.completed_waypoints, 1);
        EXPECT_LT(res.fraction, 1.0);
        EXPECT_GT(res.fraction, 0.2);
    }

    // 4. P2: Collision inside the FIRST single IK segment (1.4 m with coarse max_step) must still bisect and recover executable prefix!
    {
        auto world = std::make_shared<gp::PlanningWorld>();
        std::vector<int> parent = {-1, 0};
        std::vector<int> kinds = {4, 3};
        std::vector<int> joint_idx = {-1, 0};
        const gp::Mat3 I3 = gp::Mat3::Identity();
        std::vector<gp::Mat3> a_rot(2, I3), b_rot(2, I3);
        std::vector<gp::Vec3> a_pos(2, gp::Vec3::Zero()), b_pos(2, gp::Vec3::Zero());
        std::vector<gp::Vec3> axes = {gp::Vec3::UnitX(), gp::Vec3::UnitX()};
        std::vector<gp::KinematicsModel::Branch> branches = {{"slider", 0, gp::Vec3::Zero(), I3}};
        std::vector<std::string> bnames = {"slider::base", "slider::tip"};
        std::vector<std::string> jnames = {"slider::jx"};
        Eigen::VectorXd lo(1), hi(1), vel(1), acc(1), qinit(1);
        lo << -1.0; hi << 2.0; vel << 2.0; acc << 6.0; qinit << 0.0;
        std::unordered_map<std::string, int> lidx = {{"slider::base", 0}, {"slider::tip", 1}};
        world->init_kinematics(parent, kinds, joint_idx, a_rot, a_pos, b_rot, b_pos, axes,
                               branches, bnames, jnames, lo, hi, vel, acc, qinit, lidx);

        auto sh = std::make_shared<gp::CollisionShape>();
        sh->label = "tip_sphere";
        sh->kind = 2;
        sh->radius = 0.04;
        sh->convex = std::make_shared<coal::Sphere>(0.04);
        sh->finalize_bounds();
        gp::WorldShape tip_ws;
        tip_ws.name = "slider/tip/sphere";
        tip_ws.link_key = "slider::tip";
        tip_ws.body_id = 1;
        tip_ws.local_frame.setZero();
        tip_ws.local_frame.leftCols<3>() = I3;
        tip_ws.shape = sh;
        world->add_shape(tip_ws);

        // Place thin blocking wall at x = 0.36 m (width 0.04 -> face at x = 0.34, tip radius 0.04 -> hits at x = 0.30).
        // Notice: start (0.0), midpoint (0.70), and end (1.40) are ALL collision-free, so IK produces 1 segment [0, 1.4],
        // and continuous CCD fails on seg 0! Sub-segment bisection MUST recover the ~0.29 m executable prefix!
        add_box_obstacle(*world, "wall", gp::Vec3(0.04, 0.30, 0.30), gp::Vec3(0.36, 0.0, 0.0));

        gp::MotionPlannerNative planner(world);
        gp::CartesianPlanRequestNative req;
        req.start_q = qinit;
        req.link_id = 1;
        req.config.max_step = 2.0;  // single nominal IK segment spanning 1.4 m
        req.config.jump_threshold_prismatic = 5.0;
        req.config.max_subdivision_depth = 0;
        req.config.avoid_collisions = true;
        req.config.margin = 0.002;
        req.config.has_world_margin = true;
        req.config.world_margin = 0.005;
        req.config.strict_ccd = true;
        req.waypoints = {
            {gp::CartesianMotionType::Linear, gp::Vec3(1.4, 0.0, 0.0), false, gp::Mat3::Identity(), false, gp::Vec3::Zero(), 0.0},
        };
        const auto res = planner.plan_cartesian_path(req);
        EXPECT_FALSE(res.success);
        ASSERT_TRUE(res.has_trajectory) << res.reason;
        EXPECT_GT(res.fraction, 0.15);  // > 0.21 m out of 1.4 m
        EXPECT_LT(res.fraction, 0.25);  // < 0.35 m out of 1.4 m
        EXPECT_EQ(res.completed_waypoints, 0);

        auto snap = world->snapshot();
        std::vector<double> samples;
        res.trajectory.sample_segments(8, samples);
        const auto vres = snap->validate_trajectory_collision(
            res.trajectory.positions.data(), res.trajectory.num_waypoints(),
            samples.data(), static_cast<int64_t>(samples.size()), 8,
            req.config.margin, req.config.effective_world_margin(), true, 5e-4,
            res.trajectory.times.data(), res.trajectory.velocities.data(), true,
            res.trajectory.accelerations.data(), res.trajectory.spline_order);
        EXPECT_TRUE(vres.valid) << vres.reason;
    }

    // 5. P1: 180-deg hairpin reversal blend (forward 30 mm, back 30 mm, blend_radius=6 mm) MUST reach the 27 mm apex!
    {
        auto world = std::make_shared<gp::PlanningWorld>();
        std::vector<int> parent = {-1, 0};
        std::vector<int> kinds = {4, 3};
        std::vector<int> joint_idx = {-1, 0};
        const gp::Mat3 I3 = gp::Mat3::Identity();
        std::vector<gp::Mat3> a_rot(2, I3), b_rot(2, I3);
        std::vector<gp::Vec3> a_pos(2, gp::Vec3::Zero()), b_pos(2, gp::Vec3::Zero());
        std::vector<gp::Vec3> axes = {gp::Vec3::UnitX(), gp::Vec3::UnitX()};
        std::vector<gp::KinematicsModel::Branch> branches = {{"slider", 0, gp::Vec3::Zero(), I3}};
        std::vector<std::string> bnames = {"slider::base", "slider::tip"};
        std::vector<std::string> jnames = {"slider::jx"};
        Eigen::VectorXd lo(1), hi(1), vel(1), acc(1), qinit(1);
        lo << -1.0; hi << 2.0; vel << 2.0; acc << 6.0; qinit << 0.0;
        std::unordered_map<std::string, int> lidx = {{"slider::base", 0}, {"slider::tip", 1}};
        world->init_kinematics(parent, kinds, joint_idx, a_rot, a_pos, b_rot, b_pos, axes,
                               branches, bnames, jnames, lo, hi, vel, acc, qinit, lidx);

        gp::MotionPlannerNative planner(world);
        for (int order : {3, 5}) {
            gp::CartesianPlanRequestNative req;
            req.start_q = qinit;
            req.link_id = 1;
            req.config.max_step = 0.01;
            req.config.max_pos_error = 1e-3;
            req.config.spline_order = order;
            req.config.avoid_collisions = false;
            req.waypoints = {
                {gp::CartesianMotionType::Linear, gp::Vec3(0.030, 0.0, 0.0), false, I3, false, gp::Vec3::Zero(), 0.006},
                {gp::CartesianMotionType::Linear, gp::Vec3(0.000, 0.0, 0.0), false, I3, false, gp::Vec3::Zero(), 0.0},
            };
            const auto res = planner.plan_cartesian_path(req);
            ASSERT_TRUE(res.success) << "order=" << order << " reason=" << res.reason;
            double max_x = -1.0;
            for (int i = 0; i <= 200; ++i) {
                Eigen::VectorXd qs;
                res.trajectory.sample_single(res.trajectory.duration() * static_cast<double>(i) / 200.0, qs);
                max_x = std::max(max_x, qs(0));
            }
            // Bezier apex is at 24mm + 0.5*6mm = 27mm; playback must reach within max_pos_error (1mm) of 27mm!
            EXPECT_NEAR(max_x, 0.027, 1e-3) << "order=" << order;
        }

        // P2: 200 mm hairpin reversal with 38 mm blend and max_step=10 mm -> EVERY adjacent IK knot pair
        // must have TCP spacing <= max_step (within IK tolerance) via de Casteljau control-polygon length bound!
        gp::CartesianPlanRequestNative req_big_blend;
        req_big_blend.start_q = qinit;
        req_big_blend.link_id = 1;
        req_big_blend.config.max_step = 0.01;
        req_big_blend.config.max_pos_error = 1e-3;
        req_big_blend.config.avoid_collisions = false;
        req_big_blend.waypoints = {
            {gp::CartesianMotionType::Linear, gp::Vec3(0.200, 0.0, 0.0), false, I3, false, gp::Vec3::Zero(), 0.038},
            {gp::CartesianMotionType::Linear, gp::Vec3(0.000, 0.0, 0.0), false, I3, false, gp::Vec3::Zero(), 0.0},
        };
        const auto res_big = planner.plan_cartesian_path(req_big_blend);
        ASSERT_TRUE(res_big.success) << res_big.reason;
        double max_knot_step = 0.0;
        for (int k = 0; k + 1 < res_big.trajectory.num_waypoints(); ++k) {
            max_knot_step = std::max(max_knot_step,
                                     std::abs(res_big.trajectory.positions(k + 1, 0) - res_big.trajectory.positions(k, 0)));
        }
        EXPECT_LE(max_knot_step, req_big_blend.config.max_step + 0.5 * req_big_blend.config.max_pos_error);
    }

    // 6. P2: IK collision exhausting subdivision depth on the first interval still bisects to recover executable prefix;
    //        P2: Later IK failure must NOT overwrite earlier collision truncation reason;
    //        P3: Zero-length wp[0] == start preserves completed_waypoints=1 when wp[1] fails!
    {
        auto world = std::make_shared<gp::PlanningWorld>();
        std::vector<int> parent = {-1, 0};
        std::vector<int> kinds = {4, 3};
        std::vector<int> joint_idx = {-1, 0};
        const gp::Mat3 I3 = gp::Mat3::Identity();
        std::vector<gp::Mat3> a_rot(2, I3), b_rot(2, I3);
        std::vector<gp::Vec3> a_pos(2, gp::Vec3::Zero()), b_pos(2, gp::Vec3::Zero());
        std::vector<gp::Vec3> axes = {gp::Vec3::UnitX(), gp::Vec3::UnitX()};
        std::vector<gp::KinematicsModel::Branch> branches = {{"slider", 0, gp::Vec3::Zero(), I3}};
        std::vector<std::string> bnames = {"slider::base", "slider::tip"};
        std::vector<std::string> jnames = {"slider::jx"};
        Eigen::VectorXd lo(1), hi(1), vel(1), acc(1), qinit(1);
        lo << -1.0; hi << 2.0; vel << 2.0; acc << 6.0; qinit << 0.0;
        std::unordered_map<std::string, int> lidx = {{"slider::base", 0}, {"slider::tip", 1}};
        world->init_kinematics(parent, kinds, joint_idx, a_rot, a_pos, b_rot, b_pos, axes,
                               branches, bnames, jnames, lo, hi, vel, acc, qinit, lidx);

        auto sh = std::make_shared<gp::CollisionShape>();
        sh->label = "tip_sphere";
        sh->kind = 2;
        sh->radius = 0.04;
        sh->convex = std::make_shared<coal::Sphere>(0.04);
        sh->finalize_bounds();
        gp::WorldShape tip_ws;
        tip_ws.name = "slider/tip/sphere";
        tip_ws.link_key = "slider::tip";
        tip_ws.body_id = 1;
        tip_ws.local_frame.setZero();
        tip_ws.local_frame.leftCols<3>() = I3;
        tip_ws.shape = sh;
        world->add_shape(tip_ws);

        // Obstacle covering x=0.0875 (first depth-4 subdivision node of 1.4 m!) while leaving first ~25 mm clear:
        // box center x=0.11, size_x=0.08 -> face at 0.07, sphere r=0.04 + margin 0.005 -> hits at x=0.025 m!
        add_box_obstacle(*world, "near_wall", gp::Vec3(0.08, 0.30, 0.30), gp::Vec3(0.11, 0.0, 0.0));

        gp::MotionPlannerNative planner(world);
        gp::CartesianPlanRequestNative req_ik_col;
        req_ik_col.start_q = qinit;
        req_ik_col.link_id = 1;
        req_ik_col.config.max_step = 2.0;
        req_ik_col.config.jump_threshold_prismatic = 5.0;
        req_ik_col.config.max_subdivision_depth = 4;
        req_ik_col.config.avoid_collisions = true;
        req_ik_col.config.margin = 0.002;
        req_ik_col.config.has_world_margin = true;
        req_ik_col.config.world_margin = 0.005;
        req_ik_col.config.strict_ccd = true;
        req_ik_col.waypoints = {
            {gp::CartesianMotionType::Linear, gp::Vec3(1.4, 0.0, 0.0), false, I3, false, gp::Vec3::Zero(), 0.0},
        };
        const auto res_ik_col = planner.plan_cartesian_path(req_ik_col);
        EXPECT_FALSE(res_ik_col.success);
        ASSERT_TRUE(res_ik_col.has_trajectory) << res_ik_col.reason;
        EXPECT_GT(res_ik_col.fraction, 0.01);
        EXPECT_LT(res_ik_col.fraction, 0.03);

        // Now move wall to x=0.36 (thin width 0.04 -> seg 0 passes coarse IK, fails CCD verification at ~0.295 m)
        // AND add unreachable wp[1] in Z -> reason MUST report seg 0 collision/clearance, NOT wp[1] IK pose error!
        gp::Frame f_wall = gp::Frame::Zero();
        f_wall.leftCols<3>() = I3;
        f_wall.col(3) = gp::Vec3(0.36, 0.0, 0.0);
        world->remove_obstacle("near_wall");
        add_box_obstacle(*world, "thin_wall", gp::Vec3(0.04, 0.30, 0.30), gp::Vec3(0.36, 0.0, 0.0));
        gp::CartesianPlanRequestNative req_reason = req_ik_col;
        req_reason.config.max_subdivision_depth = 0;
        req_reason.waypoints = {
            {gp::CartesianMotionType::Linear, gp::Vec3(1.4, 0.0, 0.0), false, I3, false, gp::Vec3::Zero(), 0.0},
            {gp::CartesianMotionType::Linear, gp::Vec3(1.4, 0.0, 0.2), false, I3, false, gp::Vec3::Zero(), 0.0},
        };
        const auto res_reason = planner.plan_cartesian_path(req_reason);
        EXPECT_FALSE(res_reason.success);
        ASSERT_TRUE(res_reason.has_trajectory);
        EXPECT_NE(res_reason.reason.find("clearance"), std::string::npos) << "got reason: " << res_reason.reason;
        EXPECT_NE(res_reason.reason.find("thin_wall"), std::string::npos) << "got reason: " << res_reason.reason;

        // P3: wp[0] == start pose, wp[1] unreachable in Z -> fraction == 0, completed_waypoints == 1!
        gp::CartesianPlanRequestNative req_zero0;
        req_zero0.start_q = qinit;
        req_zero0.link_id = 1;
        req_zero0.config.avoid_collisions = false;
        req_zero0.waypoints = {
            {gp::CartesianMotionType::Linear, gp::Vec3(0.0, 0.0, 0.0), false, I3, false, gp::Vec3::Zero(), 0.0},
            {gp::CartesianMotionType::Linear, gp::Vec3(0.0, 0.0, 0.2), false, I3, false, gp::Vec3::Zero(), 0.0},
        };
        const auto res_zero0 = planner.plan_cartesian_path(req_zero0);
        EXPECT_FALSE(res_zero0.success);
        EXPECT_FALSE(res_zero0.has_trajectory);
        EXPECT_DOUBLE_EQ(res_zero0.fraction, 0.0);
        EXPECT_EQ(res_zero0.completed_waypoints, 1);
    }
}

namespace {

// Build a 4-DOF spatial arm (base yaw Z, shoulder pitch Y, elbow pitch Y, wrist pitch Y)
// carrying a cup sphere at its end-effector, for constrained global manifold planning tests.
std::shared_ptr<gp::PlanningWorld> make_upright_cup_arm_world() {
    auto world = std::make_shared<gp::PlanningWorld>();
    // Bodies:
    // 0: base (root at origin)
    // 1: turret (revolute Z at (0, 0, 0.2))
    // 2: upper_arm (revolute Y at (0, 0, 0.1))
    // 3: forearm (revolute Y at (0.45, 0, 0))
    // 4: cup (revolute Y at (0.40, 0, 0))
    std::vector<int> parent = {-1, 0, 1, 2, 3};
    std::vector<int> kinds = {4, 1, 1, 1, 1};
    std::vector<int> joint_idx = {-1, 0, 1, 2, 3};
    const gp::Mat3 I3 = gp::Mat3::Identity();
    std::vector<gp::Mat3> a_rot(5, I3), b_rot(5, I3);
    std::vector<gp::Vec3> a_pos = {
        gp::Vec3::Zero(),
        gp::Vec3(0.0, 0.0, 0.20),
        gp::Vec3(0.0, 0.0, 0.10),
        gp::Vec3(0.45, 0.0, 0.0),
        gp::Vec3(0.40, 0.0, 0.0),
    };
    std::vector<gp::Vec3> b_pos(5, gp::Vec3::Zero());
    std::vector<gp::Vec3> axes = {
        gp::Vec3::UnitX(),
        gp::Vec3::UnitZ(),
        gp::Vec3::UnitY(),
        gp::Vec3::UnitY(),
        gp::Vec3::UnitY(),
    };
    std::vector<gp::KinematicsModel::Branch> branches = {{"cup_arm", 0, gp::Vec3::Zero(), I3}};
    std::vector<std::string> bnames = {
        "cup_arm::base", "cup_arm::turret", "cup_arm::upper_arm", "cup_arm::forearm", "cup_arm::cup",
    };
    std::vector<std::string> jnames = {
        "cup_arm::yaw", "cup_arm::shoulder", "cup_arm::elbow", "cup_arm::wrist",
    };
    Eigen::VectorXd lo(4), hi(4), vel(4), acc(4), qinit(4);
    lo << -3.0, -1.6, -2.4, -2.5;
    hi <<  3.0,  1.6,  2.4,  2.5;
    vel << 2.0, 2.0, 2.5, 3.0;
    acc << 6.0, 6.0, 8.0, 10.0;
    qinit << 0.0, -0.4, 0.8, -0.4;  // shoulder + elbow + wrist == 0 => cup Z is strictly upright (+Z)!
    std::unordered_map<std::string, int> lidx = {
        {"cup_arm::base", 0}, {"cup_arm::turret", 1}, {"cup_arm::upper_arm", 2},
        {"cup_arm::forearm", 3}, {"cup_arm::cup", 4},
    };
    world->init_kinematics(parent, kinds, joint_idx, a_rot, a_pos, b_rot, b_pos, axes,
                           branches, bnames, jnames, lo, hi, vel, acc, qinit, lidx);

    // Add cup sphere collision shape on body 4 and forearm capsule on body 3.
    auto cup_sh = std::make_shared<gp::CollisionShape>();
    cup_sh->label = "cup_sphere";
    cup_sh->kind = 2;
    cup_sh->radius = 0.045;
    cup_sh->convex = std::make_shared<coal::Sphere>(0.045);
    cup_sh->finalize_bounds();
    gp::WorldShape cup_ws;
    cup_ws.name = "cup_arm/cup/sphere";
    cup_ws.link_key = "cup_arm::cup";
    cup_ws.body_id = 4;
    cup_ws.local_frame.setZero();
    cup_ws.local_frame.leftCols<3>() = I3;
    cup_ws.shape = cup_sh;
    world->add_shape(cup_ws);

    return world;
}

}  // namespace

TEST(MotionPlanningCpp, ManifoldAntiParallelPiDegeneracyAndContinuousSplineVerification) {
    auto world = make_upright_cup_arm_world();
    auto snap = world->snapshot();

    gp::PoseConstraintSpec upright;
    upright.link_id = 4;
    upright.has_rotation = true;
    upright.rotation = gp::Mat3::Identity();
    upright.rot_tol = gp::Vec3(0.05, 0.05, std::numeric_limits<double>::infinity());

    // 1. Exact anti-parallel pi configuration (shoulder + elbow + wrist == pi => cup upside down!).
    // eval_pose_constraint residual vanishes to (0, 0), yet true angle is pi -> MUST be rejected!
    Eigen::VectorXd q_upside_down(4);
    q_upside_down << 0.0, 1.0, 1.141592653589793, 1.0;  // sum == pi
    std::vector<gp::Vec3> bp(5);
    std::vector<gp::Mat3> br(5);
    snap->kin.fk_single(q_upside_down.data(), bp.data(), br.data());
    const auto ev = gp::manifold_detail::eval_constraint_norm(upright, bp.data(), br.data(), 1.0);
    EXPECT_FALSE(ev.ok);
    EXPECT_TRUE(ev.antipodal);
    EXPECT_NEAR(ev.rot_err, M_PI, 1e-4);

    // Projection operator must reject the anti-parallel pi singularity instead of returning residual=0!
    Eigen::VectorXd q_proj;
    std::vector<uint8_t> mask(4, 1);
    EXPECT_FALSE(gp::manifold_detail::project_onto_manifold(*snap, {upright}, mask, q_upside_down, q_proj));

    // 2. Mid-segment continuous spline manifold violation (both endpoints strictly upright, midpoint tilts 0.3 rad!):
    Eigen::VectorXd q_up(4);
    q_up << 0.0, -0.4, 0.8, -0.4;
    for (int order : {3, 5}) {
        gp::TrajectoryNative bad_traj;
        bad_traj.spline_order = order;
        bad_traj.times = (Eigen::VectorXd(2) << 0.0, 1.0).finished();
        bad_traj.positions.resize(2, 4);
        bad_traj.positions.row(0) = q_up.transpose();
        bad_traj.positions.row(1) = q_up.transpose();
        bad_traj.velocities = gp::RowMatrixXd::Zero(2, 4);
        // Opposing wrist velocities push midpoint wrist pitch by ~0.35 rad while endpoints stay at 0 tilt!
        bad_traj.velocities(0, 3) =  1.4;
        bad_traj.velocities(1, 3) = -1.4;
        bad_traj.accelerations = gp::RowMatrixXd::Zero(2, 4);
        const auto mchk = gp::verify_spline_manifold_constraints(*snap, bad_traj, {upright}, false);
        EXPECT_FALSE(mchk.ok) << "order=" << order;
        EXPECT_GT(mchk.max_rot_err, 0.2) << "order=" << order;
    }
}

TEST(MotionPlanningCpp, ConstrainedBiRRTUprightCupAroundPillar) {
    auto world = make_upright_cup_arm_world();
    // Place a tall blocking pillar directly on the straight yaw sweep (x ~ 0.72, y = 0.0) so
    // the arm must fold inward or lift over/around while keeping the cup strictly upright!
    add_box_obstacle(*world, "pillar", gp::Vec3(0.22, 0.16, 0.60), gp::Vec3(0.72, 0.0, 0.35));

    gp::PoseConstraintSpec upright;
    upright.link_id = 4;
    upright.has_rotation = true;
    upright.rotation = gp::Mat3::Identity();
    upright.rot_tol = gp::Vec3(0.05, 0.05, std::numeric_limits<double>::infinity());

    // Start on -Y side (yaw=-0.8), Goal on +Y side (yaw=+0.8), both strictly upright (sum of pitch == 0).
    Eigen::VectorXd q_start(4), q_goal(4);
    q_start << -0.8, -0.4, 0.8, -0.4;
    q_goal  <<  0.8, -0.4, 0.8, -0.4;

    gp::MotionPlannerNative planner(world);

    // 1. Out-of-manifold start must be cleanly rejected.
    {
        gp::MotionPlanRequestNative bad_req;
        bad_req.start_q = (Eigen::VectorXd(4) << -0.8, -0.4, 0.8, 0.2).finished();  // tilted 0.6 rad
        bad_req.goal_qs = {q_goal};
        bad_req.manifold_constraints = {upright};
        const auto bad_res = planner.plan(bad_req);
        EXPECT_FALSE(bad_res.success);
        EXPECT_NE(bad_res.reason.find("manifold"), std::string::npos);
    }

    // 2. Plan upright cup around pillar for BOTH cubic (order=3) and quintic (order=5) splines!
    for (int order : {3, 5}) {
        gp::MotionPlanRequestNative req;
        req.start_q = q_start;
        req.goal_qs = {q_goal};
        req.manifold_constraints = {upright};
        req.config.waypoints = 16;
        req.config.margin = 0.002;
        req.config.has_world_margin = true;
        req.config.world_margin = 0.01;
        req.config.spline_order = order;
        req.config.strict_ccd = true;
        req.config.stop_at_endpoints = true;
        req.config.rrt_timeout_s = 2.0;

        const auto res = planner.plan(req);
        ASSERT_TRUE(res.success) << "order=" << order << " reason=" << res.reason;
        EXPECT_GT(res.trajectory.duration(), 0.0);

        auto snap = world->snapshot();
        // Dense independent check: cup Z axis stays within 0.05 rad of world +Z everywhere along the played spline!
        std::vector<gp::Vec3> bp(5);
        std::vector<gp::Mat3> br(5);
        double worst_tilt = 0.0;
        for (int i = 0; i <= 200; ++i) {
            const double t = res.trajectory.duration() * static_cast<double>(i) / 200.0;
            Eigen::VectorXd qs;
            res.trajectory.sample_single(t, qs);
            snap->kin.fk_single(qs.data(), bp.data(), br.data());
            const double cos_ang = std::clamp(br[4].col(2).z(), -1.0, 1.0);
            const double tilt = std::acos(cos_ang);
            worst_tilt = std::max(worst_tilt, tilt);
        }
        EXPECT_LE(worst_tilt, 0.05 + 1e-5) << "order=" << order;

        // Independent certified manifold + strict CCD + bounds verification.
        const auto mchk = gp::verify_spline_manifold_constraints(*snap, res.trajectory, {upright}, false);
        EXPECT_TRUE(mchk.ok) << mchk.reason;
        std::vector<double> samples;
        res.trajectory.sample_segments(8, samples);
        const auto cchk = snap->validate_trajectory_collision(
            res.trajectory.positions.data(), res.trajectory.num_waypoints(),
            samples.data(), static_cast<int64_t>(samples.size() / 4), 8,
            req.config.margin, req.config.effective_world_margin(), true, 5e-4,
            res.trajectory.times.data(), res.trajectory.velocities.data(), true,
            res.trajectory.accelerations.data(), res.trajectory.spline_order);
        EXPECT_TRUE(cchk.valid) << cchk.reason;
    }
}

TEST(MotionPlanningCpp, ManifoldReviewRegressionsFallbackDurationSameEndpointsResampleFreeAxisPiAndTinyTol) {
    auto world = make_upright_cup_arm_world();
    add_box_obstacle(*world, "pillar", gp::Vec3(0.22, 0.16, 0.60), gp::Vec3(0.72, 0.0, 0.35));
    auto snap = world->snapshot();
    gp::MotionPlannerNative planner(world);

    gp::PoseConstraintSpec upright;
    upright.link_id = 4;
    upright.has_rotation = true;
    upright.rotation = gp::Mat3::Identity();
    upright.rot_tol = gp::Vec3(0.05, 0.05, std::numeric_limits<double>::infinity());

    Eigen::VectorXd q_start(4), q_goal(4);
    q_start << -0.8, -0.4, 0.8, -0.4;
    q_goal  <<  0.8, -0.4, 0.8, -0.4;
    std::vector<uint8_t> mask(4, 1);

    // 1. P1 #3: Identical start == goal in plan_constrained_birrt MUST emit >= 2 waypoints (never size 1 / OOB)!
    {
        const auto rrt_same = gp::plan_constrained_birrt(
            *snap, q_start, q_start, {upright}, mask, 0.002, 0.01, 0.25, 1000, 1.0, 42, false, 16, 10);
        EXPECT_TRUE(rrt_same.success);
        EXPECT_EQ(rrt_same.waypoints.size(), 10u);
        EXPECT_GE(rrt_same.dense_waypoints.size(), 2u);
    }

    // 2. P2 #5: Coarse output_waypoints=2 around blocking pillar MUST NOT mark a pillar-cutting chord as success=True!
    {
        const auto rrt_coarse = gp::plan_constrained_birrt(
            *snap, q_start, q_goal, {upright}, mask, 0.002, 0.01, 0.25, 2000, 1.5, 42, false, 32, 2);
        EXPECT_FALSE(rrt_coarse.success);
    }

    // 3. P1 #1: Raw CBiRRT fallback MUST NOT bypass user-specified waypoint pose_constraints when SQP fails!
    {
        gp::MotionPlanRequestNative req_wp;
        req_wp.start_q = q_start;
        req_wp.goal_qs = {q_goal};
        req_wp.manifold_constraints = {upright};
        gp::PoseConstraintSpec unreachable_via;
        unreachable_via.link_id = 4;
        unreachable_via.has_position = true;
        unreachable_via.position = gp::Vec3(5.0, 5.0, 5.0);  // unreachable!
        unreachable_via.pos_tol = gp::Vec3::Constant(1e-3);
        req_wp.pose_constraints.emplace_back(8, std::vector<gp::PoseConstraintSpec>{unreachable_via});
        req_wp.config.waypoints = 16;
        req_wp.config.margin = 0.002;
        req_wp.config.has_world_margin = true;
        req_wp.config.world_margin = 0.01;
        req_wp.config.rrt_timeout_s = 1.0;
        const auto res_wp = planner.plan(req_wp);
        EXPECT_FALSE(res_wp.success);
    }

    // 4. P1 #2: Fixed-dt request must preserve exact waypoint count & duration (never replaced by variable dense_rrt!).
    {
        gp::MotionPlanRequestNative req_dt;
        req_dt.start_q = q_start;
        req_dt.goal_qs = {q_goal};
        req_dt.manifold_constraints = {upright};
        req_dt.config.waypoints = 12;
        req_dt.config.has_dt = true;
        req_dt.config.dt = 0.25;
        req_dt.config.margin = 0.002;
        req_dt.config.has_world_margin = true;
        req_dt.config.world_margin = 0.01;
        req_dt.config.rrt_timeout_s = 1.5;
        const auto res_dt = planner.plan(req_dt);
        if (res_dt.success) {
            EXPECT_EQ(res_dt.trajectory.num_waypoints(), 12);
            EXPECT_NEAR(res_dt.trajectory.duration(), 11.0 * 0.25, 1e-9);
        }
    }

    // 5. P2 #7: Rotating through pi around a free axis (free_count == 2: constrain X only, Y & Z free)
    // MUST PASS continuous manifold verification instead of being falsely rejected near pi!
    {
        auto clear_world = make_upright_cup_arm_world();
        auto csnap = clear_world->snapshot();
        gp::PoseConstraintSpec free_yz;
        free_yz.link_id = 4;
        free_yz.has_rotation = true;
        free_yz.rotation = gp::Mat3::Identity();
        free_yz.rot_tol = gp::Vec3(0.05,
                                   std::numeric_limits<double>::infinity(),
                                   std::numeric_limits<double>::infinity());
        // Pure yaw sweep from 2.9 rad across pi (3.14159) to 3.3 rad (with pitch sum == 0 -> exact phi_x == 0!).
        gp::TrajectoryNative yaw_pi_traj;
        yaw_pi_traj.spline_order = 3;
        yaw_pi_traj.times = (Eigen::VectorXd(2) << 0.0, 1.0).finished();
        yaw_pi_traj.positions.resize(2, 4);
        yaw_pi_traj.positions << 2.9, -0.4, 0.8, -0.4,
                                 3.3, -0.4, 0.8, -0.4;
        yaw_pi_traj.velocities = gp::RowMatrixXd::Zero(2, 4);
        yaw_pi_traj.accelerations = gp::RowMatrixXd::Zero(2, 4);
        const auto mchk = gp::verify_spline_manifold_constraints(*csnap, yaw_pi_traj, {free_yz}, false);
        EXPECT_TRUE(mchk.ok) << mchk.reason;
    }

    // 6. P2 #9: Tiny user tolerance (1e-10) must NOT be silently relaxed to 1e-9!
    {
        auto clear_world = make_upright_cup_arm_world();
        auto csnap = clear_world->snapshot();
        Eigen::VectorXd q0(4);
        q0 << 0.0, -0.4, 0.8, -0.4;
        std::vector<gp::Vec3> bp(5);
        std::vector<gp::Mat3> br(5);
        csnap->kin.fk_single(q0.data(), bp.data(), br.data());

        gp::PoseConstraintSpec tiny_tol;
        tiny_tol.link_id = 4;
        tiny_tol.has_position = true;
        tiny_tol.position = bp[4] + gp::Vec3(5e-10, 0.0, 0.0);  // violates 1e-10 tol by 5x!
        tiny_tol.pos_tol = gp::Vec3::Constant(1e-10);

        gp::TrajectoryNative stat_traj;
        stat_traj.spline_order = 3;
        stat_traj.times = (Eigen::VectorXd(2) << 0.0, 1.0).finished();
        stat_traj.positions.resize(2, 4);
        stat_traj.positions.row(0) = q0.transpose();
        stat_traj.positions.row(1) = q0.transpose();
        stat_traj.velocities = gp::RowMatrixXd::Zero(2, 4);
        stat_traj.accelerations = gp::RowMatrixXd::Zero(2, 4);
        const auto mchk = gp::verify_spline_manifold_constraints(*csnap, stat_traj, {tiny_tol}, false);
        EXPECT_FALSE(mchk.ok);
    }

    // 7. Review Round 2 P2: Continuous manifold check MUST reject a spline that peaks slightly above
    //    tolerance (e.g. 1.0003 * tol, previously leaked by 5e-4 slack!).
    {
        auto clear_world = make_upright_cup_arm_world();
        auto csnap = clear_world->snapshot();
        // Keep base yaw=0, elbow=0, wrist=0, shoulder q1(u) = 0 at endpoints and peaks at 1.0003 * 0.05 rad
        // at midpoint via cubic Hermite velocities v0 = +4 * peak, v1 = -4 * peak!
        const double tol = 0.05;
        const double peak = tol * 1.0003;  // exceeds tol by only 0.03% (3e-4)!
        gp::TrajectoryNative slight_over;
        slight_over.spline_order = 3;
        slight_over.times = (Eigen::VectorXd(2) << 0.0, 1.0).finished();
        slight_over.positions = gp::RowMatrixXd::Zero(2, 4);
        slight_over.velocities = gp::RowMatrixXd::Zero(2, 4);
        slight_over.accelerations = gp::RowMatrixXd::Zero(2, 4);
        // For cubic Hermite with q0=q1=0, v0=v, v1=-v on h=1: q(0.5) = v * 0.25 -> v = 4 * peak.
        slight_over.velocities(0, 1) = 4.0 * peak;
        slight_over.velocities(1, 1) = -4.0 * peak;
        const auto mchk = gp::verify_spline_manifold_constraints(*csnap, slight_over, {upright}, false);
        EXPECT_FALSE(mchk.ok) << "spline peaking at 1.0003 * tol must be rejected";
    }

    // 8. Review Round 2 P2: Midpoint refinement in finalize_trajectory returns exact original waypoint_indices mapping!
    {
        auto clear_world = make_upright_cup_arm_world();
        auto csnap = clear_world->snapshot();
        // Straight joint chord between two upright poses with large shoulder/elbow sweep causes midpoint tilt
        // exceeding 0.01 rad, forcing finalize_trajectory to insert projected midpoints!
        Eigen::VectorXd qa(4), qb_seed(4), qb(4), qc(4);
        qa << -0.5, -0.8,  1.6, -0.8;
        qb_seed << 0.0, 0.8, -1.6, 0.8;
        qc <<  0.5, -0.8,  1.6, -0.8;
        std::vector<gp::Vec3> bp_a(5);
        std::vector<gp::Mat3> br_a(5);
        csnap->kin.fk_single(qa.data(), bp_a.data(), br_a.data());

        gp::PoseConstraintSpec z_plane;
        z_plane.link_id = 4;
        z_plane.has_position = true;
        z_plane.position = bp_a[4];
        z_plane.pos_tol = gp::Vec3(std::numeric_limits<double>::infinity(),
                                   std::numeric_limits<double>::infinity(),
                                   0.01);
        const std::vector<gp::PoseConstraintSpec> z_specs{z_plane};
        ASSERT_TRUE(gp::manifold_detail::project_onto_manifold(*csnap, z_specs, mask, qb_seed, qb, 2.0, 0.1, 30));

        gp::RowMatrixXd coarse(3, 4);
        coarse.row(0) = qa.transpose();
        coarse.row(1) = qb.transpose();
        coarse.row(2) = qc.transpose();

        gp::SolverConfigNative fcfg;
        fcfg.margin = 0.0;
        fcfg.has_world_margin = true;
        fcfg.world_margin = 0.0;
        const auto fin = gp::finalize_trajectory(*csnap, coarse, fcfg, &z_specs, &mask);
        ASSERT_TRUE(fin.success) << fin.reason;
        EXPECT_GT(fin.trajectory.num_waypoints(), 3);
        ASSERT_EQ(fin.waypoint_indices.size(), 3u);
        EXPECT_EQ(fin.waypoint_indices[0], 0);
        EXPECT_GT(fin.waypoint_indices[1], 0);
        EXPECT_LT(fin.waypoint_indices[1], fin.trajectory.num_waypoints() - 1);
        EXPECT_EQ(fin.waypoint_indices[2], fin.trajectory.num_waypoints() - 1);
        for (int k = 0; k < 3; ++k) {
            EXPECT_TRUE(fin.trajectory.positions.row(fin.waypoint_indices[static_cast<size_t>(k)]).isApprox(
                coarse.row(k), 1e-12));
        }
    }

    // 9. Review Round 2 P2: Multi-seed (num_seeds > 1) constrained global planning retains & uses dense CBiRRT fallback!
    {
        gp::MotionPlanRequestNative req_ms;
        req_ms.start_q = q_start;
        req_ms.goal_qs = {q_goal};
        req_ms.manifold_constraints = {upright};
        req_ms.config.waypoints = 2;  // too coarse for SQP alone; requires dense CBiRRT fallback!
        req_ms.config.num_seeds = 4;
        req_ms.config.use_global_seeding = true;
        req_ms.config.margin = 0.002;
        req_ms.config.has_world_margin = true;
        req_ms.config.world_margin = 0.01;
        req_ms.config.rrt_timeout_s = 2.0;
        req_ms.config.strict_ccd = true;
        const auto res_ms = planner.plan(req_ms);
        ASSERT_TRUE(res_ms.success) << res_ms.reason;
        EXPECT_EQ(res_ms.diagnostics.seed_strategy, "constrained_birrt");
        EXPECT_GT(res_ms.trajectory.num_waypoints(), 2);
        ASSERT_EQ(res_ms.waypoint_indices.size(), 2u);
        EXPECT_EQ(res_ms.waypoint_indices.front(), 0);
        EXPECT_EQ(res_ms.waypoint_indices.back(), res_ms.trajectory.num_waypoints() - 1);
    }

    // 10. Review Round 2 P3: Zero tolerance in manifold_constraints is explicitly rejected at entry.
    {
        gp::PoseConstraintSpec zero_pos;
        zero_pos.link_id = 4;
        zero_pos.has_position = true;
        zero_pos.position = gp::Vec3::Zero();
        zero_pos.pos_tol = gp::Vec3(0.0, 0.01, 0.01);

        gp::MotionPlanRequestNative req_z;
        req_z.start_q = q_start;
        req_z.goal_qs = {q_start};
        req_z.manifold_constraints = {zero_pos};
        EXPECT_THROW(planner.plan(req_z), std::invalid_argument);
    }
}
