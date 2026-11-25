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
