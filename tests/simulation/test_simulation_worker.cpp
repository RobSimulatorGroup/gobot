#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <limits>
#include <thread>
#include "gobot/physics/backends/null_physics_world.hpp"
#include "gobot/scene/collision_shape_3d.hpp"
#include "gobot/scene/joint_3d.hpp"
#include "gobot/scene/link_3d.hpp"
#include "gobot/scene/robot_3d.hpp"
#include "gobot/scene/rigid_body_3d.hpp"
#include "gobot/scene/resources/box_shape_3d.hpp"
#include "gobot/simulation/queued_physics_world.hpp"
#include "gobot/simulation/simulation_server.hpp"

namespace gobot {
namespace {
using Clock = std::chrono::steady_clock;

struct Probe {
    std::atomic<int> delay_ms{0};
    std::atomic<int> started{0};
    std::atomic<int> completed{0};
    std::atomic<bool> fail{false};
    std::atomic<bool> valid_partial{false};
    std::atomic<bool> fail_build{false};
    std::atomic<int> build_delay_ms{0};
    std::atomic<int> builds{0};
    std::atomic<std::size_t> construction_thread{0};
    std::atomic<std::size_t> build_thread{0};
    std::atomic<std::size_t> solve_thread{0};
    std::atomic<std::size_t> destruction_thread{0};
};

class WorkerTestWorld : public NullPhysicsWorld {
public:
    explicit WorkerTestWorld(std::shared_ptr<Probe> probe) : probe_(std::move(probe)) {
        probe_->construction_thread = std::hash<std::thread::id>{}(std::this_thread::get_id());
    }
    ~WorkerTestWorld() override { probe_->destruction_thread = std::hash<std::thread::id>{}(std::this_thread::get_id()); }
    bool Build(PhysicsSceneSnapshot snapshot) override {
        probe_->build_thread = std::hash<std::thread::id>{}(std::this_thread::get_id());
        ++probe_->builds;
        std::this_thread::sleep_for(std::chrono::milliseconds(probe_->build_delay_ms.load()));
        if (probe_->fail_build) {
            last_error_ = "Injected build failure";
            return false;
        }
        return NullPhysicsWorld::Build(std::move(snapshot));
    }
    PhysicsSolverDiagnostics GetSolverDiagnostics() const override {
        PhysicsSolverDiagnostics result;
        result.timings_available = true;
        result.linear_iterations = probe_->completed.load();
        result.stage_timings.push_back({"test_solve", 0.001, result.linear_iterations, true});
        return result;
    }
    PhysicsStepResult Step(RealType dt) override {
        probe_->solve_thread = std::hash<std::thread::id>{}(std::this_thread::get_id());
        ++probe_->started;
        std::this_thread::sleep_for(std::chrono::milliseconds(probe_->delay_ms.load()));
        ++probe_->completed;
        if (probe_->fail) {
            if (probe_->valid_partial) scene_state_.robots[0].joints[0].position = 0.5;
            return {.advanced_time = dt / 2, .state_valid = probe_->valid_partial.load(), .error = "Injected partial solve"};
        }
        for (auto& robot : scene_state_.robots) {
            for (auto& joint : robot.joints) joint.position += joint.target_effort * dt;
        }
        auto result = PhysicsWorld::Step(dt);
        result.diagnostics = GetSolverDiagnostics();
        return result;
    }
private:
    std::shared_ptr<Probe> probe_;
};

Robot3D* MakeWorkerRobot() {
    auto* robot = Node::New<Robot3D>();
    robot->SetName("robot");
    auto* base = Node::New<Link3D>();
    base->SetName("base");
    robot->AddChild(base);
    auto* joint = Node::New<Joint3D>();
    joint->SetName("joint");
    joint->SetJointType(JointType::Revolute);
    joint->SetParentLink("base");
    joint->SetChildLink("tip");
    robot->AddChild(joint);
    auto* tip = Node::New<Link3D>();
    tip->SetName("tip");
    joint->AddChild(tip);
    return robot;
}

class SimulationWorkerTest : public testing::Test {
protected:
    void SetUp() override {
        original_ = PhysicsServer::GetBackendInfo(PhysicsBackendType::Null);
        PhysicsServer::RegisterBackend(original_, [probe = probe_]() -> Ref<PhysicsWorld> {
            return MakeRef<WorkerTestWorld>(probe);
        });
        root_ = MakeWorkerRobot();
        server_.SetFixedTimeStep(0.002);
        ASSERT_TRUE(server_.SetAsyncSteppingEnabled(true));
        ASSERT_TRUE(server_.BuildWorldFromScene(root_));
        ASSERT_TRUE(PumpUntil([&] { return !server_.IsAsyncOperationPending(); }));
    }
    void TearDown() override {
        server_.ClearWorld();
        EXPECT_TRUE(PumpUntil([&] { return !server_.IsWorkerRetiring(); }));
        PhysicsServer::RegisterBackend(original_, []() -> Ref<PhysicsWorld> { return MakeRef<NullPhysicsWorld>(); });
        Node::Delete(root_);
    }
    template <typename Predicate>
    bool PumpUntil(Predicate predicate) {
        const auto deadline = Clock::now() + std::chrono::seconds(3);
        do {
            server_.AdvanceRealtime(0);
            if (predicate()) return true;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        } while (Clock::now() < deadline);
        return false;
    }
    void Tick(RealType effort) {
        ASSERT_TRUE(server_.RequestStep([&](RealType dt) {
            EXPECT_EQ(std::this_thread::get_id(), owner_);
            EXPECT_FLOAT_EQ(dt, 0.002);
            ASSERT_TRUE(server_.GetWorld()->SetJointControl("robot", "joint", PhysicsJointControlMode::Effort, effort))
                    << server_.GetWorld()->GetLastError();
        }));
        ASSERT_TRUE(PumpUntil([&] { return !server_.IsAsyncOperationPending(); }));
        ASSERT_FALSE(server_.IsFaulted()) << server_.GetLastError();
    }
    RealType Position() const { return server_.GetWorld()->GetSceneState().robots[0].joints[0].position; }
    PhysicsBackendInfo original_;
    std::shared_ptr<Probe> probe_ = std::make_shared<Probe>();
    Robot3D* root_ = nullptr;
    SimulationServer server_{PhysicsBackendType::Null, false};
    const std::thread::id owner_ = std::this_thread::get_id();
};

TEST_F(SimulationWorkerTest, SlowSolveDoesNotBlockOwnerOrStopAndRetiresOnWorker) {
    probe_->delay_ms = 200;
    auto endpoint = server_.GetWorld();
    auto retained = server_.CaptureStateFrame();
    ASSERT_TRUE(server_.RequestStep());
    ASSERT_TRUE(PumpUntil([&] { return probe_->started == 1; }));
    const auto started = Clock::now();
    for (int i = 0; i < 100; ++i) {
        server_.AdvanceRealtime(0.016);
        EXPECT_EQ(server_.CaptureStateFrame(), retained);
    }
    server_.ClearWorld();
    EXPECT_LT((std::chrono::duration<double, std::milli>(Clock::now() - started).count()), 50);
    EXPECT_FALSE(server_.HasWorld());
    EXPECT_FALSE(endpoint->IsAvailable());
    EXPECT_FALSE(endpoint->SetJointControl("robot", "joint", PhysicsJointControlMode::Effort, 1));
    EXPECT_EQ(retained->tick, 0);
    ASSERT_TRUE(PumpUntil([&] { return !server_.IsWorkerRetiring(); }));
    EXPECT_NE(probe_->solve_thread, std::hash<std::thread::id>{}(owner_));
    EXPECT_EQ(probe_->construction_thread, probe_->solve_thread);
    EXPECT_EQ(probe_->build_thread, probe_->solve_thread);
    EXPECT_EQ(probe_->destruction_thread, probe_->solve_thread);
    EXPECT_EQ(server_.GetFrameCount(), 0);
}

TEST_F(SimulationWorkerTest, ResetDuringSolveDiscardsOldEpochAndPreservesFixedDt) {
    probe_->delay_ms = 200;
    const auto old = server_.CaptureStateFrame();
    ASSERT_TRUE(server_.RequestStep());
    ASSERT_TRUE(PumpUntil([&] { return probe_->started == 1; }));
    const auto begin = Clock::now();
    ASSERT_TRUE(server_.Reset());
    EXPECT_LT((std::chrono::duration<double, std::milli>(Clock::now() - begin).count()), 50);
    ASSERT_TRUE(PumpUntil([&] { return !server_.IsAsyncOperationPending(); }));
    EXPECT_NE(server_.CaptureStateFrame()->epoch, old->epoch);
    EXPECT_EQ(server_.GetFrameCount(), 0);
    EXPECT_EQ(server_.GetSimulationTime(), 0);
    EXPECT_FLOAT_EQ(server_.GetFixedTimeStep(), 0.002);
    probe_->delay_ms = 0;
    Tick(2);
    EXPECT_NEAR(Position(), 0.004, 1e-7);
    EXPECT_EQ(old->state.robots[0].joints[0].position, 0);
}

TEST_F(SimulationWorkerTest, StageDiagnosticsAreRetainedWithTheCompletedTick) {
    Tick(1);
    auto first = server_.CaptureStateFrame();
    ASSERT_TRUE(first->step.diagnostics.timings_available);
    ASSERT_EQ(first->step.diagnostics.stage_timings.size(), 1U);
    EXPECT_EQ(first->step.diagnostics.linear_iterations, 1U);
    Tick(1);
    const auto latest = server_.GetWorld()->GetSolverDiagnostics();
    ASSERT_EQ(latest.stage_timings.size(), 1U);
    EXPECT_EQ(latest.stage_timings.front().calls, 2U);
    EXPECT_EQ(first->step.diagnostics.stage_timings.front().calls, 1U);
    server_.ClearWorld();
    EXPECT_EQ(first->step.diagnostics.stage_timings.front().name, "test_solve");
}

TEST_F(SimulationWorkerTest, CommandsAndCheckpointReplayMatchSynchronousWorld) {
    auto sync = MakeRef<WorkerTestWorld>(std::make_shared<Probe>());
    sync->SetSettings(server_.GetPhysicsWorldSettings());
    ASSERT_TRUE(sync->Build(server_.GetWorld()->GetSceneSnapshot()));
    for (int i = 0; i < 5; ++i) {
        Tick(RealType(i));
        ASSERT_TRUE(sync->SetJointControl("robot", "joint", PhysicsJointControlMode::Effort, RealType(i)));
        ASSERT_TRUE(sync->Step(0.002).completed);
    }
    EXPECT_EQ(Position(), sync->GetSceneState().robots[0].joints[0].position);
    ASSERT_TRUE(server_.RequestCheckpoint());
    ASSERT_TRUE(PumpUntil([&] { return !server_.IsAsyncOperationPending(); }));
    const auto checkpoint = server_.GetCompletedCheckpoint();
    ASSERT_TRUE(checkpoint.physics.IsValid());
    for (int i = 0; i < 10; ++i) Tick(3);
    const auto expected = Position();
    const auto expected_time = server_.GetSimulationTime();
    ASSERT_TRUE(server_.RequestRestoreCheckpoint(checkpoint));
    ASSERT_TRUE(PumpUntil([&] { return !server_.IsAsyncOperationPending(); }));
    for (int i = 0; i < 10; ++i) Tick(3);
    EXPECT_EQ(Position(), expected);
    EXPECT_EQ(server_.GetSimulationTime(), expected_time);
    EXPECT_EQ(server_.GetFrameCount(), 15);
}

TEST_F(SimulationWorkerTest, PartialFailurePausesWithoutPublishingInvalidState) {
    probe_->fail = true;
    const auto retained = server_.CaptureStateFrame();
    ASSERT_TRUE(server_.RequestStep());
    ASSERT_TRUE(PumpUntil([&] { return server_.IsFaulted(); }));
    EXPECT_TRUE(server_.IsPaused());
    EXPECT_NEAR(server_.GetSimulationTime(), 0.001, 1e-8);
    EXPECT_EQ(server_.GetFrameCount(), 0);
    EXPECT_EQ(server_.CaptureStateFrame(), retained);
    EXPECT_FALSE(server_.RequestStep());
    EXPECT_EQ(server_.GetLastError(), "Injected partial solve");
    EXPECT_TRUE(server_.SyncSceneFromWorld());
    EXPECT_EQ(server_.GetLastError(), "Injected partial solve");
}

TEST_F(SimulationWorkerTest, CheckpointIncludesCommandsAcceptedBeforeTheNextTick) {
    ASSERT_TRUE(server_.GetWorld()->SetJointControl("robot", "joint", PhysicsJointControlMode::Effort, 7));
    ASSERT_TRUE(server_.RequestCheckpoint());
    ASSERT_TRUE(PumpUntil([&] { return !server_.IsAsyncOperationPending(); }));
    const auto checkpoint = server_.GetCompletedCheckpoint();
    ASSERT_TRUE(checkpoint.physics.IsValid());
    Tick(2);
    ASSERT_TRUE(server_.RequestRestoreCheckpoint(checkpoint));
    ASSERT_TRUE(PumpUntil([&] { return !server_.IsAsyncOperationPending(); }));
    EXPECT_EQ(server_.GetFrameCount(), 0);
    ASSERT_TRUE(server_.RequestStep());
    ASSERT_TRUE(PumpUntil([&] { return !server_.IsAsyncOperationPending(); }));
    EXPECT_NEAR(Position(), 0.014, 1e-7);
}

TEST_F(SimulationWorkerTest, ValidPartialFailurePublishesAdvancedStateThenPauses) {
    probe_->fail = true;
    probe_->valid_partial = true;
    const auto retained = server_.CaptureStateFrame();
    ASSERT_TRUE(server_.RequestStep());
    ASSERT_TRUE(PumpUntil([&] { return server_.IsFaulted(); }));
    EXPECT_TRUE(server_.IsPaused());
    EXPECT_NEAR(server_.GetSimulationTime(), 0.001, 1e-8);
    EXPECT_EQ(server_.GetFrameCount(), 0);
    EXPECT_NE(server_.CaptureStateFrame(), retained);
    EXPECT_EQ(Position(), 0.5);
    EXPECT_EQ(retained->state.robots[0].joints[0].position, 0);
}

TEST_F(SimulationWorkerTest, SlowBuildAndImmediateResetStayOffOwnerThread) {
    server_.ClearWorld();
    ASSERT_TRUE(PumpUntil([&] { return !server_.IsWorkerRetiring(); }));
    server_.ClearWorld();
    probe_->build_delay_ms = 200;
    const auto started = Clock::now();
    ASSERT_TRUE(server_.BuildWorldFromScene(root_));
    EXPECT_LT((std::chrono::duration<double, std::milli>(Clock::now() - started).count()), 50);
    EXPECT_TRUE(server_.HasWorld());
    EXPECT_FALSE(server_.IsWorldReady());
    EXPECT_FALSE(server_.RequestStep());
    ASSERT_TRUE(server_.Reset());
    ASSERT_TRUE(PumpUntil([&] { return !server_.IsAsyncOperationPending(); }));
    ASSERT_FALSE(server_.IsFaulted()) << server_.GetLastError();
    EXPECT_TRUE(server_.IsWorldReady());
    EXPECT_NE(probe_->build_thread, std::hash<std::thread::id>{}(owner_));
    Tick(1);
}

TEST_F(SimulationWorkerTest, BuildFailureIsReportedAndCanStopAndBuildAgain) {
    server_.ClearWorld();
    ASSERT_TRUE(PumpUntil([&] { return !server_.IsWorkerRetiring(); }));
    probe_->fail_build = true;
    ASSERT_TRUE(server_.BuildWorldFromScene(root_));
    ASSERT_TRUE(PumpUntil([&] { return server_.IsFaulted(); }));
    EXPECT_FALSE(server_.IsWorldReady());
    EXPECT_FALSE(server_.IsAsyncOperationPending());
    EXPECT_FALSE(server_.RequestStep());
    EXPECT_EQ(server_.GetLastError(), "Injected build failure");
    server_.ClearWorld();
    ASSERT_TRUE(PumpUntil([&] { return !server_.IsWorkerRetiring(); }));
    probe_->fail_build = false;
    ASSERT_TRUE(server_.BuildWorldFromScene(root_));
    ASSERT_TRUE(PumpUntil([&] { return server_.IsWorldReady(); }));
    Tick(1);
}

TEST_F(SimulationWorkerTest, ControlCallbackChangingSessionCannotDispatchStaleCommands) {
    EXPECT_FALSE(server_.RequestStep([&](RealType) { server_.ClearWorld(); }));
    EXPECT_EQ(probe_->started, 0);
    EXPECT_EQ(server_.GetFrameCount(), 0);
}

TEST_F(SimulationWorkerTest, BoundedQueueRejectsInvalidBatchBeforeAnyCommandAndFaultsOnOverflow) {
    auto endpoint = dynamic_pointer_cast<QueuedPhysicsWorld>(server_.GetWorld());
    ASSERT_TRUE(endpoint);
    EXPECT_FALSE(endpoint->QueueCommands({PhysicsJointCommand{0, 0, PhysicsJointControlMode::Effort, 1},
            PhysicsJointCommand{0, 999, PhysicsJointControlMode::Effort, 1}}));
    EXPECT_EQ(endpoint->GetQueuedCommandCount(), 0);
    EXPECT_FALSE(endpoint->QueueCommands({PhysicsJointCommand{0, 0, PhysicsJointControlMode::Effort,
            std::numeric_limits<RealType>::quiet_NaN()}}));
    PhysicsCommands commands(QueuedPhysicsWorld::MaxCommands, PhysicsClearForcesCommand{});
    ASSERT_TRUE(endpoint->QueueCommands(std::move(commands)));
    EXPECT_FALSE(endpoint->QueueCommands({PhysicsClearForcesCommand{}}));
    EXPECT_FALSE(server_.RequestStep());
    EXPECT_TRUE(server_.IsFaulted());
    EXPECT_EQ(probe_->started, 0);
}

TEST_F(SimulationWorkerTest, StaticDragIsRejectedBeforeQueueingAndDoesNotFaultSimulation) {
    auto world = server_.GetWorld();
    EXPECT_FALSE(world->SetLinkSpringForce("robot", "base", Vector3::Zero(), {0, 0, 1}, Vector3::Zero()));
    EXPECT_NE(world->GetLastError().find("robot::base"), std::string::npos);
    EXPECT_FALSE(world->SetLinkExternalForce("robot", "base", Vector3::Zero(), {1, 0, 0}));
    auto endpoint = dynamic_pointer_cast<QueuedPhysicsWorld>(world);
    ASSERT_TRUE(endpoint);
    EXPECT_EQ(endpoint->GetQueuedCommandCount(), 0);
    EXPECT_FALSE(endpoint->QueueCommands({PhysicsJointCommand{0, 0, PhysicsJointControlMode::Effort, 2},
            PhysicsLinkForceCommand{0, 0, Vector3::Zero(), Vector3::UnitX()}}));
    EXPECT_EQ(endpoint->GetQueuedCommandCount(), 0);
    ASSERT_TRUE(world->SetLinkExternalForce("robot", "tip", Vector3::Zero(), {1, 0, 0}));
    Tick(1);
    EXPECT_FALSE(server_.IsFaulted());
    EXPECT_EQ(server_.GetFrameCount(), 1);
}

TEST(PhysicsForceValidation, FixedDescendantsInheritMotionAndMalformedCyclesTerminate) {
    PhysicsSceneSnapshot snapshot;
    PhysicsRobotSnapshot robot;
    robot.name = "robot";
    for (const auto* name : {"base", "tip", "tool", "fixture"}) {
        PhysicsLinkSnapshot link;
        link.name = name;
        link.mass = 1;
        robot.links.push_back(link);
    }
    PhysicsJointSnapshot joint;
    joint.parent_link = "base";
    joint.child_link = "tip";
    joint.joint_type = static_cast<int>(JointType::Revolute);
    robot.joints.push_back(joint);
    joint.parent_link = "tip";
    joint.child_link = "tool";
    joint.joint_type = static_cast<int>(JointType::Fixed);
    robot.joints.push_back(joint);
    joint.parent_link = "base";
    joint.child_link = "fixture";
    robot.joints.push_back(joint);
    snapshot.robots.push_back(robot);
    std::string error;
    EXPECT_FALSE(ValidatePhysicsLinkForceTarget(snapshot, "robot", "base", &error));
    EXPECT_TRUE(ValidatePhysicsLinkForceTarget(snapshot, "robot", "tip", &error));
    EXPECT_TRUE(ValidatePhysicsLinkForceTarget(snapshot, "robot", "tool", &error));
    EXPECT_FALSE(ValidatePhysicsLinkForceTarget(snapshot, "robot", "fixture", &error));
    EXPECT_FALSE(ValidatePhysicsLinkForceTarget(snapshot, "robot", "missing", &error));
    EXPECT_FALSE(ValidatePhysicsLinkForceTarget(snapshot, "missing", "base", &error));

    joint.parent_link.clear();
    joint.child_link = "base";
    joint.joint_type = static_cast<int>(JointType::Floating);
    snapshot.robots[0].joints.push_back(joint);
    EXPECT_TRUE(ValidatePhysicsLinkForceTarget(snapshot, "robot", "base", &error));
    EXPECT_TRUE(ValidatePhysicsLinkForceTarget(snapshot, "robot", "fixture", &error));

    snapshot.robots[0].joints[0].joint_type = static_cast<int>(JointType::Fixed);
    snapshot.robots[0].joints.back().joint_type = static_cast<int>(JointType::Fixed);
    snapshot.robots[0].joints.back().parent_link = "tool";
    EXPECT_FALSE(ValidatePhysicsLinkForceTarget(snapshot, "robot", "tool", &error));
}

class NativeSimulationWorkerTest : public testing::TestWithParam<PhysicsBackendType> {};

TEST_P(NativeSimulationWorkerTest, BuildStepCheckpointResetAndRetireOnOwningThread) {
    const auto backend = GetParam();
    if (!PhysicsServer::IsBackendAvailable(backend)) GTEST_SKIP() << "Native SDK not built";
    auto deleter = [](RigidBody3D* node) { Node::Delete(node); };
    std::unique_ptr<RigidBody3D, decltype(deleter)> root(Node::New<RigidBody3D>(), deleter);
    root->SetName("body");
    root->SetPosition({0, 0, 1});
    root->SetMass(1);
    root->SetInertiaDiagonal(Vector3::Constant(RealType(1) / 600));
    root->SetHasInertial(true);
    auto* collision = Node::New<CollisionShape3D>();
    auto shape = MakeRef<BoxShape3D>();
    shape->SetSize({0.1, 0.1, 0.1});
    collision->SetShape(shape);
    root->AddChild(collision);
    SimulationServer server(backend, false);
    server.SetFixedTimeStep(0.002);
    ASSERT_TRUE(server.SetAsyncSteppingEnabled(true));
    ASSERT_TRUE(server.BuildWorldFromScene(root.get()));
    auto wait = [&](auto predicate) {
        const auto deadline = Clock::now() + std::chrono::seconds(10);
        do {
            server.AdvanceRealtime(0);
            if (predicate()) return true;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        } while (Clock::now() < deadline);
        return false;
    };
    ASSERT_TRUE(wait([&] { return !server.IsAsyncOperationPending(); }));
    ASSERT_TRUE(server.IsWorldReady()) << server.GetLastError();
    EXPECT_EQ(server.GetWorld()->GetBackendType(), backend);
    auto sync = PhysicsServer::CreateWorld(backend, server.GetPhysicsWorldSettings());
    ASSERT_TRUE(sync->Build(server.GetWorld()->GetSceneSnapshot())) << sync->GetLastError();
    ASSERT_TRUE(server.GetWorld()->SetLinkExternalForce("body", "body", {0, 0, 1}, {1, 0, 0}));
    ASSERT_TRUE(sync->SetLinkExternalForce("body", "body", {0, 0, 1}, {1, 0, 0}));
    auto position = [&]() -> Vector3 {
        return server.CaptureStateFrame()->state.robots[0].links[0].global_transform.translation();
    };
    auto tick = [&] {
        if (!server.RequestStep() || !wait([&] { return !server.IsAsyncOperationPending(); })) return false;
        return !server.IsFaulted();
    };
    for (int i = 0; i < 5; ++i) {
        ASSERT_TRUE(tick()) << server.GetLastError();
        ASSERT_TRUE(sync->Step(0.002).completed) << sync->GetLastError();
        EXPECT_LT((position() - sync->GetSceneState().robots[0].links[0].global_transform.translation()).norm(), 1e-6);
    }
    EXPECT_LT(position().z(), 1);
    EXPECT_GT(position().x(), 0);
    ASSERT_TRUE(server.RequestCheckpoint());
    ASSERT_TRUE(wait([&] { return !server.IsAsyncOperationPending(); }));
    const auto checkpoint = server.GetCompletedCheckpoint();
    ASSERT_TRUE(checkpoint.physics.IsValid());
    for (int i = 0; i < 5; ++i) ASSERT_TRUE(tick());
    const Vector3 expected = position();
    ASSERT_TRUE(server.RequestRestoreCheckpoint(checkpoint));
    ASSERT_TRUE(wait([&] { return !server.IsAsyncOperationPending(); }));
    for (int i = 0; i < 5; ++i) ASSERT_TRUE(tick());
    EXPECT_LT((position() - expected).norm(), 1e-6);
    EXPECT_EQ(server.GetFrameCount(), 10);
    ASSERT_TRUE(server.Reset());
    ASSERT_TRUE(wait([&] { return !server.IsAsyncOperationPending(); }));
    EXPECT_NEAR(position().z(), 1, 1e-6);
    server.ClearWorld();
    ASSERT_TRUE(wait([&] { return !server.IsWorkerRetiring(); }));
    ASSERT_TRUE(server.BuildWorldFromScene(root.get()));
    ASSERT_TRUE(wait([&] { return !server.IsAsyncOperationPending(); }));
    ASSERT_TRUE(tick());
    server.ClearWorld();
    ASSERT_TRUE(wait([&] { return !server.IsWorkerRetiring(); }));
}

INSTANTIATE_TEST_SUITE_P(NativeBackends, NativeSimulationWorkerTest,
        testing::Values(PhysicsBackendType::MuJoCoCpu, PhysicsBackendType::SuperDex));
} // namespace
} // namespace gobot
