#include <gtest/gtest.h>
#include <future>
#include "gobot/simulation/simulation_data_runtime.hpp"

namespace gobot {
namespace {
using Request = SimulationDataRuntime::Request;
using Operation = SimulationDataRuntime::Operation;

struct Probe {
    std::thread::id built, advanced, closed, destroyed;
    std::promise<void> entered;
    std::shared_future<void> release;
    bool block{false};
    bool malformed_snapshot{false};
};
class DataExecutor : public SimulationDataExecutor {
public:
    explicit DataExecutor(std::shared_ptr<Probe> probe) : probe_(std::move(probe)) {
        probe_->built = std::this_thread::get_id();
    }
    ~DataExecutor() override { probe_->destroyed = std::this_thread::get_id(); }
    std::vector<SimulationMicrostepResult> Advance(double dt) override {
        probe_->advanced = std::this_thread::get_id();
        if (probe_->block) {
            probe_->entered.set_value();
            probe_->release.wait();
            probe_->block = false;
        }
        ++value_;
        return {{true, dt}};
    }
    void Reset(const std::vector<std::size_t>&) override { value_ = 0; }
    void Close() override { probe_->closed = std::this_thread::get_id(); }
    void ApplyCommands(const std::vector<SimulationBuffer>&) override {}
    void CaptureSnapshot(const SimulationSubscription&, std::vector<SimulationBuffer>& buffers) override {
        buffers = {{"position", {1}, {double(value_)}}};
        if (probe_->malformed_snapshot) buffers.front().shape = {1, 2};
    }
private:
    std::shared_ptr<Probe> probe_;
    int value_{0};
};

Request Install(const std::shared_ptr<Probe>& probe) {
    Request request;
    request.operation = Operation::Install;
    request.epoch = 1;
    request.subscription.interval_seconds = 0;
    request.subscription.fields = {"position"};
    request.factory = [probe] { return SimulationExecutorInstallation{
            std::make_shared<DataExecutor>(probe), 1, .002, 1}; };
    return request;
}
SimulationDataRuntime::Completion Await(SimulationDataWorker& worker) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (std::chrono::steady_clock::now() < deadline) {
        if (auto result = worker.Poll()) return std::move(*result);
        std::this_thread::yield();
    }
    throw std::runtime_error("Worker completion timeout");
}
}

TEST(SimulationDataWorkerContract, LeasedFramesAreImmutableAndFullPoolOnlySkipsPresentation) {
    auto probe = std::make_shared<Probe>();
    SimulationDataWorker worker;
    ASSERT_TRUE(worker.Submit(Install(probe)));
    auto first = Await(worker).snapshot;
    ASSERT_TRUE(first);
    ASSERT_TRUE(worker.Submit({.epoch = 1}));
    auto second = Await(worker).snapshot;
    ASSERT_TRUE(worker.Submit({.epoch = 1}));
    auto third = Await(worker).snapshot;
    ASSERT_TRUE(worker.Submit({.epoch = 1}));
    auto full = Await(worker);
    EXPECT_TRUE(full.step.Completed());
    EXPECT_TRUE(full.snapshot_skipped);
    EXPECT_FALSE(full.snapshot);
    EXPECT_EQ(full.clocks[0].tick, 3);
    EXPECT_EQ(first->buffers[0].values[0], 0);
    EXPECT_EQ(second->buffers[0].values[0], 1);
    EXPECT_EQ(third->buffers[0].values[0], 2);
    second.reset();
    ASSERT_TRUE(worker.Submit({.epoch = 1}));
    const auto reused = Await(worker);
    ASSERT_TRUE(reused.snapshot);
    EXPECT_EQ(reused.snapshot->buffers[0].values[0], 4);
    worker.Shutdown();
    EXPECT_EQ(first->buffers[0].values[0], 0);
    EXPECT_EQ(probe->built, probe->advanced);
    EXPECT_EQ(probe->built, probe->closed);
    EXPECT_EQ(probe->built, probe->destroyed);
    EXPECT_NE(probe->built, std::this_thread::get_id());
}

TEST(SimulationDataWorkerContract, ResetDuringStepPublishesOnlyNewEpoch) {
    auto probe = std::make_shared<Probe>();
    std::promise<void> release;
    probe->release = release.get_future().share();
    probe->block = true;
    auto entered = probe->entered.get_future();
    SimulationDataWorker worker;
    ASSERT_TRUE(worker.Submit(Install(probe)));
    Await(worker);
    ASSERT_TRUE(worker.Submit({.epoch = 1}));
    const auto ready = entered.wait_for(std::chrono::seconds(3));
    // Always unblock before assertions that could unwind and join the worker.
    const bool accepted = worker.RequestControl({.operation = Operation::Reset, .epoch = 2});
    release.set_value();
    ASSERT_EQ(ready, std::future_status::ready);
    ASSERT_TRUE(accepted);
    auto reset = Await(worker);
    ASSERT_EQ(reset.operation, Operation::Reset);
    EXPECT_EQ(reset.epoch, 2);
    EXPECT_EQ(reset.clocks[0].tick, 0);
    EXPECT_EQ(reset.clocks[0].episode, 1);
    EXPECT_EQ(reset.snapshot->buffers[0].values[0], 0);
    EXPECT_TRUE(worker.CanSubmit());
}

TEST(SimulationDataWorkerContract, InvalidSnapshotDoesNotEraseCommittedPhysics) {
    auto probe = std::make_shared<Probe>();
    SimulationDataWorker worker;
    ASSERT_TRUE(worker.Submit(Install(probe)));
    const auto installed = Await(worker);
    EXPECT_EQ(installed.info.environment_count, 1);
    EXPECT_DOUBLE_EQ(installed.info.fixed_time_step, .002);
    probe->malformed_snapshot = true;
    ASSERT_TRUE(worker.Submit({.epoch = 1}));
    const auto result = Await(worker);
    EXPECT_TRUE(result.step.Completed());
    EXPECT_EQ(result.clocks[0].tick, 1);
    EXPECT_FALSE(result.clocks[0].faulted);
    EXPECT_FALSE(result.snapshot);
    EXPECT_NE(result.presentation_error.find("storage"), std::string::npos);
    worker.Shutdown();
}

TEST(SimulationDataWorkerContract, RetirementDiscardsBlockedStepWithoutJoiningCaller) {
    auto probe = std::make_shared<Probe>();
    std::promise<void> release;
    probe->release = release.get_future().share();
    probe->block = true;
    auto entered = probe->entered.get_future();
    SimulationDataWorker worker;
    ASSERT_TRUE(worker.Submit(Install(probe)));
    Await(worker);
    ASSERT_TRUE(worker.Submit({.epoch = 1}));
    const auto ready = entered.wait_for(std::chrono::seconds(3));
    worker.Retire();
    EXPECT_FALSE(worker.Poll());
    EXPECT_FALSE(worker.CanSubmit());
    release.set_value();
    ASSERT_EQ(ready, std::future_status::ready);
    worker.Shutdown();
    EXPECT_FALSE(worker.Poll());
    EXPECT_EQ(probe->built, probe->destroyed);
}
} // namespace gobot
