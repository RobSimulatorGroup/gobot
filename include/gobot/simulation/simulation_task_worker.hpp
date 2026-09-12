#pragma once

#include <condition_variable>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <utility>

namespace gobot {

// Shared mailbox for native and provider runtimes. Runtime construction,
// execution and retirement all take place on one thread. Runtime supplies
// Request, Completion, IsInstall/IsControl, Execute, IsInstalled and Retire.
// Execute must turn runtime failures into a completion; Retire must not throw.
template <class Runtime>
class SimulationTaskWorker {
public:
    using Request = typename Runtime::Request;
    using Completion = typename Runtime::Completion;

    SimulationTaskWorker() : thread_([this] { Run(); }) {}
    ~SimulationTaskWorker() { Shutdown(); }
    SimulationTaskWorker(const SimulationTaskWorker&) = delete;
    SimulationTaskWorker& operator=(const SimulationTaskWorker&) = delete;

    bool CanInstall() const {
        std::lock_guard lock(mutex_);
        return !shutdown_ && !installed_ && Idle();
    }
    bool CanSubmit() const {
        std::lock_guard lock(mutex_);
        return !shutdown_ && installed_ && Idle();
    }
    bool IsPending() const {
        std::lock_guard lock(mutex_);
        return !Idle();
    }
    bool Submit(Request request) {
        std::lock_guard lock(mutex_);
        if (shutdown_ || !Idle() || (Runtime::IsInstall(request) ? installed_ : !installed_)) return false;
        request_ = std::move(request);
        wake_.notify_one();
        return true;
    }
    bool RequestControl(Request request) {
        if (!Runtime::IsControl(request)) return false;
        std::lock_guard lock(mutex_);
        if (shutdown_ || retiring_ || (!installed_ && !busy_ && !request_) || control_) return false;
        completion_.reset();
        control_ = std::move(request);
        wake_.notify_one();
        return true;
    }
    void Retire() {
        std::lock_guard lock(mutex_);
        if (shutdown_) return;
        retiring_ = true;
        completion_.reset();
        wake_.notify_one();
    }
    std::optional<Completion> Poll() {
        std::lock_guard lock(mutex_);
        auto result = std::move(completion_);
        completion_.reset();
        return result;
    }
    // Only process shutdown joins. UI Stop uses Retire(). Call on the owner
    // thread, releasing the Python GIL before joining a provider runtime.
    void Shutdown() {
        {
            std::lock_guard lock(mutex_);
            shutdown_ = true;
        }
        wake_.notify_one();
        if (thread_.joinable()) thread_.join();
    }

private:
    bool Idle() const { return !busy_ && !retiring_ && !request_ && !control_ && !completion_; }
    void Run() {
        Runtime runtime;
        std::unique_lock lock(mutex_);
        for (;;) {
            wake_.wait(lock, [&] { return shutdown_ || retiring_ || request_ || control_; });
            if (shutdown_ || retiring_) {
                const bool exit = shutdown_;
                auto abandoned = std::move(request_);
                auto abandoned_control = std::move(control_);
                request_.reset();
                control_.reset();
                completion_.reset();
                busy_ = true;
                lock.unlock();
                abandoned.reset();
                abandoned_control.reset();
                runtime.Retire();
                lock.lock();
                installed_ = false;
                retiring_ = false;
                busy_ = false;
                if (exit) return;
                continue;
            }
            // A reset queued immediately after Install must not run first.
            const bool install_pending = request_ && Runtime::IsInstall(*request_);
            Request job = control_ && !install_pending ? std::move(*control_) : std::move(*request_);
            if (control_ && !install_pending) {
                control_.reset();
                request_.reset();
            } else {
                request_.reset();
            }
            busy_ = true;
            lock.unlock();
            auto output = runtime.Execute(job);
            job = {};
            const bool installed = runtime.IsInstalled();
            lock.lock();
            installed_ = installed;
            busy_ = false;
            if (!retiring_ && !shutdown_ && !control_) completion_ = std::move(output);
        }
    }

    mutable std::mutex mutex_;
    std::condition_variable wake_;
    bool shutdown_{false};
    bool retiring_{false};
    bool busy_{false};
    bool installed_{false};
    std::optional<Request> request_;
    std::optional<Request> control_;
    std::optional<Completion> completion_;
    std::thread thread_;
};

} // namespace gobot
