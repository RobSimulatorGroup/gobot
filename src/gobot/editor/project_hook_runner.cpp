/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/editor/project_hook_runner.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <mutex>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <string_view>
#include <sys/types.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

#include "gobot/core/types.hpp"

extern char** environ;

namespace gobot {
namespace {

constexpr std::string_view kProgressPrefix = "GOBOT_PROGRESS ";
constexpr std::size_t kMaxOutputLineBytes = 16u * 1024u;
constexpr std::size_t kMaxQueuedOutputLines = 256u;
constexpr std::size_t kReadBudgetBytes = 64u * 1024u;

void CloseFileDescriptor(int& descriptor) {
    if (descriptor >= 0) {
        close(descriptor);
        descriptor = -1;
    }
}

void SetNonBlocking(int descriptor) {
    const int flags = fcntl(descriptor, F_GETFL, 0);
    if (flags >= 0) {
        fcntl(descriptor, F_SETFL, flags | O_NONBLOCK);
    }
}

std::string ExitError(int status) {
    if (WIFEXITED(status)) {
        return "Project load hook exited with code " + std::to_string(WEXITSTATUS(status)) + ".";
    }
    if (WIFSIGNALED(status)) {
        return "Project load hook was terminated by signal " + std::to_string(WTERMSIG(status)) + ".";
    }
    return "Project load hook did not exit normally.";
}

bool IsHookEnvironmentEntry(std::string_view entry) {
    const std::size_t separator = entry.find('=');
    const std::string_view key = entry.substr(0, separator);
    return key == "GOBOT_PROJECT_HOOK" ||
           key == "GOBOT_PROJECT_DIR" ||
           key == "PYTHONUNBUFFERED";
}

std::vector<std::string> BuildHookEnvironment(const std::string& working_directory) {
    std::vector<std::string> environment;
    for (char** entry = ::environ; entry != nullptr && *entry != nullptr; ++entry) {
        if (!IsHookEnvironmentEntry(*entry)) {
            environment.emplace_back(*entry);
        }
    }
    environment.emplace_back("GOBOT_PROJECT_HOOK=1");
    environment.emplace_back("GOBOT_PROJECT_DIR=" + working_directory);
    environment.emplace_back("PYTHONUNBUFFERED=1");
    return environment;
}

} // namespace

class ProjectHookRunner::Impl {
public:
    ~Impl() {
        Cancel();
        JoinWorker();
    }

    bool Start(const std::string& python_executable,
               const std::string& script_path,
               const std::string& working_directory) {
        {
            std::scoped_lock lock(mutex_);
            if (snapshot_.state == ProjectHookState::Running) {
                return false;
            }
        }
        JoinWorker();

        {
            std::scoped_lock lock(mutex_);
            snapshot_ = {};
            output_.clear();
            last_stderr_.clear();
            output_truncated_ = false;
        }
        cancel_requested_.store(false);

        int stdout_pipe[2]{-1, -1};
        int stderr_pipe[2]{-1, -1};
        if (pipe2(stdout_pipe, O_CLOEXEC) != 0 || pipe2(stderr_pipe, O_CLOEXEC) != 0) {
            const std::string error = std::string("Failed to create project hook output pipes: ") +
                                      std::strerror(errno);
            CloseFileDescriptor(stdout_pipe[0]);
            CloseFileDescriptor(stdout_pipe[1]);
            CloseFileDescriptor(stderr_pipe[0]);
            CloseFileDescriptor(stderr_pipe[1]);
            SetStartFailure(error);
            return false;
        }

        posix_spawn_file_actions_t actions;
        int spawn_error = posix_spawn_file_actions_init(&actions);
        const bool actions_initialized = spawn_error == 0;
        if (spawn_error == 0) {
            spawn_error = posix_spawn_file_actions_addopen(
                    &actions, STDIN_FILENO, "/dev/null", O_RDONLY, 0);
        }
        if (spawn_error == 0) {
            spawn_error = posix_spawn_file_actions_adddup2(&actions, stdout_pipe[1], STDOUT_FILENO);
        }
        if (spawn_error == 0) {
            spawn_error = posix_spawn_file_actions_adddup2(&actions, stderr_pipe[1], STDERR_FILENO);
        }
        if (spawn_error == 0) {
            spawn_error = posix_spawn_file_actions_addclose(&actions, stdout_pipe[0]);
        }
        if (spawn_error == 0) {
            spawn_error = posix_spawn_file_actions_addclose(&actions, stderr_pipe[0]);
        }
        if (spawn_error == 0) {
            spawn_error = posix_spawn_file_actions_addclose(&actions, stdout_pipe[1]);
        }
        if (spawn_error == 0) {
            spawn_error = posix_spawn_file_actions_addclose(&actions, stderr_pipe[1]);
        }
        if (spawn_error == 0 && !working_directory.empty()) {
            spawn_error = posix_spawn_file_actions_addchdir_np(&actions, working_directory.c_str());
        }

        posix_spawnattr_t attributes;
        const int attribute_init_error = posix_spawnattr_init(&attributes);
        const bool attributes_initialized = attribute_init_error == 0;
        if (spawn_error == 0) {
            spawn_error = attribute_init_error;
        }
        if (spawn_error == 0) {
            spawn_error = posix_spawnattr_setflags(&attributes, POSIX_SPAWN_SETPGROUP);
        }
        if (spawn_error == 0) {
            // A zero pgroup asks posix_spawn to create a process group whose
            // ID is the hook child's PID. Descendants inherit that group.
            spawn_error = posix_spawnattr_setpgroup(&attributes, 0);
        }

        std::vector<std::string> environment = BuildHookEnvironment(working_directory);
        std::vector<char*> environment_pointers;
        environment_pointers.reserve(environment.size() + 1);
        for (std::string& entry : environment) {
            environment_pointers.push_back(entry.data());
        }
        environment_pointers.push_back(nullptr);

        std::string unbuffered_flag = "-u";
        std::vector<char*> arguments{
                const_cast<char*>(python_executable.c_str()),
                unbuffered_flag.data(),
                const_cast<char*>(script_path.c_str()),
                nullptr,
        };
        pid_t child = -1;
        if (spawn_error == 0) {
            spawn_error = posix_spawnp(&child,
                                       python_executable.c_str(),
                                       &actions,
                                       &attributes,
                                       arguments.data(),
                                       environment_pointers.data());
        }
        if (attributes_initialized) {
            posix_spawnattr_destroy(&attributes);
        }
        if (actions_initialized) {
            posix_spawn_file_actions_destroy(&actions);
        }

        if (spawn_error != 0) {
            const std::string error = std::string("Failed to start project load hook: ") +
                                      std::strerror(spawn_error);
            CloseFileDescriptor(stdout_pipe[0]);
            CloseFileDescriptor(stdout_pipe[1]);
            CloseFileDescriptor(stderr_pipe[0]);
            CloseFileDescriptor(stderr_pipe[1]);
            SetStartFailure(error);
            return false;
        }

        CloseFileDescriptor(stdout_pipe[1]);
        CloseFileDescriptor(stderr_pipe[1]);
        SetNonBlocking(stdout_pipe[0]);
        SetNonBlocking(stderr_pipe[0]);

        {
            std::scoped_lock lock(mutex_);
            child_pid_ = child;
            snapshot_.state = ProjectHookState::Running;
            snapshot_.message = "Running project setup...";
        }
        worker_ = std::thread([this, stdout_fd = stdout_pipe[0], stderr_fd = stderr_pipe[0]]() mutable {
            MonitorChild(stdout_fd, stderr_fd);
        });
        return true;
    }

    void Cancel() {
        cancel_requested_.store(true);
        {
            std::scoped_lock lock(mutex_);
            if (snapshot_.state == ProjectHookState::Running) {
                snapshot_.message = "Cancelling project setup...";
            }
        }
    }

    ProjectHookSnapshot GetSnapshot() const {
        std::scoped_lock lock(mutex_);
        return snapshot_;
    }

    std::vector<ProjectHookOutputLine> DrainOutput() {
        std::scoped_lock lock(mutex_);
        std::vector<ProjectHookOutputLine> result;
        result.swap(output_);
        return result;
    }

private:
    void SetStartFailure(std::string error) {
        std::scoped_lock lock(mutex_);
        child_pid_ = -1;
        snapshot_.state = ProjectHookState::Failed;
        snapshot_.message = "Project setup could not start.";
        snapshot_.error = std::move(error);
    }

    void JoinWorker() {
        if (worker_.joinable()) {
            worker_.join();
        }
    }

    void AddOutput(std::string line, bool is_stderr) {
        if (line.empty()) {
            return;
        }
        if (line.size() > kMaxOutputLineBytes) {
            line.resize(kMaxOutputLineBytes);
            line += "... [line truncated]";
        }
        std::scoped_lock lock(mutex_);
        if (is_stderr) {
            last_stderr_ = line;
        }
        if (output_.size() >= kMaxQueuedOutputLines) {
            if (!output_truncated_) {
                output_.back() = {
                        "Project hook output limit reached; additional output was discarded.",
                        true,
                };
                output_truncated_ = true;
            }
            return;
        }
        output_.push_back({std::move(line), is_stderr});
    }

    bool ParseProgress(const std::string& line) {
        if (!line.starts_with(kProgressPrefix)) {
            return false;
        }

        try {
            const Json json = Json::parse(line.substr(kProgressPrefix.size()));
            if (!json.is_object()) {
                return false;
            }

            std::scoped_lock lock(mutex_);
            if (json.contains("current") && json["current"].is_number_unsigned()) {
                snapshot_.current = json["current"].get<std::uint64_t>();
            }
            if (json.contains("total") && json["total"].is_number_unsigned()) {
                snapshot_.total = json["total"].get<std::uint64_t>();
            }
            if (json.contains("message") && json["message"].is_string()) {
                snapshot_.message = json["message"].get<std::string>();
            }
            return true;
        } catch (const std::exception&) {
            return false;
        }
    }

    void ConsumeLines(std::string& pending, const char* data, std::size_t size, bool is_stderr) {
        pending.append(data, size);
        std::size_t line_end = 0;
        while ((line_end = pending.find_first_of("\r\n")) != std::string::npos) {
            std::string line = pending.substr(0, line_end);
            std::size_t next = line_end + 1;
            while (next < pending.size() &&
                   (pending[next] == '\r' || pending[next] == '\n')) {
                ++next;
            }
            pending.erase(0, next);

            if (!is_stderr && ParseProgress(line)) {
                continue;
            }
            AddOutput(std::move(line), is_stderr);
        }

        if (pending.size() > kMaxOutputLineBytes) {
            std::string truncated = pending.substr(0, kMaxOutputLineBytes);
            truncated += "... [line truncated]";
            AddOutput(std::move(truncated), is_stderr);
            pending.clear();
        }
    }

    void DrainDescriptor(int& descriptor, std::string& pending, bool is_stderr) {
        char buffer[4096];
        std::size_t remaining_budget = kReadBudgetBytes;
        while (descriptor >= 0 && remaining_budget > 0) {
            const std::size_t requested = std::min(sizeof(buffer), remaining_budget);
            const ssize_t count = read(descriptor, buffer, requested);
            if (count > 0) {
                ConsumeLines(pending, buffer, static_cast<std::size_t>(count), is_stderr);
                remaining_budget -= static_cast<std::size_t>(count);
                continue;
            }
            if (count == 0) {
                CloseFileDescriptor(descriptor);
                break;
            }
            if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) {
                break;
            }
            CloseFileDescriptor(descriptor);
            break;
        }
    }

    void FlushPending(std::string& pending, bool is_stderr) {
        if (pending.empty()) {
            return;
        }
        if (!is_stderr && ParseProgress(pending)) {
            pending.clear();
            return;
        }
        AddOutput(std::move(pending), is_stderr);
        pending.clear();
    }

    void MonitorChild(int stdout_fd, int stderr_fd) {
        std::string stdout_pending;
        std::string stderr_pending;
        int child_status = 0;
        bool child_reaped = false;
        bool sent_sigterm = false;
        bool sent_sigkill = false;
        std::chrono::steady_clock::time_point cancel_time{};
        std::chrono::steady_clock::time_point child_exit_time{};

        while (!child_reaped || stdout_fd >= 0 || stderr_fd >= 0) {
            pollfd descriptors[2]{};
            nfds_t count = 0;
            if (stdout_fd >= 0) {
                descriptors[count++] = {stdout_fd, POLLIN | POLLHUP | POLLERR, 0};
            }
            if (stderr_fd >= 0) {
                descriptors[count++] = {stderr_fd, POLLIN | POLLHUP | POLLERR, 0};
            }
            if (count > 0) {
                poll(descriptors, count, 100);
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }

            DrainDescriptor(stdout_fd, stdout_pending, false);
            DrainDescriptor(stderr_fd, stderr_pending, true);

            pid_t child = -1;
            {
                std::scoped_lock lock(mutex_);
                child = child_pid_;
            }

            // The monitor thread exclusively signals and reaps the process,
            // preventing Cancel from racing PID reuse after waitpid.
            if (cancel_requested_.load() && child > 0 && !sent_sigterm) {
                kill(-child, SIGTERM);
                sent_sigterm = true;
                cancel_time = std::chrono::steady_clock::now();
            }

            if (!child_reaped) {
                const pid_t result = waitpid(child, &child_status, WNOHANG);
                if (result == child) {
                    child_reaped = true;
                    child_exit_time = std::chrono::steady_clock::now();
                } else if (result < 0 && errno != EINTR) {
                    child_reaped = true;
                    child_status = 1 << 8;
                    child_exit_time = std::chrono::steady_clock::now();
                }

                if (sent_sigterm && !sent_sigkill && !child_reaped &&
                    std::chrono::steady_clock::now() - cancel_time >
                            std::chrono::seconds(2)) {
                    kill(-child, SIGKILL);
                    sent_sigkill = true;
                }
            }

            // If the direct hook exits after SIGTERM, terminate any descendant
            // that kept the process group (for example curl or aria2).
            if (child_reaped && sent_sigterm && !sent_sigkill && child > 0) {
                kill(-child, SIGKILL);
                sent_sigkill = true;
            }

            // A hook may launch a detached child that inherits its output
            // descriptors. Do not let that unrelated process keep the editor
            // waiting indefinitely after the hook itself has exited.
            if (child_reaped && child_exit_time.time_since_epoch().count() != 0 &&
                std::chrono::steady_clock::now() - child_exit_time >
                        std::chrono::milliseconds(250)) {
                DrainDescriptor(stdout_fd, stdout_pending, false);
                DrainDescriptor(stderr_fd, stderr_pending, true);
                CloseFileDescriptor(stdout_fd);
                CloseFileDescriptor(stderr_fd);
            }
        }

        FlushPending(stdout_pending, false);
        FlushPending(stderr_pending, true);
        CloseFileDescriptor(stdout_fd);
        CloseFileDescriptor(stderr_fd);

        std::scoped_lock lock(mutex_);
        child_pid_ = -1;
        if (cancel_requested_.load()) {
            snapshot_.state = ProjectHookState::Cancelled;
            snapshot_.message = "Project setup was cancelled.";
            snapshot_.error.clear();
        } else if (WIFEXITED(child_status) && WEXITSTATUS(child_status) == 0) {
            snapshot_.state = ProjectHookState::Succeeded;
            snapshot_.message = "Project setup complete.";
            if (snapshot_.total > 0) {
                snapshot_.current = snapshot_.total;
            }
            snapshot_.error.clear();
        } else {
            snapshot_.state = ProjectHookState::Failed;
            snapshot_.message = "Project setup failed.";
            snapshot_.error = ExitError(child_status);
            if (!last_stderr_.empty()) {
                snapshot_.error += "\n" + last_stderr_;
            }
        }
    }

    mutable std::mutex mutex_;
    ProjectHookSnapshot snapshot_;
    std::vector<ProjectHookOutputLine> output_;
    std::string last_stderr_;
    bool output_truncated_{false};
    std::thread worker_;
    std::atomic_bool cancel_requested_{false};
    pid_t child_pid_{-1};
};

ProjectHookRunner::ProjectHookRunner() : impl_(std::make_unique<Impl>()) {}

ProjectHookRunner::~ProjectHookRunner() = default;

bool ProjectHookRunner::Start(const std::string& python_executable,
                              const std::string& script_path,
                              const std::string& working_directory) {
    return impl_->Start(python_executable, script_path, working_directory);
}

void ProjectHookRunner::Cancel() {
    impl_->Cancel();
}

ProjectHookSnapshot ProjectHookRunner::GetSnapshot() const {
    return impl_->GetSnapshot();
}

std::vector<ProjectHookOutputLine> ProjectHookRunner::DrainOutput() {
    return impl_->DrainOutput();
}

} // namespace gobot
