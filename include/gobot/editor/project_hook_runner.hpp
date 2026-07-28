/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "gobot_export.h"

namespace gobot {

enum class ProjectHookState {
    Idle,
    Running,
    Succeeded,
    Failed,
    Cancelled,
};

struct ProjectHookSnapshot {
    ProjectHookState state{ProjectHookState::Idle};
    std::uint64_t current{0};
    std::uint64_t total{0};
    std::string message;
    std::string error;
};

struct ProjectHookOutputLine {
    std::string text;
    bool is_stderr{false};
};

// Runs editor project hooks out of process so long-running setup cannot block
// the UI or contend with Gobot's embedded Python interpreter.
class GOBOT_EXPORT ProjectHookRunner {
public:
    ProjectHookRunner();
    ~ProjectHookRunner();

    ProjectHookRunner(const ProjectHookRunner&) = delete;
    ProjectHookRunner& operator=(const ProjectHookRunner&) = delete;

    bool Start(const std::string& python_executable,
               const std::string& script_path,
               const std::string& working_directory);

    void Cancel();

    [[nodiscard]] ProjectHookSnapshot GetSnapshot() const;

    std::vector<ProjectHookOutputLine> DrainOutput();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace gobot
