/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>
#include <optional>

#include "gobot/core/ref_counted.hpp"
#include "gobot/core/types.hpp"
#include "gobot/physics/physics_types.hpp"

namespace gobot {

struct ExternalSimulationCompletion {
    PhysicsStepResult step;
    std::uint64_t tick{0};
    RealType simulation_time{0};
    bool physics_step{false};
};

class GOBOT_EXPORT ExternalSimulationDriver : public RefCounted {
    GOBCLASS(ExternalSimulationDriver, RefCounted)

public:
    ~ExternalSimulationDriver() override = default;

    virtual bool Step(RealType fixed_delta) = 0;

    // Legacy drivers promise all-or-nothing advancement. Drivers with substeps
    // must override this contract to report partial progress on failure.
    virtual PhysicsStepResult StepWithResult(RealType fixed_delta) {
        const bool completed = Step(fixed_delta);
        return {.completed = completed,
                .advanced_time = completed ? fixed_delta : RealType(0),
                .error = completed ? std::string{} : GetLastError()};
    }

    virtual bool Reset() = 0;

    // Asynchronous drivers only submit/poll on the scene owner thread. SDK
    // execution belongs to their runtime worker. No result means still busy.
    virtual bool IsAsynchronous() const { return false; }
    virtual bool CanRequestStep() { return false; }
    virtual std::optional<ExternalSimulationCompletion> PollCompletion() { return std::nullopt; }

    virtual bool SyncScene() = 0;

    virtual void Close() = 0;

    virtual const std::string& GetLastError() const = 0;
};

} // namespace gobot
