/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>

#include "gobot/core/ref_counted.hpp"
#include "gobot/core/types.hpp"

namespace gobot {

class GOBOT_EXPORT ExternalSimulationDriver : public RefCounted {
    GOBCLASS(ExternalSimulationDriver, RefCounted)

public:
    ~ExternalSimulationDriver() override = default;

    virtual bool Step(RealType fixed_delta) = 0;

    virtual bool Reset() = 0;

    virtual bool SyncScene() = 0;

    virtual void Close() = 0;

    virtual const std::string& GetLastError() const = 0;
};

} // namespace gobot
