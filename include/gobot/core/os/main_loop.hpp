/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * This file is created by Qiqi Wu, 23-2-10
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "gobot/core/object.hpp"

namespace gobot {

class GOBOT_EXPORT MainLoop : public Object {
    GOBCLASS(MainLoop, Object)
public:
    MainLoop() = default;

    ~MainLoop() override = default;

    virtual void Initialize() {};

    virtual bool PhysicsProcess(double time) {return false; };

    virtual bool Process(double time) { return false; };

    virtual void PullEvent() {};

    virtual void Finalize() {};
};



}