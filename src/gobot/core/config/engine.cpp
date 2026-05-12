/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * This file is created by Qiqi Wu, 23-2-18
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/core/config/engine.hpp"
#include "gobot/error_macros.hpp"

namespace gobot {

Engine* Engine::s_singleton = nullptr;


Engine::Engine() {
    s_singleton = this;
}

Engine::~Engine() {
    s_singleton = nullptr;
}

VersionInfo Engine::GetVersionInfo() const {
    // TODO(wqq)
    return {};
}


Engine *Engine::GetInstance() {
    ERR_FAIL_COND_V_MSG(s_singleton == nullptr, nullptr, "Must call this after initialize Engine");
    return s_singleton;
}

}
