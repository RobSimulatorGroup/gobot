/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * This file is created by Qiqi Wu, 23-2-10
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/core/os/os.hpp"
#include "gobot/error_macros.hpp"
#include <chrono>

namespace gobot {

OS* OS::s_singleton = nullptr;

OS::OS() {
    s_singleton = this;
}

OS* OS::GetInstance() {
    ERR_FAIL_COND_V_MSG(s_singleton == nullptr, nullptr, "Must call this after initialize OS");
    return s_singleton;
}

OS::~OS() {
    s_singleton = nullptr;
}

MainLoop* OS::GetMainLoop() const {
    return main_loop_;
}

void OS::SetMainLoop(MainLoop* main_loop) {
    main_loop_ = main_loop;
}

void OS::DeleteMainLoop() {
    if (main_loop_) {
        Object::Delete(main_loop_);
    }
    main_loop_ = nullptr;
}


}
