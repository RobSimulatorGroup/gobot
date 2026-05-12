/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * This file is created by Qiqi Wu, 23-2-18
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/platform/linux/os_linux.hpp"
#include "gobot/scene/window.hpp"
#include "gobot/main/main.hpp"
#include "gobot/log.hpp"

#include <csignal>

namespace gobot {
namespace {

volatile std::sig_atomic_t g_interrupt_requested = 0;

void HandleInterruptSignal(int) {
    g_interrupt_requested = 1;
}

} // namespace

void LinuxOS::Run()
{
    if (!main_loop_) {
        return;
    }

    g_interrupt_requested = 0;
    auto* previous_sigint_handler = std::signal(SIGINT, HandleInterruptSignal);
    auto* previous_sigterm_handler = std::signal(SIGTERM, HandleInterruptSignal);

    main_loop_->Initialize();

    while (!g_interrupt_requested) {
        OS::GetInstance()->GetMainLoop()->PullEvent();
        if (Main::Iteration()) {
            break;
        }
    }

    main_loop_->Finalize();

    if (previous_sigint_handler != SIG_ERR) {
        std::signal(SIGINT, previous_sigint_handler);
    }
    if (previous_sigterm_handler != SIG_ERR) {
        std::signal(SIGTERM, previous_sigterm_handler);
    }
}



}
