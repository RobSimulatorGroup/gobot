/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * This file is created by Qiqi Wu, 23-2-10
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <chrono>

namespace gobot {

class Main {
public:
    using TimePoint = std::chrono::high_resolution_clock::time_point;

    static bool Setup(int argc, char** argv);

    static bool Setup2();

    static bool Start();

    static bool Iteration();

    static void Cleanup();

private:
    static TimePoint s_last_ticks;
};

}