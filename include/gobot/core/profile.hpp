/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#if defined(GOBOT_ENABLE_TRACY) && GOBOT_ENABLE_TRACY
#include <cstdint>
#include <type_traits>

#include <tracy/Tracy.hpp>

namespace gobot {

template <typename T>
inline void ProfilePlotValue(const char* name, T value) {
    if constexpr (std::is_integral_v<T>) {
        TracyPlot(name, static_cast<int64_t>(value));
    } else {
        TracyPlot(name, value);
    }
}

} // namespace gobot

#define GOBOT_PROFILE_ZONE(name) ZoneScopedN(name)
#define GOBOT_PROFILE_FRAME(name) FrameMarkNamed(name)
#define GOBOT_PROFILE_PLOT(name, value) ::gobot::ProfilePlotValue(name, value)

#else

#define GOBOT_PROFILE_ZONE(name) ((void)0)
#define GOBOT_PROFILE_FRAME(name) ((void)0)
#define GOBOT_PROFILE_PLOT(name, value) ((void)0)

#endif
