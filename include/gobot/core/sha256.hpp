/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <string_view>

#include "gobot/core/macros.hpp"

namespace gobot {

GOBOT_EXPORT std::string Sha256Hex(std::span<const std::uint8_t> data);

GOBOT_EXPORT std::string Sha256Hex(std::string_view data);

inline std::string Sha256Digest(std::span<const std::uint8_t> data) {
    return "sha256:" + Sha256Hex(data);
}

inline std::string Sha256Digest(std::string_view data) {
    return "sha256:" + Sha256Hex(data);
}

} // namespace gobot
