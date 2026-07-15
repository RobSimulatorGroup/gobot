/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

#include "gobot/core/macros.hpp"

namespace gobot {

/// PCG64 generator with NumPy SeedSequence, uniform, and bounded-integer semantics.
class GOBOT_EXPORT Pcg64Random {
public:
    explicit Pcg64Random(std::uint64_t seed);

    std::uint64_t NextUInt64();

    std::uint32_t NextUInt32();

    double Uniform();

    double Uniform(double lower, double upper);

    /// Returns an unbiased integer in [0, bound_exclusive).
    std::uint32_t BoundedUInt32(std::uint32_t bound_exclusive);

private:
    struct UInt128 {
        std::uint64_t high{0};
        std::uint64_t low{0};
    };

    static UInt128 MultiplyAdd(UInt128 value, UInt128 multiplier, UInt128 increment);

    void Advance();

    UInt128 state_;
    UInt128 increment_;
    bool has_buffered_uint32_{false};
    std::uint32_t buffered_uint32_{0};
};

} // namespace gobot
