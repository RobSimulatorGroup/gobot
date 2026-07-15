/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/core/math/pcg_random.hpp"

#include <array>
#include <bit>
#include <limits>

namespace gobot {
namespace {

constexpr std::uint32_t kSeedInitA = 0x43b0d7e5U;
constexpr std::uint32_t kSeedMultiplierA = 0x931e8875U;
constexpr std::uint32_t kSeedInitB = 0x8b51f9ddU;
constexpr std::uint32_t kSeedMultiplierB = 0x58f38dedU;
constexpr std::uint32_t kSeedMixMultiplierLeft = 0xca01f9ddU;
constexpr std::uint32_t kSeedMixMultiplierRight = 0x4973f715U;
constexpr int kSeedShift = 16;

constexpr std::uint64_t kPcgMultiplierHigh = 0x2360ed051fc65da4ULL;
constexpr std::uint64_t kPcgMultiplierLow = 0x4385df649fccf645ULL;

std::array<std::uint32_t, 8> GenerateSeedState(std::uint64_t seed) {
    // This is the fixed-pool SeedSequence mixer used by NumPy. Keeping the
    // mixer here makes authored procedural assets independent of Python while
    // preserving the reference generator's seed contract.
    std::array<std::uint32_t, 4> entropy{
            static_cast<std::uint32_t>(seed),
            static_cast<std::uint32_t>(seed >> 32U),
            0U,
            0U,
    };
    std::array<std::uint32_t, 4> pool{};
    std::uint32_t hash_constant = kSeedInitA;
    const auto hash = [&hash_constant](std::uint32_t value) {
        value ^= hash_constant;
        hash_constant *= kSeedMultiplierA;
        value *= hash_constant;
        return value ^ (value >> kSeedShift);
    };
    const auto mix = [](std::uint32_t left, std::uint32_t right) {
        std::uint32_t value = kSeedMixMultiplierLeft * left -
                              kSeedMixMultiplierRight * right;
        return value ^ (value >> kSeedShift);
    };

    for (std::size_t index = 0; index < pool.size(); ++index) {
        pool[index] = hash(entropy[index]);
    }
    for (std::size_t source = 0; source < pool.size(); ++source) {
        for (std::size_t destination = 0; destination < pool.size(); ++destination) {
            if (source != destination) {
                pool[destination] = mix(pool[destination], hash(pool[source]));
            }
        }
    }

    std::array<std::uint32_t, 8> output{};
    hash_constant = kSeedInitB;
    for (std::size_t index = 0; index < output.size(); ++index) {
        std::uint32_t value = pool[index % pool.size()] ^ hash_constant;
        hash_constant *= kSeedMultiplierB;
        value *= hash_constant;
        output[index] = value ^ (value >> kSeedShift);
    }
    return output;
}

std::uint64_t PackLittleEndian(std::uint32_t low, std::uint32_t high) {
    return static_cast<std::uint64_t>(low) |
           (static_cast<std::uint64_t>(high) << 32U);
}

std::uint64_t MultiplyHigh(std::uint64_t left, std::uint64_t right) {
    constexpr std::uint64_t mask = 0xffffffffULL;
    const std::uint64_t left_low = left & mask;
    const std::uint64_t left_high = left >> 32U;
    const std::uint64_t right_low = right & mask;
    const std::uint64_t right_high = right >> 32U;

    const std::uint64_t low_product = left_low * right_low;
    std::uint64_t middle = left_high * right_low + (low_product >> 32U);
    const std::uint64_t middle_low = middle & mask;
    const std::uint64_t middle_high = middle >> 32U;
    middle = left_low * right_high + middle_low;
    return left_high * right_high + middle_high + (middle >> 32U);
}

} // namespace

Pcg64Random::Pcg64Random(std::uint64_t seed) {
    const std::array<std::uint32_t, 8> words = GenerateSeedState(seed);
    const std::array<std::uint64_t, 4> generated{
            PackLittleEndian(words[0], words[1]),
            PackLittleEndian(words[2], words[3]),
            PackLittleEndian(words[4], words[5]),
            PackLittleEndian(words[6], words[7]),
    };
    const UInt128 initial_state{generated[0], generated[1]};
    const UInt128 initial_sequence{generated[2], generated[3]};
    increment_.high = (initial_sequence.high << 1U) |
                      (initial_sequence.low >> 63U);
    increment_.low = (initial_sequence.low << 1U) | 1U;

    state_ = {};
    Advance();
    const std::uint64_t previous_low = state_.low;
    state_.low += initial_state.low;
    state_.high += initial_state.high + static_cast<std::uint64_t>(state_.low < previous_low);
    Advance();
}

Pcg64Random::UInt128 Pcg64Random::MultiplyAdd(UInt128 value,
                                              UInt128 multiplier,
                                              UInt128 increment) {
    const std::uint64_t low = value.low * multiplier.low;
    std::uint64_t high = MultiplyHigh(value.low, multiplier.low);
    high += value.low * multiplier.high;
    high += value.high * multiplier.low;

    UInt128 result{high, low};
    const std::uint64_t previous_low = result.low;
    result.low += increment.low;
    result.high += increment.high + static_cast<std::uint64_t>(result.low < previous_low);
    return result;
}

void Pcg64Random::Advance() {
    state_ = MultiplyAdd(state_,
                         {kPcgMultiplierHigh, kPcgMultiplierLow},
                         increment_);
}

std::uint64_t Pcg64Random::NextUInt64() {
    Advance();
    const std::uint64_t folded = state_.high ^ state_.low;
    return std::rotr(folded, static_cast<int>(state_.high >> 58U));
}

std::uint32_t Pcg64Random::NextUInt32() {
    if (has_buffered_uint32_) {
        has_buffered_uint32_ = false;
        return buffered_uint32_;
    }
    const std::uint64_t value = NextUInt64();
    buffered_uint32_ = static_cast<std::uint32_t>(value >> 32U);
    has_buffered_uint32_ = true;
    return static_cast<std::uint32_t>(value);
}

double Pcg64Random::Uniform() {
    constexpr double inverse = 0x1.0p-53;
    return static_cast<double>(NextUInt64() >> 11U) * inverse;
}

double Pcg64Random::Uniform(double lower, double upper) {
    return lower + (upper - lower) * Uniform();
}

std::uint32_t Pcg64Random::BoundedUInt32(std::uint32_t bound_exclusive) {
    if (bound_exclusive <= 1U) {
        return 0U;
    }

    std::uint64_t product = static_cast<std::uint64_t>(NextUInt32()) * bound_exclusive;
    std::uint32_t leftover = static_cast<std::uint32_t>(product);
    if (leftover < bound_exclusive) {
        const std::uint32_t range_inclusive = bound_exclusive - 1U;
        const std::uint32_t threshold =
                (std::numeric_limits<std::uint32_t>::max() - range_inclusive) %
                bound_exclusive;
        while (leftover < threshold) {
            product = static_cast<std::uint64_t>(NextUInt32()) * bound_exclusive;
            leftover = static_cast<std::uint32_t>(product);
        }
    }
    return static_cast<std::uint32_t>(product >> 32U);
}

} // namespace gobot
