/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <limits>

#include "gobot/core/io/resource.hpp"
#include "gobot/core/math/math_defs.hpp"

namespace gobot {

class GOBOT_EXPORT SensorNoiseModel : public Resource {
    GOBCLASS(SensorNoiseModel, Resource)

public:
    void SetWhiteNoiseStddev(RealType value);
    RealType GetWhiteNoiseStddev() const;
    void SetBiasMean(RealType value);
    RealType GetBiasMean() const;
    void SetBiasStddev(RealType value);
    RealType GetBiasStddev() const;
    void SetRandomWalkStddev(RealType value);
    RealType GetRandomWalkStddev() const;
    void SetQuantizationStep(RealType value);
    RealType GetQuantizationStep() const;
    void SetClipMin(RealType value);
    RealType GetClipMin() const;
    void SetClipMax(RealType value);
    RealType GetClipMax() const;
    void SetSeedOffset(std::uint32_t value);
    std::uint32_t GetSeedOffset() const;

private:
    RealType white_noise_stddev_{0.0};
    RealType bias_mean_{0.0};
    RealType bias_stddev_{0.0};
    RealType random_walk_stddev_{0.0};
    RealType quantization_step_{0.0};
    RealType clip_min_{-std::numeric_limits<RealType>::infinity()};
    RealType clip_max_{std::numeric_limits<RealType>::infinity()};
    std::uint32_t seed_offset_{0};
};

} // namespace gobot
