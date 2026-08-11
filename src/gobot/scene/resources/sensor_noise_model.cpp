/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/scene/resources/sensor_noise_model.hpp"

#include "gobot/core/registration.hpp"

namespace gobot {

#define GOBOT_NOISE_PROPERTY(Name, Field)                                      \
    void SensorNoiseModel::Set##Name(RealType value) { Field = value; MarkChanged(); } \
    RealType SensorNoiseModel::Get##Name() const { return Field; }

GOBOT_NOISE_PROPERTY(WhiteNoiseStddev, white_noise_stddev_)
GOBOT_NOISE_PROPERTY(BiasMean, bias_mean_)
GOBOT_NOISE_PROPERTY(BiasStddev, bias_stddev_)
GOBOT_NOISE_PROPERTY(RandomWalkStddev, random_walk_stddev_)
GOBOT_NOISE_PROPERTY(QuantizationStep, quantization_step_)
GOBOT_NOISE_PROPERTY(ClipMin, clip_min_)
GOBOT_NOISE_PROPERTY(ClipMax, clip_max_)

#undef GOBOT_NOISE_PROPERTY

void SensorNoiseModel::SetSeedOffset(std::uint32_t value) {
    seed_offset_ = value;
    MarkChanged();
}

std::uint32_t SensorNoiseModel::GetSeedOffset() const { return seed_offset_; }

} // namespace gobot

GOBOT_REGISTRATION {
    Class_<gobot::SensorNoiseModel>("SensorNoiseModel")
            .constructor()(CtorAsRawPtr)
            .property("white_noise_stddev", &gobot::SensorNoiseModel::GetWhiteNoiseStddev,
                      &gobot::SensorNoiseModel::SetWhiteNoiseStddev)
            .property("bias_mean", &gobot::SensorNoiseModel::GetBiasMean,
                      &gobot::SensorNoiseModel::SetBiasMean)
            .property("bias_stddev", &gobot::SensorNoiseModel::GetBiasStddev,
                      &gobot::SensorNoiseModel::SetBiasStddev)
            .property("random_walk_stddev", &gobot::SensorNoiseModel::GetRandomWalkStddev,
                      &gobot::SensorNoiseModel::SetRandomWalkStddev)
            .property("quantization_step", &gobot::SensorNoiseModel::GetQuantizationStep,
                      &gobot::SensorNoiseModel::SetQuantizationStep)
            .property("clip_min", &gobot::SensorNoiseModel::GetClipMin,
                      &gobot::SensorNoiseModel::SetClipMin)
            .property("clip_max", &gobot::SensorNoiseModel::GetClipMax,
                      &gobot::SensorNoiseModel::SetClipMax)
            .property("seed_offset", &gobot::SensorNoiseModel::GetSeedOffset,
                      &gobot::SensorNoiseModel::SetSeedOffset);

    gobot::Type::register_wrapper_converter_for_base_classes<
            gobot::Ref<gobot::SensorNoiseModel>, gobot::Ref<gobot::Resource>>();
}
