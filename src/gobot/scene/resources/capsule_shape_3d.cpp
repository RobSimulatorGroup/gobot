/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/scene/resources/capsule_shape_3d.hpp"

#include "gobot/core/registration.hpp"
#include "gobot/log.hpp"

namespace gobot {

CapsuleShape3D::CapsuleShape3D() = default;

void CapsuleShape3D::SetRadius(float radius) {
    if (radius < 0.0f) {
        LOG_ERROR("CapsuleShape3D radius cannot be negative.");
        return;
    }
    radius_ = radius;
}

float CapsuleShape3D::GetRadius() const {
    return radius_;
}

void CapsuleShape3D::SetHeight(float height) {
    if (height < 0.0f) {
        LOG_ERROR("CapsuleShape3D height cannot be negative.");
        return;
    }
    height_ = height;
}

float CapsuleShape3D::GetHeight() const {
    return height_;
}

} // namespace gobot

GOBOT_REGISTRATION {
    Class_<gobot::CapsuleShape3D>("CapsuleShape3D")
            .constructor()(CtorAsRawPtr)
            .property("height", &gobot::CapsuleShape3D::GetHeight, &gobot::CapsuleShape3D::SetHeight)
            .property("radius", &gobot::CapsuleShape3D::GetRadius, &gobot::CapsuleShape3D::SetRadius);

    gobot::Type::register_wrapper_converter_for_base_classes<gobot::Ref<gobot::CapsuleShape3D>, gobot::Ref<gobot::Shape3D>>();
};
