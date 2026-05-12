/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/scene/resources/sphere_shape_3d.hpp"

#include "gobot/core/registration.hpp"
#include "gobot/log.hpp"

namespace gobot {

SphereShape3D::SphereShape3D() = default;

void SphereShape3D::SetRadius(float radius) {
    if (radius < 0) {
        LOG_ERROR("SphereShape3D radius cannot be negative.");
        return;
    }
    radius_ = radius;
}

float SphereShape3D::GetRadius() const {
    return radius_;
}

} // namespace gobot

GOBOT_REGISTRATION {
    Class_<SphereShape3D>("SphereShape3D")
            .constructor()(CtorAsRawPtr)
            .property("radius", &SphereShape3D::GetRadius, &SphereShape3D::SetRadius);

    gobot::Type::register_wrapper_converter_for_base_classes<Ref<SphereShape3D>, Ref<Shape3D>>();

};
