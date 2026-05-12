/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/scene/resources/box_shape_3d.hpp"

#include "gobot/core/registration.hpp"
#include "gobot/log.hpp"

namespace gobot {

BoxShape3D::BoxShape3D() = default;

void BoxShape3D::SetSize(const Vector3& size) {
    if ((size.array() < 0).any()) {
        LOG_ERROR("BoxShape3D size cannot be negative.");
        return;
    }
    size_ = size;
}

const Vector3& BoxShape3D::GetSize() const {
    return size_;
}

} // namespace gobot

GOBOT_REGISTRATION {
    Class_<BoxShape3D>("BoxShape3D")
            .constructor()(CtorAsRawPtr)
            .property("size", &BoxShape3D::GetSize, &BoxShape3D::SetSize);

    gobot::Type::register_wrapper_converter_for_base_classes<Ref<BoxShape3D>, Ref<Shape3D>>();

};
