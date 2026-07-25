/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/scene/gaussian_splat_3d.hpp"

#include "gobot/core/registration.hpp"

namespace gobot {

void GaussianSplat3D::SetSplat(const Ref<GaussianSplatResource>& splat) {
    splat_ = splat;
}

const Ref<GaussianSplatResource>& GaussianSplat3D::GetSplat() const {
    return splat_;
}

void GaussianSplat3D::SetEnabled(bool enabled) {
    enabled_ = enabled;
}

bool GaussianSplat3D::IsEnabled() const {
    return enabled_;
}

}

GOBOT_REGISTRATION {
    USING_ENUM_BITWISE_OPERATORS;

    Class_<GaussianSplat3D>("GaussianSplat3D")
            .constructor()(CtorAsRawPtr)
            .property("splat", &GaussianSplat3D::GetSplat, &GaussianSplat3D::SetSplat)(
                    AddMetaPropertyInfo(PropertyInfo().SetUsageFlags(
                            PropertyUsageFlags::Storage | PropertyUsageFlags::Editor)))
            .property("enabled", &GaussianSplat3D::IsEnabled, &GaussianSplat3D::SetEnabled);
}
