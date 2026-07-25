/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "gobot/scene/node_3d.hpp"
#include "gobot/scene/resources/gaussian_splat.hpp"

namespace gobot {

class GOBOT_EXPORT GaussianSplat3D : public Node3D {
    GOBCLASS(GaussianSplat3D, Node3D)

public:
    void SetSplat(const Ref<GaussianSplatResource>& splat);
    [[nodiscard]] const Ref<GaussianSplatResource>& GetSplat() const;

    void SetEnabled(bool enabled);
    [[nodiscard]] bool IsEnabled() const;

private:
    Ref<GaussianSplatResource> splat_;
    bool enabled_ = true;
};

}
