/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * This file is created by Qiqi Wu, 23-1-13
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/scene/resources/shape_3d.hpp"

namespace gobot {

class GOBOT_EXPORT CylinderShape3D : public Shape3D {
    GOBCLASS(CylinderShape3D, Shape3D);
public:
    CylinderShape3D();

    void SetRadius(float radius);

    float GetRadius() const;

    void SetHeight(float height);

    float GetHeight() const;

private:
    float radius_{0.5};
    float height_{1.0};
};

}