/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
*/

#pragma once

#include "gobot/scene/resources/shape_3d.hpp"

namespace gobot {

class GOBOT_EXPORT SphereShape3D : public Shape3D {
    GOBCLASS(SphereShape3D, Shape3D);
public:
    SphereShape3D();

    void SetRadius(float radius);

    float GetRadius() const;

private:
    float radius_{0.5f};
};

}
