/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
*/

#pragma once

#include "gobot/core/math/geometry.hpp"
#include "gobot/scene/resources/shape_3d.hpp"

namespace gobot {

class GOBOT_EXPORT BoxShape3D : public Shape3D {
    GOBCLASS(BoxShape3D, Shape3D);
public:
    BoxShape3D();

    void SetSize(const Vector3& size);

    const Vector3& GetSize() const;

private:
    Vector3 size_{Vector3::Ones()};
};

}
