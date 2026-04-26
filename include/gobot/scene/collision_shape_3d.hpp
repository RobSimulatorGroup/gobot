/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
*/

#pragma once

#include "gobot/scene/node_3d.hpp"
#include "gobot/scene/resources/shape_3d.hpp"

namespace gobot {

class GOBOT_EXPORT CollisionShape3D : public Node3D {
    GOBCLASS(CollisionShape3D, Node3D)

public:
    CollisionShape3D() = default;

    void SetShape(const Ref<Shape3D>& shape);

    const Ref<Shape3D>& GetShape() const;

    void SetDisabled(bool disabled);

    bool IsDisabled() const;

private:
    Ref<Shape3D> shape_{nullptr};
    bool disabled_{false};
};

}
