/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
*/

#include "gobot/scene/collision_shape_3d.hpp"

#include "gobot/core/registration.hpp"

namespace gobot {

void CollisionShape3D::SetShape(const Ref<Shape3D>& shape) {
    shape_ = shape;
}

const Ref<Shape3D>& CollisionShape3D::GetShape() const {
    return shape_;
}

void CollisionShape3D::SetDisabled(bool disabled) {
    disabled_ = disabled;
}

bool CollisionShape3D::IsDisabled() const {
    return disabled_;
}

} // namespace gobot

GOBOT_REGISTRATION {

    Class_<CollisionShape3D>("CollisionShape3D")
            .constructor()(CtorAsRawPtr)
            .property("shape", &CollisionShape3D::GetShape, &CollisionShape3D::SetShape)
            .property("disabled", &CollisionShape3D::IsDisabled, &CollisionShape3D::SetDisabled);

};
