/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
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

void CollisionShape3D::SetPhysicsMaterial(const Ref<PhysicsMaterial3D>& material) {
    physics_material_ = material;
}

const Ref<PhysicsMaterial3D>& CollisionShape3D::GetPhysicsMaterial() const { return physics_material_; }

void CollisionShape3D::SetCollisionLayer(std::uint32_t layer) { collision_layer_ = layer; }

std::uint32_t CollisionShape3D::GetCollisionLayer() const { return collision_layer_; }

void CollisionShape3D::SetCollisionMask(std::uint32_t mask) { collision_mask_ = mask; }

std::uint32_t CollisionShape3D::GetCollisionMask() const { return collision_mask_; }

void CollisionShape3D::SetContactOffset(RealType offset) { contact_offset_ = offset; }

RealType CollisionShape3D::GetContactOffset() const { return contact_offset_; }

void CollisionShape3D::SetRestOffset(RealType offset) { rest_offset_ = offset; }

RealType CollisionShape3D::GetRestOffset() const { return rest_offset_; }

} // namespace gobot

GOBOT_REGISTRATION {

    Class_<CollisionShape3D>("CollisionShape3D")
            .constructor()(CtorAsRawPtr)
            .property("shape", &CollisionShape3D::GetShape, &CollisionShape3D::SetShape)
            .property("disabled", &CollisionShape3D::IsDisabled, &CollisionShape3D::SetDisabled)
            .property("physics_material",
                      &CollisionShape3D::GetPhysicsMaterial,
                      &CollisionShape3D::SetPhysicsMaterial)
            .property("collision_layer",
                      &CollisionShape3D::GetCollisionLayer,
                      &CollisionShape3D::SetCollisionLayer)
            .property("collision_mask",
                      &CollisionShape3D::GetCollisionMask,
                      &CollisionShape3D::SetCollisionMask)
            .property("contact_offset",
                      &CollisionShape3D::GetContactOffset,
                      &CollisionShape3D::SetContactOffset)
            .property("rest_offset",
                      &CollisionShape3D::GetRestOffset,
                      &CollisionShape3D::SetRestOffset);

};
