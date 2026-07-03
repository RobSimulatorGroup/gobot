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

void CollisionShape3D::SetFriction(const Vector3& friction) {
    friction_ = friction;
}

const Vector3& CollisionShape3D::GetFriction() const {
    return friction_;
}

void CollisionShape3D::SetContactType(int contype) {
    contype_ = contype;
}

int CollisionShape3D::GetContactType() const {
    return contype_;
}

void CollisionShape3D::SetContactAffinity(int conaffinity) {
    conaffinity_ = conaffinity;
}

int CollisionShape3D::GetContactAffinity() const {
    return conaffinity_;
}

void CollisionShape3D::SetContactDimension(int condim) {
    condim_ = condim;
}

int CollisionShape3D::GetContactDimension() const {
    return condim_;
}

void CollisionShape3D::SetSolref(const Vector2& solref) {
    solref_ = solref;
}

const Vector2& CollisionShape3D::GetSolref() const {
    return solref_;
}

void CollisionShape3D::SetSolimp(const std::vector<RealType>& solimp) {
    solimp_ = solimp;
}

const std::vector<RealType>& CollisionShape3D::GetSolimp() const {
    return solimp_;
}

void CollisionShape3D::SetMargin(RealType margin) {
    margin_ = margin;
}

RealType CollisionShape3D::GetMargin() const {
    return margin_;
}

void CollisionShape3D::SetGap(RealType gap) {
    gap_ = gap;
}

RealType CollisionShape3D::GetGap() const {
    return gap_;
}

void CollisionShape3D::SetPriority(int priority) {
    priority_ = priority;
}

int CollisionShape3D::GetPriority() const {
    return priority_;
}

} // namespace gobot

GOBOT_REGISTRATION {

    Class_<CollisionShape3D>("CollisionShape3D")
            .constructor()(CtorAsRawPtr)
            .property("shape", &CollisionShape3D::GetShape, &CollisionShape3D::SetShape)
            .property("disabled", &CollisionShape3D::IsDisabled, &CollisionShape3D::SetDisabled)
            .property("friction", &CollisionShape3D::GetFriction, &CollisionShape3D::SetFriction)
            .property("contype", &CollisionShape3D::GetContactType, &CollisionShape3D::SetContactType)
            .property("conaffinity", &CollisionShape3D::GetContactAffinity, &CollisionShape3D::SetContactAffinity)
            .property("condim", &CollisionShape3D::GetContactDimension, &CollisionShape3D::SetContactDimension)
            .property("solref", &CollisionShape3D::GetSolref, &CollisionShape3D::SetSolref)
            .property("solimp", &CollisionShape3D::GetSolimp, &CollisionShape3D::SetSolimp)
            .property("margin", &CollisionShape3D::GetMargin, &CollisionShape3D::SetMargin)
            .property("gap", &CollisionShape3D::GetGap, &CollisionShape3D::SetGap)
            .property("priority", &CollisionShape3D::GetPriority, &CollisionShape3D::SetPriority);

};
