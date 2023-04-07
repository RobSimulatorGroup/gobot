/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Zikun Yu, 23-3-24
*/

#include "gobot/physics/dart/dart_physics_server_3d.h"

namespace gobot {

RID DartPhysicsServer3D::BoxShapeCreate() {
    DartShape3D *shape = new DartBoxShape3D();
    RID rid = shape_owner_.MakeRID(shape);
    shape->SetSelf(rid);
    return rid;
}

void DartPhysicsServer3D::ShapeSetData(RID shape, const Variant &data) {
    DartShape3D *dart_shape = shape_owner_.GetOrNull(shape);
    ERR_FAIL_COND(!dart_shape);
    dart_shape->SetData(data);
}

PhysicsServer3D::ShapeType DartPhysicsServer3D::ShapeGetType(RID shape) const {
    const DartShape3D *dart_shape = shape_owner_.GetOrNull(shape);
    ERR_FAIL_COND_V(!dart_shape, PhysicsServer3D::ShapeType::Custom);
    return dart_shape->GetType();
}

Variant DartPhysicsServer3D::ShapeGetData(RID shape) const {
    const DartShape3D *dart_shape = shape_owner_.GetOrNull(shape);
    ERR_FAIL_COND_V(!dart_shape, Variant());
    // todo: is configured
    return dart_shape->GetData();
}

RID DartPhysicsServer3D::WorldCreate() {
    auto *world = new DartWorld3D;
    RID id = world_owner_.MakeRID(world);
    world->SetSelf(id);
    return id;
}

/* BODY API */

RID DartPhysicsServer3D::BodyCreate() {
    auto *body = new DartBody3D();
    RID rid = body_owner_.MakeRID(body);
    body->SetSelf(rid);
    return rid;
}

void DartPhysicsServer3D::BodyAddShape(RID body, RID shape, const Affine3 &transform, bool disabled) {
    auto *body_ptr = body_owner_.GetOrNull(body);
    ERR_FAIL_COND(!body_ptr);

    DartShape3D *shape_ptr = shape_owner_.GetOrNull(shape);
    ERR_FAIL_COND(!shape_ptr);

    body_ptr->AddShape(shape_ptr, transform);
}

void DartPhysicsServer3D::BodySetShapeTransform(RID body, std::size_t shape_idx, const Affine3 &transform) {
    auto *body_ptr = body_owner_.GetOrNull(body);
    ERR_FAIL_COND(!body_ptr);

    body_ptr->SetShapeTransform(shape_idx, transform);
}

std::size_t DartPhysicsServer3D::BodyGetShapeCount(RID body) const {
    auto body_ptr = body_owner_.GetOrNull(body);
    ERR_FAIL_COND_V(!body_ptr, -1);

    return body_ptr->GetShapeCount();
}

RID DartPhysicsServer3D::BodyGetShape(RID body, std::size_t shape_idx) const {
    auto body_ptr = body_owner_.GetOrNull(body);
    ERR_FAIL_COND_V(!body_ptr, RID());

    auto *shape_ptr = body_ptr->GetShape(shape_idx);
    ERR_FAIL_COND_V(!shape_ptr, RID());

    return shape_ptr->GetSelf();
}

void DartPhysicsServer3D::BodyRemoveShape(RID body, std::size_t shape_idx) {
    auto *body_ptr = body_owner_.GetOrNull(body);
    ERR_FAIL_COND(!body_ptr);

    body_ptr->RemoveShape(shape_idx);
}

void DartPhysicsServer3D::BodyClearShapes(RID body) {
    auto *body_ptr = body_owner_.GetOrNull(body);
    ERR_FAIL_COND(!body_ptr);

    while (body_ptr->GetShapeCount()) {
        body_ptr->RemoveShape(static_cast<std::size_t>(0));
    }
}

void DartPhysicsServer3D::BodySetParam(RID body, BodyParameter param, const Variant &value) {
    auto *body_ptr = body_owner_.GetOrNull(body);
    ERR_FAIL_COND(!body_ptr);

    body_ptr->SetParam(param, value);
}

Variant DartPhysicsServer3D::BodyGetParam(RID body, BodyParameter param) const {
    auto *body_ptr = body_owner_.GetOrNull(body);
    ERR_FAIL_COND_V(!body_ptr, 0);

    return body_ptr->GetParam(param);
}

RID DartPhysicsServer3D::JointCreate() {
    auto *joint = new DartJoint3D();
    RID rid = joint_owner_.MakeRID(joint);
    joint->SetSelf(rid);
    return rid;
}

//void DartPhysicsServer3D::JointClear(RID joint) {
//    DartJoint3D *joint_ptr = joint_owner_.GetOrNull(joint);
//    ERR_FAIL_NULL(joint_ptr);
//
//    // todo: delete if internal dart joint pointer is nullptr
//
//
//}

PhysicsServer3D::JointType DartPhysicsServer3D::JointGetType(RID joint) {
    DartJoint3D *joint_ptr = joint_owner_.GetOrNull(joint);
    ERR_FAIL_COND_V(!joint_ptr, PhysicsServer3D::JointType::None);
    return joint_ptr->GetType();
}









void DartPhysicsServer3D::Free(RID rid) {
    // todo: update shapes

    if (shape_owner_.Owns(rid)) {
        DartShape3D *shape = shape_owner_.GetOrNull(rid);

        // todo: DartShapeOwner3D remove shape

        shape_owner_.Free(rid);
        delete shape;
    }

    // todo: if body owner ...
    // todo: if joint owner ...

    else {
        ERR_FAIL_MSG("Invalid RID.")
    }
}

} // End of namespace gobot