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
    DartBody3D *body = new DartBody3D();
    RID rid = body_owner_.MakeRID(body);
    body->SetSelf(rid);
    return rid;
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