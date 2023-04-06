/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Zikun Yu, 23-3-24
*/

#pragma once

#include "gobot/physics/physics_server_3d.h"

#include "gobot/physics/dart/dart_world_3d.h"
#include "gobot/physics/dart/dart_shape_3d.h"
#include "gobot/physics/dart/dart_body_3d.h"
#include "gobot/core/rid_owner.h"

#include <dart/dart.hpp>

namespace gobot {

class DartPhysicsServer3D : public PhysicsServer3D {
    GOBCLASS(DartPhysicsServer3D, PhysicsServer3D);

public:
    RID BoxShapeCreate() override;

    void ShapeSetData(RID shape, const Variant &data) override;

    ShapeType ShapeGetType(RID shape) const override;
    Variant ShapeGetData(RID shape) const override;

    /* WORLD API */
    RID WorldCreate() override;

    /* BODY API */
    RID BodyCreate() override;

    void Free(RID rid) override;

private:
    mutable RID_PtrOwner<DartShape3D, true> shape_owner_;
    mutable RID_PtrOwner<DartWorld3D, true> world_owner_;
    mutable RID_PtrOwner<DartBody3D, true> body_owner_;
};

} // End of namespace gobot