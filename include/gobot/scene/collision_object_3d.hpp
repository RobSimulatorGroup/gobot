/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Zikun Yu, 23-3-24
*/

#pragma once

#include "gobot/scene/node_3d.hpp"
#include "gobot/scene/resources/shape_3d.hpp"
#include "gobot/core/rid.h"
#include "gobot/core/ref_counted.hpp"

#include "gobot/rendering/render_server.hpp"

namespace gobot {

class CollisionObject3D : public Node3D {
    GOBCLASS(CollisionObject3D, Node3D);

public:
    CollisionObject3D();
    ~CollisionObject3D() override;

    uint32_t CreateShapeOwner(Object *object_id);
    void RemoveShapeOwner(uint32_t owner_id);
//    void GetShapeOwners(std::vector<uint32_t> *owners);
    std::vector<uint32_t> GetShapeOwners();

    void ShapeOwnerSetTransform(uint32_t owner_id, const Affine3 &tfm);
    Affine3 ShapeOwnerGetTransform(uint32_t owner_id) const;
    Object *ShapeOwnerGetOwner(uint32_t owner_id) const;

    void ShapeOwnerSetDisabled(uint32_t owner_id, bool disabled);
    bool IsShapeOwnerDisabled(uint32_t owner_id) const;

    void ShapeOwnerAddShape(uint32_t owner_id, const Ref<Shape3D> &shape);
    std::size_t ShapeOwnerGetShapeCount(uint32_t owner_id) const;
    Ref<Shape3D> ShapeOwnerGetShape(uint32_t owner_id, std::size_t shape_id) const;

    void ShapeOwnerRemoveShape(uint32_t owner_id, std::size_t shape_id);
    void ShapeOwnerClearShapes(uint32_t owner_id);

//    void SetPickable(bool ray_pickable);
//    bool IsPickable() const;
//    void SetCaptureInputOnDrag(bool capture);
//    void GetCaptureInputOnDrag() const;

    FORCE_INLINE RID GetRID() const { return rid_; }

protected:
    CollisionObject3D(RID rid, bool area);

    ////////////////////////////////////////////////////////////////
    void NotificationCallBack(NotificationType notification);
//    void OnTransformChanged();

// todo: ViewPort;

// todo: SetBodyMode

private:
    RID rid_;

    bool area_;

    // todo: PhysicsServer3D::BodyMode

    struct ShapeData {
        ObjectID object_id;
        Affine3 tfm;
        std::vector<Ref<Shape3D> > shapes;
        bool disabled = false;
    };

    std::map<uint32_t, ShapeData> shape_owners_;

    void UpdateShapeData(uint32_t owner_id);

//    void ShapeChanged(const Ref<Shape3D> &shape); // for debug shapes
};

} // End of namespace gobot