/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Zikun Yu, 23-3-24
*/

#include "gobot/scene/collision_object_3d.hpp"

namespace gobot {

void CollisionObject3D::NotificationCallBack(NotificationType notification) {
    switch (notification) {
        case NotificationType::EnterTree: {
            // todo: add to debug shapes if collision shapes visible
            // todo: set notify local transform if editor hints
        } break;

        case NotificationType::ExitTree: {
            // todo: clear debug shapes
        } break;

        case NotificationType::EnterWorld: {
            if (area_) {
                // todo: PhysicsServer3D::get_singleton()->area_set_transform(rid, get_global_transform());
            } else {
                // todo: PhysicsServer3D::get_singleton()->body_set_state(rid, PhysicsServer3D::BODY_STATE_TRANSFORM, get_global_transform());
            }

            // todo: Setup disable mode

            // todo: UpdatePickable();
        } break;

        case NotificationType::TransformChanged: {
            if (area_) {
                // todo: PhysicsServer3D::get_singleton()->area_set_transform(rid, get_global_transform());
            } else {
                // todo: PhysicsServer3D::get_singleton()->body_set_state(rid, PhysicsServer3D::BODY_STATE_TRANSFORM, get_global_transform());
            }

            // todo: OnTransformChanged();
        } break;

        case NotificationType::ExitWorld: {
            // todo: Setup disable mode
        } break;
    }
}

void CollisionObject3D::UpdateShapeData(uint32_t owner_id) {
    // todo: update debug shapes if visible
}

uint32_t CollisionObject3D::CreateShapeOwner(Object *object_id) {
    ShapeData sd;
    uint32_t id;

    if (shape_owners_.empty()) {
        id = 0;
    } else {
        id = shape_owners_.end()->first + 1;
    }

    sd.object_id = object_id ? object_id->GetInstanceId() : ObjectID();

    shape_owners_[id] = sd;

    return id;
}

void CollisionObject3D::RemoveShapeOwner(uint32_t owner_id) {
    ERR_FAIL_COND(!shape_owners_.contains(owner_id));

    // todo: clear shapes

    shape_owners_.erase(owner_id);
}

//void CollisionObject3D::GetShapeOwners(std::vector<uint32_t> *owners) {
//    for ()
//}

std::vector<uint32_t> CollisionObject3D::GetShapeOwners() {
    std::vector<uint32_t> ret;
    for (const auto & e : shape_owners_) {
        ret.push_back(e.first);
    }

    return ret;
}

void CollisionObject3D::ShapeOwnerSetTransform(uint32_t owner_id, const Affine3 &tfm) {
    ERR_FAIL_COND(!shape_owners_.contains(owner_id));

    ShapeData &sd = shape_owners_[owner_id];
    sd.tfm = tfm;
    for (auto & shape : sd.shapes) {
        if (area_) {
            // todo: PhysicsServer3D::get_singleton()->area_set_shape_transform(rid, sd.shapes[i].index, p_transform);
        } else {
            // todo: PhysicsServer3D::get_singleton()->body_set_shape_transform(rid, sd.shapes[i].index, p_transform);
        }
    }

    UpdateShapeData(owner_id);
}

Affine3 CollisionObject3D::ShapeOwnerGetTransform(uint32_t owner_id) const {
    ERR_FAIL_COND_V(!shape_owners_.contains(owner_id), Affine3());

    return shape_owners_.at(owner_id).tfm;
}

Object *CollisionObject3D::ShapeOwnerGetOwner(uint32_t owner_id) const {
    ERR_FAIL_COND_V(!shape_owners_.contains(owner_id), nullptr);

    return ObjectDB::GetInstance(shape_owners_.at(owner_id).object_id);
}

void CollisionObject3D::ShapeOwnerSetDisabled(uint32_t owner_id, bool disabled) {
    ERR_FAIL_COND(!shape_owners_.contains(owner_id));

    ShapeData &sd = shape_owners_[owner_id];
    if (sd.disabled == disabled) {
        return;
    }
    sd.disabled = disabled;

    for (auto & shape : sd.shapes) {
        if (area_) {
            // todo: PhysicsServer3D::get_singleton()->area_set_shape_disabled(rid, sd.shapes[i].index, p_disabled);
        } else {
            // todo: PhysicsServer3D::get_singleton()->body_set_shape_disabled(rid, sd.shapes[i].index, p_disabled);
        }
    }

    UpdateShapeData(owner_id);
}

bool CollisionObject3D::IsShapeOwnerDisabled(uint32_t owner_id) const {
    ERR_FAIL_COND_V(!shape_owners_.contains(owner_id), false);

    return shape_owners_.at(owner_id).disabled;
}

void CollisionObject3D::ShapeOwnerAddShape(uint32_t owner_id, const Ref<Shape3D> &shape) {
    ERR_FAIL_COND(!shape_owners_.contains(owner_id));
    ERR_FAIL_COND(!shape.IsValid());

    ShapeData &sd = shape_owners_[owner_id];

    if (area_) {
        // todo: PhysicsServer3D::get_singleton()->area_add_shape(rid, p_shape->get_rid(), sd.xform, sd.disabled);
    } else {
        // todo: PhysicsServer3D::get_singleton()->body_add_shape(rid, p_shape->get_rid(), sd.xform, sd.disabled);
    }
    sd.shapes.push_back(shape);

//    total_subshapes_ ++;

    UpdateShapeData(owner_id);
}

std::size_t CollisionObject3D::ShapeOwnerGetShapeCount(uint32_t owner_id) const {
    ERR_FAIL_COND_V(!shape_owners_.contains(owner_id), 0);

    return shape_owners_.at(owner_id).shapes.size();
}

Ref<Shape3D> CollisionObject3D::ShapeOwnerGetShape(uint32_t owner_id, std::size_t shape_index) const {
    ERR_FAIL_COND_V(!shape_owners_.contains(owner_id), Ref<Shape3D>());
    ERR_FAIL_INDEX_V(shape_index, shape_owners_.at(owner_id).shapes.size(), Ref<Shape3D>());

    return shape_owners_.at(owner_id).shapes[shape_index];
}

void CollisionObject3D::ShapeOwnerRemoveShape(uint32_t owner_id, std::size_t shape_id) {
    ERR_FAIL_NULL(RenderServer::GetInstance());
    ERR_FAIL_COND(!shape_owners_.contains(owner_id));
    ERR_FAIL_INDEX(shape_id, shape_owners_.at(owner_id).shapes.size());

//    auto &s = shape_owners_.at(owner_id).shapes[shape_id];
//    std::size_t index_to_remove = s.index;

    if (area_) {
        // todo: PhysicsServer3D::get_singleton()->area_remove_shape(rid, index_to_remove);
    } else {
        // todo: PhysicsServer3D::get_singleton()->body_remove_shape(rid, index_to_remove);
    }

    auto &shapes = shape_owners_.at(owner_id).shapes;
    shapes.erase(std::remove(shapes.begin(), shapes.end(), shapes[shape_id]), shapes.end());
}

void CollisionObject3D::ShapeOwnerClearShapes(uint32_t owner_id) {
    ERR_FAIL_COND(!shape_owners_.contains(owner_id));

    shape_owners_.at(owner_id).shapes.clear();
}



CollisionObject3D::CollisionObject3D(RID rid, bool area) {
    rid_ = rid;
    area_ = area;
    SetNotifyTransform(true);

    if (area) {
        // todo: PhysicsServer3D::GetInstance()->AreaAttachObjectInstanceID(rid, GetInstanceID());
    } else {
        // todo: PhysicsServer3D::GetInstance()->BodyAttachObjectInstanceID(rid, GetInstanceID());
        // todo: PhysicsServer3D::GetInstance()->BodeSetMode(rid, body_mode_);
    }
}

CollisionObject3D::CollisionObject3D() {
    SetNotifyTransform(true);
}

CollisionObject3D::~CollisionObject3D() noexcept {
    // todo: ERR_FAIL_NULL(PhysicsServer3D::GetInstance());
    // todo: PhysicsServer3D::get_singleton()->Free(rid);
}

} // End of namespace gobot

