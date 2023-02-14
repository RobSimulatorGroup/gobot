/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Zikun Yu, 23-2-13
*/

#include "gobot/scene/node_3d.hpp"

namespace gobot {

void Node3D::NotificationCallBack(NotificationType notification) {
    switch (notification) {
        case NotificationType::EnterTree: {
            ERR_FAIL_COND(!GetTree());

        } break;

        case NotificationType::ExitTree: {

        } break;

        case NotificationType::EnterWorld: {

        } break;

        case NotificationType::ExitWorld: {

        } break;

        case NotificationType::TransformChanged: {
            // todo: tools_enabled
        } break;
    }
}

Node3D *Node3D::GetParentNode3D() const {
    return dynamic_cast<Node3D *>(GetParent());
}

void Node3D::SetPosition(const Vector3d &position) {
    local_transform_.translation() = position;
    PropagateTransformChanged(this);

    if (notify_local_transform_) {
        Notification(NotificationType::LocalTransformChanged);
    }
}

void Node3D::SetRotationEditMode(RotationEditMode mode) {
    if (rotation_edit_mode_ == mode) return;

    // todo: normalize local_transform_ if edit in rotation matrix
    bool transform_changed = false;
    if (rotation_edit_mode_ == RotationEditMode::RotationMatrix && !(dirty_ | DIRTY_LOCAL_TRANSFORM)) {
        transform_changed = true;
    }
}

void Node3D::SetRotationOrder(EulerOrder order) {
    if (euler_rotation_order_ == order)
        return;

    ERR_FAIL_INDEX(static_cast<int32_t>(order), 6);
    bool transform_changed = false;

    // todo: update rot/scale if dirty

    euler_rotation_order_ = order;

    if (transform_changed) {
        PropagateTransformChanged(this);
        if (notify_local_transform_) {
            Notification(NotificationType::LocalTransformChanged);
        }
    }
}

void Node3D::SetRotation(const Vector3d &euler_rad) {
    // todo: update scale only if rotation and scale are dirty

    euler_rotation_ = euler_rad;
    // todo: set dirty
    PropagateTransformChanged(this);

    if (notify_local_transform_) {
        Notification(NotificationType::LocalTransformChanged);
    }
}

void Node3D::SetRotationDeg(const Vector3d &euler_deg) {
    Vector3d radians{Deg2Rad(euler_deg[0]), Deg2Rad(euler_deg[1]), Deg2Rad(euler_deg[2])};
    SetRotation(radians);
}

void Node3D::SetScale(const Vector3d &scale) {
    // todo: update rotation only if rotation and scale are dirty

    scale_ = scale;
    // todo: set dirty
    PropagateTransformChanged(this);

    if (notify_local_transform_) {
        Notification(NotificationType::LocalTransformChanged);
    }
}

void Node3D::SetGlobalPosition(const Vector3d &position) {
    Transform3d transform = GetGlobalTransform();
    transform.translation() = position;
    // set global transform
}


void Node3D::SetTransform(const Transform3d &transform) {
    local_transform_ = transform;
    // todo: mark rot/scale dirty

    PropagateTransformChanged(this);

    if (notify_local_transform_) {
        Notification(NotificationType::LocalTransformChanged);
    }
}

void Node3D::SetGlobalTransform(const Transform3d &transform) {
    Transform3d local = parent_
            ? parent_->GetGlobalTransform().inverse() * transform
            : transform;

    SetTransform(local);
}




Vector3d Node3D::GetPosition() const {
    return local_transform_.translation();
}

Node3D::EulerOrder Node3D::GetRotationOrder() const {
    return euler_rotation_order_;
}

Vector3d Node3D::GetRotation() const {
    // todo: update rotation and scale if dirty

    return euler_rotation_;
}

Vector3d Node3D::GetRotationDeg() const {
    Vector3d radians = GetRotation();
    return {Rad2Deg(radians[0]), Rad2Deg(radians[1]), Rad2Deg(radians[2])};
}

Vector3d Node3D::GetScale() const {
    // todo: update rotation and scale if dirty

    return scale_;
}


Transform3d Node3D::GetGlobalTransform() const {
    ERR_FAIL_COND_V(!IsInsideTree(), Transform3d());

    // todo: update local transform if dirty

    return global_transform_;
}






void Node3D::PropagateTransformChanged(Node3D *origin) {
    if (!IsInsideTree()) return;

    for (Node3D *c : children_) {
        c->PropagateTransformChanged(origin);
    }

    // todo: do transform
}

} // End of namespace gobot