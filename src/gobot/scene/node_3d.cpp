/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Zikun Yu, 23-2-13
*/

#include "gobot/scene/node_3d.hpp"

namespace gobot {

void Node3D::NotifyDirty() {
    // todo: add to transform change list
}

void Node3D::UpdateLocalTransform() const {
    // This function is called when the local transform (data.local_transform) is dirty
    // and the right value is contained in the Euler rotation and scale.

    local_transform_.SetEulerAngleScaled(euler_, scale_, euler_order_);

    dirty_ &= ~DIRTY_LOCAL_TRANSFORM;
}

void Node3D::UpdateEulerAndScale() const {
    // This function is called when the Euler rotation (euler_rotation_) is dirty
    // and the right value is contained in the local transform.

    scale_ = local_transform_.GetScale();
    euler_ = local_transform_.GetEulerAngle(euler_order_);

    dirty_ &= ~DIRTY_EULER_AND_SCALE;
}

void Node3D::PropagateTransformChanged(Node3D *node) {
    if (!IsInsideTree()) return;

    for (Node3D *c : children_) {
        c->PropagateTransformChanged(node);
    }

    // todo: add to transform change list
    dirty_ |= DIRTY_GLOBAL_TRANSFORM;
}

void Node3D::NotificationCallBack(NotificationType notification) {
    switch (notification) {
        case NotificationType::EnterTree: {
            ERR_FAIL_COND(!GetTree());

            Node *p = GetParent();
            if (p) {
                // dynamic_cast may fail such that parent_ remains nullptr
                parent_ = dynamic_cast<Node3D *>(p);
            }

            dirty_ |= DIRTY_GLOBAL_TRANSFORM; // Global is always dirty upon entering a scene
            NotifyDirty();

            Notification(NotificationType::EnterWorld);
            // todo: update visibility
        } break;

        case NotificationType::ExitTree: {
            Notification(NotificationType::ExitWorld, true);
            // todo: remove from transform change list
            parent_ = nullptr;
            // todo: update visibility
        } break;

        case NotificationType::EnterWorld: {
            // todo: reset inside_world_ to update gizmo
            // todo: iteratively get parent as viewport
        } break;

        case NotificationType::ExitWorld: {
            // todo: clear gizmo
            // todo: reset viewport
            // todo: reset inside_world_
        } break;

        case NotificationType::TransformChanged: {
            // todo: update gizmo transform
        } break;
    }
}

Node3D *Node3D::GetParentNode3D() const {
    return dynamic_cast<Node3D *>(GetParent());
}

void Node3D::SetPosition(const Vector3 &position) {
    local_transform_.translation() = position;
    PropagateTransformChanged(this);

    if (notify_local_transform_) {
        Notification(NotificationType::LocalTransformChanged);
    }
}

Vector3 Node3D::GetPosition() const {
    return local_transform_.translation();
}

void Node3D::SetRotationEditMode(RotationEditMode mode) {
    if (rotation_edit_mode_ == mode) {
        return;
    }

    bool transform_changed = false;
    if (rotation_edit_mode_ == RotationEditMode::RotationMatrix && !(dirty_ & DIRTY_LOCAL_TRANSFORM)) {
        local_transform_.Orthogonalize();
        transform_changed = true;
    }

    rotation_edit_mode_ = mode;

    if (mode == RotationEditMode::Euler && (dirty_ & DIRTY_EULER_AND_SCALE)) {
        // If going to Euler mode, ensure that vectors are _not_ dirty, else the retrieved value may be wrong.
        // Otherwise, keep what is there, so switching back and forth between modes does not break the vectors.

        UpdateEulerAndScale();
    }

    if (transform_changed) {
        PropagateTransformChanged(this);
        if (notify_local_transform_) {
            Notification(NotificationType::LocalTransformChanged);
        }
    }

    // todo: notify property list changed
}

Node3D::RotationEditMode Node3D::GetRotationEditMode() const {
    return rotation_edit_mode_;
}

void Node3D::SetEulerOrder(EulerOrder order) {
    if (euler_order_ == order) {
        return;
    }

    bool transform_changed = false;
    if (dirty_ & DIRTY_EULER_AND_SCALE) {
        UpdateEulerAndScale();
    } else if (dirty_ & DIRTY_LOCAL_TRANSFORM) {
        // Query a new euler with order and leave local transform dirty
        Affine3 m;
        m.SetEulerAngle(euler_, euler_order_);
        euler_ = m.GetEulerAngle(order);
        transform_changed = true;
    } else {
        dirty_ |= DIRTY_LOCAL_TRANSFORM;
        transform_changed = true;
    }

    euler_order_ = order;

    if (transform_changed) {
        PropagateTransformChanged(this);
        if (notify_local_transform_) {
            Notification(NotificationType::LocalTransformChanged);
        }
    }

    // todo: notify property list changed
}

EulerOrder Node3D::GetEulerOrder() const {
    return euler_order_;
}

void Node3D::SetEuler(const EulerAngle &euler_rad) {
    if (dirty_ & DIRTY_EULER_AND_SCALE) {
        // Update scale only if rotation and scale are dirty, as rotation will be overridden.
        scale_ = local_transform_.GetScale();
        dirty_ &= ~DIRTY_EULER_AND_SCALE;
    }

    euler_ = euler_rad;
    dirty_ = DIRTY_LOCAL_TRANSFORM;
    PropagateTransformChanged(this);
    if (notify_local_transform_) {
        Notification(NotificationType::LocalTransformChanged);
    }
}

EulerAngle Node3D::GetEuler() const {
    if (dirty_ & DIRTY_EULER_AND_SCALE) {
        UpdateEulerAndScale();
    }

    return euler_;
}

void Node3D::SetEulerDegree(const EulerAngle &euler_deg) {
    EulerAngle radians{DEG_TO_RAD(euler_deg[0]), DEG_TO_RAD(euler_deg[1]), DEG_TO_RAD(euler_deg[2])};
    SetEuler(radians);
}

EulerAngle Node3D::GetEulerDegree() const {
    EulerAngle radians = GetEuler();
    return {RAD_TO_DEG(radians[0]), RAD_TO_DEG(radians[1]), RAD_TO_DEG(radians[2])};
}

void Node3D::SetScale(const Vector3 &scale) {
    if (dirty_ & DIRTY_EULER_AND_SCALE) {
        // Update rotation only if rotation and scale are dirty, as scale will be overridden.
        euler_ = local_transform_.GetEulerAngle(euler_order_);
        dirty_ &= ~DIRTY_EULER_AND_SCALE;
    }

    scale_ = scale;
    dirty_ = DIRTY_LOCAL_TRANSFORM;
    PropagateTransformChanged(this);
    if (notify_local_transform_) {
        Notification(NotificationType::LocalTransformChanged);
    }
}

Vector3 Node3D::GetScale() const {
    if (dirty_ & DIRTY_EULER_AND_SCALE) {
        UpdateEulerAndScale();
    }

    return scale_;
}

void Node3D::SetQuaternion(const Quaternion &quaternion) {
    if (dirty_ & DIRTY_EULER_AND_SCALE) {
        // We need the scale part, so update it if dirty
        scale_ = local_transform_.GetScale();
        dirty_ &= ~DIRTY_EULER_AND_SCALE;
    }

    local_transform_.linear() = Affine3(quaternion).scale(scale_).linear();
    // Rotate/scale should not be marked dirty because that would cause precision loss issues with the scale.
    // Instead, reconstruct rotation now.
    euler_ = local_transform_.GetEulerAngle(euler_order_);

    dirty_ = DIRTY_NONE;

    PropagateTransformChanged(this);
    if (notify_local_transform_) {
        Notification(NotificationType::LocalTransformChanged);
    }
}

Quaternion Node3D::GetQuaternion() const {
    if (dirty_ & DIRTY_LOCAL_TRANSFORM) {
        UpdateLocalTransform();
    }

    return local_transform_.GetQuaternion();
}

void Node3D::SetTransform(const Affine3 &transform) {
    local_transform_ = transform;
    dirty_ = DIRTY_EULER_AND_SCALE;

    PropagateTransformChanged(this);
    if (notify_local_transform_) {
        Notification(NotificationType::LocalTransformChanged);
    }
}

Affine3 Node3D::GetTransform() const {
    if (dirty_ & DIRTY_LOCAL_TRANSFORM) {
        UpdateLocalTransform();
    }

    return local_transform_;
}

void Node3D::SetGlobalTransform(const Affine3 &transform) {
    Affine3 local = parent_
                    ? parent_->GetGlobalTransform().inverse() * transform
                    : transform;

    SetTransform(local);
}

Affine3 Node3D::GetGlobalTransform() const {
    ERR_FAIL_COND_V(!IsInsideTree(), Affine3());

    if (dirty_ & DIRTY_GLOBAL_TRANSFORM) {
        if (dirty_ & DIRTY_LOCAL_TRANSFORM) {
            UpdateLocalTransform();
        }

        if (parent_) {
            global_transform_ = parent_->GetGlobalTransform() * local_transform_;
        } else {
            global_transform_ = local_transform_;
        }

        if (disable_scale_) {
            global_transform_.Orthonormalize();
        }

        dirty_ &= ~DIRTY_GLOBAL_TRANSFORM;
    }

    return global_transform_;
}

void Node3D::SetGlobalPosition(const Vector3 &position) {
    Affine3 transform = GetGlobalTransform();
    transform.translation() = position;
    SetGlobalTransform(transform);
}

Vector3 Node3D::GetGlobalPosition() const {
    return GetGlobalTransform().translation();
}

void Node3D::SetGlobalRotation(const EulerAngle &euler_rad, EulerOrder order) {
    Affine3 transform = GetGlobalTransform();
    transform.SetEulerAngle(euler_rad, order);
    SetGlobalTransform(transform);
}

EulerAngle Node3D::GetGlobalRotation(EulerOrder order) const {
    return GetGlobalTransform().GetEulerAngle(order);
}

void Node3D::SetGlobalRotationDegree(const Vector3 &euler_deg, EulerOrder order) {
    Vector3 radians{DEG_TO_RAD(euler_deg.x()), DEG_TO_RAD(euler_deg.y()), DEG_TO_RAD((euler_deg.z()))};
    SetGlobalRotation(radians, order);
}

Vector3 Node3D::GetGlobalRotationDegree(EulerOrder order) const {
    Vector3 radians = GetGlobalRotation(order);
    return {RAD_TO_DEG(radians.x()), RAD_TO_DEG(radians.y()), RAD_TO_DEG(radians.z())};
}

Matrix3 Node3D::GetRotationMatrix() const {
    return GetTransform().linear();
}

Affine3 Node3D::GetRelativeTransform(const Node *parent) const {
    if (parent == this) {
        return {};
    }

    ERR_FAIL_COND_V(!parent, Affine3());

    if (parent == parent_) {
        return GetTransform();
    } else {
        return parent_->GetRelativeTransform(parent) * GetTransform();
    }
}

void Node3D::Rotate(const Vector3 &axis, real_t angle) {
    Affine3 t = GetGlobalTransform();
    t.linear() = AngleAxis(angle, axis.normalized()) * t.linear();
    SetTransform(t);
}

void Node3D::RotateLocal(const Vector3 &axis, real_t angle) {
    Affine3 t = GetGlobalTransform();
    t.linear() = t.linear() * AngleAxis(angle, axis.normalized());
    SetTransform(t);
}

void Node3D::RotateX(real_t angle) {
    Affine3 t = GetGlobalTransform();
    t.linear() = AngleAxis(angle, Vector3::UnitX()) * t.linear();
    SetTransform(t);
}

void Node3D::RotateY(real_t angle) {
    Affine3 t = GetGlobalTransform();
    t.linear() = AngleAxis(angle, Vector3::UnitY()) * t.linear();
    SetTransform(t);
}

void Node3D::RotateZ(real_t angle) {
    Affine3 t = GetGlobalTransform();
    t.linear() = AngleAxis(angle, Vector3::UnitZ()) * t.linear();
    SetTransform(t);
}

void Node3D::Translate(const Vector3 &offset) {
    Affine3 t = GetGlobalTransform();
    t.translation() += t.linear() * offset;
    SetTransform(t);
}

void Node3D::TranslateLocal(const Vector3 &offset) {
    Affine3 t = GetGlobalTransform();
    t.translation() += offset;
    SetTransform(t);
}

void Node3D::Scale(const Vector3 &ratio) {
    Affine3 t = GetGlobalTransform();
    t.prescale(ratio);
    SetTransform(t);
}

void Node3D::ScaleLocal(const Vector3 &ratio) {
    Affine3 t = GetGlobalTransform();
    t.scale(ratio);
    SetTransform(t);
}

Vector3 Node3D::ToLocal(const Vector3 &global) const {
    return GetGlobalTransform().inverse().linear() * global;
}

Vector3 Node3D::ToGlobal(const Vector3 &local) const {
    return GetGlobalTransform().linear() * local;
}

void Node3D::SetNotifyTransform(bool enabled) {
    notify_transform_ = enabled;
}

bool Node3D::IsTransformNotificationEnabled() const {
    return notify_transform_;
}

void Node3D::SetNotifyLocalTransform(bool enabled) {
    notify_local_transform_ = enabled;
}

bool Node3D::IsLocalTransformNotificationEnabled() const {
    return notify_local_transform_;
}

void Node3D::SetIdentity() {
    SetTransform(Affine3::Identity());
}

void Node3D::SetVisible(bool visible) {
    if (visible_ == visible)
        return;

    visible_ = visible;

    if (!IsInsideTree())
        return;
    // todo: PropagateVisibilityChange();
}

void Node3D::Show() {
    SetVisible(true);
}

void Node3D::Hide() {
    SetVisible(false);
}

bool Node3D::IsVisible() const {
    return visible_;
}

bool Node3D::IsVisibleInTree() const {
    const Node3D *s = this;

    while (s) {
        if (!s->visible_) {
            return false;
        }
        s = s->parent_;
    }

    return true;
}

//void Node3D::ForceUpdateTransform() {
//    ERR_FAIL_COND(!IsInsideTree());
//
//    // todo: update change list
//
//    Notification(NotificationType::TransformChanged);
//}

void Node3D::SetVisibilityParent(const NodePath &path) {
    visibility_parent_path_ = path;

    if (IsInsideTree()) {
        // todo: update visibility parent
    }
}

NodePath Node3D::GetVisibilityParent() const {
    return visibility_parent_path_;
}


} // End of namespace gobot