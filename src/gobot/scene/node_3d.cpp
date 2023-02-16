/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Zikun Yu, 23-2-13
*/

#include "gobot/scene/node_3d.hpp"

namespace gobot {

void Node3D::NotifyDirty() {

}

void Node3D::NotificationCallBack(NotificationType notification) {
    switch (notification) {
        case NotificationType::EnterTree: {
            ERR_FAIL_COND(!GetTree());

            Node *p = GetParent();
            if (p) {
                parent_ = dynamic_cast<Node3D *>(p);
                parent_->children_.push_back(this);
            }

            dirty_ |= DIRTY_GLOBAL_TRANSFORM; // Global is always dirty upon entering a scene
            NotifyDirty();

            Notification(NotificationType::EnterWorld);
            // todo: update visibility parent

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

Matrix3d Node3D::Euler2Matrix(const Vector3d &euler, EulerOrder order) {
    Matrix3d m;
    switch (order) {
        case EulerOrder::XYZ: {
            m = Eigen::AngleAxisd(euler[0], Vector3d::UnitX())
              * Eigen::AngleAxisd(euler[1], Vector3d::UnitY())
              * Eigen::AngleAxisd(euler[2], Vector3d::UnitZ());
        } break;
        case EulerOrder::ZYX: {
            m = Eigen::AngleAxisd(euler[2], Vector3d::UnitZ())
              * Eigen::AngleAxisd(euler[1], Vector3d::UnitY())
              * Eigen::AngleAxisd(euler[0], Vector3d::UnitX());
        }
    }

    return m;
}

Vector3d Node3D::Matrix2Euler(const Matrix3d &m, EulerOrder order) {
    Vector3d euler;
    switch (order) {
        case EulerOrder::XYZ: {
            euler = m.eulerAngles(0, 1, 2);
        } break;
        case EulerOrder::ZYX: {
            euler = m.eulerAngles(2, 1, 0);
        }
    }

    return euler;
}

Vector3d Node3D::GetScaleFromTransform(const Transform3d &transform) {
    double sx = transform.linear().block<3, 1>(0, 0).norm();
    double sy = transform.linear().block<3, 1>(0, 1).norm();
    double sz = transform.linear().block<3, 1>(0, 2).norm();

    return {sx, sy, sz};
}

Vector3d Node3D::GetEulerFromTransform(const Transform3d &transform, EulerOrder order) {
    return Matrix2Euler(transform.linear(), order);
}

Quaternion Node3D::GetQuaternionFromTransform(const Transform3d &transform) {
    return Quaternion(transform.linear());
}

void Node3D::UpdateLocalTransform() const {
    // This function is called when the local transform (data.local_transform) is dirty
    // and the right value is contained in the Euler rotation and scale.

    local_transform_.linear() = Euler2Matrix(euler_rotation_, euler_rotation_order_);
    local_transform_ *= Eigen::Scaling(scale_[0], scale_[1], scale_[2]);

    dirty_ &= ~DIRTY_LOCAL_TRANSFORM;
}

void Node3D::UpdateRotationAndScale() const {
    // This function is called when the Euler rotation (euler_rotation_) is dirty
    // and the right value is contained in the local transform.

    scale_ = GetScaleFromTransform(local_transform_);
    euler_rotation_ = GetEulerFromTransform(local_transform_, euler_rotation_order_);

    dirty_ &= ~DIRTY_EULER_AND_SCALE;
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

    rotation_edit_mode_ = mode;

    if (mode == RotationEditMode::Euler && (dirty_ & DIRTY_EULER_AND_SCALE)) {
        // If going to Euler mode, ensure that vectors are _not_ dirty, else the retrieved value may be wrong.
        // Otherwise keep what is there, so switching back and forth between modes does not break the vectors.

        UpdateRotationAndScale();
    }

    // todo: notify property list change
}

Node3D::RotationEditMode Node3D::GetRotationEditMode() const {
    return rotation_edit_mode_;
}

void Node3D::SetRotationOrder(EulerOrder order) {
    if (euler_rotation_order_ == order)
        return;

    ERR_FAIL_INDEX(static_cast<int32_t>(order), 6);
    bool transform_changed = false;

    if (dirty_ & DIRTY_EULER_AND_SCALE) {
        UpdateRotationAndScale();
    } else if (dirty_ & DIRTY_LOCAL_TRANSFORM) {
        Matrix3d m = Euler2Matrix(euler_rotation_, euler_rotation_order_);
        euler_rotation_ = Matrix2Euler(m, order);
        transform_changed = true;
    } else {
        dirty_ |= DIRTY_LOCAL_TRANSFORM;
        transform_changed = true;
    }

    euler_rotation_order_ = order;

    if (transform_changed) {
        PropagateTransformChanged(this);
        if (notify_local_transform_) {
            Notification(NotificationType::LocalTransformChanged);
        }
    }
}

void Node3D::SetRotation(const Vector3d &euler_rad) {
    if (dirty_ & DIRTY_EULER_AND_SCALE) {
        // Update scale only if rotation and scale are dirty, as rotation will be overridden.
        scale_ = GetScaleFromTransform(local_transform_);
        dirty_ &= ~DIRTY_EULER_AND_SCALE;
    }

    euler_rotation_ = euler_rad;
    dirty_ = DIRTY_LOCAL_TRANSFORM;
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
    if (dirty_ & DIRTY_EULER_AND_SCALE) {
        // Update rotation only if rotation and scale are dirty, as scale will be overridden.
        euler_rotation_ = Matrix2Euler(local_transform_.linear(), euler_rotation_order_);
        dirty_ &= ~DIRTY_EULER_AND_SCALE;
    }

    scale_ = scale;
    dirty_ = DIRTY_LOCAL_TRANSFORM;
    PropagateTransformChanged(this);

    if (notify_local_transform_) {
        Notification(NotificationType::LocalTransformChanged);
    }
}

void Node3D::SetGlobalPosition(const Vector3d &position) {
    Transform3d transform = GetGlobalTransform();
    transform.translation() = position;
    SetGlobalTransform(transform);
}

void Node3D::SetGlobalRotation(const Vector3d &euler_rad) {
    Transform3d transform = GetGlobalTransform();
    transform.linear() = Euler2Matrix(euler_rad, euler_rotation_order_);
    SetGlobalTransform(transform);
}

void Node3D::SetGlobalRotationDeg(const Vector3d &euler_deg) {
    Vector3d radians(Deg2Rad(euler_deg[0]), Deg2Rad(euler_deg[1]), Deg2Rad(euler_deg[2]));
    SetGlobalRotation(radians);
}

void Node3D::SetTransform(const Transform3d &transform) {
    local_transform_ = transform;
    dirty_ = DIRTY_EULER_AND_SCALE;

    PropagateTransformChanged(this);

    if (notify_local_transform_) {
        Notification(NotificationType::LocalTransformChanged);
    }
}

void Node3D::SetQuaternion(const Quaternion &quaternion) {
    if (dirty_ & DIRTY_EULER_AND_SCALE) {
        // We need the scale part, so update it if dirty
        scale_ = GetScaleFromTransform(local_transform_);
        dirty_ &= ~DIRTY_EULER_AND_SCALE;
    }
    local_transform_.linear() = Matrix3d(quaternion) * Eigen::Scaling(scale_);
    // Rotate should not be marked dirty because that would cause precision loss issues with the scale.
    // Instead reconstruct rotation now.
    euler_rotation_ = GetEulerFromTransform(local_transform_, euler_rotation_order_);

    dirty_ = DIRTY_NONE;

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
    if (dirty_ & DIRTY_EULER_AND_SCALE) {
        UpdateRotationAndScale();
    }

    return euler_rotation_;
}

Vector3d Node3D::GetRotationDeg() const {
    Vector3d radians = GetRotation();
    return {Rad2Deg(radians[0]), Rad2Deg(radians[1]), Rad2Deg(radians[2])};
}

Vector3d Node3D::GetScale() const {
    if (dirty_ & DIRTY_EULER_AND_SCALE) {
        UpdateRotationAndScale();
    }

    return scale_;
}

Vector3d Node3D::GetGlobalPosition() const {
    return GetGlobalTransform().translation();
}

Vector3d Node3D::GetGlobalRotation() const {
    return GetEulerFromTransform(GetGlobalTransform(), euler_rotation_order_);
}

Vector3d Node3D::GetGlobalRotationDeg() const {
    Vector3d radians = GetGlobalRotation();
    return {Rad2Deg(radians[0]), Rad2Deg(radians[1]), Rad2Deg(radians[2])};
}

Transform3d Node3D::GetTransform() const {
    if (dirty_ & DIRTY_LOCAL_TRANSFORM) {
        UpdateLocalTransform();
    }

    return local_transform_;
}

Matrix3d Node3D::GetRotationMatrix() const {
    return GetTransform().linear();
}

Quaternion Node3D::GetQuaternion() const {
    if (dirty_ & DIRTY_LOCAL_TRANSFORM) {
        UpdateLocalTransform();
    }

    return GetQuaternionFromTransform(local_transform_);
}

Transform3d Node3D::GetGlobalTransform() const {
    ERR_FAIL_COND_V(!IsInsideTree(), Transform3d());

    // todo: update local transform if dirty
    if (dirty_ & DIRTY_GLOBAL_TRANSFORM) {
        UpdateLocalTransform();

        if (parent_) {
            global_transform_ = parent_->global_transform_ * local_transform_;
        } else {
            global_transform_ = local_transform_;
        }

        if (disable_scale_) {
            global_transform_.linear().normalize();
        }

        dirty_ &= ~DIRTY_GLOBAL_TRANSFORM;
    }

    return global_transform_;
}

Transform3d Node3D::GetRelativeTransform(const Node *parent) const {
    if (parent == this) {
        return {};
    }

    ERR_FAIL_COND_V(!parent, Transform3d());

    if (parent == parent_) {
        return GetTransform();
    } else {
        return parent_->GetRelativeTransform(parent) * GetTransform();
    }
}

void Node3D::Rotate(const Vector3d &axis, double angle) {
    Transform3d t = GetGlobalTransform();
    t.prerotate(Eigen::AngleAxisd(angle, axis.normalized()));
    SetTransform(t);
}

void Node3D::RotateX(double angle) {
    Transform3d t = GetGlobalTransform();
    t.prerotate(Eigen::AngleAxisd(angle, Vector3d::UnitX()));
    SetTransform(t);
}

void Node3D::RotateY(double angle) {
    Transform3d t = GetGlobalTransform();
    t.prerotate(Eigen::AngleAxisd(angle, Vector3d::UnitY()));
    SetTransform(t);
}

void Node3D::RotateZ(double angle) {
    Transform3d t = GetGlobalTransform();
    t.prerotate(Eigen::AngleAxisd(angle, Vector3d::UnitZ()));
    SetTransform(t);
}

void Node3D::Translate(const Vector3d &offset) {
    Transform3d t = GetGlobalTransform();
    t.pretranslate(offset);
    SetTransform(t);
}

void Node3D::Scale(const Vector3d &ratio) {
    Transform3d t = GetGlobalTransform();
    t.prescale(ratio);
    SetTransform(t);
}

void Node3D::RotateObjectLocal(const Vector3d &axis, double angle) {
    Transform3d t = GetGlobalTransform();
    t.rotate(Eigen::AngleAxisd(angle, Vector3d::UnitZ()));
    SetTransform(t);
}

void Node3D::ScaleObjectLocal(const Vector3d &ratio) {
    Transform3d t = GetGlobalTransform();
    t.scale(ratio);
    SetTransform(t);
}

void Node3D::TranslateObjectLocal(const Vector3d &offset) {
    Transform3d t = GetGlobalTransform();
    t.translate(offset);
    SetTransform(t);
}

Vector3d Node3D::ToLocal(const Vector3d &global) const {
    return GetGlobalTransform().inverse() * global;
}

Vector3d Node3D::ToGlobal(const Vector3d &local) const {
    return GetGlobalTransform() * local;
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
    SetTransform(Transform3d::Identity());
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

void Node3D::ForceUpdateTransform() {
    ERR_FAIL_COND(!IsInsideTree());

    // todo: update change list

    Notification(NotificationType::TransformChanged);
}

void Node3D::SetVisibilityParent(const NodePath &path) {
    visibility_parent_path_ = path;

    if (IsInsideTree()) {
        // todo: update visibility parent
    }
}

NodePath Node3D::GetVisibilityParent() const {
    return visibility_parent_path_;
}

void Node3D::PropagateTransformChanged(Node3D *origin) {
    if (!IsInsideTree()) return;

    for (Node3D *c : children_) {
        c->PropagateTransformChanged(origin);
    }

    // todo: do transform
}

} // End of namespace gobot