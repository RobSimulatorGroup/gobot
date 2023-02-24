/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Zikun Yu, 23-2-13
*/

#pragma once

#include "gobot/scene/node.hpp"

#include "gobot/core/math/geometry.hpp"

namespace gobot {

class ViewPort;

class GOBOT_EXPORT Node3D : public Node {
    GOBCLASS(Node3D, Node)

public:
    enum class RotationEditMode {
        Euler,
        Quaternion,
        RotationMatrix
    };

    Node3D() = default;

    Node3D *GetParentNode3D() const;

    // todo: Ref<World3D> GetWorld3D() const;

    void SetPosition(const Vector3 &position);
    Vector3 GetPosition() const;

    void SetRotationEditMode(RotationEditMode mode);
    RotationEditMode GetRotationEditMode() const;

    void SetEulerOrder(EulerOrder order);
    EulerOrder GetEulerOrder() const;

    void SetEuler(const EulerAngle &euler_rad);
    EulerAngle GetEuler() const;

    void SetEulerDegree(const EulerAngle &euler_deg);
    EulerAngle GetEulerDegree() const;

    void SetScale(const Vector3 &scale);
    Vector3 GetScale() const;

    void SetQuaternion(const Quaternion &quaternion);
    Quaternion GetQuaternion() const;

    void SetTransform(const Affine3 &transform);
    Affine3 GetTransform() const;

    void SetGlobalTransform(const Affine3 &transform);
    Affine3 GetGlobalTransform() const;

    void SetGlobalPosition(const Vector3 &position);
    Vector3 GetGlobalPosition() const;

    void SetGlobalRotation(const EulerAngle &euler_rad, EulerOrder order = EulerOrder::SXYZ);
    EulerAngle GetGlobalRotation(EulerOrder order = EulerOrder::SXYZ) const;

    void SetGlobalRotationDegree(const Vector3 &euler_deg, EulerOrder order = EulerOrder::SXYZ);
    Vector3 GetGlobalRotationDegree(EulerOrder order = EulerOrder::SXYZ) const;

    Matrix3 GetRotationMatrix() const;

    Affine3 GetRelativeTransform(const Node *parent) const;

    void Rotate(const Vector3 &axis, real_t angle);
    void RotateX(real_t angle);
    void RotateY(real_t angle);
    void RotateZ(real_t angle);
    void RotateLocal(const Vector3 &axis, real_t angle);

    void Translate(const Vector3 &offset);
    void TranslateLocal(const Vector3 &offset);

    void Scale(const Vector3 &ratio);
    void ScaleLocal(const Vector3 &ratio);

    Vector3 ToLocal(const Vector3 &global) const;
    Vector3 ToGlobal(const Vector3 &local) const;

    void SetNotifyTransform(bool enabled);
    bool IsTransformNotificationEnabled() const;

    void SetNotifyLocalTransform(bool enabled);
    bool IsLocalTransformNotificationEnabled() const;

    void SetIdentity();

    void SetVisible(bool visible);
    void Show();
    void Hide();
    bool IsVisible() const;
    bool IsVisibleInTree() const;

    // todo: void ForceUpdateTransform()

    void SetVisibilityParent(const NodePath &path);
    NodePath GetVisibilityParent() const;

protected:
    FORCE_INLINE void UpdateLocalTransform() const;
    FORCE_INLINE void UpdateEulerAndScale() const;

    void NotificationCallBack(NotificationType notification);

private:
    enum TransformDirty {
        DIRTY_NONE = 0,
        DIRTY_EULER_AND_SCALE = 1,
        DIRTY_LOCAL_TRANSFORM = 2,
        DIRTY_GLOBAL_TRANSFORM = 4
    };

    void NotifyDirty();
    void PropagateTransformChanged(Node3D *node);

//    void UpdateVisibilityParent(bool update_root);

    mutable Affine3 global_transform_ = Affine3::Identity();
    mutable Affine3 local_transform_ = Affine3::Identity();
    mutable EulerOrder euler_order_ = EulerOrder::SXYZ;
    mutable EulerAngle euler_ = EulerAngle::Zero();
    mutable Vector3 scale_ = Vector3{1, 1, 1};
    mutable RotationEditMode rotation_edit_mode_ = RotationEditMode::Euler;
    mutable int dirty_ = DIRTY_NONE;

    NodePath visibility_parent_path_;

    Node3D *parent_ = nullptr;
    std::vector<Node3D *> children_;

    bool notify_local_transform_ = false;
    bool notify_transform_ = false;

    bool visible_ = false;
    bool disable_scale_ = false;
};

} // End of namespace gobot