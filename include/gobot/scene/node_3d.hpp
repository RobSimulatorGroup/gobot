/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Zikun Yu, 23-2-13
*/

#pragma once

#include "gobot/scene/node.hpp"

#include <Eigen/Dense>
#include <Eigen/Geometry>
//#include <cmath>

namespace gobot {

using Matrix3d = Eigen::Matrix3d;
using Vector3d = Eigen::Vector3d;
using Transform3d = Eigen::Affine3d;
using Quaternion = Eigen::Quaterniond;

class ViewPort;

class GOBOT_EXPORT Node3D : public Node {
    GOBCLASS(Node3D, Node)

public:
    enum class RotationEditMode {
        Euler,
        Quaternion,
    };

    enum class EulerOrder {
        XYZ,
        ZYX
    };

    Node3D() = default;

    Node3D *GetParentNode3D() const;

//    Ref<World3D> GetWorld3D() const;

    void SetPosition(const Vector3d &position);

    void SetRotationEditMode(RotationEditMode mode);
    RotationEditMode GetRotationEditMode() const;

    void SetRotationOrder(EulerOrder order);
    void SetRotation(const Vector3d &euler_rad);
    void SetRotationDeg(const Vector3d &euler_deg);
    void SetScale(const Vector3d &scale);

    void SetGlobalPosition(const Vector3d &position);
//    void SetGlobalRotation(const Vector3d &euler_rad);
//    void SetGlobalRotationDeg(const Vector3d &euler_deg);


    void SetTransform(const Transform3d &transform);
//    void SetQuaternion(const Quaternion &quaternion);
    void SetGlobalTransform(const Transform3d &transform);

    Vector3d GetPosition() const;

    EulerOrder GetRotationOrder() const;
    Vector3d GetRotation() const;
    Vector3d GetRotationDeg() const;
    Vector3d GetScale() const;

//    Vector3d GetGlobalPosition() const;
//    Vector3d GetGlobalRotation() const;
//    Vector3d GetGlobalRotationDeg() const;

//
//    Transform3d GetTransform() const;
//    Transform3d GetBasis() const;
//    Quaternion GetQuaternion() const;
    Transform3d GetGlobalTransform() const;

//    Transform3d GetRelativeTransform(const Node *parent) const;
//
//    void Rotate(const Vector3d &axis, double angle);
//    void RotateX(double angle);
//    void RotateY(double angle);
//    void RotateZ(double angle);
//    void Translate(const Vector3d &offset);
//    void Scale(const Vector3d &ratio);
//
//    void RotateObjectLocal(const Vector3d &axis, double angle);
//    void ScaleObjectLocal(const Vector3d &ratio);
//    void TranslateObjectLocal(const Vector3d &offset);
//
//    Vector3d ToLocal(Vector3d global) const;
//    Vector3d ToGlobal(Vector3d local) const;
//
//    void SetNotifyTransform(bool enabled);
//    bool IsTransformNotificationEnabled() const;
//
//    void SetNotifyLocalTransform(bool enabled);
//    bool IsLocalTransformNotificationEnabled() const;
//
//    void Orthonormalize();
//    void SetIdentity();
//
//    void SetVisible(bool visible);
//    void Show();
//    void Hide();
//    bool IsVisible() const;
//    bool IsVisibleInTree() const;
//
//    void ForceUpdateTransform();
//
//    void SetVisibilityParent(const NodePath &path);
//    NodePath GetVisibilityParent() const;

protected:
    FORCE_INLINE void UpdateLocalTransform() const;
    FORCE_INLINE void UpdateRotationAndScale() const;

    void NotificationCallBack(NotificationType notification);

private:
    enum TransformDirty {
        DIRTY_NONE = 0,
        DIRTY_EULER_AND_SCALE = 1,
        DIRTY_LOCAL_TRANSFORM = 2,
        DIRTY_GLOBAL_TRANSFORM = 4,
    };

    void PropagateTransformChanged(Node3D *origin);

    FORCE_INLINE double Deg2Rad(double deg) const { return deg * (M_PI / 180.0); };
    FORCE_INLINE double Rad2Deg(double rad) const { return rad * (180.0 / M_PI); };

    static FORCE_INLINE Matrix3d Euler2Matrix(const Vector3d &euler, EulerOrder order);
    static FORCE_INLINE Vector3d Matrix2Euler(const Matrix3d &m, EulerOrder order);

    static FORCE_INLINE Vector3d GetScaleFromTransform(const Transform3d &transform) ;
    static FORCE_INLINE Vector3d GetEulerFromTransform(const Transform3d &transform, EulerOrder order) ;

    mutable Transform3d global_transform_ = Transform3d::Identity();
    mutable Transform3d local_transform_ = Transform3d::Identity();
    mutable EulerOrder euler_rotation_order_ = EulerOrder::XYZ;
    mutable Vector3d euler_rotation_ = Vector3d::Zero();
    mutable Vector3d scale_ = Vector3d{1, 1, 1};
    mutable RotationEditMode rotation_edit_mode_ = RotationEditMode::Euler;

    mutable int dirty_ = DIRTY_NONE;

    ViewPort *viewport_ = nullptr;

    bool inside_world_ = false;

    Node3D *parent_ = nullptr;
    std::vector<Node3D *> children_;

//    bool ignore_notification_ = false;
    bool notify_local_transform_ = false;
//    bool notify_transform_ = false;

    bool visible_ = false;
    bool disable_scale_ = false;
};

} // End of namespace gobot