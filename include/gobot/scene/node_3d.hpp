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

    /**
     *
     * @return the parent Node3D, or nullptr if no parent exists or parent is not of type Node3D.
     */
    Node3D *GetParentNode3D() const;

    // todo: Ref<World3D> GetWorld3D() const;

    /**
     * @brief Specify the local position or translation of this node relative to the parent.
     *  This is equivalent to Affine.translation().
     *
     * @param position[Vector3]: Local position or translation.
     */
    void SetPosition(const Vector3 &position);
    Vector3 GetPosition() const;

    /**
     * @brief Specify how rotation (and scale) will be presented in the editor.
     *
     * @param mode[RotationEditMode]: Euler, Quaternion or RotationMatrix.
     */
    void SetRotationEditMode(RotationEditMode mode);
    RotationEditMode GetRotationEditMode() const;

    /**
     * @brief Specify the axis rotation order of Euler rotations. The final orientation is constructed
     *  by rotating the Euler angles in the order specified by this property.
     *
     * @param order[EulerOrder]: See gobot::geometry.
     */
    void SetEulerOrder(EulerOrder order);
    EulerOrder GetEulerOrder() const;

    /**
     * @brief Specify rotation part of the local transform in radians, specified in terms of Euler Angles.
     *  The angles construct a rotation in the order specified by the EulerOrder property.
     *  Note that this changes local rotation only while translation part will not follow up.
     *
     * @param euler_rad[EulerAngle/Vector3]: Vector of Euler angles in radians.
     */
    void SetEuler(const EulerAngle &euler_rad);
    EulerAngle GetEuler() const;

    /**
     * @brief Helper property to access Euler rotation in degrees instead of radians.
     *
     * @param euler_deg[EulerAngle/Vector3]: Vector of Euler angles in degrees.
     */
    void SetEulerDegree(const EulerAngle &euler_deg);
    EulerAngle GetEulerDegree() const;

    /**
     * @brief Specify the scale part of the local transform.
     *  Note that mixed negative scales in 3D are not decomposable from the transformation matrix, which means the scale values
     *  will either be all positive or all negative.
     *
     * @param scale[Vector3]: Scale factors as x, y, z components.
     */
    void SetScale(const Vector3 &scale);
    Vector3 GetScale() const;

    /**
     * @brief Access to the node rotation as a Quaternion.
     *  Node:
     *  Quaternion is not an attribute Conversion to Quaternion back and forth a lot may cause precision loss as Quaternion must be
     *  normalized before returning to a rotation matrix.
     *
     * @param quaternion[Quaternion]: Quaternion components in w, x, y, z order.
     */
    void SetQuaternion(const Quaternion &quaternion);
    Quaternion GetQuaternion() const;

    /**
     * @brief Specify local space transform of this node, with respect to the parent node.
     *
     * @param transform[Affine3]: Affine transform with (rotation, translation and scaling).
     */
    void SetTransform(const Affine3 &transform);
    Affine3 GetTransform() const;

    /**
     * @brief Specify global space transform of this node in World3D.
     *
     * @param transform[Affine3]: Affine transform with (rotation, translation and scaling).
     */
    void SetGlobalTransform(const Affine3 &transform);
    Affine3 GetGlobalTransform() const;

    /**
     * @brief Specify global position of this node, equivalent to global_transform_.translation().
     *
     * @param position[Vector3]: Global position.
     */
    void SetGlobalPosition(const Vector3 &position);
    Vector3 GetGlobalPosition() const;

    /**
     * @brief Rotation part of the global transformation in radians, specified in terms of SXYZ order.
     *
     * @param euler_rad[EulerAngle/Vector3]: Vector of Euler angles in radians.
     * @param order[EulerOrder]: See gobot::geometry.
     */
    void SetGlobalRotation(const EulerAngle &euler_rad, EulerOrder order = EulerOrder::SXYZ);
    EulerAngle GetGlobalRotation(EulerOrder order = EulerOrder::SXYZ) const;

    /**
     * @brief Helper property to access global Euler rotation in degrees instead of radians.
     *
     * @param euler_rad[EulerAngle/Vector3]: Vector of Euler angles in degrees.
     * @param order[EulerOrder]: See gobot::geometry.
     */
    void SetGlobalRotationDegree(const Vector3 &euler_deg, EulerOrder order = EulerOrder::SXYZ);
    Vector3 GetGlobalRotationDegree(EulerOrder order = EulerOrder::SXYZ) const;

    /**
     *
     * @return a rotation matrix of local transform, equivalent to trasform_.linear().
     */
    Matrix3 GetRotationMatrix() const;

    /**
     * @brief Get a relative space transform of this node with respect to a specified parent node.
     *
     * @param parent[Affine3]: Affine transform with (rotation, translation and scaling).
     *
     * @return an affine transform.
     */
    Affine3 GetRelativeTransform(const Node *parent) const;

    /**
     * @brief Rotates the local transformation around axis, a unit Vector3, by specified angle in radians.
     *
     * @param axis[Vector3]: Axis of this rotation.
     * @param angle[real_t]: Rotation angle in radians.
     */
    void Rotate(const Vector3 &axis, RealType angle);

    /**
     * @brief Rotates the local transformation around the X axis by angle in radians.
     *
     * @param angle[RealType]: Rotation angle in radians.
     */
    void RotateX(RealType angle);

    /**
     * @brief Rotates the local transformation around the Y axis by angle in radians.
     *
     * @param angle[RealType]: Rotation angle in radians.
     */
    void RotateY(RealType angle);

    /**
     * @brief Rotates the local transformation around the Z axis by angle in radians.
     *
     * @param angle[real_t]: Rotation angle in radians.
     */
    void RotateZ(RealType angle);

    /**
     * @brief Rotates the local transformation by angle in radians around an axis in object-local
     *  coordinate system.
     *
     * @param angle[RealType]: Rotation angle in radians.
     */
    void RotateLocal(const Vector3 &axis, RealType angle);

    /**
     * @brief Changes the node's position by the given offset Vector3.
     *  Note that the translation offset is affected by the node's scale, so if scaled by e.g. (10, 1, 1),
     *  a translation by an offset of (2, 0, 0) would actually add 20 (2 * 10) to the X coordinate.
     *
     * @param offset[Vector3]: Offset the original translation.
     */
    void Translate(const Vector3 &offset);

    /**
     * @brief Changes the node's position by the given offset Vector3 in local space.
     *
     * @param offset[Vector3]: Offset the original translation.
     */
    void TranslateLocal(const Vector3 &offset);

    /**
     * @brief Uniformly scales the local transformation by Vector3 offset.
     *
     * @param ratio[Vector3]: A uniform scale in x, y, z directions.
     */
    void Scale(const Vector3 &ratio);

    /**
     * @brief Scales the global (world) transformation by the given Vector3 scale factors.
     *
     * @param ratio[Vector3]: A uniform scale in x, y, z directions.
     */
    void ScaleLocal(const Vector3 &ratio);

    /**
     * @brief Transforms global position from world space to this node's local space.
     *
     * @param global[Vector3]: Global position in world space.
     *
     * @return a global position in local space.
     */
    Vector3 ToLocal(const Vector3 &global) const;

    /**
     * @brief Transforms local position from this node's local space to global space.
     *
     * @param global[Vector3]: Global position in local space.
     *
     * @return a global position in world space.
     */
    Vector3 ToGlobal(const Vector3 &local) const;

    /**
     * @brief Sets whether the node notifies about its global and local transformation changes.
     *  Node3D will not propagate this by default, unless it is in the editor context and it has a valid gizmo.
     *
     * @param enabled[bool]: Enable notification for transformation changes.
     */
    void SetNotifyTransform(bool enabled);
    bool IsTransformNotificationEnabled() const;

    /**
     * @brief Sets whether the node notifies about its local transformation changes.
     *  Node3D will not propagate this by default.
     *
     * @param enabled[bool]: Enable notification for transformation changes.
     */
    void SetNotifyLocalTransform(bool enabled);
    bool IsLocalTransformNotificationEnabled() const;

    /**
     * @brief Reset all transformations for this node (sets its Affine3 to the identity matrix).
     */
    void SetIdentity();

    /**
     * @brief The node is only visible if all of its antecedents are visible as well.
     *
     * @param visible[bool]: If true, the node is drawn.
     */
    void SetVisible(bool visible);

    /**
     * @brief Enables rendering of this node. Changes visible to true.
     */
    void Show();

    /**
     * @brief Disables rendering of this node. Changes visible to false.
     */
    void Hide();

    bool IsVisible() const;

    /**
     *
     * @return true if the node is present in the SceneTree, its visible property is true and all its antecedents
     *  are also visible. If any antecedent is hidden, this node will not be visible in the scene tree.
     */
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