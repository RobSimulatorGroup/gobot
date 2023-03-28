/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Zikun Yu, 23-3-24
*/

#pragma once

#include "gobot/scene/collision_object_3d.hpp"

namespace gobot {

class RigidBody3D : public CollisionObject3D {
    GOBCLASS(RigidBody3D, CollisionObject3D);

public:
    RigidBody3D() = default;
    ~RigidBody3D() override = default;

    void SetMass(real_t mass);
    real_t GetMass() const;

    // todo: GetInverseMass

    void SetInertia(const Vector3 &inertia);
    const Vector3 &GetInertia() const;

    // todo: SetCenterOfMassMode

    void SetCenterOfMass(const Vector3 &center_of_mass);
    const Vector3 &GetCenterOfMass() const;

    // todo: SetPhysicsMaterial

    void SetLinearVelocity(const Vector3 &velocity);
    Vector3 GetLinearVelocity() const;

    // todo: SetAxisVelocity

    void SetAngularVelocity(const Vector3 &velocity);
    Vector3 GetAngularVelocity() const;

    // todo: GetInverseInertiaTensor
    // todo: Set/GetGravityScale

    void SetLinearDamp(real_t damp);
    real_t GetLinearDamp() const;

    void SetAngularDamp(real_t damp);
    real_t GetAngularDamp() const;

    // todo: SetContactMonitor
    // todo: GetContactCount
    // todo: SetUseContinuousCollisionDetection
    // todo: GetCollidingBodies

    // todo: ApplyInpulse/Force/Torque

protected:
    void NotificationCallBack(NotificationType notification);

private:
    // todo: BodyStateChangedCallback

    real_t mass_ = 1.0;
    Vector3 inertia_;
    Vector3 center_of_mass_;

    Vector3 linear_velocity_;
    Vector3 angular_velocity_;

    real_t linear_damp_;
    real_t angular_damp_;
};

} // End of namespace gobot