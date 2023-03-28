/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Zikun Yu, 23-3-24
*/

#include "gobot/scene/rigid_body_3d.hpp"

namespace gobot {

void RigidBody3D::NotificationCallBack(NotificationType notification) {

}

void RigidBody3D::SetMass(real_t mass) {
    ERR_FAIL_COND(mass <= 0);
    mass_ = mass;
    // todo: PhysicsServer3D::GetInstance()->BodySetParam(GetRID(), PhysicsServer3D::BODY_PARAM_MASS, mass_);
}

real_t RigidBody3D::GetMass() const {
    return mass_;
}

void RigidBody3D::SetInertia(const Vector3 &inertia) {
    ERR_FAIL_COND(inertia.x() < 0);
    ERR_FAIL_COND(inertia.y() < 0);
    ERR_FAIL_COND(inertia.z() < 0);

    inertia_ = inertia;
    // todo: PhysicsServer3D::GetInstance()->BodySetParam(GetRID(), PhysicsServer3D::BODY_PARAM_INERTIA, inertia_);
}

const Vector3 &RigidBody3D::GetInertia() const {
    return inertia_;
}

void RigidBody3D::SetCenterOfMass(const Vector3 &center_of_mass) {
    if (center_of_mass_ == center_of_mass) {
        return;
    }

    center_of_mass_ = center_of_mass;
    // todo: PhysicsServer3D::GetInstance()->BodySetParam(GetRID(), PhysicsServer3D::BODY_PARAM_CENTER_OF_MASS, center_of_mass_);
}

const Vector3 &RigidBody3D::GetCenterOfMass() const {
    return center_of_mass_;
}

void RigidBody3D::SetLinearVelocity(const Vector3 &velocity) {
    linear_velocity_ = velocity;
    // todo: PhysicsServer3D::GetInstance()->BodySetState(GetRID(), PhysicsServer3D::BODY_STATE_LINEAR_VELOCITY, linear_velocity_);
}

Vector3 RigidBody3D::GetLinearVelocity() const {
    return linear_velocity_;
}

void RigidBody3D::SetAngularVelocity(const Vector3 &velocity) {
    angular_velocity_ = velocity;
    // todo: PhysicsServer3D::GetInstance()->BodySetState(GetRID(), PhysicsServer3D::BODY_STATE_ANGULAR_VELOCITY, angular_velocity_);
}

Vector3 RigidBody3D::GetAngularVelocity() const {
    return angular_velocity_;
}

void RigidBody3D::SetLinearDamp(real_t damp) {
    ERR_FAIL_COND(damp < 0.0);
    linear_damp_ = damp;
    // todo: PhysicsServer3D::GetInstance()->BodySetParam(GetRID(), PhysicsServer3D::BODY_PARAM_LINEAR_DAMP, linear_damp_);
}

real_t RigidBody3D::GetLinearDamp() const {
    return linear_damp_;
}

void RigidBody3D::SetAngularDamp(real_t damp) {
    ERR_FAIL_COND(damp < 0.0);
    angular_damp_ = damp;
    // todo: PhysicsServer3D::GetInstance()->BodySetParam(GetRID(), PhysicsServer3D::BODY_PARAM_ANGULAR_DAMP, angular_damp_);
}

real_t RigidBody3D::GetAngularDamp() const {
    return angular_damp_;
}

} // End of namespace gobot