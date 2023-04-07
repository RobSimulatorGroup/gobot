/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Zikun Yu, 23-4-3
*/

#include "gobot/physics/dart/dart_joint_3d.h"

#include <dart/dart.hpp>

namespace gobot {

void DartRevoluteJoint3D::SetParam(PhysicsServer3D::RevoluteJointParam param, real_t value) {
    switch (param) {
        case PhysicsServer3D::RevoluteJointParam::PositionUpperLimit: {
            joint_->setPositionUpperLimit(0, value);
        } break;
        case PhysicsServer3D::RevoluteJointParam::PositionLowerLimit: {
            joint_->setPositionLowerLimit(0, value);
        } break;
        case PhysicsServer3D::RevoluteJointParam::VelocityUpperLimit: {
            joint_->setVelocityUpperLimit(0, value);
        } break;
        case PhysicsServer3D::RevoluteJointParam::VelocityLowerLimit: {
            joint_->setVelocityLowerLimit(0, value);
        } break;
        case PhysicsServer3D::RevoluteJointParam::AccelerationUpperLimit: {
            joint_->setAccelerationUpperLimit(0, value);
        } break;
        case PhysicsServer3D::RevoluteJointParam::AccelerationLowerLimit: {
            joint_->setAccelerationLowerLimit(0, value);
        } break;
        case PhysicsServer3D::RevoluteJointParam::Stiffness: {
            joint_->setSpringStiffness(0, value);
        } break;
        case PhysicsServer3D::RevoluteJointParam::Damping: {
            joint_->setDampingCoefficient(0, value);
        } break;
        case PhysicsServer3D::RevoluteJointParam::Friction: {
            joint_->setCoulombFriction(0, value);
        } break;
    }
}

real_t DartRevoluteJoint3D::GetParam(PhysicsServer3D::RevoluteJointParam param) const {
    switch (param) {
        case PhysicsServer3D::RevoluteJointParam::PositionUpperLimit: {
            return static_cast<real_t>(joint_->getPositionUpperLimit(0));
        } break;
        case PhysicsServer3D::RevoluteJointParam::PositionLowerLimit: {
            return static_cast<real_t>(joint_->getPositionLowerLimit(0));
        } break;
        case PhysicsServer3D::RevoluteJointParam::VelocityUpperLimit: {
            return static_cast<real_t>(joint_->getVelocityUpperLimit(0));
        } break;
        case PhysicsServer3D::RevoluteJointParam::VelocityLowerLimit: {
            return static_cast<real_t>(joint_->getVelocityLowerLimit(0));
        } break;
        case PhysicsServer3D::RevoluteJointParam::AccelerationUpperLimit: {
            return static_cast<real_t>(joint_->getAccelerationUpperLimit(0));
        } break;
        case PhysicsServer3D::RevoluteJointParam::AccelerationLowerLimit: {
            return static_cast<real_t>(joint_->getAccelerationLowerLimit(0));
        } break;
        case PhysicsServer3D::RevoluteJointParam::Stiffness: {
            return static_cast<real_t>(joint_->getSpringStiffness(0));
        } break;
        case PhysicsServer3D::RevoluteJointParam::Damping: {
            return static_cast<real_t>(joint_->getDampingCoefficient(0));
        } break;
        case PhysicsServer3D::RevoluteJointParam::Friction: {
            return static_cast<real_t>(joint_->getCoulombFriction(0));
        } break;
    }
}

}