/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Zikun Yu, 23-4-3
*/

#pragma once

#include "gobot/physics/physics_server_3d.h"
#include "gobot/physics/dart/dart_shape_3d.h"

#include <dart/dart.hpp>

namespace gobot {

using DartJoint = dart::dynamics::Joint;

class DartJoint3D {
public:
    DartJoint3D() = default;
    virtual ~DartJoint3D() = default;

    FORCE_INLINE void SetSelf(const RID &self) { self_ = self; }
    [[nodiscard]] FORCE_INLINE RID GetSelf() const { return self_; }

    [[nodiscard]] virtual PhysicsServer3D::JointType GetType() const {
        return PhysicsServer3D::JointType::None;
    };

protected:
    DartJoint *joint_;

private:
    RID self_;
};

class DartFreeJoint3D : public DartJoint3D {
public:
    [[nodiscard]] PhysicsServer3D::JointType GetType() const override {
        return PhysicsServer3D::JointType::Free;
    }
};

class DartFixedJoint3D : public DartJoint3D {
    [[nodiscard]] PhysicsServer3D::JointType GetType() const override {
        return PhysicsServer3D::JointType::Fixed;
    }
};

class DartRevoluteJoint3D : public DartJoint3D {
    [[nodiscard]] PhysicsServer3D::JointType GetType() const override {
        return PhysicsServer3D::JointType::Revolute;
    }

    void SetParam(PhysicsServer3D::RevoluteJointParam param, real_t value);
    [[nodiscard]] real_t GetParam(PhysicsServer3D::RevoluteJointParam param) const;
};

} // End of namespace gobot