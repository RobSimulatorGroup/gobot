/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
*/

#pragma once

#include <string>

#include "gobot/core/math/geometry.hpp"
#include "gobot/scene/node_3d.hpp"

namespace gobot {

enum class JointType {
    Fixed,
    Revolute,
    Continuous,
    Prismatic,
    Floating,
    Planar
};

class GOBOT_EXPORT Joint3D : public Node3D {
    GOBCLASS(Joint3D, Node3D)

public:
    Joint3D() = default;

    void SetJointType(JointType joint_type);

    JointType GetJointType() const;

    void SetParentLink(std::string parent_link);

    const std::string& GetParentLink() const;

    void SetChildLink(std::string child_link);

    const std::string& GetChildLink() const;

    void SetAxis(const Vector3& axis);

    const Vector3& GetAxis() const;

    void SetLowerLimit(RealType lower_limit);

    RealType GetLowerLimit() const;

    void SetUpperLimit(RealType upper_limit);

    RealType GetUpperLimit() const;

    void SetEffortLimit(RealType effort_limit);

    RealType GetEffortLimit() const;

    void SetVelocityLimit(RealType velocity_limit);

    RealType GetVelocityLimit() const;

private:
    JointType joint_type_{JointType::Fixed};
    std::string parent_link_;
    std::string child_link_;
    Vector3 axis_{Vector3::UnitX()};
    RealType lower_limit_{0.0};
    RealType upper_limit_{0.0};
    RealType effort_limit_{0.0};
    RealType velocity_limit_{0.0};
};

}
