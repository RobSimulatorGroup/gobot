/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
*/

#pragma once

#include "gobot/core/math/geometry.hpp"
#include "gobot/scene/node_3d.hpp"

namespace gobot {

enum class LinkRole {
    Physical,
    VirtualRoot,
};

class GOBOT_EXPORT Link3D : public Node3D {
    GOBCLASS(Link3D, Node3D)

public:
    Link3D() = default;

    void SetMass(RealType mass);

    RealType GetMass() const;

    void SetCenterOfMass(const Vector3& center_of_mass);

    const Vector3& GetCenterOfMass() const;

    void SetInertiaDiagonal(const Vector3& inertia_diagonal);

    const Vector3& GetInertiaDiagonal() const;

    void SetInertiaOffDiagonal(const Vector3& inertia_off_diagonal);

    const Vector3& GetInertiaOffDiagonal() const;

    void SetRole(LinkRole role);

    LinkRole GetRole() const;

private:
    RealType mass_{0.0};
    Vector3 center_of_mass_{Vector3::Zero()};
    Vector3 inertia_diagonal_{Vector3::Zero()};
    Vector3 inertia_off_diagonal_{Vector3::Zero()};
    LinkRole role_{LinkRole::Physical};
};

}
