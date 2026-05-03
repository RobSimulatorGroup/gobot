/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
*/

#include "gobot/scene/link_3d.hpp"

#include "gobot/core/registration.hpp"
#include "gobot/log.hpp"

namespace gobot {

void Link3D::SetMass(RealType mass) {
    if (mass < 0) {
        LOG_ERROR("Link3D mass cannot be negative.");
        return;
    }
    mass_ = mass;
}

RealType Link3D::GetMass() const {
    return mass_;
}

void Link3D::SetCenterOfMass(const Vector3& center_of_mass) {
    center_of_mass_ = center_of_mass;
}

const Vector3& Link3D::GetCenterOfMass() const {
    return center_of_mass_;
}

void Link3D::SetInertiaDiagonal(const Vector3& inertia_diagonal) {
    if ((inertia_diagonal.array() < 0).any()) {
        LOG_ERROR("Link3D inertia diagonal cannot be negative.");
        return;
    }
    inertia_diagonal_ = inertia_diagonal;
}

const Vector3& Link3D::GetInertiaDiagonal() const {
    return inertia_diagonal_;
}

void Link3D::SetInertiaOffDiagonal(const Vector3& inertia_off_diagonal) {
    inertia_off_diagonal_ = inertia_off_diagonal;
}

const Vector3& Link3D::GetInertiaOffDiagonal() const {
    return inertia_off_diagonal_;
}

void Link3D::SetRole(LinkRole role) {
    role_ = role;
}

LinkRole Link3D::GetRole() const {
    return role_;
}

} // namespace gobot

GOBOT_REGISTRATION {

    QuickEnumeration_<LinkRole>("LinkRole");

    Class_<Link3D>("Link3D")
            .constructor()(CtorAsRawPtr)
            .property("mass", &Link3D::GetMass, &Link3D::SetMass)
            .property("center_of_mass", &Link3D::GetCenterOfMass, &Link3D::SetCenterOfMass)
            .property("inertia_diagonal", &Link3D::GetInertiaDiagonal, &Link3D::SetInertiaDiagonal)
            .property("inertia_off_diagonal", &Link3D::GetInertiaOffDiagonal, &Link3D::SetInertiaOffDiagonal)
            .property("role", &Link3D::GetRole, &Link3D::SetRole);

};
