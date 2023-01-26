/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-1-13
*/

#include "gobot/scene/resources/cylinder_shape_3d.hpp"
#include "gobot/core/registration.hpp"
#include "gobot/log.hpp"


namespace gobot {

CylinderShape3D::CylinderShape3D() {

}

void CylinderShape3D::SetRadius(float radius) {
    if (radius < 0) {
        LOG_ERROR("CylinderShape3D radius cannot be negative.");
        return;
    }
    radius_ = radius;
}

float CylinderShape3D::GetRadius() const {
    return radius_;
}

void CylinderShape3D::SetHeight(float height) {
    if (height < 0) {
        LOG_ERROR("CylinderShape3D height cannot be negative.");
        return;
    }
    height_ = height;
}

float CylinderShape3D::GetHeight() const {
    return height_;
}


}

GOBOT_REGISTRATION {
    Class_<CylinderShape3D>("CylinderShape3D")
            .constructor()(CtorAsRawPtr)
            .property("height", &CylinderShape3D::GetHeight, &CylinderShape3D::SetHeight)
            .property("radius", &CylinderShape3D::GetRadius, &CylinderShape3D::SetRadius);

};