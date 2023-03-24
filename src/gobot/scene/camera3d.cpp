/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-28
*/

#include "gobot/scene/camera3d.hpp"
#include "gobot/rendering/render_server.hpp"

namespace gobot {

Camera3D::Camera3D() {

}

void Camera3D::SetFovy(const real_t& fovy) {
    fovy_ = fovy;
}

void Camera3D::SetAspect(const real_t& aspect) {
    aspect_ = aspect;
}

void Camera3D::SetNear(const real_t& near) {
    near_ = near;
}

void Camera3D::SetFar(const real_t& far) {
    far_ = far;
}


void Camera3D::SetPerspective(real_t fovy_degrees, real_t z_near, real_t z_far) {
    if (!force_change_ && fovy_ == fovy_degrees && z_near == near_ && z_far == far_ && mode_ == ProjectionType::Perspective) {
        return;
    }

    fovy_ = fovy_degrees;
    near_ = z_near;
    far_ = z_far;
    mode_ = Perspective;

    force_change_ = false;
}

}
