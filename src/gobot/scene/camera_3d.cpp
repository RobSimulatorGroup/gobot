/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-28
*/

#include "gobot/scene/camera_3d.hpp"
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
    fovy_ = fovy_degrees;
    near_ = z_near;
    far_ = z_far;
}

void Camera3D::SetViewMatrix(const Vector3& eye, const Vector3& at, const Vector3& up) {
    eye_ = eye;
    at_ = at;
    up_ = up;
    SetGlobalTransform(Affine3(Matrix4::LookAt(eye_, at_, up_)));
}

Matrix4 Camera3D::GetViewMatrix() const {
    // TODO(wqq): Add cache
    return Matrix4::LookAt(eye_, at_, up_);
}

Matrix4 Camera3D::GetProjectionMatrix() const {
    // TODO(wqq): Add cache
    // TODO(wqq): add Ortho projection
    return Matrix4f::Perspective(fovy_, aspect_, near_, far_);
}

}
