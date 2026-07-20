/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * This file is created by Qiqi Wu, 23-2-28
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/scene/camera_3d.hpp"
#include "gobot/core/registration.hpp"
#include "gobot/rendering/render_server.hpp"

namespace gobot {

Camera3D::Camera3D() {

}

void Camera3D::SetFovy(const RealType& fovy) {
    fovy_ = fovy;
}

void Camera3D::SetAspect(const RealType& aspect) {
    aspect_ = aspect;
}

void Camera3D::SetNear(const RealType& near) {
    near_ = near;
}

void Camera3D::SetFar(const RealType& far) {
    far_ = far;
}


void Camera3D::SetPerspective(RealType fovy_degrees, RealType z_near, RealType z_far) {
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

void Camera3D::SetViewMatrixEye(const Vector3& eye) {
    eye_ = eye;
}

void Camera3D::SetViewMatrixAt(const Vector3& at) {
    at_ = at;
}

void Camera3D::SetViewMatrixUp(const Vector3& up) {
    up_ = up;
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

} // namespace gobot

GOBOT_REGISTRATION {
    Class_<gobot::Camera3D>("Camera3D")
            .constructor()(gobot::CtorAsRawPtr)
            .property("fov_y", &gobot::Camera3D::GetFovy, &gobot::Camera3D::SetFovy)
            .property("z_near", &gobot::Camera3D::GetNear, &gobot::Camera3D::SetNear)
            .property("z_far", &gobot::Camera3D::GetFar, &gobot::Camera3D::SetFar)
            .property("eye", &gobot::Camera3D::GetViewMatrixEye, &gobot::Camera3D::SetViewMatrixEye)
            .property("target", &gobot::Camera3D::GetViewMatrixAt, &gobot::Camera3D::SetViewMatrixAt)
            .property("up", &gobot::Camera3D::GetViewMatrixUp, &gobot::Camera3D::SetViewMatrixUp);
};
