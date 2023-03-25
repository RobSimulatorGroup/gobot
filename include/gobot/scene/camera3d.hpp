/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-28
*/

#pragma once

#include "gobot/scene/node_3d.hpp"

namespace gobot {

class GOBOT_EXPORT Camera3D : public Node3D {
    GOBCLASS(Camera3D, Node3D)
public:
    enum ProjectionType {
        Perspective,
        Orthogonal
    };

    Camera3D();

    void SetFovy(const real_t& fovy);

    FORCE_INLINE real_t GetFovy() const { return fovy_; }

    void SetAspect(const real_t& aspect);

    FORCE_INLINE real_t GetAspect() const { return aspect_; }

    void SetNear(const real_t& near);

    FORCE_INLINE real_t GetNear() const { return near_; }

    void SetFar(const real_t& far);

    FORCE_INLINE real_t GetFar() const { return far_; }

    FORCE_INLINE ProjectionType GetProjectionType() const { return mode_; }

    void SetPerspective(real_t fovy_degrees, real_t z_near, real_t z_far);

    void SetViewMatrix(const Vector3& eye, const Vector3& at, const Vector3& up);

    FORCE_INLINE Vector3 GetViewMatrixEye() const { return eye_; };

    FORCE_INLINE Vector3 GetViewMatrixAt() const { return at_; };

    FORCE_INLINE Vector3 GetViewMatrixUp() const { return up_; };

    Matrix4 GetViewMatrix(Handedness handedness = Handedness::Left) const;

    Matrix4 GetProjectionMatrix(Handedness handedness = Handedness::Left) const;

private:
    real_t fovy_ = 75.0;
    real_t aspect_ = 1.0;
    real_t near_ = 0.05;
    real_t far_ = 4000.0;

    ProjectionType mode_ = Perspective;

    // redundant info for ViewMatrix
    Vector3 eye_{};
    Vector3 at_{};
    Vector3 up_{};
};


}
