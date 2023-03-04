/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-28
*/

#pragma once

#include "gobot/scene/node_3d.hpp"

namespace gobot {

class Camera3D : public Node3D {
    GOBCLASS(Camera3D, Node3D)
public:
    enum ProjectionType {
        Perspective,
        Orthogonal
    };

    void SetFovy(const real_t& fovy);

    FORCE_INLINE real_t GetFovy() const { return fovy_; }

    void SetAspect(const real_t& aspect);

    FORCE_INLINE real_t GetAspect() const { return fovy_; }

    void SetNear(const real_t& near);

    FORCE_INLINE real_t GetNear() const { return fovy_; }

    void SetFar(const real_t& far);

    FORCE_INLINE real_t GetFar() const { return fovy_; }

    void SetPerspective(real_t fovy_degrees, real_t z_near, real_t z_far);

public:
    Camera3D();

private:
    real_t fovy_ = 75.0;
    real_t aspect_ = 1.0;
    real_t near_ = 0.05;
    real_t far_ = 4000.0;

    ProjectionType mode_ = Perspective;

    bool force_change_{false};
};


}
