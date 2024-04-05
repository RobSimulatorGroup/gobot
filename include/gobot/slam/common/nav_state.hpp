/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 24-3-23.
*/

#pragma once

#include "gobot/core/math/matrix.hpp"
#include "gobot/core/math/lie_groups.hpp"

namespace gobot::slam {

template <typename T>
struct NavState {
    using Vec3 = Eigen::Matrix<T, 3, 1>;
    using SO3 = Sophus::SO3<T>;

    NavState() = default;

    // from time, R, p, v, bg, ba
    explicit NavState(double time,
                      const SO3& R = SO3(),
                      const Vec3& t = Vec3::Zero(),
                      const Vec3& v = Vec3::Zero(),
                      const Vec3& bg = Vec3::Zero(),
                      const Vec3& ba = Vec3::Zero())
            : timestamp_(time), R_(R), p_(t), v_(v), bg_(bg), ba_(ba) { }

    // from pose and vel
    NavState(double time, const SE3& pose, const Vec3& vel = Vec3::Zero())
            : timestamp_(time), R_(pose.so3()), p_(pose.translation()), v_(vel) {}

    Sophus::SE3<T> GetSE3() const { return SE3(R_, p_); }

    friend std::ostream& operator<<(std::ostream& os, const NavState<T>& s) {
        os << "p: " << s.p_.transpose() << ", v: " << s.v_.transpose()
           << ", q: " << s.R_.unit_quaternion().coeffs().transpose() << ", bg: " << s.bg_.transpose()
           << ", ba: " << s.ba_.transpose();
        return os;
    }

    double timestamp_ = 0;    // timestamp
    SO3 R_;                   // rotation
    Vec3 p_ = Vec3::Zero();   // translation
    Vec3 v_ = Vec3::Zero();   // speed
    Vec3 bg_ = Vec3::Zero();  // gyro bias
    Vec3 ba_ = Vec3::Zero();  // acce bias
};

using NavStated = NavState<double>;
using NavStatef = NavState<float>;

}
