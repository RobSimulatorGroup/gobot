/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 24-3-23.
*/

#pragma once

#include "gobot/core/math/matrix.hpp"


namespace gobot::slam {

struct IMU {
    IMU() = default;
    IMU(double t, const Vector3d& gyro, const Vector3d& acce) : timestamp_(t), gyro_(gyro), acce_(acce) {}

    double timestamp_ = 0.0;
    Vector3d gyro_ = Vector3d::Zero();
    Vector3d acce_ = Vector3d::Zero();
};

}
