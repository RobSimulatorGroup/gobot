/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 24-3-23.
*/

#pragma once

#include "gobot/core/math/matrix.hpp"
#include "gobot/core/math/lie_groups.hpp"
#include "gobot/slam/common/imu.hpp"
#include "gobot/slam/common/nav_state.hpp"

namespace gobot::slam {

class IMUPreintegration {
public:
    struct Options {
        Options() {}
        Vector3d init_bg_ = Vector3d::Zero();  // bias
        Vector3d init_ba_ = Vector3d::Zero();  // bias
        double noise_gyro_ = 1e-2;       // std of gyro
        double noise_acce_ = 1e-1;       // std of acce
    };

    IMUPreintegration(Options options = Options());

    /**
     * add new imu data
     * @param imu
     * @param dt
     */
    void Integrate(const IMU &imu, double dt);

    /**
     * 从某个起始点开始预测积分之后的状态
     * @param start 起始时时刻状态
     * @return  预测的状态
     */
    NavStated Predict(const NavStated &start, const Vector3d &grav = Vector3d(0, 0, -9.81)) const;


    /// 获取修正之后的观测量，bias可以与预积分时期的不同，会有一阶修正
    SO3 GetDeltaRotation(const Vector3d &bg);

    Vector3d GetDeltaVelocity(const Vector3d &bg, const Vector3d &ba);

    Vector3d GetDeltaPosition(const Vector3d &bg, const Vector3d &ba);

public:
    double dt_ = 0;                          // 整体预积分时间
    Matrix9d cov_ = Matrix9d::Zero();              // 累计噪声矩阵
    Matrix6d noise_gyro_acce_ = Matrix6d::Zero();  // 测量噪声矩阵

    // 零偏
    Vector3d bg_ = Vector3d::Zero();
    Vector3d ba_ = Vector3d::Zero();

    // 预积分观测量
    SO3 dR_;
    Vector3d dv_ = Vector3d::Zero();
    Vector3d dp_ = Vector3d::Zero();

    // 雅可比矩阵
    Matrix3d dR_dbg_ = Matrix3d::Zero();
    Matrix3d dV_dbg_ = Matrix3d::Zero();
    Matrix3d dV_dba_ = Matrix3d::Zero();
    Matrix3d dP_dbg_ = Matrix3d::Zero();
    Matrix3d dP_dba_ = Matrix3d::Zero();
};


}
