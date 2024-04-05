/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 24-4-4.
*/

#pragma once

#include "imu.hpp"
#include "odom.hpp"

namespace gobot::slam {


class StaticIMUInit {
public:
    struct Options {
        Options() {}
        double init_time_seconds_ = 10.0;     // time of static init
        int init_imu_queue_max_size_ = 2000;
        int static_odom_pulse_ = 5;           // odom noise of static
        double max_static_gyro_var = 0.5;
        double max_static_acce_var = 0.05;
        double gravity_norm_ = 9.81;
        bool use_speed_for_static_checking_ = true;
    };

    explicit StaticIMUInit(Options options = Options()) : options_(options) {}

    bool AddIMU(const IMU& imu);
    bool AddOdom(const Odom& odom);

    [[nodiscard]] bool InitSuccess() const { return init_success_; }

    [[nodiscard]] const Vector3d& GetCovGyro() const { return cov_gyro_; }
    [[nodiscard]] const Vector3d& GetCovAcce() const { return cov_acce_; }
    [[nodiscard]] const Vector3d& GetInitBg() const { return init_bg_; }
    [[nodiscard]] const Vector3d& GetInitBa() const { return init_ba_; }
    [[nodiscard]] const Vector3d& GetGravity() const { return gravity_; }

private:
    bool TryInit();

    Options options_;
    bool init_success_ = false;
    Vector3d cov_gyro_ = Vector3d::Zero();
    Vector3d cov_acce_ = Vector3d::Zero();
    Vector3d init_bg_  = Vector3d::Zero();
    Vector3d init_ba_  = Vector3d::Zero();
    Vector3d gravity_  = Vector3d::Zero();
    bool is_static_ = false;
    std::deque<IMU> init_imu_deque_;
    double current_time_ = 0.0;
    double init_start_time_ = 0.0;
};


}
