/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 24-4-4.
*/

#include "gobot/slam/common/static_imu_init.hpp"
#include "gobot/core/math/math_util.hpp"
#include "gobot/error_macros.hpp"


namespace gobot::slam {

bool StaticIMUInit::AddIMU(const IMU& imu) {
    if (init_success_) {
        return true;
    }

    if (options_.use_speed_for_static_checking_ && !is_static_) {
        LOG_WARN("Waiting vehicle to be static...");
        init_imu_deque_.clear();
        return false;
    }

    if (init_imu_deque_.empty()) {
        init_start_time_ = imu.timestamp_;
    }

    init_imu_deque_.push_back(imu);

    double init_time = imu.timestamp_ - init_start_time_;
    if (init_time > options_.init_time_seconds_) {
        TryInit();
    }

    while (init_imu_deque_.size() > options_.init_imu_queue_max_size_) {
        init_imu_deque_.pop_front();
    }

    current_time_ = imu.timestamp_;
    return false;
}

bool StaticIMUInit::AddOdom(const Odom& odom) {
    if (init_success_) {
        return true;
    }

    if (odom.left_pulse_ < options_.static_odom_pulse_ && odom.right_pulse_ < options_.static_odom_pulse_) {
        is_static_ = true;
    } else {
        is_static_ = false;
    }

    current_time_ = odom.timestamp_;
    return true;
}


bool StaticIMUInit::TryInit() {
    if (init_imu_deque_.size() < 10) {
        return false;
    }

    // caculate the mean and covariance of gyro and acce
    Vector3d mean_gyro, mean_acce;
    ComputeMeanAndCovDiag(init_imu_deque_, mean_gyro, cov_gyro_,
                          [](const IMU& imu) { return imu.gyro_; });
    ComputeMeanAndCovDiag(init_imu_deque_, mean_acce, cov_acce_,
                          [this](const IMU& imu) { return imu.acce_; });

    std::cout << "mean gyro: " << mean_gyro.transpose() << " acce: " << mean_acce.transpose() << std::endl;
    LOG_INFO("mean gyro: {}", mean_gyro.transpose());
    gravity_ = -mean_acce / mean_acce.norm() * options_.gravity_norm_;

    // 重新计算加计的协方差
    ComputeMeanAndCovDiag(init_imu_deque_, mean_acce, cov_acce_,
                                [this](const IMU& imu) { return imu.acce_ + gravity_; });

    // 检查IMU噪声
    if (cov_gyro_.norm() > options_.max_static_gyro_var) {
//        LOG_ERROR()
//        LOG(ERROR) << "陀螺仪测量噪声太大" << cov_gyro_.norm() << " > " << options_.max_static_gyro_var;
        return false;
    }

    if (cov_acce_.norm() > options_.max_static_acce_var) {
//        LOG_ERROR()
//        LOG(ERROR) << "加计测量噪声太大" << cov_acce_.norm() << " > " << options_.max_static_acce_var;
        return false;
    }

    // 估计测量噪声和零偏
    init_bg_ = mean_gyro;
    init_ba_ = mean_acce;

//    LOG(INFO) << "IMU 初始化成功，初始化时间= " << current_time_ - init_start_time_ << ", bg = " << init_bg_.transpose()
//              << ", ba = " << init_ba_.transpose() << ", gyro sq = " << cov_gyro_.transpose()
//              << ", acce sq = " << cov_acce_.transpose() << ", grav = " << gravity_.transpose()
//              << ", norm: " << gravity_.norm();
//    LOG(INFO) << "mean gyro: " << mean_gyro.transpose() << " acce: " << mean_acce.transpose();
    init_success_ = true;
    return true;
}


}
