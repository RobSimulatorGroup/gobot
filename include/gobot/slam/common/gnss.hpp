/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 24-3-24.
*/

#pragma once

#include <utility>

#include "gobot/core/math/matrix.hpp"
#include "gobot/core/math/lie_groups.hpp"

namespace gobot::slam {

/// GNSS状态位信息
/// 通常由GNSS厂商提供，这里使用千寻提供的状态位
enum class GpsStatusType {
    GNSS_FLOAT_SOLUTION = 5,         // 浮点解（cm到dm之间）
    GNSS_FIXED_SOLUTION = 4,         // 固定解（cm级）
    GNSS_PSEUDO_SOLUTION = 2,        // 伪距差分解（分米级）
    GNSS_SINGLE_POINT_SOLUTION = 1,  // 单点解（10m级）
    GNSS_NOT_EXIST = 0,              // GPS无信号
    GNSS_OTHER = -1,                 // 其他
};

/// UTM 坐标
struct UTMCoordinate {
    UTMCoordinate() = default;
    explicit UTMCoordinate(int zone, Vector2d xy = Vector2d::Zero(), bool north = true)
            : zone_(zone), xy_(std::move(xy)), north_(north) {}

    int zone_ = 0;              // utm 区域
    Vector2d xy_ = Vector2d::Zero();  // utm xy
    double z_ = 0;              // z 高度（直接来自于gps）
    bool north_ = true;         // 是否在北半球
};

struct GNSS {
    GNSS() = default;
    GNSS(double unix_time, int status, Vector3d lat_lon_alt, double heading, bool heading_valid)
            : unix_time_(unix_time), lat_lon_alt_(std::move(lat_lon_alt)), heading_(heading), heading_valid_(heading_valid) {
        status_ = GpsStatusType(status);
    }

    double unix_time_ = 0;                                  // unix系统时间
    GpsStatusType status_ = GpsStatusType::GNSS_NOT_EXIST;  // GNSS 状态位
    Vector3d lat_lon_alt_ = Vector3d::Zero();                     // 经度、纬度、高度，前二者单位为度
    double heading_ = 0.0;                                  // 双天线读到的方位角，单位为度
    bool heading_valid_ = false;                            // 方位角是否有效

    UTMCoordinate utm_;       // UTM 坐标（区域之类的也在内）
    bool utm_valid_ = false;  // UTM 坐标是否已经计算（若经纬度给出错误数值，此处也为false）

    SE3 utm_pose_;  // 用于后处理的6DoF Pose
};

}
