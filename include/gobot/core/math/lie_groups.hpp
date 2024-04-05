/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 24-3-23.
*/

#pragma once

#include <sophus/se3.hpp>
#include <sophus/se2.hpp>
#include "geometry.hpp"

namespace gobot {

using SE2d = Sophus::SE2d;
using SE2f = Sophus::SE2f;
using SO2d = Sophus::SO2d;
using SE3d = Sophus::SE3d;
using SE3f = Sophus::SE3f;
using SO3d = Sophus::SO3d;

using SE2 = SE2d;
using SO2 = SO2d;
using SE3 = SE3d;
using SO3 = SO3d;

}
