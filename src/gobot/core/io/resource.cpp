/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 22-11-20
*/

#include "gobot/core/io/resource.hpp"

#include "gobot/core/registration.hpp"

namespace gobot {

Resource::Resource() {

}

}

GOBOT_REGISTRATION {
    gobot::Class_<gobot::Resource>("gobot::core::Resource");

};
