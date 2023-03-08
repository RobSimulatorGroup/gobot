/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-23
*/

#include "gobot/core/color.hpp"
#include "gobot/core/registration.hpp"

GOBOT_REGISTRATION {
    Class_<Color>("Color")
        .constructor()(CtorAsObject)
        .property("red", &Color::r_)
        .property("green", &Color::g_)
        .property("blue", &Color::b_)
        .property("alpha", &Color::a_);

};
