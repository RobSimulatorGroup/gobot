/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-17
*/

#include "gobot/core/math/geometry.hpp"
#include "gobot/core/registration.hpp"

GOBOT_REGISTRATION {
    Class_<Quaterniond>("Quaterniond")
            .constructor()(CtorAsObject)
            .property("x", &Quaterniond::GetX, &Quaterniond::SetX)
            .property("y", &Quaterniond::GetY, &Quaterniond::SetY)
            .property("z", &Quaterniond::GetZ, &Quaterniond::SetZ)
            .property("w", &Quaterniond::GetW, &Quaterniond::SetW);

};