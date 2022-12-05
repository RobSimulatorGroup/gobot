/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 22-11-20
*/

#include "gobot/core/object.hpp"
#include "gobot/core/notification_enum.hpp"

#include "gobot/core/registration.hpp"

namespace gobot {

Object::Object() {

}


}

GOBOT_REGISTRATION {

    gobot::QuickEnumeration_<gobot::PropertyHint>("PropertyHint");

    gobot::QuickEnumeration_<gobot::PropertyUsageFlags>("PropertyUsageFlags");

    gobot::Class_<gobot::PropertyInfo>("PropertyInfo")
            .constructor()(gobot::CtorAsObject)
            .property("name", &gobot::PropertyInfo::name)
            .property("hint", &gobot::PropertyInfo::hint)
            .property("hint_string", &gobot::PropertyInfo::hint_string)
            .property("PropertyUsageFlags", &gobot::PropertyInfo::usage);

    gobot::Class_<gobot::Object>("Object")
            .constructor()(gobot::CtorAsRawPtr)
            .property_readonly("class_name", &gobot::Object::GetClassName);


    gobot::QuickEnumeration_<gobot::NotificationType>("NotificationType");

};
