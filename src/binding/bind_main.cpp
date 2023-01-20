/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 22-11-20
*/

#ifdef BUILD_WITH_PYBIND11
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#endif

#include "gobot/core/object.hpp"
#include "gobot/core/registration.hpp"

#ifdef BUILD_WITH_PYBIND11
PYBIND11_EMBEDDED_MODULE(gobot, m) {
    using namespace gobot;
#else
GOBOT_REGISTRATION {
#endif



QuickEnumeration_<PropertyHint>("PropertyHint");

QuickEnumeration_<PropertyUsageFlags>("PropertyUsageFlags");

Class_<PropertyInfo>("PropertyInfo")
    .constructor()(CtorAsObject)
    .property("name", &gobot::PropertyInfo::name)
    .property("hint", &gobot::PropertyInfo::hint)
    .property("hint_string", &gobot::PropertyInfo::hint_string)
    .property("PropertyUsageFlags", &gobot::PropertyInfo::usage);

Class_<Object>("Object")
    .constructor()(CtorAsRawPtr)
    .property_readonly("class_name", &Object::GetClassName);


QuickEnumeration_<NotificationType>("NotificationType");

};