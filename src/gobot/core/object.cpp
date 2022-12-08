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

bool Object::SetProperty(const StringName& name, Argument argument) {
    auto type = GetType();
    for (const auto& prop : type.get_properties()) {
        if (prop.is_readonly() && prop.get_name() == name) {
            return prop.set_value(Instance(this), argument);
        }
    }

    return false;
}


bool Object::GetProperty(const StringName& name, Variant& variant) {
    auto type = GetType();
    for (const auto& prop : type.get_properties()) {
        if (prop.get_name() == name) {
            variant = prop.get_value(Instance(this));
            return true;
        }
    }

    return false;
}

}

GOBOT_REGISTRATION {

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
