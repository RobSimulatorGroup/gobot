/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 22-11-20
*/

#include "gobot/core/object.hpp"

#include <utility>
#include "gobot/core/notification_enum.hpp"

#include "gobot/core/registration.hpp"

namespace gobot {

Object::Object() {

}

bool Object::Set(const String& name, Argument arg) {
    auto type = GetType();
    return type.set_property_value(name.toStdString(), Instance(this), std::move(arg));
}


Variant Object::Get(const String& name) const {
    Variant res;

    auto type = GetType();
    res = type.get_property_value(name.toStdString(), Instance(this));

    return res;
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
