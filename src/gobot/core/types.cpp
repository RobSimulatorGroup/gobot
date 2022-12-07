/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 22-11-20
*/

#include "gobot/core/types.hpp"
#include "gobot/core/registration.hpp"

GOBOT_REGISTRATION {

    Class_<String>("String")
            .constructor()(CtorAsObject)
            .constructor<const String&>();

    Class_<Uuid>("Uuid")
            .constructor()(CtorAsObject)
            .method("isNull", &Uuid::isNull)
            .method("toString", overload_cast<>(&Uuid::toString, const_));

    Type::register_converter_func([](const String& value, bool& ok) {
        ok = true;
        return value.toStdString();
    });

    Type::register_converter_func([](const Uuid& value, bool& ok) {
        ok = true;
        return value.toString().toStdString();
    });

};
