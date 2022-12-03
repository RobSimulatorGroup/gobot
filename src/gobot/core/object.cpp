/* Copyright(c) 2020-2022, Qiqi Wu<1258552199@qq.com>.
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 * This file is created by Qiqi Wu, 22-11-20
*/

#include "gobot/core/object.hpp"

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

};
