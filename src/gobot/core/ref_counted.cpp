/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 22-11-26
*/


#include "gobot/core/ref_counted.hpp"

#include "gobot/core/registration.hpp"

namespace gobot {

RefCounted::RefCounted()
    : Object(true)
{
}

}

GOBOT_REGISTRATION {

    Class_<RefCounted>("RefCounted")
            .constructor()(CtorAsRawPtr)
            .property_readonly("use_count", &RefCounted::use_count)
            .property_readonly("weak_count", &RefCounted::weak_count)
            .method("is_unique", &RefCounted::unique);

};
