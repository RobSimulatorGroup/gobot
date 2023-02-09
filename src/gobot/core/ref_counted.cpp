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
    : Object(true),
      rc_(0)
{
}

RefCounted::RefCounted(const RefCounted &)
        : rc_(0) {
}

RefCounted::~RefCounted() {
}

RefCounted& RefCounted::operator=(const RefCounted &) {
    // intentionally don't copy reference count
    return *this;
}

void RefCounted::AddRef() noexcept {
    ++rc_;
}

void RefCounted::RemoveRef() noexcept {
    if (--rc_ == 0)
        delete this;
}

}

GOBOT_REGISTRATION {

    Class_<RefCounted>("RefCounted")
            .constructor()(CtorAsRawPtr)
            .property_readonly("use_count", &RefCounted::GetReferenceCount)
            .method("is_unique", &RefCounted::Unique);

};
