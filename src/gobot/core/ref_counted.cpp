/* Copyright(c) 2020-2022, Qiqi Wu<1258552199@qq.com>.
 * This file is created by Qiqi Wu, 22-11-26
*/


#include "gobot/core/ref_counted.hpp"

namespace gobot::core {

RefCounted::RefCounted() {
    refcount_.Init(); // count = 1
    refcount_init_.Init(); // count = 1
}

bool RefCounted::IsReferenced() const {
    return refcount_init_.GetCount() != 1;
}

bool RefCounted::InitRef() {
    if (Reference()) {
        if (!IsReferenced() && refcount_init_.UnRef()) {
            UnReference(); // first referencing is already 1, so compensate for the ref above
        }

        return true;
    } else {
        return false;
    }
}

bool RefCounted::Reference() {
    uint32_t rc_val = refcount_.RefValue();
    bool success = rc_val != 0;
    return success;
}
bool RefCounted::UnReference() {
    uint32_t rc_val = refcount_.UnRefValue();
    bool die = rc_val == 0;
    return die;
}

}