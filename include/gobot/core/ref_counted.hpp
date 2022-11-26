/* Copyright(c) 2020-2022, Qiqi Wu<1258552199@qq.com>.
 * This file is created by Qiqi Wu, 22-11-26
*/

#pragma once

#include "gobot/core/object.hpp"
#include <atomic>

namespace gobot::core {


class SafeRefCount {
    std::atomic<uint32_t> count;

public:
//    ALWAYS_INLINE bool ref() { // true on success
//        return count.conditional_increment() != 0;
//    }
//
//    ALWAYS_INLINE uint32_t refval() { // none-zero on success
//        return count.conditional_increment();
//    }
//
//    ALWAYS_INLINE bool unref() { // true if must be disposed of
//        return count.decrement() == 0;
//    }

    ALWAYS_INLINE uint32_t UnRefVal() { // 0 if must be disposed of
        return count.fetch_sub(1, std::memory_order_acq_rel) - 1;
    }

    [[nodiscard]] ALWAYS_INLINE uint32_t Get() const {
        return count.load(std::memory_order_acquire);
    }

    ALWAYS_INLINE void Init(uint32_t value = 1) {
        count.store(value);
    }
};

class RefCounted : public Object {
    GOBCLASS(RefCounted, Object)
public:

    RefCounted();

private:
    SafeRefCount refcount_;
    SafeRefCount refcount_init_;
};

}