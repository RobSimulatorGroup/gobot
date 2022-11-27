/* Copyright(c) 2020-2022, Qiqi Wu<1258552199@qq.com>.
 * This file is created by Qiqi Wu, 22-11-27
*/

#pragma once

#include <atomic>
#include "gobot/core/marcos.hpp"

namespace gobot::core {

template <class T, typename SFINAE = std::enable_if_t<std::atomic<T>::is_always_lock_free>>
class SafeNumeric {
public:
    ALWAYS_INLINE explicit SafeNumeric(T value = static_cast<T>(0)) {
        SetValue(value);
    }

    ALWAYS_INLINE void SetValue(T p_value) {
        value_.store(p_value, std::memory_order_release);
    }

    [[nodiscard]] ALWAYS_INLINE T GetValue() const {
        return value_.load(std::memory_order_acquire);
    }

    // Return the value after increment
    ALWAYS_INLINE T Increment() {
        return value_.fetch_add(1, std::memory_order_acq_rel) + 1;
    }

    // Returns the original value before increment
    ALWAYS_INLINE T PostIncrement() {
        return value_.fetch_add(1, std::memory_order_acq_rel);
    }

    // Return the value after decrement
    ALWAYS_INLINE T Decrement() {
        return value_.fetch_sub(1, std::memory_order_acq_rel) - 1;
    }

    // Returns the original value before decrement
    ALWAYS_INLINE T PostDecrement() {
        return value_.fetch_sub(1, std::memory_order_acq_rel);
    }


    ALWAYS_INLINE T Add(T value) {
        return value_.fetch_add(value, std::memory_order_acq_rel) + value;
    }

    // Returns the original value before add
    ALWAYS_INLINE T PostAdd(T value) {
        return value_.fetch_add(value, std::memory_order_acq_rel);
    }

    ALWAYS_INLINE T Sub(T value) {
        return value_.fetch_sub(value, std::memory_order_acq_rel) - value;
    }

    // Returns the original value before sub
    ALWAYS_INLINE T PostSub(T value) {
        return value_.fetch_sub(value, std::memory_order_acq_rel);
    }

    // Exchange current_value with input_value if input_value is >= current_value
    ALWAYS_INLINE T ExchangeIfGreater(T value) {
        while (true) {
            T tmp = value_.load(std::memory_order_acquire);
            if (tmp >= value) {
                return tmp; // already greater, or equal
            }

            if (value_.compare_exchange_weak(tmp, value, std::memory_order_acq_rel)) {
                return value;
            }
        }
    }

    // Increment if current value is not 0
    ALWAYS_INLINE T ConditionalIncrement() {
        while (true) {
            T c = value_.load(std::memory_order_acquire);
            if (c == 0) {
                return 0;
            }
            if (value_.compare_exchange_weak(c, c + 1, std::memory_order_acq_rel)) {
                return c + 1;
            }
        }
    }

private:
    std::atomic<T> value_;
};

class SafeFlag {
public:

    ALWAYS_INLINE explicit SafeFlag(bool value = false) {
        SetTo(value);
    }

    [[nodiscard]] ALWAYS_INLINE bool IsSet() const {
        return flag_.load(std::memory_order_acquire);
    }

    ALWAYS_INLINE void Set() {
        flag_.store(true, std::memory_order_release);
    }

    ALWAYS_INLINE void Clear() {
        flag_.store(false, std::memory_order_release);
    }

    ALWAYS_INLINE void SetTo(bool p_value) {
        flag_.store(p_value, std::memory_order_release);
    }

private:
    std::atomic_bool flag_;
};

class SafeRefCount {
public:
    // Add reference once if it is already referenced.
    // return true if is a referenced count.
    ALWAYS_INLINE bool Ref() {
        return count_.ConditionalIncrement() != 0;
    }

    // Add reference once if it is already referenced.
    // return the referenced count value.
    ALWAYS_INLINE uint32_t RefValue() {
        return count_.ConditionalIncrement();
    }

    // Remove reference once
    // return true if referenced count -1 =0.
    ALWAYS_INLINE bool UnRef() {
        return count_.Decrement() == 0;
    }

    // Remove reference once
    // return the referenced count value.
    ALWAYS_INLINE uint32_t UnRefValue() {
        return count_.Decrement();
    }

    // return the referenced count value.
    [[nodiscard]] ALWAYS_INLINE uint32_t GetCount() const {
        return count_.GetValue();
    }

    ALWAYS_INLINE void Init(uint32_t value = 1) {
        count_.SetValue(value);
    }

private:
    SafeNumeric<uint32_t> count_;
};


}
