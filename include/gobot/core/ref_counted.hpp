/* Copyright(c) 2020-2022, Qiqi Wu<1258552199@qq.com>.
 * This file is created by Qiqi Wu, 22-11-26
*/

#pragma once

#include <intrusive_ptr.hpp>

#include "gobot/core/object.hpp"

namespace gobot::core {

template <typename T>
using Ref = third_part::intrusive_ptr<T>;

template <typename T>
using RefWeak = third_part::intrusive_weak_ptr<T>;

class RefCounted : public third_part::intrusive_base<RefCounted>, public Object  {
    GOBCLASS(RefCounted, Object)
public:

};



}

namespace godot {

template<typename T, typename ...Args>
auto make_ref(Args &&... args){
    return third_part::make_intrusive<T>(args...);
}

template<typename U, typename T>
gobot::core::Ref<U> static_pointer_cast(gobot::core::Ref<T> ref) noexcept {
    const auto u = static_cast<U *>(ref.get());
    ref.release();
    return gobot::core::Ref<U>(u);
}

template<typename U, typename T>
gobot::core::Ref<U> dynamic_pointer_cast(gobot::core::Ref<T> ref) noexcept {
    const auto u = dynamic_cast<U *>(ref.get());
    if(u){
        ref.release();
    }
    return gobot::core::Ref<U>(u);
}

template<typename U, typename T>
gobot::core::Ref<U> const_pointer_cast(gobot::core::Ref<T> ref) noexcept {
    const auto u = const_cast<U *>(ref.get());
    ref.release();
    return gobot::core::Ref<U>(u);
}

}
