/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 22-11-26
*/

#pragma once

#include <intrusive_ptr/intrusive_ptr.hpp>

#include "gobot/core/object.hpp"

namespace gobot {

template <typename T>
using Ref = typename third_party::intrusive_ptr<T>;

template <typename T>
using RefWeak = third_party::intrusive_weak_ptr<T>;

class RefCounted : public third_party::intrusive_base<RefCounted>, public Object  {
    GOBCLASS(RefCounted, Object)
public:
    RefCounted();
};


template<typename T, typename ...Args>
auto MakeRef(Args &&... args){
    return third_party::make_intrusive<T>(args...);
}

template<typename U, typename T>
gobot::Ref<U> static_pointer_cast(gobot::Ref<T> ref) noexcept {
    const auto u = static_cast<U *>(ref.get());
    ref.release();
    return gobot::Ref<U>(u);
}

template<typename U, typename T>
gobot::Ref<U> dynamic_pointer_cast(gobot::Ref<T> ref) noexcept {
    const auto u = dynamic_cast<U *>(ref.get());
    if(u){
        ref.release();
    }
    return gobot::Ref<U>(u);
}

template<typename U, typename T>
gobot::Ref<U> const_pointer_cast(gobot::Ref<T> ref) noexcept {
    const auto u = const_cast<U *>(ref.get());
    ref.release();
    return gobot::Ref<U>(u);
}

} // end of namespace gobot

#include "gobot/core/ref_wrapper_mapper.hpp"
