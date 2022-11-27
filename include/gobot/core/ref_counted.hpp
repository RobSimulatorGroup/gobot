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

namespace gobot {
template<typename T, typename ...Args>
auto make_intrusive(Args &&... args){
    return third_part::make_intrusive<T>(args...);
}

//template<typename _U, typename _T>
//intrusive_ptr<_U> static_pointer_cast(intrusive_ptr<_T> __r) noexcept {
//    const auto __u = static_cast<_U *>(__r.get());
//    __r.release();
//    return intrusive_ptr<_U>(__u);
//}
//template<typename _U, typename _T>
//intrusive_ptr<_U> dynamic_pointer_cast(intrusive_ptr<_T> __r) noexcept {
//    const auto __u = dynamic_cast<_U *>(__r.get());
//    if(__u){
//        __r.release();
//    }
//    return intrusive_ptr<_U>(__u);
//}
//template<typename _U, typename _T>
//intrusive_ptr<_U> const_pointer_cast(intrusive_ptr<_T> __r) noexcept {
//    const auto __u = const_cast<_U *>(__r.get());
//    __r.release();
//    return intrusive_ptr<_U>(__u);
//}

}