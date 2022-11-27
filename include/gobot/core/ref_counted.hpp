/* Copyright(c) 2020-2022, Qiqi Wu<1258552199@qq.com>.
 * This file is created by Qiqi Wu, 22-11-26
*/

#pragma once

#include <intrusive_ptr.hpp>
#include "gobot/log.hpp"

namespace gobot::core {

class RefCounted : public third_parts::intrusive_base<RefCounted> {


};

template <typename T>
using Ref = third_parts::intrusive_ptr<T>;

template <typename T>
using RefWeak = third_parts::intrusive_weak_ptr<T>;

}