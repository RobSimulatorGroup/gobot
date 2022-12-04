/* Copyright(c) 2020-2022, Qiqi Wu<1258552199@qq.com>.
 * This file is created by Qiqi Wu, 22-11-22
*/

#pragma once

#include "gobot/core/types.hpp"

namespace gobot {

Varint ObjectFromJson(const Type& type,  const Json& json);

Json ObjectToJson(rttr::instance obj);

}