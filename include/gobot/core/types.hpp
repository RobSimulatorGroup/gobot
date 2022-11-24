/* Copyright(c) 2020-2022, Qiqi Wu<1258552199@qq.com>.
 * This file is created by Qiqi Wu, 22-11-6
*/

#pragma once

#include <rttr/variant.h>
#include <QString>
#include <QUuid>
#include <nlohmann/json.hpp>

namespace gobot::core {

using Varint = rttr::variant;
using Type = rttr::type;
using VarintListView = rttr::variant_sequential_view;
using VarintMapView  = rttr::variant_associative_view;
using Instance = rttr::instance;
using Method = rttr::method;
using Argument = rttr::argument;


using String = QString;
using Uuid = QUuid;

using Json = nlohmann::json;

}