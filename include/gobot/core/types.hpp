/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 22-11-6
*/

#pragma once

#include <rttr/variant.h>
#include <QString>
#include <QUuid>
#include <nlohmann/json.hpp>

namespace gobot {

using Varint = rttr::variant;
using Type = rttr::type;
using VarintListView = rttr::variant_sequential_view;
using VarintMapView  = rttr::variant_associative_view;
using Instance = rttr::instance;
using Method = rttr::method;
using Argument = rttr::argument;
using Enumeration = rttr::enumeration;


using String = QString;
using Uuid = QUuid;

using Json = nlohmann::json;

}