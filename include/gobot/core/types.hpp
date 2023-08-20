/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 22-11-6
*/

#pragma once

#include <rttr/variant.h>
#include <rttr/enumeration.h>
#include <QString>
#include <QUuid>
#include <QFile>
#include <QByteArray>
#include <nlohmann/json.hpp>

namespace gobot {

using Variant = rttr::variant;
using Type = rttr::type;
using VariantListView = rttr::variant_sequential_view;
using VariantMapView  = rttr::variant_associative_view;
using Instance = rttr::instance;
using Property = rttr::property;
using Method = rttr::method;
using Argument = rttr::argument;
using Enumeration = rttr::enumeration;
using MetaData = rttr::detail::metadata;
using WrapperHolderType = rttr::wrapper_holder_type;


using String = QString;
using Uuid = QUuid;
using FileIODevice = QFile;
using ByteArray = QByteArray;

using Json = nlohmann::json;

}