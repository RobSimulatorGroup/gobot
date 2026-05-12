/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * This file is created by Qiqi Wu, 22-11-6
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <rttr/variant.h>
#include <rttr/enumeration.h>
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


using Json = nlohmann::json;

}