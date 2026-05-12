/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * This file is created by Qiqi Wu, 23-3-31
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "gobot/core/types.hpp"
#include "gobot/core/object.hpp"

namespace gobot {

struct VariantCache {
    Variant& variant;
    Instance instance;
    Type type;
    Object* object{nullptr};

    explicit VariantCache(Variant& p_variant)
            : variant(p_variant),
              instance(Instance(p_variant).get_type().get_raw_type().is_wrapper() ?
                       Instance(p_variant).get_wrapped_instance() :
                       Instance(p_variant)),
              type(instance.get_derived_type().get_raw_type()),
              object(instance.try_convert<Object>())
    {
    }
};


}
