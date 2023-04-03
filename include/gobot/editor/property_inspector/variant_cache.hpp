/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-3-31
*/

#pragma once

#include "gobot/core/types.hpp"
#include "gobot/core/object.hpp"

namespace gobot {

struct VariantCache {
    Variant variant;
    Instance instance;
    Type type;
    Object* object{nullptr};

    explicit VariantCache(Variant p_variant, Instance p_instance)
            : variant(p_variant),
              instance(p_instance.get_type().get_raw_type().is_wrapper() ?
                       p_instance.get_wrapped_instance() :
                       p_instance),
              type(instance.get_derived_type().get_raw_type()),
              object(instance.try_convert<Object>())
    {
    }
};


}
