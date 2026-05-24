/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * This file is created by Qiqi Wu, 23-3-28
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/editor/property_inspector/editor_property.hpp"

#include <fmt/format.h>

namespace gobot {

std::string EditorProperty::GetImGuiID(const char* suffix) const {
    std::string suffix_text = suffix != nullptr ? suffix : "";
    if (const auto* property_data_model = dynamic_cast<const PropertyDataModel*>(data_model_.get())) {
        const auto& cache = property_data_model->GetVariantCache();
        if (cache.object != nullptr) {
            return fmt::format("##{}:{}{}",
                               static_cast<std::uint64_t>(cache.object->GetInstanceId()),
                               property_data_model->GetPropertyName(),
                               suffix_text);
        }

        return fmt::format("##{}:{}{}",
                           fmt::ptr(&cache.variant),
                           property_data_model->GetPropertyName(),
                           suffix_text);
    }

    return fmt::format("##{}{}", fmt::ptr(this), suffix_text);
}

}
