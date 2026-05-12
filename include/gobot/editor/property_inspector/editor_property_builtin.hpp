/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * This file is created by Qiqi Wu, 23-3-30
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "gobot/editor/property_inspector/editor_property.hpp"
#include "gobot/editor/imgui/imgui_utilities.hpp"
#include "gobot/error_macros.hpp"

namespace gobot {

class EditorBuiltInProperty : public EditorProperty {
    GOBCLASS(EditorBuiltInProperty, EditorProperty)
public:
    explicit EditorBuiltInProperty(TypeCategory type_category,
                                   std::unique_ptr<VariantDataModel> variant_data_model)
        : EditorProperty(type_category, std::move(variant_data_model)),
          property_data_model_(dynamic_cast<PropertyDataModel*>(data_model_.get()))
    {
        CRASH_COND_MSG(property_data_model_ == nullptr, "Input data_model must be PropertyDataModel");
    }


    bool Begin() override {
        const bool grid_open = ImGuiUtilities::BeginPropertyGrid(property_data_model_->GetPropertyNameStr().c_str(),
                                                                 property_data_model_->GetPropertyToolTipStr().c_str(),
                                                                 right_align_next_column_);
        disabled_open_ = grid_open && property_data_model_->IsPropertyReadOnly();
        if (disabled_open_) {
            ImGui::BeginDisabled();
        }
        return grid_open;
    }

    void End() override {
        if (disabled_open_) {
            ImGui::EndDisabled();
            disabled_open_ = false;
        }
        ImGuiUtilities::EndPropertyGrid();
    }


protected:
    PropertyDataModel* property_data_model_{nullptr};
    bool disabled_open_{false};
};

}
