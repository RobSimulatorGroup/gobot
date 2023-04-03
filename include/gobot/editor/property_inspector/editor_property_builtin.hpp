/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-3-30
*/

#pragma once

#include "gobot/editor/property_inspector/editor_property.hpp"
#include "gobot/editor/imgui/imgui_utilities.hpp"
#include "gobot/error_macros.hpp"

namespace gobot {

class EditorBuiltInProperty : public EditorProperty {
    GOBCLASS(EditorBuiltInProperty, EditorProperty)
public:
    explicit EditorBuiltInProperty(std::unique_ptr<VariantDataModel> variant_data_model, bool using_grid = true)
        : EditorProperty(std::move(variant_data_model), using_grid),
          property_data_model_(dynamic_cast<PropertyDataModel*>(data_model_.get()))
    {
        CRASH_COND_MSG(property_data_model_ == nullptr, "Input data_model must be PropertyDataModel");

    }


    bool Begin() override {
        if (property_data_model_->IsPropertyReadOnly()) {
            ImGui::BeginDisabled();
        }
        return ImGuiUtilities::BeginPropertyGrid(property_data_model_->GetPropertyNameCStr(),
                                                 property_data_model_->GetPropertyToolTipCStr());
    }

    void End() override {
        ImGuiUtilities::EndPropertyGrid();
        if (property_data_model_->IsPropertyReadOnly()) {
            ImGui::EndDisabled();
        }
    }


protected:
    PropertyDataModel* property_data_model_{nullptr};
};

}
