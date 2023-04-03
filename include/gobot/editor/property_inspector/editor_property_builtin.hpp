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

template<typename T>
class EditorBuiltInProperty : public EditorProperty {
public:
    using BaseClass = EditorBuiltInProperty<T>;

    explicit EditorBuiltInProperty(std::unique_ptr<VariantDataModel> variant_data_model, bool using_grid = true)
        : EditorProperty(std::move(variant_data_model), using_grid),
          property_data_model_(dynamic_cast<PropertyDataModel*>(data_model_.get()))
    {
        CRASH_COND_MSG(property_data_model_ == nullptr, "Input data_model must be PropertyDataModel");
        ERR_FAIL_COND_MSG(SaveDataToProperty(), "Cannot save data to property");
    }


    virtual void OnDataImGui() = 0;


    bool SaveDataToProperty() {
        return property_data_model_->SetValue(data_);
    }

    bool LoadDataFromProperty() {
        return property_data_model_->GetValue().template convert(data_);
    }

    virtual void OnImGuiContent() {
        if (ImGuiUtilities::BeginPropertyGrid(property_data_model_->GetPropertyName().toLocal8Bit().data(),
                                              property_data_model_->GetPropertyInfo().tool_tip.toLocal8Bit().data())) {
            // TODO(wqq): Do we need load every time?
            LoadDataFromProperty();
            OnDataImGui();
            ImGuiUtilities::EndPropertyGrid();
        };

    }


protected:
    PropertyDataModel* property_data_model_{nullptr};
    T data_;
};

}
