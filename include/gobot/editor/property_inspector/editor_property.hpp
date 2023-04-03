/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-3-28
*/

#pragma once

#include "gobot/scene/imgui_node.hpp"
#include "gobot/core/types.hpp"
#include "gobot/editor/property_inspector/variant_data_model.hpp"


namespace gobot {

class EditorProperty : public ImGuiNode {
public:
    explicit EditorProperty(std::unique_ptr<VariantDataModel> variant_data_model, bool using_grid = true)
      : data_model_(std::move(variant_data_model)),
        using_grid_(using_grid)
    {
    }

protected:
    std::unique_ptr<VariantDataModel> data_model_{nullptr};
    bool using_grid_;
};

}