/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-3-28
*/

#include "gobot/editor/property_inspector/editor_property_primitives.hpp"
#include "imgui_stdlib.h"
#include "gobot/log.hpp"

namespace gobot {

void EditorPropertyNil::OnDataImGui() {

}

/////////////////////////////////////////////


void EditorPropertyText::OnDataImGui() {
    if (property_data_model_->IsPropertyReadOnly()) {
        ImGui::BeginDisabled();
    }

    std::string str = data_.toStdString();
    if (ImGui::InputText(fmt::format("##{}", property_data_model_->GetPropertyName()).c_str(), &str)) {
        data_ = str.c_str();
        SaveDataToProperty();
    }

    if (property_data_model_->IsPropertyReadOnly()) {
        ImGui::EndDisabled();
    }
}

}