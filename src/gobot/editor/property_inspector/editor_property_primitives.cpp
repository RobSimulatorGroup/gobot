/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-3-28
*/

#include "gobot/editor/property_inspector/editor_property_primitives.hpp"
#include "imgui_stdlib.h"
#include "imgui.h"
#include "gobot/log.hpp"

namespace gobot {

void EditorPropertyBool::OnImGuiContent() {
    auto value = property_data_model_->GetValue().to_bool();
    if (ImGui::Checkbox(fmt::format("##{}", fmt::ptr(this)).c_str(), &value)) {
        property_data_model_->SetValue(value);
    }

}

/////////////////////////////////////////////


void EditorPropertyText::OnImGuiContent() {
    auto value = property_data_model_->GetValue().to_string();
    if (ImGui::InputText(fmt::format("##{}", fmt::ptr(this)).c_str(), &value)) {
        property_data_model_->SetValue(value);
    }

}

}