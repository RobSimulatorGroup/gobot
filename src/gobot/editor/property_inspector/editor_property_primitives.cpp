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

/////////////////////////////////////////////////////////////


void EditorPropertyInteger::OnImGuiContent() {
    if (type_category_ == TypeCategory::UInt8) {
        auto value = property_data_model_->GetValue().to_uint8();
        if (ImGui::DragScalar("drag u8", ImGuiDataType_U8, &value,  drag_speed_)) {
            property_data_model_->SetValue(value);
        }
    } else if (type_category_ == TypeCategory::UInt16) {
        auto value = property_data_model_->GetValue().to_uint16();
        if (ImGui::DragScalar("drag u16", ImGuiDataType_U16, &value,  drag_speed_)) {
            property_data_model_->SetValue(value);
        }
    } else if (type_category_ == TypeCategory::UInt32) {
        auto value = property_data_model_->GetValue().to_uint32();
        if (ImGui::DragScalar("drag u32", ImGuiDataType_U32, &value,  drag_speed_)) {
            property_data_model_->SetValue(value);
        }
    } else if (type_category_ == TypeCategory::UInt64) {
        auto value = property_data_model_->GetValue().to_uint64();
        if (ImGui::DragScalar("drag u64", ImGuiDataType_U64, &value,  drag_speed_)) {
            property_data_model_->SetValue(value);
        }
    } else if (type_category_ == TypeCategory::Int8) {
        auto value = property_data_model_->GetValue().to_int8();
        if (ImGui::DragScalar("drag int8", ImGuiDataType_S8, &value,  drag_speed_)) {
            property_data_model_->SetValue(value);
        }
    } else if (type_category_ == TypeCategory::Int8) {
        auto value = property_data_model_->GetValue().to_int8();
        if (ImGui::DragScalar("drag int8", ImGuiDataType_S8, &value,  drag_speed_)) {
            property_data_model_->SetValue(value);
        }
    } else if (type_category_ == TypeCategory::Int16) {
        auto value = property_data_model_->GetValue().to_int16();
        if (ImGui::DragScalar("drag int8", ImGuiDataType_S16, &value,  drag_speed_)) {
            property_data_model_->SetValue(value);
        }
    } else if (type_category_ == TypeCategory::Int32) {
        auto value = property_data_model_->GetValue().to_int32();
        if (ImGui::DragScalar("drag int8", ImGuiDataType_S32, &value,  drag_speed_)) {
            property_data_model_->SetValue(value);
        }
    } else if (type_category_ == TypeCategory::Int64) {
        auto value = property_data_model_->GetValue().to_int64();
        if (ImGui::DragScalar("drag int8", ImGuiDataType_S64, &value,  drag_speed_)) {
            property_data_model_->SetValue(value);
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////

void EditorPropertyFloat::OnImGuiContent() {
    if (type_category_ == TypeCategory::Float) {
        auto value = property_data_model_->GetValue().to_float();
        if (ImGui::DragScalar("float", ImGuiDataType_Float, &value,  drag_speed_)) {
            property_data_model_->SetValue(value);
        }
    } else if (type_category_ == TypeCategory::Double) {
        auto value = property_data_model_->GetValue().to_double();
        if (ImGui::DragScalar("double", ImGuiDataType_Double, &value,  drag_speed_)) {
            property_data_model_->SetValue(value);
        }
    }
}

/////////////////////////////////////////////


void EditorPropertyText::OnImGuiContent() {
    auto value = property_data_model_->GetValue().to_string();
    if (ImGui::InputText(fmt::format("##{}", fmt::ptr(this)).c_str(), &value)) {
        property_data_model_->SetValue(value);
    }
}

//////////////////////////////////////////////////

void EditorPropertyMultilineText::OnImGuiContent() {
    auto value = property_data_model_->GetValue().to_string();
    if (ImGui::InputTextMultiline(fmt::format("##{}", fmt::ptr(this)).c_str(), &value)) {
        property_data_model_->SetValue(value);
    }
}

//////////////////////////////////////////////////////

void EditorPropertyPath::OnImGuiContent() {
    // TODO
}

/////////////////////////////////////////////////////


void EditorPropertyNodePath::OnImGuiContent() {
    // TODO
}

////////////////////////////////////////////////////////


void EditorPropertyRID::OnImGuiContent() {

}


void EditorPropertyRenderRID::OnImGuiContent() {

}

}