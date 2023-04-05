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
    if (ImGui::Checkbox(GetPtrImGuiID(), &value)) {
        property_data_model_->SetValue(value);
    }
}

/////////////////////////////////////////////////////////////


void EditorPropertyInteger::OnImGuiContent() {
    if (type_category_ == TypeCategory::UInt8) {
        auto value = property_data_model_->GetValue().to_uint8();
        if (ImGui::DragScalar(GetPtrImGuiID(), ImGuiDataType_U8, &value,  drag_speed_)) {
            property_data_model_->SetValue(value);
        }
    } else if (type_category_ == TypeCategory::UInt16) {
        auto value = property_data_model_->GetValue().to_uint16();
        if (ImGui::DragScalar(GetPtrImGuiID(), ImGuiDataType_U16, &value,  drag_speed_)) {
            property_data_model_->SetValue(value);
        }
    } else if (type_category_ == TypeCategory::UInt32) {
        auto value = property_data_model_->GetValue().to_uint32();
        if (ImGui::DragScalar(GetPtrImGuiID(), ImGuiDataType_U32, &value,  drag_speed_)) {
            property_data_model_->SetValue(value);
        }
    } else if (type_category_ == TypeCategory::UInt64) {
        auto value = property_data_model_->GetValue().to_uint64();
        if (ImGui::DragScalar(GetPtrImGuiID(), ImGuiDataType_U64, &value,  drag_speed_)) {
            property_data_model_->SetValue(value);
        }
    } else if (type_category_ == TypeCategory::Int8) {
        auto value = property_data_model_->GetValue().to_int8();
        if (ImGui::DragScalar(GetPtrImGuiID(), ImGuiDataType_S8, &value,  drag_speed_)) {
            property_data_model_->SetValue(value);
        }
    } else if (type_category_ == TypeCategory::Int8) {
        auto value = property_data_model_->GetValue().to_int8();
        if (ImGui::DragScalar(GetPtrImGuiID(), ImGuiDataType_S8, &value,  drag_speed_)) {
            property_data_model_->SetValue(value);
        }
    } else if (type_category_ == TypeCategory::Int16) {
        auto value = property_data_model_->GetValue().to_int16();
        if (ImGui::DragScalar(GetPtrImGuiID(), ImGuiDataType_S16, &value,  drag_speed_)) {
            property_data_model_->SetValue(value);
        }
    } else if (type_category_ == TypeCategory::Int32) {
        auto value = property_data_model_->GetValue().to_int32();
        if (ImGui::DragScalar(GetPtrImGuiID(), ImGuiDataType_S32, &value,  drag_speed_)) {
            property_data_model_->SetValue(value);
        }
    } else if (type_category_ == TypeCategory::Int64) {
        auto value = property_data_model_->GetValue().to_int64();
        if (ImGui::DragScalar(GetPtrImGuiID(), ImGuiDataType_S64, &value,  drag_speed_)) {
            property_data_model_->SetValue(value);
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////

void EditorPropertyFloat::OnImGuiContent() {
    if (type_category_ == TypeCategory::Float) {
        auto value = property_data_model_->GetValue().to_float();
        if (ImGui::DragScalar(GetPtrImGuiID(), ImGuiDataType_Float, &value,  drag_speed_)) {
            property_data_model_->SetValue(value);
        }
    } else if (type_category_ == TypeCategory::Double) {
        auto value = property_data_model_->GetValue().to_double();
        if (ImGui::DragScalar(GetPtrImGuiID(), ImGuiDataType_Double, &value,  drag_speed_)) {
            property_data_model_->SetValue(value);
        }
    }
}

/////////////////////////////////////////////

// https://stackoverflow.com/questions/108318/how-can-i-test-whether-a-number-is-a-power-of-2
inline bool IsPowerOf2(int x) {
    return x > 0 && !(x & (x - 1));
}

EditorPropertyFlags::EditorPropertyFlags(TypeCategory type_category,
                                         std::unique_ptr<VariantDataModel> variant_data_model)
    : EditorBuiltInProperty(type_category, std::move(variant_data_model)),
      enumeration_(data_model_->GetValueType().get_enumeration())
{
    auto type = enumeration_.get_underlying_type();
    if (type == Type::get<std::uint8_t>()) {
        underlying_type_ = UInt8;
    } else if (type == Type::get<std::uint16_t>()) {
        underlying_type_ = UInt16;
    } else if (type == Type::get<std::uint32_t>()) {
        underlying_type_ = UInt32;
    } else if (type == Type::get<std::int8_t>()) {
        underlying_type_ = Int8;
    } else if (type == Type::get<std::int16_t>()) {
        underlying_type_ = Int16;
    } else if (type == Type::get<std::int32_t>()) {
        underlying_type_ = Int32;
    }

    for (const auto& name: enumeration_.get_names()) {
        auto value = enumeration_.name_to_value(name);
        // remove the data that are not inside {0, 2^n}
        switch (underlying_type_) {
            case UInt8: {
                if (IsPowerOf2(value.to_uint8())) {
                    names_.emplace_back(name.data(), value);
                }
            } break;
            case UInt16: {
                if (IsPowerOf2(value.to_uint16())) {
                    names_.emplace_back(name.data(), value);
                }
            } break;
            case UInt32: {
                if (IsPowerOf2(value.to_uint32())) {
                    names_.emplace_back(name.data(), value);
                }
            } break;
            case Int8: {
                if (IsPowerOf2(value.to_int8())) {
                    names_.emplace_back(name.data(), value);
                }
            } break;
            case Int16: {
                if (IsPowerOf2(value.to_int16())) {
                    names_.emplace_back(name.data(), value);
                }
            } break;
            case Int32: {
                if (IsPowerOf2(value.to_int32())) {
                    names_.emplace_back(name.data(), value);
                }
            } break;
        }
    }
}

void EditorPropertyFlags::OnImGuiContent() {
    switch (underlying_type_) {
        case UInt8: {
            uint_data_ = property_data_model_->GetValue().to_uint8();
            for (const auto& [name, value] : names_) {
                if (ImGui::CheckboxFlags(name.data(), &uint_data_, value.to_uint8())) {
                    auto new_flags = enumeration_.value_to_enum(static_cast<uint8_t>(int_data_));
                    if (!(property_data_model_->SetValue(new_flags))) {
                        LOG_ERROR("Set flags: {:b} to {} failed", int_data_, property_data_model_->GetPropertyName());
                    }
                }
            }
        } break;
        case UInt16: {
            uint_data_ = property_data_model_->GetValue().to_uint16();
            for (const auto &[name, value]: names_) {
                if (ImGui::CheckboxFlags(name.data(), &uint_data_, value.to_uint16())) {
                    auto new_flags = enumeration_.value_to_enum(static_cast<uint16_t>(int_data_));
                    if (!(property_data_model_->SetValue(new_flags))) {
                        LOG_ERROR("Set flags: {:b} to {} failed", int_data_, property_data_model_->GetPropertyName());
                    }
                }
            }
            break;
        }
        case UInt32: {
            uint_data_ = property_data_model_->GetValue().to_uint32();
            for (const auto& [name, value] : names_) {
                if (ImGui::CheckboxFlags(name.data(), &uint_data_, value.to_uint32())) {
                    auto new_flags = enumeration_.value_to_enum(static_cast<uint32_t>(int_data_));
                    if (!(property_data_model_->SetValue(new_flags))) {
                        LOG_ERROR("Set flags: {:b} to {} failed", int_data_, property_data_model_->GetPropertyName());
                    }
                }
            }
        } break;
        case Int8: {
            int_data_ = property_data_model_->GetValue().to_int8();
            for (const auto& [name, value] : names_) {
                if (ImGui::CheckboxFlags(name.data(), &int_data_, value.to_int8())) {
                    auto new_flags = enumeration_.value_to_enum(static_cast<int8_t>(int_data_));
                    if (!(property_data_model_->SetValue(new_flags))) {
                        LOG_ERROR("Set flags: {:b} to {} failed", int_data_, property_data_model_->GetPropertyName());
                    }
                }
            }
        } break;
        case Int16: {
            int_data_ = property_data_model_->GetValue().to_int16();
            for (const auto& [name, value] : names_) {
                if (ImGui::CheckboxFlags(name.data(), &int_data_, value.to_int16())) {
                    auto new_flags = enumeration_.value_to_enum(static_cast<int16_t>(int_data_));
                    if (!(property_data_model_->SetValue(new_flags))) {
                        LOG_ERROR("Set flags: {:b} to {} failed", int_data_, property_data_model_->GetPropertyName());
                    }
                }
            }
        } break;
        case Int32: {
            int_data_ = property_data_model_->GetValue().to_int32();
            for (const auto& [name, value] : names_) {
                if (ImGui::CheckboxFlags(name.data(), &int_data_, value.to_int32())) {
                    auto new_flags = enumeration_.value_to_enum(static_cast<int32_t>(int_data_));
                    if (!(property_data_model_->SetValue(new_flags))) {
                        LOG_ERROR("Set flags: {:b} to {} failed", int_data_, property_data_model_->GetPropertyName());
                    }
                }
            }
        } break;
    }
}

/////////////////////////////////////////////

EditorPropertyEnum::EditorPropertyEnum(TypeCategory type_category,
                                       std::unique_ptr<VariantDataModel> variant_data_model)
    : EditorBuiltInProperty(type_category, std::move(variant_data_model)),
      enumeration_(data_model_->GetValueType().get_enumeration())
{
    int i = 0;
    for (const auto& name: enumeration_.get_names()) {
        names_.emplace_back(name.data());
        names_map_.emplace(name.data(), i++);
    }
}

void EditorPropertyEnum::OnImGuiContent() {
    auto index = names_map_.at(property_data_model_->GetValue().to_string());
    if (ImGui::Combo("combo", &index, &names_[0], names_.size())) {
        std::string changed_name = names_.at(index);
        Variant data(changed_name);
        if (!(data.convert(data_model_->GetValueType()) && property_data_model_->SetValue(data))) {
            LOG_ERROR("Set enum {} to {} failed", changed_name, property_data_model_->GetPropertyName());
        }
    }
}

/////////////////////////////////////////////

void EditorPropertyText::OnImGuiContent() {
    auto value = property_data_model_->GetValue().to_string();
    if (ImGui::InputText(GetPtrImGuiID(), &value)) {
        property_data_model_->SetValue(String::fromStdString(value));
    }
}

//////////////////////////////////////////////////

void EditorPropertyMultilineText::OnImGuiContent() {
    auto value = property_data_model_->GetValue().to_string();
    if (ImGui::InputTextMultiline(GetPtrImGuiID(), &value, ImVec2(-FLT_MIN, ImGui::GetTextLineHeight() * 6),
                                  ImGuiInputTextFlags_AllowTabInput | ImGuiInputTextFlags_CtrlEnterForNewLine)) {
        property_data_model_->SetValue(String::fromStdString(value));
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

void EditorPropertyColor::OnImGuiContent() {
    auto color = property_data_model_->GetValue().convert<Color>();
    color_[0] = color.red();
    color_[1] = color.green();
    color_[2] = color.blue();
    color_[3] = color.alpha();
    if (ImGui::ColorEdit4(GetPtrImGuiID(), (float*)&color_, ImGuiColorEditFlags_Float)) {
        color.red() = color_[0];
        color.green() = color_[1];
        color.blue() = color_[2];
        color.alpha() = color_[3];
        if (!property_data_model_->SetValue(color)) {
            LOG_ERROR("Set color to {} failed", property_data_model_->GetPropertyName());
        }
    }
}

////////////////////////////////////////////////////////


void EditorPropertyRID::OnImGuiContent() {

}


void EditorPropertyRenderRID::OnImGuiContent() {

}

}