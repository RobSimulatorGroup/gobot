/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-3-28
*/

#include "gobot/editor/property_inspector/editor_property_primitives.hpp"
#include "gobot/editor/editor.hpp"
#include "gobot/scene/joint_3d.hpp"
#include "imgui_stdlib.h"
#include "imgui.h"
#include "gobot/log.hpp"
#include "imgui_extension/file_browser/ImFileBrowser.h"

#include <algorithm>
#include <cstring>

namespace gobot {
namespace {

bool InputScalarCommit(const char* id,
                       ImGuiDataType data_type,
                       void* value,
                       const char* format = nullptr,
                       ImGuiInputTextFlags flags = ImGuiInputTextFlags_EnterReturnsTrue) {
    const bool enter_pressed = ImGui::InputScalar(id, data_type, value, nullptr, nullptr, format, flags);
    return enter_pressed || ImGui::IsItemDeactivatedAfterEdit();
}

Joint3D* GetJoint3DPropertyHolder(PropertyDataModel* model) {
    if (!model) {
        return nullptr;
    }

    auto* object = model->GetVariantCache().object;
    return Object::PointerCastTo<Joint3D>(object);
}

bool IsAngularJoint(JointType joint_type) {
    return joint_type == JointType::Revolute || joint_type == JointType::Continuous;
}

bool IsJointPositionProperty(const std::string& property_name) {
    return property_name == "joint_position" ||
           property_name == "lower_limit" ||
           property_name == "upper_limit";
}

double JointValueToDisplay(JointType joint_type,
                           const std::string& property_name,
                           double value) {
    if ((IsJointPositionProperty(property_name) || property_name == "velocity_limit") &&
        IsAngularJoint(joint_type)) {
        return RAD_TO_DEG(static_cast<RealType>(value));
    }

    return value;
}

double JointValueFromDisplay(JointType joint_type,
                             const std::string& property_name,
                             double value) {
    if ((IsJointPositionProperty(property_name) || property_name == "velocity_limit") &&
        IsAngularJoint(joint_type)) {
        return DEG_TO_RAD(static_cast<RealType>(value));
    }

    return value;
}

const char* JointPropertyUnit(JointType joint_type, const std::string& property_name) {
    if (IsJointPositionProperty(property_name)) {
        if (IsAngularJoint(joint_type)) {
            return "deg";
        }
        if (joint_type == JointType::Prismatic) {
            return "m";
        }
    }

    if (property_name == "velocity_limit") {
        if (IsAngularJoint(joint_type)) {
            return "deg/s";
        }
        if (joint_type == JointType::Prismatic) {
            return "m/s";
        }
    }

    if (property_name == "effort_limit") {
        if (IsAngularJoint(joint_type)) {
            return "Nm";
        }
        if (joint_type == JointType::Prismatic) {
            return "N";
        }
    }

    return "";
}

bool JointPropertyHasUnlimitedDisplay(JointType joint_type, const std::string& property_name) {
    return joint_type == JointType::Continuous &&
           (property_name == "lower_limit" || property_name == "upper_limit");
}

bool DrawUnitInput(const char* id,
                   ImGuiDataType data_type,
                   void* value,
                   const char* unit,
                   const char* format = nullptr) {
    const float spacing = ImGui::GetStyle().ItemInnerSpacing.x;
    const float unit_width = (unit && unit[0] != '\0') ? ImGui::CalcTextSize(unit).x + spacing : 0.0f;
    const float input_width = std::max(1.0f, ImGui::GetContentRegionAvail().x - unit_width);

    ImGui::SetNextItemWidth(input_width);
    const bool commit = InputScalarCommit(id, data_type, value, format);
    if (unit && unit[0] != '\0') {
        ImGui::SameLine(0.0f, spacing);
        ImGui::TextDisabled("%s", unit);
    }
    return commit;
}

bool DrawReadonlyUnitText(const char* id, const char* text, const char* unit) {
    const float spacing = ImGui::GetStyle().ItemInnerSpacing.x;
    const float unit_width = (unit && unit[0] != '\0') ? ImGui::CalcTextSize(unit).x + spacing : 0.0f;
    const float input_width = std::max(1.0f, ImGui::GetContentRegionAvail().x - unit_width);

    ImGui::SetNextItemWidth(input_width);
    ImGui::BeginDisabled();
    ImGui::InputText(id, const_cast<char*>(text), std::strlen(text), ImGuiInputTextFlags_ReadOnly);
    ImGui::EndDisabled();
    if (unit && unit[0] != '\0') {
        ImGui::SameLine(0.0f, spacing);
        ImGui::TextDisabled("%s", unit);
    }
    return false;
}

bool DrawJointFloatProperty(PropertyDataModel* model, TypeCategory type_category) {
    auto* joint = GetJoint3DPropertyHolder(model);
    if (!joint) {
        return false;
    }

    const std::string& property_name = model->GetPropertyName();
    if (!IsJointPositionProperty(property_name) &&
        property_name != "velocity_limit" &&
        property_name != "effort_limit") {
        return false;
    }

    const JointType joint_type = joint->GetJointType();
    const char* unit = JointPropertyUnit(joint_type, property_name);
    if (JointPropertyHasUnlimitedDisplay(joint_type, property_name)) {
        DrawReadonlyUnitText("##joint_unlimited", "unlimited", unit);
        return true;
    }

    if (type_category == TypeCategory::Float) {
        float value = static_cast<float>(JointValueToDisplay(joint_type, property_name, model->GetValue().to_float()));
        if (DrawUnitInput("##joint_float", ImGuiDataType_Float, &value, unit, "%.6g")) {
            model->SetValue(static_cast<float>(JointValueFromDisplay(joint_type, property_name, value)));
        }
    } else if (type_category == TypeCategory::Double) {
        double value = JointValueToDisplay(joint_type, property_name, model->GetValue().to_double());
        if (DrawUnitInput("##joint_double", ImGuiDataType_Double, &value, unit, "%.6g")) {
            model->SetValue(JointValueFromDisplay(joint_type, property_name, value));
        }
    }

    return true;
}

} // namespace

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
        if (InputScalarCommit(GetPtrImGuiID(), ImGuiDataType_U8, &value)) {
            property_data_model_->SetValue(value);
        }
    } else if (type_category_ == TypeCategory::UInt16) {
        auto value = property_data_model_->GetValue().to_uint16();
        if (InputScalarCommit(GetPtrImGuiID(), ImGuiDataType_U16, &value)) {
            property_data_model_->SetValue(value);
        }
    } else if (type_category_ == TypeCategory::UInt32) {
        auto value = property_data_model_->GetValue().to_uint32();
        if (InputScalarCommit(GetPtrImGuiID(), ImGuiDataType_U32, &value)) {
            property_data_model_->SetValue(value);
        }
    } else if (type_category_ == TypeCategory::UInt64) {
        auto value = property_data_model_->GetValue().to_uint64();
        if (InputScalarCommit(GetPtrImGuiID(), ImGuiDataType_U64, &value)) {
            property_data_model_->SetValue(value);
        }
    } else if (type_category_ == TypeCategory::Int8) {
        auto value = property_data_model_->GetValue().to_int8();
        if (InputScalarCommit(GetPtrImGuiID(), ImGuiDataType_S8, &value)) {
            property_data_model_->SetValue(value);
        }
    } else if (type_category_ == TypeCategory::Int8) {
        auto value = property_data_model_->GetValue().to_int8();
        if (InputScalarCommit(GetPtrImGuiID(), ImGuiDataType_S8, &value)) {
            property_data_model_->SetValue(value);
        }
    } else if (type_category_ == TypeCategory::Int16) {
        auto value = property_data_model_->GetValue().to_int16();
        if (InputScalarCommit(GetPtrImGuiID(), ImGuiDataType_S16, &value)) {
            property_data_model_->SetValue(value);
        }
    } else if (type_category_ == TypeCategory::Int32) {
        auto value = property_data_model_->GetValue().to_int32();
        if (InputScalarCommit(GetPtrImGuiID(), ImGuiDataType_S32, &value)) {
            property_data_model_->SetValue(value);
        }
    } else if (type_category_ == TypeCategory::Int64) {
        auto value = property_data_model_->GetValue().to_int64();
        if (InputScalarCommit(GetPtrImGuiID(), ImGuiDataType_S64, &value)) {
            property_data_model_->SetValue(value);
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////

void EditorPropertyFloat::OnImGuiContent() {
    if (DrawJointFloatProperty(property_data_model_, type_category_)) {
        return;
    }

    if (type_category_ == TypeCategory::Float) {
        auto value = property_data_model_->GetValue().to_float();
        if (InputScalarCommit(GetPtrImGuiID(), ImGuiDataType_Float, &value, "%.6g")) {
            property_data_model_->SetValue(value);
        }
    } else if (type_category_ == TypeCategory::Double) {
        auto value = property_data_model_->GetValue().to_double();
        if (InputScalarCommit(GetPtrImGuiID(), ImGuiDataType_Double, &value, "%.6g")) {
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
    if (ImGui::Combo(GetPtrImGuiID(), &index, &names_[0], names_.size())) {
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
        property_data_model_->SetValue(value);
    }
}

//////////////////////////////////////////////////

void EditorPropertyMultilineText::OnImGuiContent() {
    auto value = property_data_model_->GetValue().to_string();
    if (ImGui::InputTextMultiline(GetPtrImGuiID(), &value, ImVec2(-FLT_MIN, ImGui::GetTextLineHeight() * 6),
                                  ImGuiInputTextFlags_AllowTabInput | ImGuiInputTextFlags_CtrlEnterForNewLine)) {
        property_data_model_->SetValue(value);
    }
}


//////////////////////////////////////////////////////

void EditorPropertyPath::OnImGuiContent() {
    auto file_browser = Editor::GetInstance()->GetFileBrowser();
    auto path = property_data_model_->GetValue().to_string();
    if (ImGui::Button(path.c_str())) {
        file_browser->Open();
    }
    if (file_browser->HasSelected()) {
        property_data_model_->SetValue(file_browser->GetPath().string());
        file_browser->ClearSelected();
    }
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
        } else {
            LOG_TRACE("Inspector set Color property '{}' on '{}' to ({}, {}, {}, {}).",
                      property_data_model_->GetPropertyName(),
                      property_data_model_->GetHolderType().get_name().data(),
                      color.red(),
                      color.green(),
                      color.blue(),
                      color.alpha());
        }
    }
}

////////////////////////////////////////////////////////

void EditorPropertyObjectID::OnImGuiContent() {
    auto object_id = property_data_model_->GetValue().convert<ObjectID>();
    ImGui::TextUnformatted(object_id.IsValid() ?
                                     fmt::format("Object id: {}", object_id.operator int64_t()).c_str() :
                                     "Invalid id");
}

////////////////////////////////////////////////////////

void EditorPropertyRID::OnImGuiContent() {
    auto rid = property_data_model_->GetValue().convert<RID>();
    ImGui::TextUnformatted(rid.IsValid() ?
                           fmt::format("RID: {}", rid.GetID()).c_str() :
                           "Invalid id");
}


}
