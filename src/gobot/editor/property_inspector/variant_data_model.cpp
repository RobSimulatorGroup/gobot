/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-3-31
*/

#include "gobot/editor/property_inspector/variant_data_model.hpp"

#include <utility>

#include "gobot/editor/editor.hpp"
#include "gobot/main/engine_context.hpp"
#include "gobot/log.hpp"
#include "gobot/scene/node.hpp"
#include "gobot/scene/scene_command.hpp"

namespace gobot {
namespace {

bool VariantValuesEqual(const Variant& left, const Variant& right) {
    if (!left.is_valid() || !right.is_valid()) {
        return left.is_valid() == right.is_valid();
    }

    Variant converted_right = right;
    if (left.get_type() != converted_right.get_type() && !converted_right.convert(left.get_type())) {
        return false;
    }

    return left == converted_right;
}

} // namespace

PropertyDataModel::PropertyDataModel(VariantCache& variant, const Property& property)
        : VariantDataModel(variant),
          property_(property),
          property_cache_(property_.get_type(),
                          property_.get_name().data(),
                          property_.is_readonly(),
                          property_.get_metadata(PROPERTY_INFO_KEY))
{
}

const Type& PropertyDataModel::GetValueType() const {
    return property_cache_.property_type;
}

const std::string& PropertyDataModel::GetPropertyName() const {
    return property_cache_.property_name;
}

const std::string& PropertyDataModel::GetPropertyNameStr() const {
    return property_cache_.property_name_str;
}

bool PropertyDataModel::IsPropertyReadOnly() const {
    return property_cache_.property_readonly;
}

const PropertyInfo& PropertyDataModel::GetPropertyInfo() const {
    return property_cache_.property_info;
}

const std::string& PropertyDataModel::GetPropertyToolTipStr() const {
    return property_cache_.tool_tip_str;
}

bool PropertyDataModel::SetValue(Argument argument) {
    Variant old_value = GetValue();
    Variant new_value = argument;
    if (VariantValuesEqual(old_value, new_value)) {
        return true;
    }

    if (auto* node = Object::PointerCastTo<Node>(variant_cache_.object)) {
        if (auto* editor = Editor::GetInstanceOrNull()) {
            auto* context = editor->GetEngineContext();
            return context != nullptr &&
                   context->ExecuteSceneCommand(std::make_unique<SetNodePropertyCommand>(
                           node->GetInstanceId(),
                           property_cache_.property_name_str,
                           new_value));
        }
    }

    return false;
}

Variant PropertyDataModel::GetValue() const {
    return property_.get_value(variant_cache_.instance);
}

///////////////////////////////////////////////////////////////////////////////////

SequenceContainerDataModel::SequenceContainerDataModel(VariantCache& variant)
        : VariantDataModel(variant),
          sc_cache_(variant_cache_.variant.create_sequential_view())
{
}

const Type& SequenceContainerDataModel::GetValueType() const {
    return sc_cache_.value_type;
}

std::size_t SequenceContainerDataModel::GetSize() const {
    return sc_cache_.variant_list_view.get_size();
}

Variant SequenceContainerDataModel::GetValue(std::size_t index) const {
    return sc_cache_.variant_list_view.get_value(index);
}

void SequenceContainerDataModel::SetValue(std::size_t index, Argument argument) {
    GOB_UNUSED(index);
    GOB_UNUSED(argument);
    LOG_ERROR("Sequence property editing is not command-backed yet.");
}

///////////////////////////////////////////////////////////////////////////////////

AssociativeContainerDataModel::AssociativeContainerDataModel(VariantCache& variant)
        : VariantDataModel(variant),
          ac_cache_(variant_cache_.variant.create_associative_view())
{
}

bool AssociativeContainerDataModel::IsKeyOnlyType() const {
    return ac_cache_.is_key_only_type;
}

const Type& AssociativeContainerDataModel::GetValueType() const {
    return ac_cache_.value_type;
}

const Type& AssociativeContainerDataModel::GetKeyType() const {
    return ac_cache_.key_type;
}

///////////////////////////////////////////////////////////////////////////////////

FunctionDataModel::FunctionDataModel(VariantCache& variant, const Method& method)
        : VariantDataModel(variant),
          method_(method),
          method_cache_{method.get_name().data(),
                        method.get_declaring_type(),
                        method.get_return_type()}
{
}

const Type& FunctionDataModel::GetValueType() const {
    return method_cache_.declaring_type;
}

const std::string& FunctionDataModel::GetMethodName() const {
    return method_cache_.method_name;
}

void FunctionDataModel::DoMethodCall() {
    method_.invoke(variant_cache_.instance);
}

}
