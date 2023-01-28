/* Copyright(c) 2020-2022, Qiqi Wu<1258552199@qq.com>.
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 * This file is created by Qiqi Wu, 22-11-20
*/

#include "gobot/core/io/variant_serializer.hpp"
#include "gobot/core/object.hpp"
#include "gobot/log.hpp"
#include "gobot/core/io/resource_format_scene.hpp"
#include "rttr/enumeration.h"

namespace gobot {

const char* CLASS_TAG = "__CLASS__";

ResourceFormatSaverSceneInstance* VariantSerializer::s_resource_format_saver_ = nullptr;
ResourceFormatLoaderSceneInstance* VariantSerializer::s_resource_format_loader_ = nullptr;

bool VariantSerializer::WriteAtomicTypesToJson(const Type& t, const Variant& var, Json& writer)
{
    if (t.is_arithmetic()) {
        if (t == Type::get<bool>())
            writer = var.to_bool();
        else if (t == Type::get<int8_t>())
            writer = var.to_int8();
        else if (t == Type::get<int16_t>())
            writer = var.to_int16();
        else if (t == Type::get<int32_t>())
            writer = var.to_int32();
        else if (t == Type::get<int64_t>())
            writer = var.to_int64();
        else if (t == Type::get<uint8_t>())
            writer = var.to_uint8();
        else if (t == Type::get<uint16_t>())
            writer = var.to_uint16();
        else if (t == Type::get<uint32_t>())
            writer = var.to_uint32();
        else if (t == Type::get<uint64_t>())
            writer = var.to_uint64();
        else if (t == Type::get<float>())
            writer = var.to_float();
        else if (t == Type::get<double>())
            writer = var.to_double();
        return true;
    } else if (t.is_enumeration()) {
        bool ok = false;
        auto result = var.to_string(&ok);
        if (ok) {
            writer = var.to_string();
            return true;
        } else {
            return false;
        }
    } else if (t == Type::get<std::string>() || t == Type::get<String>()) {
        writer = var.to_string();
        return true;
    }

    return false;
}


void VariantSerializer::WriteArray(const VariantListView& view, Json& writer)
{
    writer = Json::array();
    for (const auto& item : view) {
        Json value;
        if (item.is_sequential_container()) {
            WriteArray(item.create_sequential_view(), value);
        } else {
            Variant wrapped_var = item.extract_wrapped_value();
            Type value_type = wrapped_var.get_type();
            if (value_type.is_arithmetic() || value_type == Type::get<std::string>() || value_type.is_enumeration()) {
                WriteAtomicTypesToJson(value_type, wrapped_var, value);
            } else { // object
                ToJsonRecursively(wrapped_var, value);
            }
        }
        writer.emplace_back(value);
    }
}

void VariantSerializer::WriteAssociativeContainer(const VariantMapView& view, Json& writer)
{
    static const char* key_name("key");
    static const char* value_name("value");

    if (view.is_key_only_type()) {
        for (auto& item : view) {
            Json value;
            WriteVariant(item.first.extract_wrapped_value(), value);
            writer.emplace_back(value);
        }
    } else {
        for (auto& item : view) {
            Json node;
            Json key;
            WriteVariant(item.first.extract_wrapped_value(), key);
            if (!key.is_null()) {
                node[key_name] = key;
            }
            Json value;
            WriteVariant(item.second.extract_wrapped_value(), value);
            if (!value.is_null()) {
                node[value_name] = value;
            }
            writer.emplace_back(node);
        }
    }
}

bool VariantSerializer::WriteVariant(const Variant& var, Json& writer)
{
    auto value_type = var.get_type();
    auto wrapped_type = value_type.is_wrapper() ? value_type.get_wrapped_type() : value_type;
    bool is_wrapper = wrapped_type != value_type;

    if (wrapped_type.is_arithmetic() || wrapped_type.is_enumeration() || wrapped_type == Type::get<String>()) {
        return WriteAtomicTypesToJson(is_wrapper ? wrapped_type : value_type,
                                      is_wrapper ? var.extract_wrapped_value() : var, writer);
    } else if (var.is_sequential_container()) {
        WriteArray(var.create_sequential_view(), writer);
    } else if (var.is_associative_container()) {
        WriteAssociativeContainer(var.create_associative_view(), writer);
    } else {
        auto child_props = is_wrapper ? Instance(var.extract_wrapped_value())
                                                 .get_derived_type().get_properties() : value_type.get_properties();
        if (!child_props.empty()) {
            ToJsonRecursively(var, writer);
        } else {
            bool ok = false;
            auto text = var.to_string(&ok);
            if (ok) {
                writer = text;
                return true;
            } else {
                return false;
            }
        }
    }

    return true;
}


bool VariantSerializer::SaveResource(const Variant& variant, const Type& type, Json& writer) {
    if (s_resource_format_saver_) {
        bool success{false};
        auto res = variant.convert<Ref<Resource>>(&success);
        if (!success) {
            LOG_ERROR("Cannot convert object to Resource {}", type.get_name().data());
            return false;
        }
        if (s_resource_format_saver_->external_resources_.contains(res)) {
            writer = fmt::format("ExtResource({})", s_resource_format_saver_->external_resources_[res]);
        } else if (s_resource_format_saver_->internal_resources_.contains(res)) {
            writer = fmt::format("SubResource({})", s_resource_format_saver_->external_resources_[res]);
        }
    } else {
        LOG_ERROR("Unsupported wrapper type: {}", type.get_name().data());
        return false;
    }

    return true;
}

void VariantSerializer::ToJsonRecursively(const Variant& variant, Json& writer)
{
    Instance object(variant);
    auto raw_type = object.get_type().get_raw_type();
    Instance obj = raw_type.is_wrapper() ? object.get_wrapped_instance() : object;
    if (raw_type.is_wrapper() && raw_type.get_wrapper_holder_type() == WrapperHolderType::Ref) {
        SaveResource(variant, raw_type, writer);
        return;
    }

    auto prop_list = obj.get_derived_type().get_properties();

    for (auto prop : prop_list) {
        if (prop.is_readonly()) {
            // Not load readonly key
            continue;
        }
        PropertyInfo property_info;
        auto meta_data = prop.get_metadata(PROPERTY_INFO_KEY);
        if (meta_data.is_valid()) {
            property_info = meta_data.get_value<PropertyInfo>();
        }
        USING_ENUM_BITWISE_OPERATORS;
        if (static_cast<bool>(property_info.usage & PropertyUsageFlags::Storage)) {
            Variant prop_value = prop.get_value(obj);
            if (!prop_value)
                continue; // cannot serialize, because we cannot retrieve the value

            Json prop_json;
            const auto prop_name = prop.get_name();
            if (!WriteVariant(prop_value, prop_json)) {
                LOG_ERROR("Cannot serialize property: {}", prop_name.data());
            } else {
                writer[prop_name.data()] = prop_json;
            }
        }
    }

}

Json VariantSerializer::VariantToJson(const Variant& variant,
                                      ResourceFormatSaverSceneInstance* resource_format_saver) {
    if (!variant.is_valid()) {
        return {};
    }

    s_resource_format_saver_ = resource_format_saver;
    Json json;

    auto type = variant.get_type();
    if (type.is_arithmetic() || type.is_enumeration() ||
        type == Type::get<String>() || type == Type::get<std::string>()) {
        WriteAtomicTypesToJson(type, variant, json);
    } else if (type.is_sequential_container()) {
        WriteArray(variant.create_sequential_view(), json);
    } else if (type.is_associative_container()) {
        WriteAssociativeContainer(variant.create_associative_view(), json);
    } else {
        ToJsonRecursively(variant, json);
    }

    return json;
}


Variant VariantSerializer::ExtractPrimitiveTypes(const Type& type, const Json& json_value)
{
    if (json_value.is_boolean()) {
        if (type == Type::get<bool>()){
            return json_value.get<bool>();
        } else {
            LOG_ERROR("json_value:{} and type: {} is unmatched.", json_value, type.get_name().data());
        }
    } else if (json_value.is_number_unsigned()) {
        if (type == Type::get<uint8_t>()) {
            return json_value.get<uint8_t>();
        } else if (type == Type::get<uint16_t>()) {
            return json_value.get<uint16_t>();
        } else if (type == Type::get<uint32_t>()) {
            return json_value.get<uint32_t>();
        } else if (type == Type::get<uint64_t>()) {
            return json_value.get<uint64_t>();
        } else {
            LOG_ERROR("json_value:{} and type: {} is unmatched.", json_value, type.get_name().data());
        }
    } else if (json_value.is_number_integer()) {
        if (type == Type::get<int8_t>()) {
            return json_value.get<int8_t>();
        } else if (type == Type::get<int16_t>()) {
            return json_value.get<int16_t>();
        } else if (type == Type::get<int32_t>()) {
            return json_value.get<int32_t>();
        } else if (type == Type::get<int64_t>()) {
            return json_value.get<int64_t>();
        } else {
            LOG_ERROR("json_value:{} and type: {} is unmatched.", json_value, type.get_name().data());
        }
    } else if (json_value.is_number_float()) {
        if (type == Type::get<float>()) {
            return json_value.get<float>();
        } else if (type == Type::get<double>()) {
            return json_value.get<double>();
        } else {
            LOG_ERROR("json_value:{} and type: {} is unmatched.", json_value, type.get_name().data());
        }
    } else if (json_value.is_string()) {
        if (type == Type::get<std::string>()) {
            return json_value.get<std::string>();
        } else if (type == Type::get<String>()) {
            return String(json_value.get<std::string>().c_str());
        } else if (type.is_enumeration()) {
            auto enum_class = type.get_enumeration();
            return enum_class.name_to_value(json_value.get<std::string>());
        }
        else {
            LOG_ERROR("json_value:{} and type: {} is unmatched.", json_value, type.get_name().data());
        }
    }

    return {};
}


bool VariantSerializer::WriteArrayRecursively(VariantListView& view, const Json& json_array_value)
{
    view.set_size(json_array_value.size());
    const auto array_value_type = view.get_rank_type(1);
    bool set_success{true};

    for (std::size_t i = 0; i < json_array_value.size(); ++i) {
        auto& json_index_value = json_array_value[i];
        if (json_index_value.is_array()) {
            auto sub_array_view = view.get_value(i).create_sequential_view();
            set_success &= WriteArrayRecursively(sub_array_view, json_index_value);
        } else if (json_index_value.is_object()) {
            Variant var_tmp = view.get_value(i);
            Variant wrapped_var = var_tmp.extract_wrapped_value();
            set_success &= FromJsonRecursively(wrapped_var, json_index_value);
            set_success &= view.set_value(i, wrapped_var);
        } else {
            auto extracted_value = ExtractPrimitiveTypes(array_value_type, json_index_value);
            if (extracted_value.is_valid()) {
                set_success &= view.set_value(i, extracted_value);
            }
        }
    }

    return set_success;
}

Variant VariantSerializer::ExtractValue(const Type& type, const Json& json)
{
    if (json.is_primitive()) {
        return ExtractPrimitiveTypes(type, json);
    } else if (json.is_object()) {
        rttr::constructor ctor = type.get_constructor();
        for (auto& item : type.get_constructors()) {
            if (item.get_instantiated_type() == type)
                ctor = item;
        }
        auto value = ctor.invoke();
        FromJsonRecursively(value, json);
        return value;
    } else if (json.is_array()) {
        rttr::constructor ctor = type.get_constructor();
        for (auto& item : type.get_constructors()) {
            if (item.get_instantiated_type() == type)
                ctor = item;
        }
        auto value = ctor.invoke();
        if (type.is_sequential_container()) {
            auto view = value.create_sequential_view();
            WriteArrayRecursively(view, json);
        } else if (type.is_associative_container()) {
            auto view = value.create_associative_view();
            WriteAssociativeViewRecursively(view, json);
        }

        return value;
    }

    return {};
}

bool VariantSerializer::WriteAssociativeViewRecursively(VariantMapView& view, const Json& json_array_value)
{
    bool set_success{true};

    for (const auto& json_index_value : json_array_value) {
        if (json_index_value.is_object())  { // a key-value associative view
            if (json_index_value.contains("key") && json_index_value.contains("value")) {
                auto key_var = ExtractValue(view.get_key_type(), json_index_value["key"]);
                auto value_var = ExtractValue(view.get_value_type(), json_index_value["value"]);
                if (key_var && value_var) {
                    view.insert(key_var, value_var);
                } else {
                    set_success = false;
                }
            }
        } else  { // a key-only associative view
            Variant extracted_value = ExtractPrimitiveTypes(view.get_key_type(), json_index_value);
            if (extracted_value) {
                view.insert(extracted_value);
            } else {
                set_success = false;
            }
        }
    }
    return set_success;
}


bool VariantSerializer::LoadSubResource(Variant& variant, const String& id) {
    if (s_resource_format_loader_) {
        auto it = s_resource_format_loader_->sub_resources_.find(id);
        if (it == s_resource_format_loader_->sub_resources_.end()) {
            LOG_ERROR("Cannot find sub-resource of id: {}", id);
            return false;
        }
        variant = it->second;
        return true;
    }
    return false;
}

bool VariantSerializer::LoadExtResource(Variant& variant, const String& id) {
    if (s_resource_format_loader_) {
        auto it = s_resource_format_loader_->ext_resources_.find(id);
        if (it == s_resource_format_loader_->ext_resources_.end()) {
            LOG_ERROR("Cannot find ext-resource of id: {}", id);
            return false;
        }
        variant = it->second;
        return true;
    }
    return false;
}

bool VariantSerializer::LoadResource(Variant& variant, const Json& json) {
    if (json.is_string()) {
        auto str = String::fromStdString(json.get<std::string>());
        auto left_bracket = str.indexOf("(");
        auto right_bracket = str.indexOf(")");
        auto key_word = str.left(left_bracket);
        auto id = str.mid(left_bracket + 1, right_bracket - left_bracket - 1);
        if (key_word == "SubResource") {
            return LoadSubResource(variant, id);
        } else if (key_word == "ExtResource") {
            return LoadExtResource(variant, id);
        } else {
            LOG_ERROR("Cannot load Resource of type: {}, from json: {}", variant.get_type().get_name().data(), json.dump(4));
            return false;
        }
    }

    return false;
}

bool VariantSerializer::FromJsonRecursively(Variant& variant, const Json& json) {
    Instance instance(variant);
    auto raw_type = instance.get_type().get_raw_type();
    Instance obj = instance.get_type().get_raw_type().is_wrapper() ? instance.get_wrapped_instance() : instance;
    if (raw_type.is_wrapper() && raw_type.get_wrapper_holder_type() == WrapperHolderType::Ref) {
        return LoadResource(variant, json);
    }

    const auto prop_list = obj.get_derived_type().get_properties();

    bool set_success{true};

    for (auto prop : prop_list) {
        if (prop.is_readonly()) {
            continue;
        }

        PropertyInfo property_info;
        auto meta_data = prop.get_metadata(PROPERTY_INFO_KEY);
        if (meta_data.is_valid()) {
            property_info = meta_data.get_value<PropertyInfo>();
        }
        USING_ENUM_BITWISE_OPERATORS;
        if (static_cast<bool>(property_info.usage & PropertyUsageFlags::Storage)) {
            const auto prop_name = prop.get_name();
            if (!json.contains(prop_name.data())) {
                LOG_ERROR("Cannot find key: {} in json: {}", prop_name.data(), json);
                continue;
            }
            const auto& child_json = json[prop_name.data()];
            const auto& prop_type = prop.get_type();
            if (child_json.is_array()) {
                if (prop_type.is_sequential_container()) {
                    auto value = prop.get_value(obj);
                    auto view = value.create_sequential_view();
                    set_success &= WriteArrayRecursively(view, child_json);
                } else if (prop_type.is_associative_container()) {
                    auto value = prop.get_value(obj);
                    auto view = value.create_associative_view();
                    set_success &= WriteAssociativeViewRecursively(view, child_json);
                }
            } else if (child_json.is_object()) {
                auto var = prop.get_value(obj);
                set_success &= FromJsonRecursively(var, child_json);
                set_success &= prop.set_value(obj, var);
            } else {
                auto extracted_value = ExtractPrimitiveTypes(prop.get_type(), child_json);
                if (extracted_value) // REMARK: CONVERSION WORKS ONLY WITH "const type", check whether this is correct or not!
                    set_success &= prop.set_value(obj, extracted_value);
            }
        }
    }

    return set_success;
}


bool VariantSerializer::JsonToVariant(Variant& variant,
                                      const Json& json,
                                      ResourceFormatLoaderSceneInstance* s_resource_format_loader) {
    if (json.is_null()) {
        LOG_ERROR("Input json is null");
        return {};
    }
    if (!variant.is_valid()) {
        LOG_ERROR("Input type is invalid");
        return {};
    }

    s_resource_format_loader_ = s_resource_format_loader;

    auto type = variant.get_type();
    if (type.is_enumeration() || type.is_arithmetic() ||
        type == Type::get<std::string>() || type == Type::get<String>()) {
        auto var = ExtractPrimitiveTypes(type, json);
        if (var.is_valid()) {
            variant = var;
            return true;
        }
    } else if (type.is_sequential_container()) {
            auto view = variant.create_sequential_view();
            return WriteArrayRecursively(view, json);
    } else if (type.is_associative_container()) {
            auto view = variant.create_associative_view();
            return WriteAssociativeViewRecursively(view, json);
    } else {
        return FromJsonRecursively(variant, json);
    }

    return false;
}

}