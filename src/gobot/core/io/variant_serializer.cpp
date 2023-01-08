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

namespace gobot {

ResourceFormatSaverSceneInstance* VariantSerializer::s_resource_format_saver_ = nullptr;

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
            writer == var.to_int32();
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
            ok = false;
            auto value = var.to_uint64(&ok);
            if (ok) {
                writer = value;
                return true;
            }
        }
    } else if (t == Type::get<std::string>()) {
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
            if (key) {
                node[key_name] = key;
            }
            Json value;
            WriteVariant(item.second.extract_wrapped_value(), value);
            if (value) {
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


void VariantSerializer::SaveResource(Instance instance, const Type& type, Json& writer) {
    if (type.get_wrapper_holder_type() == rttr::wrapper_holder_type::Ref) {
        if (s_resource_format_saver_) {
            auto res = Ref<Resource>(instance.try_convert<Resource>());
            if (!res) {
                LOG_ERROR("Cannot convert object to Resource {}", type.get_name().data());
                return;
            }
            if (s_resource_format_saver_->external_resources_.contains(res)) {
                writer = fmt::format("ExtResource({})", res->GetResourceUuid().toString());
            } else if (s_resource_format_saver_->internal_resources_.contains(res)) {
                writer = fmt::format("SubResource({})", res->GetResourceUuid().toString());
            }
        } else {
            LOG_ERROR("Unsupported wrapper type: {}", type.get_name().data());
            return;
        }
    } else {
        LOG_ERROR("Unsupported wrapper type: {}", type.get_name().data());
        return;
    }
}

void VariantSerializer::ToJsonRecursively(Instance object, Json& writer)
{
    auto raw_type = object.get_type().get_raw_type();
    Instance obj = raw_type.is_wrapper() ? object.get_wrapped_instance() : object;
    if (raw_type.is_wrapper() && raw_type.get_wrapper_holder_type() == rttr::wrapper_holder_type::Ref) {
        SaveResource(obj, raw_type, writer);
        return;
    }

    auto prop_list = obj.get_derived_type().get_properties();

    for (auto prop : prop_list) {
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

Json VariantSerializer::VariantToJson(Instance obj,
                                      ResourceFormatSaverSceneInstance* resource_format_saver) {
    if (!obj.is_valid()) {
        return {};
    }

    Json json;
    s_resource_format_saver_ = resource_format_saver;
    ToJsonRecursively(obj, json);
    s_resource_format_saver_ = nullptr;
    return json;
}



Variant VariantSerializer::JsonToVariant(const Json& json) {

}

}