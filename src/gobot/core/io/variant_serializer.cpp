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

//namespace gobot::io {
//
//
//core::Varint ObjectFromJson(const core::Type& type,  const core::Json& json) {
//
//}
//
//void ToJsonRecursively(const core::Instance& obj, core::Json& writer);
//
//bool WriteVariant(const core::Varint& var, core::Json& writer);
//
//
//bool WriteAtomicTypesToJson(const core::Type& t, const core::Varint& var, core::Json& writer)
//{
//    if (t.is_arithmetic())
//    {
//        if (t == core::Type::get<bool>())
//            writer = var.to_bool()
//        else if (t == core::Type::get<char>())
//            writer.Bool(var.to_bool());
//        else if (t == type::get<int8_t>())
//            writer.Int(var.to_int8());
//        else if (t == type::get<int16_t>())
//            writer.Int(var.to_int16());
//        else if (t == type::get<int32_t>())
//            writer.Int(var.to_int32());
//        else if (t == type::get<int64_t>())
//            writer.Int64(var.to_int64());
//        else if (t == type::get<uint8_t>())
//            writer.Uint(var.to_uint8());
//        else if (t == type::get<uint16_t>())
//            writer.Uint(var.to_uint16());
//        else if (t == type::get<uint32_t>())
//            writer.Uint(var.to_uint32());
//        else if (t == type::get<uint64_t>())
//            writer.Uint64(var.to_uint64());
//        else if (t == type::get<float>())
//            writer.Double(var.to_double());
//        else if (t == type::get<double>())
//            writer.Double(var.to_double());
//
//        return true;
//    }
//    else if (t.is_enumeration())
//    {
//        bool ok = false;
//        auto result = var.to_string(&ok);
//        if (ok)
//        {
//            writer.String(var.to_string());
//        }
//        else
//        {
//            ok = false;
//            auto value = var.to_uint64(&ok);
//            if (ok)
//                writer.Uint64(value);
//            else
//                writer.Null();
//        }
//
//        return true;
//    }
//    else if (t == type::get<std::string>())
//    {
//        writer.String(var.to_string());
//        return true;
//    }
//
//    return false;
//}
//
///////////////////////////////////////////////////////////////////////////////////////////
//
//static void write_array(const variant_sequential_view& view, PrettyWriter<StringBuffer>& writer)
//{
//    writer.StartArray();
//    for (const auto& item : view)
//    {
//        if (item.is_sequential_container())
//        {
//            write_array(item.create_sequential_view(), writer);
//        }
//        else
//        {
//            variant wrapped_var = item.extract_wrapped_value();
//            type value_type = wrapped_var.get_type();
//            if (value_type.is_arithmetic() || value_type == type::get<std::string>() || value_type.is_enumeration())
//            {
//                write_atomic_types_to_json(value_type, wrapped_var, writer);
//            }
//            else // object
//            {
//                to_json_recursively(wrapped_var, writer);
//            }
//        }
//    }
//    writer.EndArray();
//}
//
//
///////////////////////////////////////////////////////////////////////////////////////////
//
//static void write_associative_container(const variant_associative_view& view, PrettyWriter<StringBuffer>& writer)
//{
//    static const string_view key_name("key");
//    static const string_view value_name("value");
//
//    writer.StartArray();
//
//    if (view.is_key_only_type())
//    {
//        for (auto& item : view)
//        {
//            write_variant(item.first, writer);
//        }
//    }
//    else
//    {
//        for (auto& item : view)
//        {
//            writer.StartObject();
//            writer.String(key_name.data(), static_cast<rapidjson::SizeType>(key_name.length()), false);
//
//            write_variant(item.first, writer);
//
//            writer.String(value_name.data(), static_cast<rapidjson::SizeType>(value_name.length()), false);
//
//            write_variant(item.second, writer);
//
//            writer.EndObject();
//        }
//    }
//
//    writer.EndArray();
//}
//
///////////////////////////////////////////////////////////////////////////////////////////
//
//bool write_variant(const variant& var, PrettyWriter<StringBuffer>& writer)
//{
//    auto value_type = var.get_type();
//    auto wrapped_type = value_type.is_wrapper() ? value_type.get_wrapped_type() : value_type;
//    bool is_wrapper = wrapped_type != value_type;
//
//    if (write_atomic_types_to_json(is_wrapper ? wrapped_type : value_type,
//                                   is_wrapper ? var.extract_wrapped_value() : var, writer))
//    {
//    }
//    else if (var.is_sequential_container())
//    {
//        write_array(var.create_sequential_view(), writer);
//    }
//    else if (var.is_associative_container())
//    {
//        write_associative_container(var.create_associative_view(), writer);
//    }
//    else
//    {
//        auto child_props = is_wrapper ? wrapped_type.get_properties() : value_type.get_properties();
//        if (!child_props.empty())
//        {
//            to_json_recursively(var, writer);
//        }
//        else
//        {
//            bool ok = false;
//            auto text = var.to_string(&ok);
//            if (!ok)
//            {
//                writer.String(text);
//                return false;
//            }
//
//            writer.String(text);
//        }
//    }
//
//    return true;
//}
//
///////////////////////////////////////////////////////////////////////////////////////////
//
//void to_json_recursively(const instance& obj2, PrettyWriter<StringBuffer>& writer)
//{
//    writer.StartObject();
//    instance obj = obj2.get_type().get_raw_type().is_wrapper() ? obj2.get_wrapped_instance() : obj2;
//
//    auto prop_list = obj.get_derived_type().get_properties();
//    for (auto prop : prop_list)
//    {
//        if (prop.get_metadata("NO_SERIALIZE"))
//            continue;
//
//        variant prop_value = prop.get_value(obj);
//        if (!prop_value)
//            continue; // cannot serialize, because we cannot retrieve the value
//
//        const auto name = prop.get_name();
//        writer.String(name.data(), static_cast<rapidjson::SizeType>(name.length()), false);
//        if (!write_variant(prop_value, writer))
//        {
//            std::cerr << "cannot serialize property: " << name << std::endl;
//        }
//    }
//
//    writer.EndObject();
//}
//
//} // end namespace anonymous
//
///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////
//
//namespace io
//{
//
///////////////////////////////////////////////////////////////////////////////////////////
//
//std::string to_json(rttr::instance obj)
//{
//    if (!obj.is_valid())
//        return std::string();
//
//    StringBuffer sb;
//    PrettyWriter<StringBuffer> writer(sb);
//
//    to_json_recursively(obj, writer);
//
//    return sb.GetString();
//}
//
//core::Json ObjectToJson(rttr::instance obj);
//
//}

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
    } else if (t == Type::get<std::string>())
    {
        writer = var.to_string();
        return true;
    }

    return false;
}


void VariantSerializer::WriteArray(const VariantListView& view, Json& writer)
{
//    writer.StartArray();
    for (const auto& item : view) {
        if (item.is_sequential_container()) {
            WriteArray(item.create_sequential_view(), writer);
        } else {
            Variant wrapped_var = item.extract_wrapped_value();
            Type value_type = wrapped_var.get_type();
            if (value_type.is_arithmetic() || value_type == Type::get<std::string>() || value_type.is_enumeration()) {
                WriteAtomicTypesToJson(value_type, wrapped_var, writer);
            } else { // object
                ToJsonRecursively(wrapped_var, writer);
            }
        }
    }
//    writer.EndArray();
}

void VariantSerializer::WriteAssociativeContainer(const VariantMapView& view, Json& writer)
{
//    static const string_view key_name("key");
//    static const string_view value_name("value");

//    writer.StartArray();

    if (view.is_key_only_type()) {
        for (auto& item : view) {
            WriteVariant(item.first, writer);
        }
    } else {
        for (auto& item : view) {
//            writer.StartObject();
//            writer.String(key_name.data(), static_cast<rapidjson::SizeType>(key_name.length()), false);

            WriteVariant(item.first, writer);

//            writer.String(value_name.data(), static_cast<rapidjson::SizeType>(value_name.length()), false);

            WriteVariant(item.second, writer);

//            writer.EndObject();
        }
    }

//    writer.EndArray();
}

bool VariantSerializer::WriteVariant(const Variant& var, Json& writer)
{
    auto value_type = var.get_type();
    auto wrapped_type = value_type.is_wrapper() ? value_type.get_wrapped_type() : value_type;
    bool is_wrapper = wrapped_type != value_type;

    if (value_type.is_arithmetic() || value_type.is_enumeration() || value_type == Type::get<String>()) {
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
    if (raw_type.is_wrapper()) {
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
    Json json;
    s_resource_format_saver_ = resource_format_saver;
    ToJsonRecursively(obj, json);
    s_resource_format_saver_ = nullptr;
    return json;
}

}