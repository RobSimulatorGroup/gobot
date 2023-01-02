/* Copyright(c) 2020-2022, Qiqi Wu<1258552199@qq.com>.
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 * This file is created by Qiqi Wu, 22-11-20
*/

#include "gobot/core/io/variant_serializer.hpp"

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


void ToJsonRecursively(Instance obj2, Json& writer)
{
//    Instance obj = obj2.get_type().get_raw_type().is_wrapper() ? obj2.get_wrapped_instance() : obj2;
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

}

ResourceFormatSaverSceneInstance* s_resource_format_saver = nullptr;

Json VariantToJson(Instance obj,
                   ResourceFormatSaverSceneInstance* resource_format_saver) {

    Json json;
    s_resource_format_saver = resource_format_saver;
    ToJsonRecursively(obj, json);
    return json;
}

}