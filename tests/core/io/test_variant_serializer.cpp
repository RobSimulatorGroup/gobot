/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-1-15
*/

#include <gtest/gtest.h>

#include <gobot/core/io/resource.hpp>
#include <gobot/core/registration.hpp>
#include <gobot/core/io/variant_serializer.hpp>
#include <gobot/core/types.hpp>
#include <gobot/log.hpp>

TEST(TestVariantSerializer, test_primitive_type) {
    // test int
    {
        int i{1};
        auto json = gobot::VariantSerializer::VariantToJson(i);
        int i_2;
        gobot::Variant variant(i_2);
        ASSERT_TRUE(gobot::VariantSerializer::JsonToVariant(variant, json));
        ASSERT_TRUE(variant.get_value<int>() == i);
    }
    // test uint
    {
        uint16_t u16{1};
        auto json = gobot::VariantSerializer::VariantToJson(u16);
        uint16_t u16_2;
        gobot::Variant variant(u16_2);
        ASSERT_TRUE(gobot::VariantSerializer::JsonToVariant(variant, json));
        ASSERT_TRUE(variant.get_value<uint16_t>() == u16);
    }

    // test float/double
    {
        float f{0.1};
        auto json = gobot::VariantSerializer::VariantToJson(f);
        float f_2;
        gobot::Variant variant(f_2);
        ASSERT_TRUE(gobot::VariantSerializer::JsonToVariant(variant, json));
        ASSERT_TRUE(variant.get_value<float>() == f);
    }

    // test std::string
    {
        std::string str{"111"};
        auto json = gobot::VariantSerializer::VariantToJson(str);
        std::string str_2;
        gobot::Variant variant(str_2);
        ASSERT_TRUE(gobot::VariantSerializer::JsonToVariant(variant, json));
        ASSERT_TRUE(variant.get_value<std::string>() == str);
    }


    // test gobot::String
    {
        gobot::String str{"111"};
        auto json = gobot::VariantSerializer::VariantToJson(str);
        gobot::String str_2;
        gobot::Variant variant(str_2);
        ASSERT_TRUE(gobot::VariantSerializer::JsonToVariant(variant, json));
        ASSERT_TRUE(variant.get_value<gobot::String>() == str);
    }

    // test enum
    {
        gobot::PropertyUsageFlags flags{gobot::PropertyUsageFlags::Editor};
        auto json = gobot::VariantSerializer::VariantToJson(flags);
        gobot::PropertyUsageFlags flags_2;
        gobot::Variant variant(flags_2);
        ASSERT_TRUE(gobot::VariantSerializer::JsonToVariant(variant, json));
        ASSERT_TRUE(variant.get_value<gobot::PropertyUsageFlags>() == flags);
    }

}

TEST(TestVariantSerializer, test_vector_int) {
    std::vector<int> vector_int{1, 2, 3};
    gobot::Variant var(vector_int);
    auto json = gobot::VariantSerializer::VariantToJson(var);
    std::vector<int> vector_int_2;
    gobot::Variant variant(vector_int_2);
    ASSERT_TRUE(gobot::VariantSerializer::JsonToVariant(variant, json));
    ASSERT_TRUE(variant.get_value<std::vector<int>>() == vector_int);
}

TEST(TestVariantSerializer, test_struct) {
    gobot::PropertyInfo property_info;
    property_info.name = "name1";
    property_info.hint_string = "hint";
    auto json = gobot::VariantSerializer::VariantToJson(property_info);
    gobot::PropertyInfo property_info_2;
    gobot::Variant variant(property_info_2);
    ASSERT_TRUE(gobot::VariantSerializer::JsonToVariant(variant, json));
    ASSERT_TRUE(variant.get_value<gobot::PropertyInfo>() == property_info);
}

TEST(TestVariantSerializer, test_vector_struct) {
    std::vector<gobot::PropertyInfo> property_info_vec{{"name1"}, {"name2"}};
    gobot::Variant var(property_info_vec);
    auto json = gobot::VariantSerializer::VariantToJson(var);
    std::vector<gobot::PropertyInfo> property_info_vec_2;
    gobot::Variant variant(property_info_vec_2);
    ASSERT_TRUE(gobot::VariantSerializer::JsonToVariant(variant, json));
    ASSERT_TRUE(variant.get_value<std::vector<gobot::PropertyInfo>>() == property_info_vec);
}


TEST(TestVariantSerializer, test_simple_map) {
    std::map<gobot::String, gobot::String> map{{"111", "222"}, {"2222", "22"}};
    auto json = gobot::VariantSerializer::VariantToJson(map);
    std::map<gobot::String, gobot::String> map_2;
    gobot::Variant variant(map_2);
    ASSERT_TRUE(gobot::VariantSerializer::JsonToVariant(variant, json));
    auto equal = variant.get_value<std::map<gobot::String, gobot::String>>() == map;
    ASSERT_TRUE(equal);
}

TEST(TestVariantSerializer, test_struct_map) {
    std::map<gobot::String, gobot::PropertyInfo> property_info_map{{"111", {"name1"}}, {"2222", {"name2"}}};
    auto json = gobot::VariantSerializer::VariantToJson(property_info_map);
    std::map<gobot::String, gobot::PropertyInfo> map_2;
    gobot::Variant variant(map_2);
    ASSERT_TRUE(gobot::VariantSerializer::JsonToVariant(variant, json));
    auto equal = variant.get_value<std::map<gobot::String, gobot::PropertyInfo>>() == property_info_map;
    ASSERT_TRUE(equal);
}