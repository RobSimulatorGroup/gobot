/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-1-15
*/

#include <gtest/gtest.h>


#include <gobot/core/io/resource.hpp>
#include <gobot/core/io/variant_serializer.hpp>
#include <gobot/core/types.hpp>
#include <gobot/log.hpp>



TEST(TestVariantSerializer, test_primitive_type) {
    // test int
    {
        int i{1};
        auto json = gobot::VariantSerializer::VariantToJson(i);
        gobot::Variant var = gobot::VariantSerializer::JsonToVariant(gobot::Type::get<int>(), json);
        ASSERT_TRUE(var.get_value<int>() == i);
        auto json2 = gobot::VariantSerializer::VariantToJson(var);
        ASSERT_TRUE(json == json2);
    }
    // test uint
    {
        uint16_t u16{1};
        auto json = gobot::VariantSerializer::VariantToJson(u16);
        gobot::Variant var = gobot::VariantSerializer::JsonToVariant(gobot::Type::get<uint16_t>(), json);
        ASSERT_TRUE(var.get_value<uint16_t>() == u16);
        auto json2 = gobot::VariantSerializer::VariantToJson(var);
        ASSERT_TRUE(json == json2);
    }

    // test float/double
    {
        float f{0.1};
        auto json = gobot::VariantSerializer::VariantToJson(f);
        gobot::Variant var = gobot::VariantSerializer::JsonToVariant(gobot::Type::get<float>(), json);
        ASSERT_TRUE(var.get_value<float>() == f);
        auto json2 = gobot::VariantSerializer::VariantToJson(var);
        ASSERT_TRUE(json == json2);
    }

    // test std::string
    {
        std::string str{"111"};
        auto json = gobot::VariantSerializer::VariantToJson(str);
        gobot::Variant var = gobot::VariantSerializer::JsonToVariant(gobot::Type::get<std::string>(), json);
        ASSERT_TRUE(var.get_value<std::string>() == str);
        auto json2 = gobot::VariantSerializer::VariantToJson(var);
        ASSERT_TRUE(json == json2);
    }

    // test std::string
    {
        std::string str{"111"};
        auto json = gobot::VariantSerializer::VariantToJson(str);
        gobot::Variant var = gobot::VariantSerializer::JsonToVariant(gobot::Type::get<std::string>(), json);
        ASSERT_TRUE(var.get_value<std::string>() == str);
        auto json2 = gobot::VariantSerializer::VariantToJson(var);
        ASSERT_TRUE(json == json2);
    }

    // test gobot::String
    {
        gobot::String str{"111"};
        auto json = gobot::VariantSerializer::VariantToJson(str);
        gobot::Variant var = gobot::VariantSerializer::JsonToVariant(gobot::Type::get<gobot::String>(), json);
        ASSERT_TRUE(var.get_value<gobot::String>() == str);
        auto json2 = gobot::VariantSerializer::VariantToJson(var);
        ASSERT_TRUE(json == json2);
    }

}

TEST(TestVariantSerializer, test_vector_int) {
    std::vector<int> vector_int{1, 2, 3};
    gobot::Variant var(vector_int);
    auto json = gobot::VariantSerializer::VariantToJson(var);
    gobot::Variant var2 = gobot::VariantSerializer::JsonToVariant(var.get_type(), json);
    auto json2 = gobot::VariantSerializer::VariantToJson(var2);
    ASSERT_TRUE(json == json2);
}