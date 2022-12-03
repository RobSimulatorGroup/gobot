/* Copyright(c) 2020-2022, Qiqi Wu<1258552199@qq.com>.
 * This file is created by Qiqi Wu, 22-12-3
*/

#include <gtest/gtest.h>

#include <gobot/core/object.hpp>
#include <gobot/core/types.hpp>
#include <rttr/enumeration.h>


TEST(TestRefCounted, test_registration) {
    auto property_usage_flags = gobot::Type::get_by_name("PropertyUsageFlags");
    ASSERT_TRUE(property_usage_flags.is_enumeration());
    auto enumeration = property_usage_flags.get_enumeration();
    auto names = enumeration.get_names();
    std::vector<std::string> string_names;
    for(const auto& name: names) {
        string_names.emplace_back(name.data());
    }
    ASSERT_TRUE(string_names.size() == 4);
    ASSERT_TRUE(string_names.at(0) == "None");
    ASSERT_TRUE(string_names.at(1) == "Storage");
    ASSERT_TRUE(string_names.at(2) == "Editor");
    ASSERT_TRUE(string_names.at(3) == "UsageDefault");

    auto object_type = gobot::Type::get_by_name("Object");
    auto object_type2 = gobot::Type::get<gobot::Object>();
    ASSERT_TRUE(object_type.is_valid());
    ASSERT_TRUE(object_type == object_type2);

}