/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-1-15
 */

#include <gtest/gtest.h>

#include <gobot/core/io/resource.hpp>
#include <gobot/core/io/variant_serializer.hpp>
#include <gobot/core/registration.hpp>
#include <gobot/core/types.hpp>
#include <gobot/log.hpp>

namespace gobot {
struct VectorTestStruct {
  std::vector<int> test_int;
  std::vector<float> test_float;
  std::vector<std::string> test_string;

  bool operator==(const VectorTestStruct& rhs) const = default;

  friend std::ostream& operator<<(std::ostream& os, const VectorTestStruct& data) {
    os << "VectorTestStruct: [";
    for (auto value : data.test_int) {
      os << " " << std::to_string(value);
    }

    for (auto value : data.test_float) {
      os << " " << std::to_string(value);
    }

    for (auto value : data.test_string) {
      os << " " << value;
    }

    return os << "]";
  }
};
}  // namespace gobot

GOBOT_REGISTRATION {
  Class_<VectorTestStruct>("VectorTestStruct")
      .constructor()(CtorAsObject)
      .property("test_int", &VectorTestStruct::test_int)
      .property("test_float", &VectorTestStruct::test_float)
      .property("test_string", &VectorTestStruct::test_string);
}

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
    std::string str{"111"};
    auto json = gobot::VariantSerializer::VariantToJson(str);
      std::string str_2;
    gobot::Variant variant(str_2);
    ASSERT_TRUE(gobot::VariantSerializer::JsonToVariant(variant, json));
    ASSERT_TRUE(variant.get_value<std::string>() == str);
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
  ASSERT_TRUE(variant.get_value<std::vector<gobot::PropertyInfo>>() ==
              property_info_vec);
}

TEST(TestVariantSerializer, test_simple_map) {
  std::map<std::string, std::string> map{{"111", "222"}, {"2222", "22"}};
  auto json = gobot::VariantSerializer::VariantToJson(map);
  std::map<std::string, std::string> map_2;
  gobot::Variant variant(map_2);
  ASSERT_TRUE(gobot::VariantSerializer::JsonToVariant(variant, json));
  auto equal =
      variant.get_value<std::map<std::string, std::string>>() == map;
  ASSERT_TRUE(equal);
}

TEST(TestVariantSerializer, test_struct_map) {
  std::map<std::string, gobot::PropertyInfo> property_info_map{
      {"111", {"name1"}}, {"2222", {"name2"}}};
  auto json = gobot::VariantSerializer::VariantToJson(property_info_map);
  std::map<std::string, gobot::PropertyInfo> map_2;
  gobot::Variant variant(map_2);
  ASSERT_TRUE(gobot::VariantSerializer::JsonToVariant(variant, json));
  auto equal =
      variant.get_value<std::map<std::string, gobot::PropertyInfo>>() ==
      property_info_map;
  ASSERT_TRUE(equal);
}

TEST(TestVariantSerializer, test_struct_of_vector) {
  using gobot::VectorTestStruct;
  VectorTestStruct test{{1, 2, 3}, {4.0, 5.0}, {"hello"}};
  gobot::Variant var(test);
  auto json = gobot::VariantSerializer::VariantToJson(var);
  VectorTestStruct deserialized;
  gobot::Variant variant(deserialized);
  ASSERT_TRUE(gobot::VariantSerializer::JsonToVariant(variant, json));
  ASSERT_EQ(variant.get_value<VectorTestStruct>(), test);
}
