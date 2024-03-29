/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 22-12-3
*/

#include <gtest/gtest.h>

#include <gobot/core/object.hpp>
#include <gobot/core/types.hpp>
#include <gobot/log.hpp>
#include <rttr/enumeration.h>
#include <rttr/library.h>

class ReflectionTestEnvironment : public ::testing::Environment {
 public:
  virtual void SetUp() {
    lib_ = std::make_unique<rttr::library>("gobot");
    lib_->load();
    std::cerr << "LIB: " << lib_->get_file_name() << "\n";
    std::cerr << "IS LOADED: " << lib_->is_loaded() << "\n";
    auto types = lib_->get_types();
    std::cerr << "We have type num: " << types.size() << "\n";
    for (auto& type : types) {
      std::cerr << "Type " << type.get_name() << "\n";
    }
  }

  virtual void TearDown() { 
     std::cerr << "Teardown"
              << "\n";
      lib_->unload();
  }

 private:
  std::unique_ptr<rttr::library> lib_;
};

TEST(TestRegistration, test_registration) {
    auto property_usage_flags = gobot::Type::get_by_name("PropertyUsageFlags");
    ASSERT_TRUE(property_usage_flags.is_enumeration());
    auto enumeration = property_usage_flags.get_enumeration();
    auto names = enumeration.get_names();
    std::vector<std::string> string_names;
    for(const auto& name: names) {
        string_names.emplace_back(name.data());
    }
    ASSERT_TRUE(string_names.at(0) == "None");
    ASSERT_TRUE(string_names.at(1) == "Storage");
    ASSERT_TRUE(string_names.at(2) == "Editor");

    auto object_type = gobot::Type::get_by_name("Object");
    auto object_type2 = gobot::Type::get<gobot::Object>();
    ASSERT_TRUE(object_type.is_valid());
    ASSERT_TRUE(object_type == object_type2);

}

TEST(TestRegistration, test_types) {
//    auto uuid = gobot::Uuid::createUuid();
//    gobot::Variant var_uuid = uuid;
//    ASSERT_TRUE(uuid.toString().toStdString() == var_uuid.to_string());
}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    // gtest takes ownership of the ReflectionTestEnvironment ptr - we don't delete it.
    ::testing::AddGlobalTestEnvironment(new ReflectionTestEnvironment);
    return RUN_ALL_TESTS();
}
