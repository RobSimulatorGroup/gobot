/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-1-15
*/

#include <gtest/gtest.h>
#include <gobot/core/string_utils.hpp>
#include <gobot/log.hpp>

TEST(TestStringUtils, test_get_file_extension) {
    ASSERT_TRUE(gobot::GetFileExtension("1111.json") == "json");
    ASSERT_TRUE(gobot::GetFileExtension("res://1111.json") == "json");
}

TEST(TestStringUtils, test_is_absolute_path) {
    ASSERT_FALSE(gobot::IsAbsolutePath(""));
    ASSERT_TRUE(gobot::IsAbsolutePath("/"));
    ASSERT_TRUE(gobot::IsAbsolutePath("/1111"));
    ASSERT_TRUE(gobot::IsAbsolutePath("C:/1111.json"));
    ASSERT_TRUE(gobot::IsAbsolutePath("C:\\1111.json"));
}

TEST(TestStringUtils, test_simplify_path) {
    ASSERT_TRUE(gobot::SimplifyPath("/usr/lib/../") == "/usr");
    ASSERT_TRUE(gobot::SimplifyPath("res://test.jscn") == "res://test.jscn");
    ASSERT_TRUE(gobot::SimplifyPath("/usr/lib/./dpkg") == "/usr/lib/dpkg");
    ASSERT_TRUE(gobot::SimplifyPath("res://font/test/../") == "res://font");
    ASSERT_TRUE(gobot::SimplifyPath("res://font/./test.jscn") == "res://font/test.jscn");
}

TEST(TestStringUtils, test_validate_local_path) {
    ASSERT_TRUE(gobot::ValidateLocalPath("font") == "res://font");
}

TEST(TestStringUtils, test_base_dir) {
    ASSERT_TRUE(gobot::GetBaseDir("font.json") == "");
    ASSERT_TRUE(gobot::GetBaseDir("C:\\font.json") == "C:\\");
    ASSERT_TRUE(gobot::GetBaseDir("/usr/lib/font.json") == "/usr/lib");
}

TEST(TestStringUtils, test_path_join) {
    ASSERT_TRUE(gobot::PathJoin("/usr", "font.json") == "/usr/font.json");
    ASSERT_TRUE(gobot::PathJoin("/usr/", "font.json") == "/usr/font.json");
}


int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}