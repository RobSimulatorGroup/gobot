/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 22-11-27
*/

#include <gtest/gtest.h>

#include <gobot/log.hpp>
#include <gobot/scene/node_path.hpp>


TEST(TestNodePath, relative_path) {
    const gobot::NodePath node_path_relative = gobot::NodePath("Path2D/PathFollow2D/Sprite2D:position:x");

    // The constructor should return the same node path.
    ASSERT_TRUE(
            gobot::NodePath(node_path_relative.GetNames(),
                            node_path_relative.GetSubNames(),
                            false) ==
            node_path_relative);
    // The returned property path should match the expected value.
    ASSERT_TRUE(node_path_relative.GetAsPropertyPath() ==
        gobot::NodePath(":Path2D/PathFollow2D/Sprite2D:position:x"));
    // The returned concatenated names should match the expected value.
    ASSERT_TRUE(node_path_relative.GetConcatenatedNames() == "Path2D/PathFollow2D/Sprite2D");
    // The returned concatenated subnames should match the expected value.
    ASSERT_TRUE(node_path_relative.GetConcatenatedSubNames() == "position:x");

    // The returned name at index 0 should match the expected value.
    ASSERT_TRUE(node_path_relative.GetName(0) == "Path2D");
    // The returned name at index 1 should match the expected value.
    ASSERT_TRUE(node_path_relative.GetName(1) == "PathFollow2D");
    // The returned name at index 2 should match the expected value.
    ASSERT_TRUE(node_path_relative.GetName(2) == "Sprite2D");
LOG_OFF;
    // The returned name at invalid index 3 should match the expected value.
    ASSERT_TRUE(node_path_relative.GetName(3) == "");
    // The returned name at invalid index -1 should match the expected value.
    ASSERT_TRUE(node_path_relative.GetName(-1) == "");
LOG_ON;
    // The returned number of names should match the expected value.
    ASSERT_TRUE(node_path_relative.GetNameCount() == 3);

    // The returned subname at index 0 should match the expected value.
    ASSERT_TRUE(node_path_relative.GetSubName(0) == "position");
    // The returned subname at index 1 should match the expected value.
    ASSERT_TRUE(node_path_relative.GetSubName(1) == "x");
LOG_OFF;
    // The returned subname at invalid index 2 should match the expected value.
    ASSERT_TRUE(node_path_relative.GetSubName(2) == "");
    // The returned subname at invalid index -1 should match the expected value.
    ASSERT_TRUE(node_path_relative.GetSubName(-1) == "");
LOG_ON;
    // The returned number of subnames should match the expected value.
    ASSERT_TRUE(node_path_relative.GetSubNameCount() == 2);

    // The node path should be considered relative.
    ASSERT_TRUE(node_path_relative.IsAbsolute() == false);
    // The node path shouldn't be considered empty.
    ASSERT_TRUE(node_path_relative.IsEmpty() == false);
}

TEST(TestNodePath, absolute_path) {
    const gobot::NodePath node_path_absolute = gobot::NodePath("/root/Sprite2D");

    // The constructor should return the same node path.
    ASSERT_TRUE(gobot::NodePath(node_path_absolute.GetNames(), true) == node_path_absolute);
    // The returned property path should match the expected value.
    ASSERT_TRUE(node_path_absolute.GetAsPropertyPath() == gobot::NodePath(":root/Sprite2D"));
    // The returned concatenated names should match the expected value.
    ASSERT_TRUE(node_path_absolute.GetConcatenatedNames() == "/root/Sprite2D");
    // The returned concatenated subnames should match the expected value.
    ASSERT_TRUE(node_path_absolute.GetConcatenatedSubNames() == "");

    // The returned name at index 0 should match the expected value.
    ASSERT_TRUE(node_path_absolute.GetName(0) == "root");
    // The returned name at index 1 should match the expected value.
    ASSERT_TRUE(node_path_absolute.GetName(1) == "Sprite2D");
LOG_OFF;
    // The returned name at invalid index 2 should match the expected value.
    ASSERT_TRUE(node_path_absolute.GetName(2) == "");
    // The returned name at invalid index -1 should match the expected value.
    ASSERT_TRUE(node_path_absolute.GetName(-1) == "");
LOG_ON;
    // The returned number of names should match the expected value.
    ASSERT_TRUE(node_path_absolute.GetNameCount() == 2);
    // The returned number of subnames should match the expected value.
    ASSERT_TRUE(node_path_absolute.GetSubNameCount() == 0);

    // The node path should be considered absolute.
    ASSERT_TRUE(node_path_absolute.IsAbsolute() == true);
    // The node path shouldn't be considered empty.
    ASSERT_TRUE(node_path_absolute.IsEmpty() == false);
}

TEST(TestNodePath, empty_path) {
    const gobot::NodePath node_path_empty = gobot::NodePath();

    // The constructor should return the same node path.
    ASSERT_TRUE(
            gobot::NodePath(node_path_empty.GetNames(),
                            node_path_empty.GetSubNames(),
                            false) ==
                    node_path_empty);
    // The returned property path should match the expected value.
    ASSERT_TRUE(node_path_empty.GetAsPropertyPath() == gobot::NodePath());
LOG_OFF;
    // The returned concatenated names should match the expected value.
    ASSERT_TRUE(node_path_empty.GetConcatenatedNames() == "");
    // The returned concatenated subnames should match the expected value.
    ASSERT_TRUE(node_path_empty.GetConcatenatedSubNames() == "");
LOG_ON;

    // The returned number of names should match the expected value.
    ASSERT_TRUE(node_path_empty.GetNameCount() == 0);
    // The returned number of subnames should match the expected value.
    ASSERT_TRUE(node_path_empty.GetSubNameCount() == 0);

    // The node path shouldn't be considered absolute.
    ASSERT_TRUE(node_path_empty.IsAbsolute() == false);
    // The node path should be considered empty.
    ASSERT_TRUE(node_path_empty.IsEmpty() == true);
}

TEST(TestNodePath, complex_path) {
    const gobot::NodePath node_path_complex = gobot::NodePath("Path2D/./PathFollow2D/../Sprite2D:position:x");
    const gobot::NodePath node_path_simplified = node_path_complex.Simplified();

    // The returned concatenated names should match the expected value.
    ASSERT_TRUE(node_path_simplified.GetConcatenatedNames() == "Path2D/Sprite2D");
    // The returned concatenated subnames should match the expected value.
    ASSERT_TRUE(node_path_simplified.GetConcatenatedSubNames() == "position:x");
}

TEST(TestNodePath, test_type) {
    gobot::Variant var = gobot::NodePath("/home/gobot");
    auto node_path = var.convert<gobot::NodePath>();
    ASSERT_TRUE(node_path.operator gobot::
    String() == gobot::String("/home/gobot"));

    auto prop = var.get_type().get_property("str_data");
    ASSERT_TRUE(prop.get_value(var).to_string() == "/home/gobot");

    auto type = gobot::Type::get_by_name("NodePath");
    auto var2 = type.create();
    prop.set_value(var2, gobot::String("/home/gobot"));
    ASSERT_TRUE(prop.get_value(var2).to_string() == "/home/gobot");
}
