/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 22-11-27
*/

#include <gtest/gtest.h>
#include <iostream>

#include <gobot/log.hpp>
#include <gobot/scene/node_path.hpp>

namespace {

class TestNodePath : public gobot::NodePath {
public:
    TestNodePath() = default;
};

}

//TEST(TestNodePath, test_ctor) {
//    gobot::String path = "/root";
//    gobot::NodePath node_path(path);

TEST(TestNodePath, rel_path) {
    const gobot::NodePath node_path_relative = gobot::NodePath("/Path2D/PathFollow2D/Sprite2D:position:x");

    ASSERT_TRUE(node_path_relative.IsAbsolute() == true);

    std::cout << "GetNames: ";
    for (const auto& str : node_path_relative.GetNames()) {
        std::cout << str.toStdString() << " ";
    }
    std::cout << std::endl;

    std::cout << "GetSubNames: ";
    for (const auto& str : node_path_relative.GetSubNames()) {
        std::cout << str.toStdString() << " ";
    }
    std::cout << std::endl;
}
