/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-4-4
*/

#pragma once

#include "gobot/scene/node_3d.hpp"

namespace gobot {

class TestPropertyNode : public Node3D {
    GOBCLASS(TestPropertyNode, Node3D)
public:

    TestPropertyNode() = default;

private:
    GOBOT_REGISTRATION_FRIEND

    uint8_t uint8_;
    std::int64_t int64_;

};

}