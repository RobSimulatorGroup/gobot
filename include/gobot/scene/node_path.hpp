/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 22-11-6
*/

#pragma once

#include "gobot/core/types.hpp"

namespace gobot {

class NodePath {
public:
    NodePath();

    bool IsAbsolute() const;

private:
    std::vector<String> path_;
    bool absolute_;
};

}