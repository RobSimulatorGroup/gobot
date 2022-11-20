/* Copyright(c) 2020-2022, Qiqi Wu<1258552199@qq.com>.
 * This file is created by Qiqi Wu, 22-11-6
*/


#include "gobot/scene/node_path.hpp"

namespace gobot::scene {

NodePath::NodePath() {
}

bool NodePath::IsAbsolute() const {
    return absolute_;
}

}