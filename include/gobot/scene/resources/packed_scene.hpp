/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 22-12-21
*/

#pragma once

#include "gobot/core/io/resource.hpp"

namespace gobot {

class PackedScene;

class GOBOT_EXPORT SceneState : public RefCounted {
    GOBCLASS(SceneState, RefCounted)
public:
    struct NodeData {

    };

    std::size_t GetNodeCount() const;

    Ref<PackedScene> GetNodeInstance(std::size_t idx) const;

private:
    std::vector<NodeData> nodes_;
};

class GOBOT_EXPORT PackedScene : public Resource {
    GOBCLASS(PackedScene, Resource)
public:

    PackedScene();

    Ref<SceneState> GetState() const;

private:
    Ref<SceneState> state_;
};

}