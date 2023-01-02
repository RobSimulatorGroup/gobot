/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 22-12-21
*/

#include "gobot/scene/resources/packed_scene.hpp"

namespace gobot {

std::size_t SceneState::GetNodeCount() const {
    return nodes_.size();
}

Ref<PackedScene> SceneState::GetNodeInstance(std::size_t idx) const {
    if (idx >= nodes_.size()) {
        return {};
    }


    return Ref<PackedScene>();
}


PackedScene::PackedScene() {
    state_ = Ref<SceneState>(gobot::Object::New<SceneState>());
}

Ref<SceneState> PackedScene::GetState() const {
    return state_;
}

}