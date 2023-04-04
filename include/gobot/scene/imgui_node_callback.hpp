/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-4-3
*/

#pragma once

#include "gobot/scene/imgui_node.hpp"

namespace gobot {

class ImGuiCustomNode : public ImGuiNode {
    GOBCLASS(ImGuiCustomNode, ImGuiNode)
public:
    ImGuiCustomNode(const std::function<void(void)>& callback)
        : callback_(callback)
    {
    }

    void SetCallBack(const std::function<void(void)>& callback) {
        callback_ = callback;
    }

    void OnImGuiContent() override {
        if (callback_) {
            callback_();
        }
    }

private:
    std::function<void(void)> callback_{};
};


}
