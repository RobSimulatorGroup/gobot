/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * This file is created by Qiqi Wu, 23-4-3
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <utility>

#include "gobot/scene/imgui_node.hpp"

namespace gobot {

// This class allow user add callback function when process imgui.
class ImGuiCustomNode : public ImGuiNode {
    GOBCLASS(ImGuiCustomNode, ImGuiNode)
public:
    ImGuiCustomNode(std::function<void(void)> callback)
        : content_callback_(std::move(callback))
    {
    }

    void SetCallBack(std::function<void(void)> callback) {
        content_callback_ = std::move(callback);
    }

    void OnImGuiContent() override {
        if (content_callback_) {
            content_callback_();
        }
    }

private:
    std::function<void(void)> content_callback_{};
};


}
