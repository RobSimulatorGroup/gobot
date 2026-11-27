/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "gobot/scene/imgui_window.hpp"

#include <string>

namespace gobot {

class GOBOT_EXPORT MotionPlanningPanel : public ImGuiWindow {
    GOBCLASS(MotionPlanningPanel, ImGuiWindow)
public:
    MotionPlanningPanel();

    void OnImGuiContent() override;

private:
    std::string last_action_note_;
};

} // namespace gobot
