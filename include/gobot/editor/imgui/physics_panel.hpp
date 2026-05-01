/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
*/

#pragma once

#include "gobot/physics/physics_types.hpp"
#include "gobot/scene/imgui_window.hpp"

namespace gobot {

class GOBOT_EXPORT PhysicsPanel : public ImGuiWindow {
    GOBCLASS(PhysicsPanel, ImGuiWindow)

public:
    PhysicsPanel();

    ~PhysicsPanel() override = default;

    void OnImGuiContent() override;

private:
    PhysicsBackendType selected_backend_{PhysicsBackendType::MuJoCoCpu};
};

} // namespace gobot
