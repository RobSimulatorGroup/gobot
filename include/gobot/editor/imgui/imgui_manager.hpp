/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-3-8
*/

#pragma once

#include "gobot/core/object.hpp"
#include "gobot/rendering/imgui_renderer.hpp"

namespace gobot {

class GOBOT_EXPORT ImGuiManager : public Object {
    GOBCLASS(ImGuiManager, Object)
public:
    ImGuiManager();

    ~ImGuiManager();

    static ImGuiManager* GetInstance();

    void BeginFrame();

    void EndFrame();

private:
    void SetImGuiStyle();

    void AddIconFont();

private:
    static ImGuiManager* s_singleton;

    std::unique_ptr<ImGuiRenderer> imgui_renderer_{nullptr};

    float font_size_;
};

}
