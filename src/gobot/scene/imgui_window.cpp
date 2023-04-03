/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-3-10
*/

#include "gobot/scene/imgui_window.hpp"
#include "imgui.h"


namespace gobot {

ImGuiWindow::ImGuiWindow() {

}

bool ImGuiWindow::Begin() {
    //   [Important: due to legacy reason, this is inconsistent with most other functions such as BeginMenu/EndMenu,
    //    BeginPopup/EndPopup, etc. where the EndXXX call should only be called if the corresponding BeginXXX function
    //    returned true. Begin and BeginChild are the only odd ones out. Will be fixed in a future update.]

    if (window_size_.has_value()) {
        const auto& [size, cond] = window_size_.value();
        ImGui::SetNextWindowSize(size, cond);
    }
    if (window_pos_.has_value()) {
        const auto& [size, cond, pivot] = window_pos_.value();
        ImGui::SetNextWindowPos(size, cond, pivot);
    }

    collapsed_ = ImGui::Begin(GetName().toStdString().c_str(), &open_, imgui_window_flags_);
    return true;
};

void ImGuiWindow::End() {
    ImGui::End();
};


}
