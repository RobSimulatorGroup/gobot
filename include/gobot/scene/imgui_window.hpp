/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-3-10
*/

#pragma once

#include "gobot/core/object.hpp"
#include "gobot/scene/imgui_node.hpp"

namespace gobot {

class Editor;

// This class wraps ImGui window.
class ImGuiWindow : public ImGuiNode {
    GOBCLASS(ImGuiWindow, ImGuiNode)
public:
    ImGuiWindow();

    FORCE_INLINE bool& IsOpened() { return open_; }

    FORCE_INLINE void SetOpen(bool open) { open_ = open; }

    FORCE_INLINE void SetImGuiWindowFlag(int flags) { imgui_window_flags_ = flags; }

    FORCE_INLINE void SetImGuiWindowSize(const Vector2f& size, int imgui_cond) {
        window_size_ = std::make_pair(size, imgui_cond);
    }

    FORCE_INLINE void SetImGuiWindowPos(const Vector2f& pos, int imgui_cond = 0, const Vector2f& pivot = Vector2f(0, 0)) {
        window_pos_ = std::tuple(pos, imgui_cond, pivot);
    }

    bool Begin() override;

    void End() override;

protected:
    bool collapsed_{false};
    bool open_ = true;
    int imgui_window_flags_{0};
    std::optional<std::pair<Vector2f, int>> window_size_;
    std::optional<std::tuple<Vector2f, int, Vector2f>> window_pos_;
};

}
