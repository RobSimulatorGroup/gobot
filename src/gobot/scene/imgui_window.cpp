/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * This file is created by Qiqi Wu, 23-3-10
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/scene/imgui_window.hpp"
#include "imgui.h"


namespace gobot {

ImGuiWindow::ImGuiWindow() {

}

void ImGuiWindow::SetImGuiWindow(const std::string& title, const std::string& id) {
    imgui_window_title_ = title;
    imgui_window_id_ = id;
    imgui_window_label_cache_.clear();
}

void ImGuiWindow::RequestFocus() {
    open_ = true;
    request_focus_ = true;
}

const std::string& ImGuiWindow::GetImGuiWindowTitle() const {
    return imgui_window_title_.empty() ? GetName() : imgui_window_title_;
}

const std::string& ImGuiWindow::GetImGuiWindowLabel() const {
    if (imgui_window_title_.empty() && imgui_window_id_.empty()) {
        return GetName();
    }

    if (imgui_window_label_cache_.empty()) {
        imgui_window_label_cache_ = imgui_window_title_;
        if (!imgui_window_id_.empty()) {
            imgui_window_label_cache_ += "###";
            imgui_window_label_cache_ += imgui_window_id_;
        }
    }

    return imgui_window_label_cache_;
}

bool ImGuiWindow::Begin() {
    if (!open_) {
        return false;
    }

    ImGuiNode::Begin();

    if (window_size_.has_value()) {
        const auto& [size, cond] = window_size_.value();
        ImGui::SetNextWindowSize(size, cond);
    }
    if (window_pos_.has_value()) {
        const auto& [size, cond, pivot] = window_pos_.value();
        ImGui::SetNextWindowPos(size, cond, pivot);
    }
    if (request_focus_) {
        ImGui::SetNextWindowFocus();
        request_focus_ = false;
    }

    collapsed_ = ImGui::Begin(GetImGuiWindowLabel().c_str(), &open_, imgui_window_flags_);
    //   [Important: due to legacy reason, this is inconsistent with most other functions such as BeginMenu/EndMenu,
    //    BeginPopup/EndPopup, etc. where the EndXXX call should only be called if the corresponding BeginXXX function
    //    returned true. Begin and BeginChild are the only odd ones out. Will be fixed in a future update.]
    return true;
}

void ImGuiWindow::End() {
    ImGui::End();
    ImGuiNode::End();
}


}
