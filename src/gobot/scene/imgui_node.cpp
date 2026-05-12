/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * This file is created by Qiqi Wu, 23-4-2
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/scene/imgui_node.hpp"
#include "imgui.h"

namespace gobot {


void ImGuiNode::NotificationCallBack(NotificationType notification) {
    switch (notification) {
        case NotificationType::EnterTree: {
            ERR_FAIL_COND(!GetTree());

            Node *p = GetParent();
            if (p) {
                parent_ = Object::PointerCastTo<ImGuiNode>(p);
            }

            if (parent_) {
                parent_->imgui_children_.emplace_back(this);
            }

        } break;

        case NotificationType::ExitTree: {
            Notification(NotificationType::ExitWorld, true);
            // todo: remove from transform change list
            if (parent_) {
                parent_->imgui_children_.remove(this);
            }
            parent_ = nullptr;
            // todo: update visibility
        } break;
    }
}

void ImGuiNode::OnImGui() {

    if (Begin()) {
        // Do itself, then call child
        OnImGuiContent();
        for (auto* child : imgui_children_) {
            child->OnImGui();
        }
        End();
    }
}

void ImGuiNode::SetImGuiStyleVar(int var, const Vector2f& value) {
    imgui_style_var_stack_.emplace_back(var, value);
}

void ImGuiNode::SetImGuiStyleVar(int var, float value) {
    imgui_style_var_stack_.emplace_back(var, value);
}

void ImGuiNode::SetImGuiStyleColor(int col, const Color& color) {
    imgui_style_color_stack_.emplace_back(col, color);
}


bool ImGuiNode::Begin() {
    for (const auto& [var, value]: imgui_style_var_stack_) {
        std::visit([=](auto&& arg) {
                       ImGui::PushStyleVar(var, arg);
                   }, value);
    }

    for (const auto& [var, color]: imgui_style_color_stack_) {
        ImGui::PushStyleColor(var, color);
    }

    return true;
}

void ImGuiNode::End() {
    ImGui::PopStyleVar(static_cast<int>(imgui_style_var_stack_.size()));
    ImGui::PopStyleColor(static_cast<int>(imgui_style_color_stack_.size()));
}



}
