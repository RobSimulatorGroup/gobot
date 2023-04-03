/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-4-2
*/

#pragma once

#include "gobot/scene/node.hpp"
#include "gobot/core/math/matrix.hpp"
#include "gobot/core/color.hpp"

namespace gobot {

class ImGuiNode : public Node {
    GOBCLASS(ImGuiNode, Node)
public:

    void OnImGui();

    void SetImGuiStyleVar(int var, const Vector2f& value);

    void SetImGuiStyleVar(int var, float value);

    void SetImGuiStyleColor(int col, const Color& color);

protected:
    virtual bool Begin();

    virtual void OnImGuiContent() {}

    virtual void End();

    void NotificationCallBack(NotificationType notification);

private:

    ImGuiNode* parent_{nullptr};
    std::list<ImGuiNode*> imgui_children_{};

    std::vector<std::pair<int, std::variant<Vector2f, float>>> imgui_style_var_stack_;
    std::vector<std::pair<int, Color>> imgui_style_color_stack_;
};

}
