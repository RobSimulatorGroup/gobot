/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-3-20
*/

#pragma once

#include <functional>
#include <string>
#include <vector>

#include "gobot/scene/imgui_window.hpp"

class ImGuiTextFilter;

namespace gobot {

class GOBOT_EXPORT SceneEditorPanel : public ImGuiWindow {
public:
    SceneEditorPanel();

    ~SceneEditorPanel() override;

    void OnImGuiContent() override;

    void DrawNode(Node* node);

private:
    struct AddNodeEntry {
        std::string label;
        std::function<Node*()> create;
    };

    static std::vector<AddNodeEntry> BuildAddNodeEntries();

    Node* GetAddChildTarget(Node* scene_root) const;

    void RequestOpenAddChildDialog(Node* parent);

    void DrawAddChildDialog();

    bool CreateSelectedAddNode();

    ImGuiTextFilter* filter_{nullptr};
    Node* double_clicked_{nullptr};
    Node* current_{nullptr};
    Node* add_child_parent_{nullptr};
    bool open_add_child_dialog_{false};
    std::string add_node_search_;
    int selected_add_node_index_{-1};
    bool select_up_;
    bool select_down_;

};


}
