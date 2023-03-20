/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-3-20
*/

#include "gobot/editor/imgui/scene_editor_panel.hpp"
#include "imgui_extension/icon_fonts/icons_material_design_icons.h"
#include "imgui.h"

namespace gobot {

SceneEditorPanel::SceneEditorPanel() {
    name_ = ICON_MDI_FILE_TREE " SceneTree###scene_editor";
}

void SceneEditorPanel::OnImGui() {

    ImGui::Begin(name_.toStdString().c_str());
    ImGui::End();

}


}
