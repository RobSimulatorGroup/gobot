/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-3-20
*/

#include "gobot/editor/imgui/inspector_panel.hpp"
#include "gobot/editor/property_inspector/editor_inspector.hpp"
#include "imgui_extension/icon_fonts/icons_material_design_icons.h"
#include "gobot/scene/node_3d.hpp"

#include "imgui.h"

namespace gobot {

InspectorPanel::InspectorPanel() {
    name_ = ICON_MDI_INFORMATION " Inspector###inspector";
    node_3d_  = Node3D::New<Node3D>();
    node_3d_->SetName("node3d");
    Variant variant(node_3d_);
    editor_inspector_ = EditorInspector::New<EditorInspector>(variant);
}

void InspectorPanel::OnImGui() {
    ImGui::Begin(name_.toStdString().c_str());

    editor_inspector_->OnImGui();

    ImGui::End();
}

}