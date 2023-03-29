/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-3-29
*/

#include "gobot/editor/imgui/type_icons.hpp"
#include "imgui_extension/icon_fonts/icons_material_design_icons.h"

namespace gobot {

const char* GetTypeIcon(const Type& type) {
    auto name = type.get_name();
    if (name == "Node3D") {
        return ICON_MDI_CUBE;
    }

    return ICON_MDI_HELP_CIRCLE_OUTLINE;
}


}
