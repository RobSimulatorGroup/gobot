/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * This file is created by Qiqi Wu, 23-3-29
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/editor/imgui/type_icons.hpp"
#include "imgui_extension/icon_fonts/icons_material_design_icons.h"

namespace gobot {

const char* GetTypeIcon(const Type& type) {
    const auto name = type.get_name();
    if (name == "Node" || name == "Node3D" || name == "MeshInstance3D") {
        return ICON_MDI_CUBE;
    }

    if (name == "Mesh" || name == "ArrayMesh" || name == "PrimitiveMesh") {
        return ICON_MDI_VECTOR_TRIANGLE;
    }
    if (name == "BoxMesh") {
        return ICON_MDI_CUBE_OUTLINE;
    }
    if (name == "SphereMesh") {
        return ICON_MDI_CIRCLE_OUTLINE;
    }
    if (name == "CylinderMesh") {
        return ICON_MDI_DATABASE;
    }

    if (name == "Material" || name == "PBRMaterial3D" || name == "ShaderMaterial") {
        return ICON_MDI_PALETTE;
    }

    if (name == "Shape3D" || name == "BoxShape3D" || name == "SphereShape3D" || name == "CylinderShape3D") {
        return ICON_MDI_SHAPE_OUTLINE;
    }

    if (name == "Resource") {
        return ICON_MDI_FILE_OUTLINE;
    }

    return ICON_MDI_CUBE;
}


}
