/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * This file is created by Qiqi Wu, 23-3-29
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/editor/imgui/type_icons.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <string>
#include <string_view>
#include <unordered_map>

#include "gobot/scene/node.hpp"
#include "imgui_extension/icon_fonts/icons_material_design_icons.h"

namespace gobot {
namespace {

constexpr std::uint32_t IconColor(int r, int g, int b, int a = 255) {
    return static_cast<std::uint32_t>(r) |
           (static_cast<std::uint32_t>(g) << 8U) |
           (static_cast<std::uint32_t>(b) << 16U) |
           (static_cast<std::uint32_t>(a) << 24U);
}

constexpr std::uint32_t kNodeColor = IconColor(154, 172, 190);
constexpr std::uint32_t kSceneColor = IconColor(118, 169, 255);
constexpr std::uint32_t kRobotColor = IconColor(89, 214, 155);
constexpr std::uint32_t kPhysicsColor = IconColor(242, 158, 75);
constexpr std::uint32_t kSensorColor = IconColor(64, 205, 210);
constexpr std::uint32_t kMeshColor = IconColor(128, 176, 255);
constexpr std::uint32_t kMaterialColor = IconColor(226, 142, 255);
constexpr std::uint32_t kScriptColor = IconColor(255, 216, 96);
constexpr std::uint32_t kResourceColor = IconColor(198, 205, 215);
constexpr std::uint32_t kImageColor = IconColor(104, 190, 255);
constexpr std::uint32_t kDataColor = IconColor(142, 213, 132);

EditorIcon FontIcon(const char* glyph, std::uint32_t color) {
    return {glyph, EditorIconKind::Font, color};
}

EditorIcon AxisIcon() {
    return {ICON_MDI_AXIS, EditorIconKind::Axis3D, kSceneColor};
}

EditorIcon JointIcon() {
    return {ICON_MDI_SOURCE_COMMIT, EditorIconKind::Joint3D, kRobotColor};
}

const std::unordered_map<std::string_view, EditorIcon>& TypeIconMap() {
    static const std::unordered_map<std::string_view, EditorIcon> icons = {
            {"Object", FontIcon(ICON_MDI_CUBE, kResourceColor)},
            {"RefCounted", FontIcon(ICON_MDI_LINK_VARIANT, kResourceColor)},
            {"Node", FontIcon(ICON_MDI_CIRCLE_OUTLINE, kNodeColor)},
            {"Node3D", AxisIcon()},
            {"Camera3D", FontIcon(ICON_MDI_CAMERA_IRIS, kSceneColor)},
            {"Window", FontIcon(ICON_MDI_APPLICATION, kNodeColor)},
            {"SceneTree", FontIcon(ICON_MDI_FILE_TREE, kSceneColor)},

            {"MeshInstance3D", FontIcon(ICON_MDI_CUBE_OUTLINE, kMeshColor)},
            {"BoxMeshInstance3D", FontIcon(ICON_MDI_CUBE_OUTLINE, kMeshColor)},
            {"CylinderMeshInstance3D", FontIcon(ICON_MDI_PIPE, kMeshColor)},
            {"SphereMeshInstance3D", FontIcon(ICON_MDI_VECTOR_CIRCLE, kMeshColor)},
            {"CollisionShape3D", FontIcon(ICON_MDI_SHAPE_OUTLINE, kPhysicsColor)},
            {"Terrain3D", FontIcon(ICON_MDI_TERRAIN, kPhysicsColor)},

            {"Robot3D", FontIcon(ICON_MDI_ROBOT_INDUSTRIAL, kRobotColor)},
            {"Link3D", FontIcon(ICON_MDI_LINK, kRobotColor)},
            {"Joint3D", JointIcon()},

            {"Sensor3D", FontIcon(ICON_MDI_RADAR, kSensorColor)},
            {"IMUSensor3D", FontIcon(ICON_MDI_GAUGE, kSensorColor)},
            {"AngularMomentumSensor3D", FontIcon(ICON_MDI_ROTATE_3D, kSensorColor)},
            {"ContactSensor3D", FontIcon(ICON_MDI_CROSSHAIRS, kSensorColor)},
            {"RayCastSensor3D", FontIcon(ICON_MDI_RAY_START_END, kSensorColor)},
            {"TerrainHeightSensor3D", FontIcon(ICON_MDI_TERRAIN, kSensorColor)},
            {"HeightScanner3D", FontIcon(ICON_MDI_RAY_START_END, kSensorColor)},

            {"Resource", FontIcon(ICON_MDI_FILE_OUTLINE, kResourceColor)},
            {"PackedScene", FontIcon(ICON_MDI_FILE_TREE, kSceneColor)},
            {"PythonScript", FontIcon(ICON_MDI_LANGUAGE_PYTHON, kScriptColor)},
            {"ProjectSettings", FontIcon(ICON_MDI_COGS, kSceneColor)},

            {"Mesh", FontIcon(ICON_MDI_VECTOR_TRIANGLE, kMeshColor)},
            {"ArrayMesh", FontIcon(ICON_MDI_VECTOR_TRIANGLE, kMeshColor)},
            {"PrimitiveMesh", FontIcon(ICON_MDI_VECTOR_TRIANGLE, kMeshColor)},
            {"BoxMesh", FontIcon(ICON_MDI_CUBE_OUTLINE, kMeshColor)},
            {"PlaneMesh", FontIcon(ICON_MDI_VECTOR_RECTANGLE, kMeshColor)},
            {"CapsuleMesh", FontIcon(ICON_MDI_PILL, kMeshColor)},
            {"SphereMesh", FontIcon(ICON_MDI_VECTOR_CIRCLE, kMeshColor)},
            {"CylinderMesh", FontIcon(ICON_MDI_PIPE, kMeshColor)},

            {"Material", FontIcon(ICON_MDI_PALETTE, kMaterialColor)},
            {"PBRMaterial3D", FontIcon(ICON_MDI_PALETTE_SWATCH, kMaterialColor)},
            {"ShaderMaterial", FontIcon(ICON_MDI_PALETTE_ADVANCED, kMaterialColor)},
            {"Shader", FontIcon(ICON_MDI_SCRIPT_TEXT, kScriptColor)},
            {"ShaderProgram", FontIcon(ICON_MDI_CHIP, kScriptColor)},
            {"RasterizerShaderProgram", FontIcon(ICON_MDI_CHIP, kScriptColor)},
            {"ComputeShaderProgram", FontIcon(ICON_MDI_CHIP, kScriptColor)},

            {"Image", FontIcon(ICON_MDI_IMAGE, kImageColor)},
            {"BitMap", FontIcon(ICON_MDI_CHECKBOX_MULTIPLE_BLANK_OUTLINE, kImageColor)},
            {"Texture", FontIcon(ICON_MDI_TEXTURE, kImageColor)},
            {"Texture2D", FontIcon(ICON_MDI_TEXTURE, kImageColor)},
            {"Texture3D", FontIcon(ICON_MDI_TEXTURE, kImageColor)},
            {"TextureCube", FontIcon(ICON_MDI_CUBE_SCAN, kImageColor)},

            {"Shape3D", FontIcon(ICON_MDI_SHAPE_OUTLINE, kPhysicsColor)},
            {"BoxShape3D", FontIcon(ICON_MDI_CUBE_OUTLINE, kPhysicsColor)},
            {"CapsuleShape3D", FontIcon(ICON_MDI_PILL, kPhysicsColor)},
            {"SphereShape3D", FontIcon(ICON_MDI_VECTOR_CIRCLE, kPhysicsColor)},
            {"CylinderShape3D", FontIcon(ICON_MDI_PIPE, kPhysicsColor)},

            {"PhysicsServer", FontIcon(ICON_MDI_COGS, kPhysicsColor)},
            {"PhysicsWorld", FontIcon(ICON_MDI_CUBE_SCAN, kPhysicsColor)},
            {"MuJoCoPhysicsWorld", FontIcon(ICON_MDI_CUBE_SCAN, kPhysicsColor)},
            {"NullPhysicsWorld", FontIcon(ICON_MDI_CUBE_OUTLINE, kPhysicsColor)},
            {"SimulationServer", FontIcon(ICON_MDI_PLAY_NETWORK, kPhysicsColor)},
            {"RenderServer", FontIcon(ICON_MDI_EYE_CIRCLE_OUTLINE, kSceneColor)},
            {"Input", FontIcon(ICON_MDI_GAMEPAD_VARIANT, kSceneColor)},
    };
    return icons;
}

std::string Lower(std::string value) {
    std::ranges::transform(value, value.begin(), [](unsigned char character) {
        return static_cast<char>(std::tolower(character));
    });
    return value;
}

std::string LowerExtension(std::string_view path) {
    return Lower(std::filesystem::path(std::string(path)).extension().string());
}

std::string LowerFilename(std::string_view path) {
    return Lower(std::filesystem::path(std::string(path)).filename().string());
}

EditorIcon FolderIcon() {
    return FontIcon(ICON_MDI_FOLDER, kResourceColor);
}

} // namespace

const char* GetTypeIcon(const Type& type) {
    const auto name = type.get_name();
    return GetTypeIcon(std::string_view(name.data(), name.size()));
}

const char* GetTypeIcon(std::string_view type_name) {
    return GetTypeEditorIcon(type_name).glyph;
}

const char* GetNodeIcon(const Node& node) {
    return GetNodeEditorIcon(node).glyph;
}

const char* GetResourcePathIcon(std::string_view path, bool is_directory) {
    return GetResourcePathEditorIcon(path, is_directory).glyph;
}

EditorIcon GetTypeEditorIcon(const Type& type) {
    const auto name = type.get_name();
    return GetTypeEditorIcon(std::string_view(name.data(), name.size()));
}

EditorIcon GetTypeEditorIcon(std::string_view type_name) {
    if (const auto iter = TypeIconMap().find(type_name); iter != TypeIconMap().end()) {
        return iter->second;
    }

    if (type_name.ends_with("Sensor3D")) {
        return FontIcon(ICON_MDI_RADAR, kSensorColor);
    }
    if (type_name.ends_with("Shape3D")) {
        return FontIcon(ICON_MDI_SHAPE_OUTLINE, kPhysicsColor);
    }
    if (type_name.ends_with("Mesh")) {
        return FontIcon(ICON_MDI_VECTOR_TRIANGLE, kMeshColor);
    }
    if (type_name.ends_with("Material")) {
        return FontIcon(ICON_MDI_PALETTE, kMaterialColor);
    }
    if (type_name.ends_with("Texture")) {
        return FontIcon(ICON_MDI_TEXTURE, kImageColor);
    }
    if (type_name.ends_with("3D")) {
        return AxisIcon();
    }

    return FontIcon(ICON_MDI_FILE_OUTLINE, kResourceColor);
}

EditorIcon GetNodeEditorIcon(const Node& node) {
    return GetTypeEditorIcon(node.GetType());
}

EditorIcon GetResourcePathEditorIcon(std::string_view path, bool is_directory) {
    if (is_directory) {
        return FolderIcon();
    }

    const std::string extension = LowerExtension(path);
    const std::string filename = LowerFilename(path);
    if (filename == "project.gobot") {
        return FontIcon(ICON_MDI_APPLICATION, kSceneColor);
    }
    if (extension == ".jscn") {
        return FontIcon(ICON_MDI_FILE_TREE, kSceneColor);
    }
    if (extension == ".jres" || extension == ".res") {
        return FontIcon(ICON_MDI_FILE_OUTLINE, kResourceColor);
    }
    if (extension == ".py") {
        return FontIcon(ICON_MDI_LANGUAGE_PYTHON, kScriptColor);
    }
    if (extension == ".usd" || extension == ".usda" || extension == ".usdc" ||
        extension == ".urdf" || extension == ".mjcf" || extension == ".xml") {
        return FontIcon(ICON_MDI_FILE_XML, kSceneColor);
    }
    if (extension == ".obj" || extension == ".stl" || extension == ".dae" ||
        extension == ".fbx" || extension == ".gltf" || extension == ".glb") {
        return FontIcon(ICON_MDI_VECTOR_TRIANGLE, kMeshColor);
    }
    if (extension == ".png" || extension == ".jpg" || extension == ".jpeg" ||
        extension == ".bmp" || extension == ".tga" || extension == ".hdr") {
        return FontIcon(ICON_MDI_FILE_IMAGE, kImageColor);
    }
    if (extension == ".vert" || extension == ".frag" || extension == ".glsl" ||
        extension == ".shader") {
        return FontIcon(ICON_MDI_SCRIPT_TEXT, kScriptColor);
    }
    if (extension == ".onnx" || extension == ".pt" || extension == ".pth") {
        return FontIcon(ICON_MDI_CHIP, kDataColor);
    }
    if (extension == ".json" || extension == ".yaml" || extension == ".yml" ||
        extension == ".toml") {
        return FontIcon(ICON_MDI_CODE_BRACES, kDataColor);
    }
    if (extension == ".md" || extension == ".txt" || extension == ".log") {
        return FontIcon(ICON_MDI_FILE_DOCUMENT_OUTLINE, kResourceColor);
    }
    if (extension == ".csv") {
        return FontIcon(ICON_MDI_FILE_TABLE, kDataColor);
    }
    if (extension == ".zip" || extension == ".tar" || extension == ".gz") {
        return FontIcon(ICON_MDI_ARCHIVE, kResourceColor);
    }
    if (extension == ".mp4" || extension == ".mov" || extension == ".webm") {
        return FontIcon(ICON_MDI_FILE_VIDEO, kImageColor);
    }
    if (extension == ".wav" || extension == ".mp3" || extension == ".ogg") {
        return FontIcon(ICON_MDI_FILE_MUSIC, kResourceColor);
    }

    return FontIcon(ICON_MDI_FILE_OUTLINE, kResourceColor);
}

}
