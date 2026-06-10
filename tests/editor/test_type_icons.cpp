#include "gobot/editor/imgui/type_icons.hpp"

#include "gtest/gtest.h"
#include "imgui_extension/icon_fonts/icons_material_design_icons.h"

TEST(TestTypeIcons, uses_distinct_icons_for_editor_node_categories) {
    EXPECT_STREQ(gobot::GetTypeIcon("Object"), ICON_MDI_CUBE);
    EXPECT_STREQ(gobot::GetTypeIcon("Node"), ICON_MDI_CIRCLE_OUTLINE);
    EXPECT_STREQ(gobot::GetTypeIcon("Node3D"), ICON_MDI_AXIS);
    EXPECT_EQ(gobot::GetTypeEditorIcon("Node3D").kind, gobot::EditorIconKind::Axis3D);
    EXPECT_STREQ(gobot::GetTypeIcon("Robot3D"), ICON_MDI_ROBOT_INDUSTRIAL);
    EXPECT_STREQ(gobot::GetTypeIcon("Terrain3D"), ICON_MDI_TERRAIN);
    EXPECT_STREQ(gobot::GetTypeIcon("Joint3D"), ICON_MDI_SOURCE_COMMIT);
    EXPECT_EQ(gobot::GetTypeEditorIcon("Joint3D").kind, gobot::EditorIconKind::Joint3D);
    EXPECT_STREQ(gobot::GetTypeIcon("ContactSensor3D"), ICON_MDI_CROSSHAIRS);
    EXPECT_STREQ(gobot::GetTypeIcon("RayCastSensor3D"), ICON_MDI_RAY_START_END);
    EXPECT_STREQ(gobot::GetTypeIcon("TerrainHeightSensor3D"), ICON_MDI_TERRAIN);
    EXPECT_STREQ(gobot::GetTypeIcon("HeightScanner3D"), ICON_MDI_RAY_START_END);
    EXPECT_STREQ(gobot::GetTypeIcon("CylinderMesh"), ICON_MDI_PIPE);
}

TEST(TestTypeIcons, uses_resource_icons_from_paths) {
    EXPECT_STREQ(gobot::GetResourcePathIcon("res://go1_scene.jscn", false), ICON_MDI_FILE_TREE);
    EXPECT_STREQ(gobot::GetResourcePathIcon("res://scripts/go1.py", false), ICON_MDI_LANGUAGE_PYTHON);
    EXPECT_STREQ(gobot::GetResourcePathIcon("res://meshes/body.obj", false), ICON_MDI_VECTOR_TRIANGLE);
    EXPECT_STREQ(gobot::GetResourcePathIcon("res://textures/body.png", false), ICON_MDI_FILE_IMAGE);
    EXPECT_STREQ(gobot::GetResourcePathIcon("res://scripts", true), ICON_MDI_FOLDER);
    EXPECT_STREQ(gobot::GetResourcePathIcon("res://policies", true), ICON_MDI_FOLDER);
    EXPECT_STREQ(gobot::GetResourcePathIcon("res://policies/go1.onnx", false), ICON_MDI_CHIP);
    EXPECT_STREQ(gobot::GetResourcePathIcon("res://train/config.yaml", false), ICON_MDI_CODE_BRACES);
    EXPECT_STREQ(gobot::GetResourcePathIcon("res://project.gobot", false), ICON_MDI_APPLICATION);
}

TEST(TestTypeIcons, editor_icons_carry_category_colors) {
    EXPECT_NE(gobot::GetTypeEditorIcon("Robot3D").color, gobot::GetTypeEditorIcon("Sensor3D").color);
    EXPECT_NE(gobot::GetTypeEditorIcon("MeshInstance3D").color, gobot::GetTypeEditorIcon("CollisionShape3D").color);
    EXPECT_NE(gobot::GetResourcePathEditorIcon("res://scripts/go1.py", false).color,
              gobot::GetResourcePathEditorIcon("res://textures/body.png", false).color);
}
