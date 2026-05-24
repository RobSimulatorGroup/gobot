/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * This file is created by Qiqi Wu, 22-11-27
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <fstream>

#include <gobot/core/config/project_setting.hpp>
#include <gobot/log.hpp>
#include <gobot/core/string_utils.hpp>

TEST(TestDir, test_dir) {
    ASSERT_TRUE(std::filesystem::weakly_canonical("/bin/") == std::filesystem::weakly_canonical("/bin"));
}

TEST(TestProjectSetting, test_localize_path) {

#ifdef _WIN32
    QDir project_path_dir = gobot::PathJoin(QDir::home().path(), "test_project");

    if (!project_path_dir.exists()){
        project_path_dir.mkpath(project_path_dir.path());
    }
    gobot::ProjectSettings project_settings;

    gobot::String project_path = project_path_dir.path();

    auto* project_setting = gobot::ProjectSettings::GetInstance();
    project_setting->SetProjectPath(project_path);

    ASSERT_TRUE(project_setting->LocalizePath(project_path) == "res://");
    ASSERT_TRUE(project_setting->LocalizePath(project_path) == "res://");
    ASSERT_TRUE(project_setting->LocalizePath( gobot::PathJoin(project_path, "test")) == "res://test");

    ASSERT_TRUE(project_setting->LocalizePath("res://test") == "res://test");
    ASSERT_TRUE(project_setting->LocalizePath("res://test/..") == "res://");

    ASSERT_TRUE(project_setting->LocalizePath("test") == "res://test");
    ASSERT_TRUE(project_setting->LocalizePath("test/tt") == "res://test/tt");

#else
    if (!std::filesystem::exists("/tmp/test_project")){
        std::filesystem::create_directories("/tmp/test_project");
    }
    gobot::ProjectSettings project_settings;

    auto* project_setting = gobot::ProjectSettings::GetInstance();
    project_setting->SetProjectPath("/tmp/test_project");

    ASSERT_TRUE(project_setting->LocalizePath("/tmp/test_project/") == "res://");
    ASSERT_TRUE(project_setting->LocalizePath("/tmp/test_project") == "res://");
    ASSERT_TRUE(project_setting->LocalizePath("/tmp/test_project/test") == "res://test");
    {
        std::ofstream file("/tmp/test_project/scene.jscn");
        file << "{}";
    }
    ASSERT_TRUE(project_setting->LocalizePath("/tmp/test_project/scene.jscn") == "res://scene.jscn");

    ASSERT_TRUE(project_setting->LocalizePath("res://test") == "res://test");
    ASSERT_TRUE(project_setting->LocalizePath("res://test/..") == "res://");

    ASSERT_TRUE(project_setting->LocalizePath("test") == "res://test");
    ASSERT_TRUE(project_setting->LocalizePath("test/tt") == "res://test/tt");
#endif

}

TEST(TestProjectSetting, saves_and_loads_main_scene) {
    const std::filesystem::path project_path =
            std::filesystem::temp_directory_path() / "gobot_project_settings_main_scene_test";
    std::error_code error;
    std::filesystem::remove_all(project_path, error);
    std::filesystem::create_directories(project_path);
    {
        std::ofstream scene(project_path / "main.jscn");
        scene << "{}";
    }

    {
        gobot::ProjectSettings project_settings;
        ASSERT_TRUE(project_settings.SetProjectPath(project_path.string()));
        ASSERT_TRUE(project_settings.SetMainScenePath("res://main.jscn"));
        EXPECT_EQ(project_settings.GetMainScenePath(), "res://main.jscn");
    }

    {
        gobot::ProjectSettings project_settings;
        ASSERT_TRUE(project_settings.SetProjectPath(project_path.string()));
        EXPECT_EQ(project_settings.GetMainScenePath(), "res://main.jscn");
    }
}

TEST(TestProjectSetting, saves_and_loads_editor_scene_view_state) {
    const std::filesystem::path project_path =
            std::filesystem::temp_directory_path() / "gobot_project_settings_scene_view_test";
    std::error_code error;
    std::filesystem::remove_all(project_path, error);
    std::filesystem::create_directories(project_path);
    {
        std::ofstream scene(project_path / "main.jscn");
        scene << "{}";
    }

    const gobot::EditorSceneViewState state{
            .eye = gobot::Vector3(1.0, 2.0, 3.0),
            .at = gobot::Vector3(4.0, 5.0, 6.0),
            .up = gobot::Vector3(0.0, 0.0, 1.0),
    };

    {
        gobot::ProjectSettings project_settings;
        ASSERT_TRUE(project_settings.SetProjectPath(project_path.string()));
        ASSERT_TRUE(project_settings.SetMainScenePath("res://main.jscn"));
        ASSERT_TRUE(project_settings.SetEditorSceneViewState("res://main.jscn", state));
    }

    {
        gobot::ProjectSettings project_settings;
        ASSERT_TRUE(project_settings.SetProjectPath(project_path.string()));
        EXPECT_EQ(project_settings.GetMainScenePath(), "res://main.jscn");

        std::optional<gobot::EditorSceneViewState> loaded =
                project_settings.GetEditorSceneViewState("res://main.jscn");
        ASSERT_TRUE(loaded.has_value());
        EXPECT_TRUE(loaded->eye.isApprox(state.eye, CMP_EPSILON));
        EXPECT_TRUE(loaded->at.isApprox(state.at, CMP_EPSILON));
        EXPECT_TRUE(loaded->up.isApprox(state.up, CMP_EPSILON));
    }
}
