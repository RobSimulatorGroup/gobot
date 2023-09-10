/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 22-11-27
*/

#include <gtest/gtest.h>

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

    ASSERT_TRUE(project_setting->LocalizePath("res://test") == "res://test");
    ASSERT_TRUE(project_setting->LocalizePath("res://test/..") == "res://");

    ASSERT_TRUE(project_setting->LocalizePath("test") == "res://test");
    ASSERT_TRUE(project_setting->LocalizePath("test/tt") == "res://test/tt");
#endif

}