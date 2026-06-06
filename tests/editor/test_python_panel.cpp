#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>

#include <gobot/core/config/project_setting.hpp>
#include <gobot/editor/imgui/python_panel.hpp>

class TestPythonPanel : public testing::Test {
protected:
    void SetUp() override {
        project_settings = gobot::Object::New<gobot::ProjectSettings>();
        project_path = std::filesystem::temp_directory_path() / "gobot_python_panel_test";
        std::filesystem::remove_all(project_path);
        std::filesystem::create_directories(project_path / "scripts");
        ASSERT_TRUE(project_settings->SetProjectPath(project_path.string()));
    }

    void TearDown() override {
        gobot::Object::Delete(project_settings);
        std::filesystem::remove_all(project_path);
    }

    void WriteScript(const std::string& text) const {
        std::ofstream stream(project_path / "scripts" / "panel.py", std::ios::out | std::ios::trunc);
        stream << text;
    }

    void TouchScript() const {
        const std::filesystem::path path = project_path / "scripts" / "panel.py";
        std::filesystem::last_write_time(path,
                                         std::filesystem::file_time_type::clock::now() + std::chrono::seconds(5));
    }

    gobot::ProjectSettings* project_settings{nullptr};
    std::filesystem::path project_path;
};

TEST_F(TestPythonPanel, reloads_clean_script_after_external_change) {
    WriteScript("value = 1\n");

    gobot::PythonPanel panel;
    ASSERT_TRUE(panel.LoadScript("res://scripts/panel.py"));
    ASSERT_FALSE(panel.IsScriptDirty());
    ASSERT_FALSE(panel.IsScriptExternallyModified());

    WriteScript("value = 2\n");
    TouchScript();
    panel.CheckExternalScriptModificationForTesting();

    EXPECT_FALSE(panel.IsScriptDirty());
    EXPECT_FALSE(panel.IsScriptExternallyModified());
    EXPECT_NE(panel.GetEditorTextForTesting().find("value = 2"), std::string::npos);
}

TEST_F(TestPythonPanel, flags_external_change_when_panel_has_unsaved_edits) {
    WriteScript("value = 1\n");

    gobot::PythonPanel panel;
    ASSERT_TRUE(panel.LoadScript("res://scripts/panel.py"));

    panel.SetEditorTextForTesting("value = 99\n");
    ASSERT_TRUE(panel.IsScriptDirty());

    WriteScript("value = 2\n");
    TouchScript();
    panel.CheckExternalScriptModificationForTesting();

    EXPECT_TRUE(panel.IsScriptDirty());
    EXPECT_TRUE(panel.IsScriptExternallyModified());
    EXPECT_NE(panel.GetEditorTextForTesting().find("value = 99"), std::string::npos);
    EXPECT_EQ(panel.GetEditorTextForTesting().find("value = 2"), std::string::npos);
}

TEST_F(TestPythonPanel, failed_load_keeps_current_script_state) {
    WriteScript("value = 1\n");

    gobot::PythonPanel panel;
    ASSERT_TRUE(panel.LoadScript("res://scripts/panel.py"));

    EXPECT_FALSE(panel.LoadScript("res://scripts/missing.py"));

    EXPECT_EQ(panel.GetScriptLocalPath(), "res://scripts/panel.py");
    EXPECT_FALSE(panel.IsScriptDirty());
    EXPECT_FALSE(panel.IsScriptExternallyModified());
    EXPECT_NE(panel.GetEditorTextForTesting().find("value = 1"), std::string::npos);
}
