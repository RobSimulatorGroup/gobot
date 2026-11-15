#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>

#include <gobot/core/config/project_setting.hpp>
#include <gobot/editor/imgui/python_panel.hpp>
#include <gobot/python/python_script_runner.hpp>

class TestPythonPanel : public testing::Test {
protected:
    void SetUp() override {
        project_settings = gobot::Object::New<gobot::ProjectSettings>();
        const testing::TestInfo* test_info = testing::UnitTest::GetInstance()->current_test_info();
        ASSERT_NE(test_info, nullptr);
        project_path = std::filesystem::temp_directory_path() /
                       (std::string("gobot_python_panel_test_") + test_info->name());
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

    std::string ReadText(const std::string& filename) const {
        std::ifstream stream(project_path / filename);
        return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
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

TEST_F(TestPythonPanel, embedded_runtime_imports_public_gobot_package) {
    setenv("PYTHONNOUSERSITE", "1", 1);
    setenv("PYTHONPATH", GOBOT_TEST_PYTHON_PATH, 1);
    setenv("GOBOT_PYTHON_EXECUTABLE", GOBOT_TEST_PYTHON_EXECUTABLE, 1);

    const std::filesystem::path result_path = project_path / "scripts" / "embedded_import.txt";
    setenv("GOBOT_EMBEDDED_IMPORT_RESULT", result_path.string().c_str(), 1);

    gobot::python::PythonExecutionResult result =
            gobot::python::PythonScriptRunner::ExecuteString(R"PY(
import os
import pathlib
import sys

import gobot
import gobot.rl
import gobot.terrain

cfg = gobot.terrain.go1_rough_terrain_cfg(seed=7, curriculum=False)
path = pathlib.Path(os.environ["GOBOT_EMBEDDED_IMPORT_RESULT"])
path.write_text(
    f"{gobot.__name__}:{gobot._core.__name__}:"
    f"{hasattr(gobot, 'rl')}:{hasattr(gobot, 'terrain')}:{cfg.num_rows}:{cfg.num_cols}:"
    f"{'gobot' in sys.builtin_module_names}:{'gobot._core' in sys.builtin_module_names}"
)
)PY",
                                                          nullptr,
                                                          "<embedded-import-test>");

    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_EQ(ReadText("scripts/embedded_import.txt"), "gobot:gobot._core:True:True:10:20:False:True");

    gobot::python::PythonScriptRunner::Shutdown();
}


TEST_F(TestPythonPanel, drains_async_workers_before_engine_contexts_are_destroyed) {
    setenv("PYTHONNOUSERSITE", "1", 1);
    setenv("PYTHONPATH", GOBOT_TEST_PYTHON_PATH, 1);
    setenv("GOBOT_PYTHON_EXECUTABLE", GOBOT_TEST_PYTHON_EXECUTABLE, 1);
    setenv("GOBOT_WORKER_CLOSE_RESULT", (project_path / "worker-closed.txt").string().c_str(), 1);
    const auto result = gobot::python::PythonScriptRunner::ExecuteString(R"PY(
import os, pathlib, sys, types, time
from gobot.sim import AsyncSimulationSession, SimulationRuntimeSpec
module = types.ModuleType("gobot_shutdown_fixture")
class Provider:
    num_envs = 1
    fixed_time_step = 0.02
    def close(self):
        time.sleep(0.01)
        pathlib.Path(os.environ["GOBOT_WORKER_CLOSE_RESULT"]).write_text("closed")
module.create = Provider
sys.modules[module.__name__] = module
session = AsyncSimulationSession(SimulationRuntimeSpec("gobot_shutdown_fixture:create", {}))
deadline = time.monotonic() + 10
while session.poll() is None:
    if time.monotonic() > deadline:
        raise RuntimeError("worker did not initialize")
    time.sleep(0.001)
session.close()
)PY");
    ASSERT_TRUE(result.ok) << result.error;
    gobot::python::PythonScriptRunner::ShutdownSimulationWorkers();
    EXPECT_EQ(ReadText("worker-closed.txt"), "closed");
    gobot::python::PythonScriptRunner::Shutdown();
}
