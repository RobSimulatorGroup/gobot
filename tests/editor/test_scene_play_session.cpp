#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <cstdlib>

#include <gobot/core/config/project_setting.hpp>
#include <gobot/core/io/python_script.hpp>
#include <gobot/editor/scene_play_session.hpp>
#include <gobot/main/engine_context.hpp>
#include <gobot/physics/physics_server.hpp>
#include <gobot/scene/node.hpp>
#include <gobot/scene/scene_initializer.hpp>
#include <gobot/scene/scene_tree.hpp>
#include <gobot/scene/window.hpp>
#include <gobot/simulation/simulation_server.hpp>

class TestScenePlaySession : public testing::Test {
protected:
    void SetUp() override {
        project_settings = gobot::Object::New<gobot::ProjectSettings>();
        physics_server = gobot::Object::New<gobot::PhysicsServer>();
        simulation_server = gobot::Object::New<gobot::SimulationServer>();
        context = std::make_unique<gobot::EngineContext>(project_settings,
                                                         physics_server,
                                                         simulation_server);
        setenv("PYTHONNOUSERSITE", "1", 1);
        setenv("PYTHONPATH", GOBOT_TEST_BUILD_PYTHON_DIR, 1);
        setenv("HOME", "/tmp/gobot-test-home", 1);
        project_path = std::filesystem::temp_directory_path() / "gobot_scene_play_session_test";
        std::filesystem::create_directories(project_path);
        ASSERT_TRUE(project_settings->SetProjectPath(project_path.string()));
        tree = gobot::SceneTree::New<gobot::SceneTree>(false);
        tree->Initialize();
        root = tree->GetRoot();
        context->SetSceneRoot(root, false, "res://scene.jscn");
        gobot::SceneInitializer::Init();
    }

    void TearDown() override {
        session.Stop();
        context.reset();
        gobot::SceneInitializer::Destroy();
        tree->Finalize();
        gobot::SceneTree::Delete(tree);
        gobot::Object::Delete(simulation_server);
        gobot::Object::Delete(physics_server);
        gobot::Object::Delete(project_settings);
        std::filesystem::remove_all(project_path);
    }

    gobot::Ref<gobot::PythonScript> MakeScript(const std::string& path, const std::string& source) {
        const std::filesystem::path global_path = project_path / path;
        std::filesystem::create_directories(global_path.parent_path());
        std::ofstream stream(global_path, std::ios::out | std::ios::trunc);
        stream << source;
        stream.close();

        auto script = gobot::MakeRef<gobot::PythonScript>();
        script->SetPath("res://" + path, false);
        script->SetSourceCode(source);
        return script;
    }

    std::string ReadText(const std::string& filename) const {
        std::ifstream stream(project_path / filename);
        return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
    }

    gobot::ProjectSettings* project_settings{nullptr};
    gobot::PhysicsServer* physics_server{nullptr};
    gobot::SimulationServer* simulation_server{nullptr};
    std::unique_ptr<gobot::EngineContext> context;
    gobot::SceneTree* tree{nullptr};
    gobot::Node* root{nullptr};
    gobot::ScenePlaySession session;
    std::filesystem::path project_path;
};

TEST_F(TestScenePlaySession, node_scripts_run_only_inside_play_session) {
    auto script = MakeScript("scripts/session_counter.py", R"PY(
import gobot
import os
import pathlib

COUNTS = pathlib.Path(os.environ["GOBOT_SCENE_PLAY_SESSION_COUNTS"])

def _read():
    if not COUNTS.exists():
        return [0, 0, 0, 0]
    return [int(item) for item in COUNTS.read_text().strip().split(",")]

def _write(values):
    COUNTS.write_text(",".join(str(item) for item in values))

class Script(gobot.NodeScript):
    def _ready(self):
        values = _read()
        values[0] += 1
        _write(values)

    def _process(self, delta):
        values = _read()
        values[1] += 1
        _write(values)

    def _physics_process(self, delta):
        values = _read()
        values[2] += 1
        _write(values)

    def _exit_tree(self):
        values = _read()
        values[3] += 1
        _write(values)
)PY");

    auto* child = gobot::Object::New<gobot::Node>();
    child->SetName("scripted");
    child->SetScript(script);
    root->AddChild(child);
    setenv("GOBOT_SCENE_PLAY_SESSION_COUNTS",
           (project_path / "scripts" / "counts.txt").string().c_str(),
           1);

    tree->Process(0.016);
    tree->PhysicsProcess(0.004);
    EXPECT_FALSE(std::filesystem::exists(project_path / "scripts/counts.txt"));

    ASSERT_TRUE(session.Start(root, context.get()));
    EXPECT_TRUE(session.IsRunning());
    EXPECT_EQ(session.GetActiveScriptCount(), 1);
    EXPECT_EQ(ReadText("scripts/counts.txt"), "1,0,0,0");

    session.NotifyProcess(0.016);
    EXPECT_EQ(ReadText("scripts/counts.txt"), "1,1,0,0");

    session.NotifyPhysicsProcess(0.004);
    EXPECT_EQ(ReadText("scripts/counts.txt"), "1,1,1,0");

    session.Stop();
    EXPECT_FALSE(session.IsRunning());
    EXPECT_EQ(ReadText("scripts/counts.txt"), "1,1,1,1");
}
