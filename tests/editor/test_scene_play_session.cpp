#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <cstdlib>

#include <gobot/core/config/project_setting.hpp>
#include <gobot/core/io/python_script.hpp>
#include <gobot/editor/scene_play_session.hpp>
#include <gobot/main/engine_context.hpp>
#include <gobot/physics/physics_server.hpp>
#include <gobot/scene/node_3d.hpp>
#include <gobot/scene/node.hpp>
#include <gobot/scene/resources/packed_scene.hpp>
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
        root = gobot::Object::New<gobot::Node3D>();
        root->SetName("EditedRoot");
        tree->GetRoot()->AddChild(root);
        context->SetSceneRoot(root, false, "res://scene.jscn");
        gobot::SceneInitializer::Init();
    }

    void TearDown() override {
        session.Stop();
        context.reset();
        gobot::SceneInitializer::Destroy();
        if (tree != nullptr) {
            tree->Finalize();
            gobot::SceneTree::Delete(tree);
        }
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
    gobot::Node3D* root{nullptr};
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
    ASSERT_NE(session.GetRuntimeRoot(), nullptr);
    EXPECT_NE(session.GetRuntimeRoot(), root);
    EXPECT_TRUE(session.GetRuntimeRoot()->IsInsideTree());
    EXPECT_EQ(session.GetRuntimeRoot()->GetName(), "EditedRoot");
    EXPECT_EQ(ReadText("scripts/counts.txt"), "1,0,0,0");

    session.NotifyProcess(0.016);
    EXPECT_EQ(ReadText("scripts/counts.txt"), "1,1,0,0");

    session.NotifyPhysicsProcess(0.004);
    EXPECT_EQ(ReadText("scripts/counts.txt"), "1,1,1,0");

    session.Stop();
    EXPECT_FALSE(session.IsRunning());
    EXPECT_EQ(session.GetRuntimeRoot(), nullptr);
    EXPECT_EQ(ReadText("scripts/counts.txt"), "1,1,1,1");
}

TEST_F(TestScenePlaySession, stop_after_scene_tree_finalize_does_not_touch_deleted_runtime_scene) {
    auto script = MakeScript("scripts/lifecycle.py", R"PY(
import gobot

class Script(gobot.NodeScript):
    def _ready(self):
        pass

    def _exit_tree(self):
        pass
)PY");
    root->SetScript(script);

    ASSERT_TRUE(session.Start(root, context.get()));
    ASSERT_NE(session.GetRuntimeRoot(), nullptr);

    tree->Finalize();
    EXPECT_EQ(session.GetRuntimeRoot(), nullptr);

    session.Stop();
    EXPECT_FALSE(session.IsRunning());

    gobot::SceneTree::Delete(tree);
    tree = nullptr;
    root = nullptr;
}

TEST_F(TestScenePlaySession, node_scripts_mutate_runtime_clone_without_dirtying_edited_scene) {
    auto script = MakeScript("scripts/runtime_clone.py", R"PY(
import gobot

class Script(gobot.NodeScript):
    def _ready(self):
        self.node.name = "runtime_scripted"
        self.context.root.name = "runtime_root"
)PY");

    auto* child = gobot::Object::New<gobot::Node>();
    child->SetName("edited_scripted");
    child->SetScript(script);
    root->AddChild(child);
    const std::size_t command_version = context->GetSceneCommandVersion();

    ASSERT_TRUE(session.Start(root, context.get()));
    ASSERT_TRUE(session.IsRunning());
    ASSERT_NE(session.GetRuntimeRoot(), nullptr);
    EXPECT_NE(session.GetRuntimeRoot(), root);
    EXPECT_EQ(session.GetRuntimeRoot()->GetName(), "runtime_root");
    ASSERT_EQ(session.GetRuntimeRoot()->GetChildCount(), root->GetChildCount());
    EXPECT_EQ(session.GetRuntimeRoot()->GetChild(0)->GetName(), "runtime_scripted");

    EXPECT_EQ(root->GetName(), "EditedRoot");
    EXPECT_EQ(child->GetName(), "edited_scripted");
    EXPECT_FALSE(context->IsSceneDirty());
    EXPECT_EQ(context->GetSceneCommandVersion(), command_version);

    session.Stop();
    EXPECT_EQ(session.GetRuntimeRoot(), nullptr);
}

TEST_F(TestScenePlaySession, runtime_clone_expands_scene_instance_children_for_playback) {
    auto* prefab_root = gobot::Object::New<gobot::Node3D>();
    prefab_root->SetName("RobotPrefab");
    auto* prefab_link = gobot::Object::New<gobot::Node>();
    prefab_link->SetName("RobotLink");
    prefab_root->AddChild(prefab_link);

    gobot::Ref<gobot::PackedScene> prefab = gobot::MakeRef<gobot::PackedScene>();
    ASSERT_TRUE(prefab->Pack(prefab_root));
    gobot::Object::Delete(prefab_root);

    gobot::Node* robot_instance = prefab->Instantiate();
    ASSERT_NE(robot_instance, nullptr);
    robot_instance->SetName("RobotInstance");
    robot_instance->SetSceneInstance(prefab);
    ASSERT_EQ(robot_instance->GetChildCount(), 1);
    root->AddChild(robot_instance);

    ASSERT_TRUE(session.Start(root, context.get()));
    ASSERT_NE(session.GetRuntimeRoot(), nullptr);
    ASSERT_EQ(session.GetRuntimeRoot()->GetChildCount(), 1);

    gobot::Node* runtime_robot = session.GetRuntimeRoot()->GetChild(0);
    ASSERT_NE(runtime_robot, nullptr);
    EXPECT_EQ(runtime_robot->GetName(), "RobotInstance");
    ASSERT_EQ(runtime_robot->GetChildCount(), 1);
    EXPECT_EQ(runtime_robot->GetChild(0)->GetName(), "RobotLink");
    EXPECT_FALSE(runtime_robot->GetSceneInstance().IsValid());
    EXPECT_TRUE(robot_instance->GetSceneInstance().IsValid());
}

TEST_F(TestScenePlaySession, node_script_stdout_is_returned_from_notifications) {
    auto script = MakeScript("scripts/prints.py", R"PY(
import gobot

class Script(gobot.NodeScript):
    def _ready(self):
        print("_ready")
)PY");

    auto* child = gobot::Object::New<gobot::Node>();
    child->SetName("scripted");
    root->AddChild(child);

    gobot::python::PythonScriptRunner::SetSceneScriptContext(context.get());
    gobot::python::PythonScriptRunner::SetSceneScriptRoot(root, context->GetSceneEpoch());
    gobot::python::PythonExecutionResult attach =
            gobot::python::PythonScriptRunner::AttachSceneScript(child, script);
    ASSERT_TRUE(attach.ok) << attach.error;

    gobot::python::PythonExecutionResult ready =
            gobot::python::PythonScriptRunner::NotifySceneScript(child,
                                                                 gobot::NotificationType::Ready,
                                                                 0.0);
    EXPECT_TRUE(ready.ok) << ready.error;
    EXPECT_NE(ready.output.find("_ready"), std::string::npos);

    gobot::python::PythonScriptRunner::DetachSceneScript(child);
}
