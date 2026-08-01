#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <cstdlib>
#include <stdexcept>
#include <vector>

#include <gobot/core/config/project_setting.hpp>
#include <gobot/core/io/python_script.hpp>
#include <gobot/editor/python_script_sync.hpp>
#include <gobot/editor/scene_play_session.hpp>
#include <gobot/main/engine_context.hpp>
#include <gobot/python/python_app_context.hpp>
#include <gobot/scene/joint_3d.hpp>
#include <gobot/scene/link_3d.hpp>
#include <gobot/scene/node_3d.hpp>
#include <gobot/scene/node.hpp>
#include <gobot/scene/robot_3d.hpp>
#include <gobot/scene/resources/packed_scene.hpp>
#include <gobot/scene/scene_initializer.hpp>
#include <gobot/scene/scene_tree.hpp>
#include <gobot/scene/window.hpp>
#include <gobot/simulation/simulation_server.hpp>

class TestScenePlaySession : public testing::Test {
protected:
    void SetUp() override {
        project_settings = gobot::Object::New<gobot::ProjectSettings>();
        simulation_server = gobot::Object::New<gobot::SimulationServer>();
        context = std::make_unique<gobot::EngineContext>(project_settings,
                                                         simulation_server);
        gobot::python::RegisterExternalAppContext(context.get());
        setenv("PYTHONNOUSERSITE", "1", 1);
        setenv("GOBOT_PYTHON_EXECUTABLE", GOBOT_TEST_PYTHON_EXECUTABLE, 1);
        setenv("PYTHONPATH", GOBOT_TEST_PYTHON_PATH, 1);
        setenv("HOME", "/tmp/gobot-test-home", 1);
        const auto* test_info = testing::UnitTest::GetInstance()->current_test_info();
        ASSERT_NE(test_info, nullptr);
        project_path = std::filesystem::temp_directory_path() /
                       (std::string{"gobot_scene_play_session_test_"} + test_info->name());
        std::filesystem::create_directories(project_path);
        std::ofstream(project_path / "gobot.py")
                << "raise RuntimeError('project gobot.py must not shadow the engine package')\n";
        setenv("GOBOT_SCENE_PLAY_SESSION_COUNTS",
               (project_path / "scripts" / "counts.txt").string().c_str(),
               1);
        setenv("GOBOT_SCENE_PLAY_SESSION_ROOT_LOOKUP",
               (project_path / "scripts" / "root_lookup.txt").string().c_str(),
               1);
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
        gobot::python::UnregisterExternalAppContext(context.get());
        context.reset();
        gobot::SceneInitializer::Destroy();
        if (tree != nullptr) {
            tree->Finalize();
            gobot::SceneTree::Delete(tree);
        }
        gobot::Object::Delete(simulation_server);
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

TEST_F(TestScenePlaySession, failing_node_script_receives_exit_before_detach) {
    auto script = MakeScript("scripts/failure_cleanup.py", R"PY(
import pathlib

import gobot

class Script(gobot.NodeScript):
    def _result(self):
        return pathlib.Path(self.context.project_path) / "scripts" / "failure_cleanup.txt"

    def _process(self, delta):
        del delta
        self._result().write_text("process")
        raise RuntimeError("process failed")

    def _exit_tree(self):
        result = self._result()
        result.write_text(result.read_text() + ",exit")
)PY");
    root->SetScript(script);

    ASSERT_TRUE(session.Start(root, context.get())) << session.GetLastError();
    ASSERT_EQ(session.GetActiveScriptCount(), 1);

    session.NotifyProcess(0.016);

    EXPECT_EQ(session.GetActiveScriptCount(), 0);
    EXPECT_EQ(ReadText("scripts/failure_cleanup.txt"), "process,exit");
    EXPECT_NE(session.GetLastError().find("process failed"), std::string::npos);

    session.Stop();
    EXPECT_EQ(ReadText("scripts/failure_cleanup.txt"), "process,exit");
}

TEST_F(TestScenePlaySession, external_provider_exception_is_reported_without_outer_python_gil) {
    auto script = MakeScript("scripts/failing_provider.py", R"PY(
import gobot

class Provider:
    def step(self, *, nsteps=1):
        del nsteps
        raise RuntimeError("external provider exploded")

    def close(self):
        pass

class Script(gobot.NodeScript):
    def _ready(self):
        self.session = gobot.sim.ProviderPlaySession(
            self.context,
            Provider(),
            fixed_dt=0.005,
            sync_scene=lambda: None,
        ).start()

    def _exit_tree(self):
        self.session.close()
)PY");
    root->SetScript(script);

    ASSERT_TRUE(session.Start(root, context.get())) << session.GetLastError();
    ASSERT_TRUE(simulation_server->HasExternalSession());

    EXPECT_FALSE(simulation_server->StepOnce([&](gobot::RealType fixed_delta) {
        session.NotifyPhysicsProcess(fixed_delta);
    }));
    EXPECT_NE(simulation_server->GetLastError().find("external provider exploded"),
              std::string::npos);

    session.Stop();
    EXPECT_FALSE(simulation_server->HasActiveSession());
}

TEST_F(TestScenePlaySession, external_provider_sync_resolves_runtime_link_handles) {
    auto* parent_link = gobot::Object::New<gobot::Link3D>();
    parent_link->SetName("parent_link");
    auto* child_link = gobot::Object::New<gobot::Link3D>();
    child_link->SetName("child_link");
    parent_link->AddChild(child_link);
    root->AddChild(parent_link);
    auto script = MakeScript("scripts/provider_sync.py", R"PY(
import gobot
import numpy as np

class Provider:
    def step(self, *, nsteps=1):
        del nsteps

    def close(self):
        pass

class Script(gobot.NodeScript):
    def _ready(self):
        self.parent_link = self.get_root().child(0)
        self.child_link = self.parent_link.child(0)
        self.session = gobot.sim.ProviderPlaySession(
            self.context,
            Provider(),
            fixed_dt=0.005,
            sync_scene=self._sync,
        ).start()

    def _sync(self):
        original_parent = self.parent_link.position.copy()
        invalid = np.asarray([
            [9.0, 9.0, 9.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ], dtype=np.float32)
        try:
            self.context._apply_link_pose_batch(
                (self.parent_link, self.child_link), invalid
            )
            raise AssertionError("invalid quaternion should reject the whole batch")
        except ValueError:
            pass
        assert np.allclose(self.parent_link.position, original_parent)

        non_finite = np.asarray(
            [[np.nan, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32
        )
        try:
            self.context._apply_link_pose_batch((self.parent_link,), non_finite)
            raise AssertionError("non-finite positions must be rejected")
        except ValueError:
            pass

        poses = np.asarray([
            [4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 1.0],
            [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float32)
        self.context._apply_link_pose_batch(
            (self.child_link, self.parent_link), poses
        )

    def _exit_tree(self):
        self.session.close()
)PY");
    root->SetScript(script);

    ASSERT_TRUE(session.Start(root, context.get())) << session.GetLastError();
    ASSERT_TRUE(simulation_server->HasExternalSession());
    ASSERT_TRUE(simulation_server->StepOnce([&](gobot::RealType fixed_delta) {
        session.NotifyPhysicsProcess(fixed_delta);
    })) << simulation_server->GetLastError();

    auto* runtime_parent = gobot::Object::PointerCastTo<gobot::Link3D>(
            session.GetRuntimeRoot()->GetChild(0));
    ASSERT_NE(runtime_parent, nullptr);
    auto* runtime_child = gobot::Object::PointerCastTo<gobot::Link3D>(
            runtime_parent->GetChild(0));
    ASSERT_NE(runtime_child, nullptr);
    EXPECT_TRUE(runtime_parent->GetGlobalPosition().isApprox(
            gobot::Vector3{1.0, 2.0, 3.0}, CMP_EPSILON));
    EXPECT_TRUE(runtime_child->GetGlobalPosition().isApprox(
            gobot::Vector3{4.0, 5.0, 6.0}, CMP_EPSILON));

    session.Stop();
    EXPECT_FALSE(simulation_server->HasActiveSession());
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

TEST_F(TestScenePlaySession, node_script_artifact_compile_uses_runtime_clone) {
#ifndef GOBOT_HAS_MUJOCO
    GTEST_SKIP() << "MuJoCo support is not enabled.";
#else
    auto* robot = gobot::Object::New<gobot::Robot3D>();
    robot->SetName("edited_robot");
    auto* base = gobot::Object::New<gobot::Link3D>();
    base->SetName("base");
    base->SetMass(1.0);
    base->SetInertiaDiagonal({0.01, 0.01, 0.01});
    robot->AddChild(base);
    root->AddChild(robot);

    auto script = MakeScript("scripts/runtime_artifact.py", R"PY(
import gobot

class Script(gobot.NodeScript):
    def _ready(self):
        robot = self.context.root.find("edited_robot")
        robot.name = "runtime_robot"
        artifact = self.context.compile_scene_artifact(
            gobot.PhysicsBackendType.MuJoCoCpu
        )
        result = self.context.project_path + "/scripts/runtime_artifact.txt"
        with open(result, "w", encoding="utf-8") as stream:
            stream.write(",".join(artifact["robot_names"]))
)PY");
    root->SetScript(script);

    ASSERT_TRUE(session.Start(root, context.get())) << session.GetLastError();
    EXPECT_EQ(ReadText("scripts/runtime_artifact.txt"), "runtime_robot");
    EXPECT_EQ(robot->GetName(), "edited_robot");

    gobot::PhysicsSceneArtifact edited_artifact;
    ASSERT_TRUE(context->CompileSceneArtifact(gobot::PhysicsBackendType::MuJoCoCpu,
                                              &edited_artifact))
            << context->GetLastError();
    EXPECT_EQ(edited_artifact.robot_names,
              std::vector<std::string>{"edited_robot"});
#endif
}

TEST_F(TestScenePlaySession, node_script_root_handle_uses_live_external_context) {
    auto script = MakeScript("scripts/root_lookup.py", R"PY(
import gobot
import os
import pathlib

RESULT = pathlib.Path(os.environ["GOBOT_SCENE_PLAY_SESSION_ROOT_LOOKUP"])

class Script(gobot.NodeScript):
    def _ready(self):
        root = self.get_root()
        robot = root.find("robot")
        RESULT.write_text(f"{root.name}:{robot.name}")
)PY");

    auto* robot = gobot::Object::New<gobot::Node>();
    robot->SetName("robot");
    root->AddChild(robot);
    root->SetScript(script);

    ASSERT_TRUE(session.Start(root, context.get())) << session.GetLastError();
    EXPECT_EQ(ReadText("scripts/root_lookup.txt"), "EditedRoot:robot");
}

TEST_F(TestScenePlaySession, node_script_debug_arrows_are_context_runtime_data) {
    auto script = MakeScript("scripts/debug_arrows.py", R"PY(
import gobot

class Script(gobot.NodeScript):
    def _ready(self):
        gobot.render.set_debug_arrows([
            gobot.render.DebugArrow(
                start=(1.0, 2.0, 3.0),
                vector=(0.5, 0.0, 0.0),
                color=(0.1, 0.2, 0.3, 1.0),
                scale=2.0,
                label="test_arrow",
            )
        ])

    def _exit_tree(self):
        gobot.render.clear_debug_arrows()
)PY");

    root->SetScript(script);

    ASSERT_TRUE(session.Start(root, context.get())) << session.GetLastError();
    ASSERT_EQ(context->GetDebugArrows().size(), 1);
    EXPECT_EQ(context->GetDebugArrows()[0].label, "test_arrow");
    EXPECT_TRUE(context->GetDebugArrows()[0].start.isApprox(gobot::Vector3(1.0, 2.0, 3.0)));
    EXPECT_TRUE(context->GetDebugArrows()[0].vector.isApprox(gobot::Vector3(0.5, 0.0, 0.0)));
    EXPECT_FLOAT_EQ(context->GetDebugArrows()[0].color.red(), 0.1f);
    EXPECT_DOUBLE_EQ(context->GetDebugArrows()[0].scale, 2.0);

    session.Stop();
    EXPECT_TRUE(context->GetDebugArrows().empty());
    EXPECT_FALSE(context->IsSceneDirty());
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

TEST_F(TestScenePlaySession, runtime_clone_preserves_motion_robot_joint_positions) {
    auto* robot = gobot::Object::New<gobot::Robot3D>();
    robot->SetName("cartpole");
    auto* rail = gobot::Object::New<gobot::Link3D>();
    rail->SetName("rail");
    rail->SetRole(gobot::LinkRole::VirtualRoot);
    auto* slider = gobot::Object::New<gobot::Joint3D>();
    slider->SetName("slider");
    slider->SetJointType(gobot::JointType::Prismatic);
    slider->SetParentLink("rail");
    slider->SetChildLink("cart");
    slider->SetLowerLimit(-2.4);
    slider->SetUpperLimit(2.4);
    slider->SetJointPosition(0.75);

    auto* cart = gobot::Object::New<gobot::Link3D>();
    cart->SetName("cart");

    root->AddChild(robot);
    robot->AddChild(rail);
    rail->AddChild(slider);
    slider->AddChild(cart);
    robot->SetMode(gobot::RobotMode::Motion);

    ASSERT_TRUE(session.Start(root, context.get()));
    ASSERT_NE(session.GetRuntimeRoot(), nullptr);
    auto* runtime_robot = gobot::Object::PointerCastTo<gobot::Robot3D>(
            session.GetRuntimeRoot()->GetChild(0));
    ASSERT_NE(runtime_robot, nullptr);
    ASSERT_EQ(runtime_robot->GetChildCount(), 1);
    auto* runtime_rail = runtime_robot->GetChild(0);
    ASSERT_NE(runtime_rail, nullptr);
    ASSERT_EQ(runtime_rail->GetChildCount(), 1);
    auto* runtime_slider = gobot::Object::PointerCastTo<gobot::Joint3D>(
            runtime_rail->GetChild(0));
    ASSERT_NE(runtime_slider, nullptr);
    EXPECT_DOUBLE_EQ(runtime_slider->GetJointPosition(), 0.75);
}

TEST_F(TestScenePlaySession, python_script_sync_refreshes_attached_script_source) {
    auto script = MakeScript("scripts/reload.py", R"PY(
import gobot

class Script(gobot.NodeScript):
    def _ready(self):
        self.node.name = "old"
)PY");
    root->SetScript(script);

    const std::string updated_source = R"PY(
import gobot

class Script(gobot.NodeScript):
    def _ready(self):
        self.node.name = "new"
)PY";
    {
        std::ofstream stream(project_path / "scripts" / "reload.py", std::ios::out | std::ios::trunc);
        stream << updated_source;
    }

    gobot::SyncPythonScriptResourceSource("res://scripts/reload.py", updated_source, root);
    EXPECT_EQ(script->GetSourceCode(), updated_source);
}

TEST_F(TestScenePlaySession, node_script_stdout_is_returned_from_notifications) {
    auto script = MakeScript("scripts/prints.py", R"PY(
import gobot

class Script(gobot.NodeScript):
    def _ready(self):
        print("_ready")

    def _process(self, delta):
        print("_process")
)PY");

    auto* child = gobot::Object::New<gobot::Node>();
    child->SetName("scripted");
    root->AddChild(child);

    gobot::python::PythonScriptRunner::SetSceneScriptContext(context.get());
    gobot::python::PythonScriptRunner::SetSceneScriptRoot(root, context->GetSceneEpoch());
    gobot::python::PythonExecutionResult attach =
            gobot::python::PythonScriptRunner::AttachSceneScript(child, script);
    ASSERT_TRUE(attach.ok) << attach.error;

    std::vector<std::string> streamed_output;
    gobot::python::PythonExecutionResult ready =
            gobot::python::PythonScriptRunner::NotifySceneScript(child,
                                                                 gobot::NotificationType::Ready,
                                                                 0.0,
                                                                 [&](const std::string& message,
                                                                     bool is_stderr) {
                                                                     EXPECT_FALSE(is_stderr);
                                                                     streamed_output.push_back(message);
                                                                 });
    EXPECT_TRUE(ready.ok) << ready.error;
    EXPECT_NE(ready.output.find("_ready"), std::string::npos);
    ASSERT_EQ(streamed_output.size(), 1);
    EXPECT_NE(streamed_output[0].find("_ready"), std::string::npos);

    gobot::python::PythonExecutionResult process =
            gobot::python::PythonScriptRunner::NotifySceneScript(
                    child,
                    gobot::NotificationType::Process,
                    0.0,
                    [](const std::string&, bool) { throw std::runtime_error("callback failed"); });
    EXPECT_TRUE(process.ok) << process.error;
    EXPECT_NE(process.output.find("_process"), std::string::npos);

    gobot::python::PythonScriptRunner::DetachSceneScript(child);
}

TEST_F(TestScenePlaySession, node_script_output_streams_through_session_callback) {
    auto script = MakeScript("scripts/streamed_output.py", R"PY(
import sys

import gobot

class Script(gobot.NodeScript):
    def _ready(self):
        print("initializing provider", flush=True)
        print("diagnostic warning", file=sys.stderr, flush=True)
)PY");
    root->SetScript(script);

    std::vector<std::pair<std::string, bool>> messages;
    session.SetScriptOutputCallback(
            [&](const std::string& message, bool is_stderr, const std::string&) {
                messages.emplace_back(message, is_stderr);
            });

    ASSERT_TRUE(session.Start(root, context.get())) << session.GetLastError();
    ASSERT_EQ(messages.size(), 2);
    EXPECT_NE(messages[0].first.find("initializing provider"), std::string::npos);
    EXPECT_FALSE(messages[0].second);
    EXPECT_NE(messages[1].first.find("diagnostic warning"), std::string::npos);
    EXPECT_TRUE(messages[1].second);
}
