#include <gtest/gtest.h>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <thread>
#include "gobot/python/python_script_runner.hpp"
#include "gobot/python/python_app_context.hpp"
#include "gobot/core/config/project_setting.hpp"
#include "gobot/main/engine_context.hpp"
#include "gobot/scene/node_3d.hpp"
#include "gobot/simulation/simulation_server.hpp"

TEST(PythonSimulationRuntime, WorkerRunsBetweenEditorCallsAndClosesBeforeInterpreterFinalization) {
    using Runner = gobot::python::PythonScriptRunner;
    setenv("GOBOT_PYTHON_EXECUTABLE", GOBOT_TEST_PYTHON_EXECUTABLE, 1);
    setenv("PYTHONPATH", GOBOT_TEST_PYTHON_PATH, 1);
    char directory_template[] = "/tmp/gobot-runtime-gil-XXXXXX";
    const char* directory = mkdtemp(directory_template);
    ASSERT_NE(directory, nullptr);
    const std::filesystem::path path(directory);
    const std::string source = std::string("runtime_test_directory = '") + directory + "'\n" + R"PY(
import sys, types
from gobot.sim import AsyncSimulationSession, SimulationRuntimeSpec
factory = types.ModuleType('gobot_runtime_gil_test')
sys.modules[factory.__name__] = factory
exec('''
import pathlib, time
class Provider:
    num_envs = 1
    fixed_time_step = .002
    def __init__(self, directory):
        self.path = pathlib.Path(directory)
        while not (self.path / 'go').exists():
            time.sleep(.001)
        (self.path / 'built').touch()
    def close(self):
        time.sleep(.02)
        (self.path / 'closed').touch()
def create(directory):
    return Provider(directory)
''', factory.__dict__)
runtime_session = AsyncSimulationSession(SimulationRuntimeSpec(
    'gobot_runtime_gil_test:create', {'directory': runtime_test_directory}))
)PY";
    const auto result = Runner::ExecuteString(source);
    std::ofstream(path / "go") << "go";
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (!std::filesystem::exists(path / "built") && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    const bool ran_between_calls = std::filesystem::exists(path / "built");
    Runner::Shutdown();
    EXPECT_TRUE(result.ok) << result.error;
    EXPECT_TRUE(ran_between_calls);
    EXPECT_TRUE(std::filesystem::exists(path / "closed"));
    std::filesystem::remove_all(path);
}

TEST(PythonSimulationRuntime, PlayBridgeCommitsCompletedClocksAndResetsThroughWorker) {
    using namespace gobot;
    using Runner = python::PythonScriptRunner;
    setenv("GOBOT_PYTHON_EXECUTABLE", GOBOT_TEST_PYTHON_EXECUTABLE, 1);
    setenv("PYTHONPATH", GOBOT_TEST_PYTHON_PATH, 1);
    ProjectSettings project(false);
    SimulationServer server;
    EngineContext context(&project, &server);
    python::RegisterExternalAppContext(&context);
    context.SetSceneRoot(Object::New<Node3D>(), true);
    const auto install = Runner::ExecuteString(R"PY(
import sys, types, threading
import gobot
factory = types.ModuleType('gobot_runtime_play_bridge')
sys.modules[factory.__name__] = factory
exec('''
import threading, time
events = []
fail_snapshot = False
class Provider:
    num_envs = 1
    fixed_time_step = .002
    def __init__(self):
        events.append(('build', threading.get_ident()))
    def step(self, actions=None, *, nsteps=1):
        time.sleep(.01)
        events.append(('step', threading.get_ident()))
    def reset(self, mask):
        events.append(('reset', threading.get_ident()))
    def close(self):
        events.append(('close', threading.get_ident()))
    def snapshot_arrays(self, fields, environments):
        if fail_snapshot:
            raise RuntimeError('injected snapshot failure')
        return {'positions': [[0., 0., 0.]]}
def create():
    return Provider()
''', factory.__dict__)
runtime = gobot.sim.AsyncSimulationSession(gobot.sim.SimulationRuntimeSpec(
    'gobot_runtime_play_bridge:create', {}), fields=('positions',), max_hz=1e9)
play = gobot.sim.ProviderPlaySession(gobot.app.context(), runtime, fixed_dt=.002).start()
)PY", &context);
    EXPECT_TRUE(install.ok) << install.error;
    EXPECT_TRUE(server.IsSessionAsynchronous());
    EXPECT_FALSE(server.IsSessionReady());
    const auto poll_until = [&](auto condition) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
        while (!condition() && std::chrono::steady_clock::now() < deadline) {
            server.AdvanceRealtime(0);
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        return condition();
    };
    if (install.ok) {
        EXPECT_TRUE(poll_until([&] { return !server.IsAsyncOperationPending(); }));
        EXPECT_EQ(server.GetFrameCount(), 0);
        EXPECT_TRUE(server.IsSessionReady());
        EXPECT_TRUE(server.RequestStep());
        EXPECT_EQ(server.GetFrameCount(), 0);
        EXPECT_TRUE(poll_until([&] { return server.GetFrameCount() == 1; })) << server.GetLastError();
        EXPECT_NEAR(server.GetSimulationTime(), .002, 1e-8);
        EXPECT_TRUE(server.Reset()) << server.GetLastError();
        EXPECT_FALSE(server.IsSessionReady());
        EXPECT_TRUE(poll_until([&] { return !server.IsAsyncOperationPending(); }));
        EXPECT_EQ(server.GetFrameCount(), 0);
        EXPECT_EQ(server.GetSimulationTime(), 0);
        const auto inject = Runner::ExecuteString("factory.fail_snapshot = True", &context);
        EXPECT_TRUE(inject.ok) << inject.error;
        EXPECT_TRUE(server.RequestStep());
        EXPECT_TRUE(poll_until([&] { return server.GetFrameCount() == 1; }));
        EXPECT_NEAR(server.GetSimulationTime(), .002, 1e-8);
        EXPECT_TRUE(server.IsPaused());
        EXPECT_EQ(server.GetLastStepCount(), 1);
        EXPECT_NE(server.GetLastError().find("injected snapshot failure"), std::string::npos);
        const auto close = Runner::ExecuteString(R"PY(
play.close()
runtime.shutdown()
assert {name for name, _ in factory.events} == {'build', 'step', 'reset', 'close'}
assert len({owner for _, owner in factory.events}) == 1
assert threading.get_ident() not in {owner for _, owner in factory.events}
)PY", &context);
        EXPECT_TRUE(close.ok) << close.error;
        EXPECT_FALSE(server.IsSessionReady());
        EXPECT_FALSE(server.IsSessionAsynchronous());
    }
    context.ClearWorld();
    python::UnregisterExternalAppContext(&context);
    Runner::Shutdown();
}
