#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <signal.h>
#include <string>
#include <thread>

#include <gobot/editor/project_hook_runner.hpp>

namespace {

gobot::ProjectHookSnapshot WaitForCompletion(gobot::ProjectHookRunner& runner) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (std::chrono::steady_clock::now() < deadline) {
        gobot::ProjectHookSnapshot snapshot = runner.GetSnapshot();
        if (snapshot.state != gobot::ProjectHookState::Running) {
            return snapshot;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return runner.GetSnapshot();
}

std::filesystem::path MakeTestDirectory(const std::string& name) {
    const std::filesystem::path path =
            std::filesystem::temp_directory_path() / ("gobot_project_hook_runner_" + name);
    std::error_code error;
    std::filesystem::remove_all(path, error);
    std::filesystem::create_directories(path);
    return path;
}

} // namespace

TEST(TestProjectHookRunner, reports_progress_and_output) {
    const std::filesystem::path directory = MakeTestDirectory("success");
    const std::filesystem::path script = directory / "hook.py";
    std::ofstream(script) << R"PY(
import json
import os
import pathlib
import sys

assert os.environ["GOBOT_PROJECT_HOOK"] == "1"
assert pathlib.Path.cwd() == pathlib.Path(os.environ["GOBOT_PROJECT_DIR"])
print("hook output")
print("GOBOT_PROGRESS " + json.dumps({"current": 4, "total": 8, "message": "Downloading"}))
print("hook warning", file=sys.stderr)
)PY";

    gobot::ProjectHookRunner runner;
    ASSERT_TRUE(runner.Start(GOBOT_TEST_PYTHON_EXECUTABLE,
                             script.string(),
                             directory.string()));
    const gobot::ProjectHookSnapshot snapshot = WaitForCompletion(runner);
    ASSERT_EQ(snapshot.state, gobot::ProjectHookState::Succeeded);
    EXPECT_EQ(snapshot.current, 8u);
    EXPECT_EQ(snapshot.total, 8u);

    const std::vector<gobot::ProjectHookOutputLine> output = runner.DrainOutput();
    ASSERT_EQ(output.size(), 2u);
    EXPECT_EQ(output[0].text, "hook output");
    EXPECT_FALSE(output[0].is_stderr);
    EXPECT_EQ(output[1].text, "hook warning");
    EXPECT_TRUE(output[1].is_stderr);
}

TEST(TestProjectHookRunner, reports_failure_and_can_cancel) {
    const std::filesystem::path directory = MakeTestDirectory("failure");
    const std::filesystem::path failing_script = directory / "fail.py";
    std::ofstream(failing_script) << "import sys\nprint('download failed', file=sys.stderr)\nraise SystemExit(3)\n";

    gobot::ProjectHookRunner runner;
    EXPECT_FALSE(runner.Start("/definitely/missing/gobot-python",
                              failing_script.string(),
                              directory.string()));
    gobot::ProjectHookSnapshot snapshot = runner.GetSnapshot();
    EXPECT_EQ(snapshot.state, gobot::ProjectHookState::Failed);
    EXPECT_FALSE(snapshot.error.empty());

    ASSERT_TRUE(runner.Start(GOBOT_TEST_PYTHON_EXECUTABLE,
                             failing_script.string(),
                             directory.string()));
    snapshot = WaitForCompletion(runner);
    ASSERT_EQ(snapshot.state, gobot::ProjectHookState::Failed);
    EXPECT_NE(snapshot.error.find("code 3"), std::string::npos);

    const std::filesystem::path sleeping_script = directory / "sleep.py";
    std::ofstream(sleeping_script) << "import time\ntime.sleep(30)\n";
    ASSERT_TRUE(runner.Start(GOBOT_TEST_PYTHON_EXECUTABLE,
                             sleeping_script.string(),
                             directory.string()));
    runner.Cancel();
    snapshot = WaitForCompletion(runner);
    EXPECT_EQ(snapshot.state, gobot::ProjectHookState::Cancelled);
}

TEST(TestProjectHookRunner, cancel_terminates_descendants) {
    const std::filesystem::path directory = MakeTestDirectory("descendant_cancel");
    const std::filesystem::path pid_file = directory / "descendant.pid";
    const std::filesystem::path script = directory / "spawn_child.py";
    std::ofstream(script) << R"PY(
import pathlib
import subprocess
import sys
import time

pid_file = pathlib.Path("descendant.pid")
child = subprocess.Popen([
    sys.executable,
    "-c",
    "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)",
])
pid_file.write_text(str(child.pid))
time.sleep(30)
)PY";

    gobot::ProjectHookRunner runner;
    ASSERT_TRUE(runner.Start(GOBOT_TEST_PYTHON_EXECUTABLE,
                             script.string(),
                             directory.string()));
    const auto pid_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!std::filesystem::exists(pid_file) && std::chrono::steady_clock::now() < pid_deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    ASSERT_TRUE(std::filesystem::exists(pid_file));
    std::ifstream pid_stream(pid_file);
    int descendant_pid = -1;
    pid_stream >> descendant_pid;
    ASSERT_GT(descendant_pid, 0);

    runner.Cancel();
    const gobot::ProjectHookSnapshot snapshot = WaitForCompletion(runner);
    ASSERT_EQ(snapshot.state, gobot::ProjectHookState::Cancelled);

    const auto exit_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (kill(descendant_pid, 0) == 0 && std::chrono::steady_clock::now() < exit_deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    EXPECT_NE(kill(descendant_pid, 0), 0);
}

TEST(TestProjectHookRunner, bounds_unterminated_output) {
    const std::filesystem::path directory = MakeTestDirectory("bounded_output");
    const std::filesystem::path script = directory / "flood.py";
    std::ofstream(script) << "import sys\nsys.stdout.write('x' * (2 * 1024 * 1024))\nsys.stdout.flush()\n";

    gobot::ProjectHookRunner runner;
    ASSERT_TRUE(runner.Start(GOBOT_TEST_PYTHON_EXECUTABLE,
                             script.string(),
                             directory.string()));
    const gobot::ProjectHookSnapshot snapshot = WaitForCompletion(runner);
    ASSERT_EQ(snapshot.state, gobot::ProjectHookState::Succeeded);

    const std::vector<gobot::ProjectHookOutputLine> output = runner.DrainOutput();
    EXPECT_LE(output.size(), 256u);
    for (const gobot::ProjectHookOutputLine& line : output) {
        EXPECT_LT(line.text.size(), 17u * 1024u);
    }
}

TEST(TestProjectHookRunner, starting_while_running_returns_without_waiting) {
    const std::filesystem::path directory = MakeTestDirectory("nonblocking_start");
    const std::filesystem::path ready_file = directory / "ready";
    const std::filesystem::path script = directory / "ignore_term.py";
    std::ofstream(script) << R"PY(
import pathlib
import signal
import time

signal.signal(signal.SIGTERM, signal.SIG_IGN)
pathlib.Path("ready").write_text("ready")
time.sleep(30)
)PY";

    gobot::ProjectHookRunner runner;
    ASSERT_TRUE(runner.Start(GOBOT_TEST_PYTHON_EXECUTABLE,
                             script.string(),
                             directory.string()));
    const auto ready_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!std::filesystem::exists(ready_file) &&
           std::chrono::steady_clock::now() < ready_deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    ASSERT_TRUE(std::filesystem::exists(ready_file));

    const auto start = std::chrono::steady_clock::now();
    EXPECT_FALSE(runner.Start(GOBOT_TEST_PYTHON_EXECUTABLE,
                              script.string(),
                              directory.string()));
    EXPECT_LT(std::chrono::steady_clock::now() - start, std::chrono::milliseconds(250));

    runner.Cancel();
    EXPECT_EQ(WaitForCompletion(runner).state, gobot::ProjectHookState::Cancelled);
}
