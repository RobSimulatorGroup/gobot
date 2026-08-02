#include "gobot/editor/detail/example_project_discovery.hpp"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <vector>

namespace {

class ExampleDirectoryTest : public testing::Test {
protected:
    void SetUp() override {
        root_ = std::filesystem::temp_directory_path() /
                "gobot_example_project_discovery_test";
        std::error_code error;
        std::filesystem::remove_all(root_, error);
        ASSERT_TRUE(std::filesystem::create_directories(root_));
    }

    void TearDown() override {
        std::error_code error;
        std::filesystem::remove_all(root_, error);
    }

    std::filesystem::path MakeProject(const std::string& name) const {
        const std::filesystem::path project = root_ / name;
        std::filesystem::create_directories(project);
        return project;
    }

    std::filesystem::path root_;
};

TEST_F(ExampleDirectoryTest, discovers_configured_gaussian_and_native_scenes) {
    const std::filesystem::path gaussian = MakeProject("gaussian_splatting");
    std::filesystem::create_directories(gaussian / "scenes");
    std::ofstream(gaussian / "scenes" / "playroom.gsplat") << "{}";
    std::ofstream(gaussian / "project.gobot")
            << R"({"main_scene":"res://scenes/playroom.gsplat"})";

    const std::filesystem::path native = MakeProject("native");
    std::ofstream(native / "main.jscn") << "{}";

    const std::filesystem::path missing = MakeProject("missing");
    std::ofstream(missing / "project.gobot")
            << R"({"main_scene":"res://missing.gsplat"})";

    const std::filesystem::path escaped = MakeProject("escaped");
    std::ofstream(root_ / "outside.gsplat") << "{}";
    std::ofstream(escaped / "project.gobot")
            << R"({"main_scene":"res://../outside.gsplat"})";

    const std::filesystem::path unrelated = MakeProject("unrelated");
    std::ofstream(unrelated / "README.md") << "not a scene";

    std::vector<std::string> projects;
    gobot::editor_detail::AppendExampleProjectDirectories(root_, projects);

    std::set<std::string> names;
    for (const std::string& project : projects) {
        names.insert(std::filesystem::path(project).filename().string());
    }
    EXPECT_EQ(names, (std::set<std::string>{"gaussian_splatting", "native"}));
}

TEST_F(ExampleDirectoryTest, recognizes_scene_extensions_case_insensitively) {
    EXPECT_TRUE(gobot::editor_detail::IsSceneResourcePath("scene.jscn"));
    EXPECT_TRUE(gobot::editor_detail::IsSceneResourcePath("scene.GSPLAT"));
    EXPECT_FALSE(gobot::editor_detail::IsSceneResourcePath("scene.jres"));
    EXPECT_FALSE(gobot::editor_detail::IsSceneResourcePath("scene.usda"));
}

} // namespace
