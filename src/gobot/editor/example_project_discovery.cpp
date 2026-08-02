/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/editor/detail/example_project_discovery.hpp"

#include "gobot/core/types.hpp"
#include "gobot/log.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <ranges>
#include <string_view>

namespace gobot::editor_detail {
namespace {

std::string ToLower(std::string value) {
    std::ranges::transform(value, value.begin(), [](unsigned char character) {
        return static_cast<char>(std::tolower(character));
    });
    return value;
}

bool IsPathWithinDirectory(const std::filesystem::path& path,
                           const std::filesystem::path& directory) {
    const std::filesystem::path normalized_path = path.lexically_normal();
    const std::filesystem::path normalized_directory = directory.lexically_normal();
    auto path_iter = normalized_path.begin();
    auto directory_iter = normalized_directory.begin();
    for (; directory_iter != normalized_directory.end(); ++directory_iter, ++path_iter) {
        if (path_iter == normalized_path.end() || *path_iter != *directory_iter) {
            return false;
        }
    }
    return true;
}

bool HasConfiguredMainScene(const std::filesystem::path& project_dir) {
    std::ifstream input(project_dir / "project.gobot");
    if (!input.is_open()) {
        return false;
    }

    Json config;
    try {
        input >> config;
    } catch (const std::exception&) {
        return false;
    }
    if (!config.is_object() || !config.contains("main_scene") ||
        !config["main_scene"].is_string()) {
        return false;
    }

    const std::string main_scene = config["main_scene"].get<std::string>();
    constexpr std::string_view resource_prefix = "res://";
    if (!main_scene.starts_with(resource_prefix)) {
        return false;
    }

    const std::filesystem::path relative_path =
            std::filesystem::path(main_scene.substr(resource_prefix.size())).lexically_normal();
    if (relative_path.empty() || relative_path.is_absolute()) {
        return false;
    }

    std::error_code error;
    const std::filesystem::path canonical_project =
            std::filesystem::weakly_canonical(project_dir, error);
    if (error) {
        return false;
    }
    const std::filesystem::path scene_path =
            std::filesystem::weakly_canonical(project_dir / relative_path, error);
    if (error || !IsPathWithinDirectory(scene_path, canonical_project) ||
        !std::filesystem::is_regular_file(scene_path, error)) {
        return false;
    }
    return IsSceneResourcePath(scene_path);
}

bool HasTopLevelScene(const std::filesystem::path& project_dir) {
    std::error_code error;
    std::filesystem::directory_iterator iterator(project_dir, error);
    if (error) {
        return false;
    }
    for (const auto& entry : iterator) {
        error.clear();
        if (entry.is_regular_file(error) && !error && IsSceneResourcePath(entry.path())) {
            return true;
        }
    }
    return false;
}

} // namespace

bool IsSceneResourcePath(const std::filesystem::path& path) {
    const std::string extension = ToLower(path.extension().string());
    return extension == ".jscn" || extension == ".gsplat";
}

void AppendExampleProjectDirectories(const std::filesystem::path& examples_dir,
                                     std::vector<std::string>& projects) {
    std::error_code error;
    if (examples_dir.empty() || !std::filesystem::is_directory(examples_dir, error)) {
        return;
    }

    std::filesystem::directory_iterator iterator(examples_dir, error);
    if (error) {
        LOG_WARN("Skipping examples directory '{}': {}.", examples_dir.string(), error.message());
        return;
    }
    for (const auto& entry : iterator) {
        error.clear();
        if (!entry.is_directory(error) || error) {
            continue;
        }
        if (!HasConfiguredMainScene(entry.path()) && !HasTopLevelScene(entry.path())) {
            continue;
        }

        const std::string project_path =
                std::filesystem::weakly_canonical(entry.path(), error).string();
        if (error || project_path.empty()) {
            continue;
        }
        const std::string project_name = ToLower(entry.path().filename().string());
        const bool duplicate_name =
                std::ranges::any_of(projects, [&](const std::string& existing_project) {
                    return ToLower(std::filesystem::path(existing_project).filename().string()) ==
                           project_name;
                });
        if (!duplicate_name && std::ranges::find(projects, project_path) == projects.end()) {
            projects.push_back(project_path);
        }
    }
}

} // namespace gobot::editor_detail
