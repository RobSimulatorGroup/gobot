/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * This file is created by Qiqi Wu, 22-12-14
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/core/config/project_setting.hpp"
#include "gobot/core/string_utils.hpp"
#include "gobot/error_macros.hpp"

#include <fstream>

namespace gobot {

ProjectSettings *ProjectSettings::s_singleton = nullptr;

ProjectSettings::ProjectSettings() {
    s_singleton = this;
}

ProjectSettings::~ProjectSettings() {
    s_singleton = nullptr;
}

ProjectSettings* ProjectSettings::GetInstance() {
    ERR_FAIL_COND_V_MSG(s_singleton == nullptr, nullptr, "Must call this after initialize ProjectSettings");
    return s_singleton;
}

bool ProjectSettings::SetProjectPath(const std::string& project_path) {
    project_path_ = std::filesystem::weakly_canonical(project_path);
    if (!std::filesystem::exists(project_path_)) {
        LOG_ERROR("Invalid project path specified: {}", project_path);
        project_path_.clear();
        main_scene_path_.clear();
        return false;
    }
    LoadProjectConfig();
    return true;
}

void ProjectSettings::ClearProjectPath() {
    project_path_.clear();
    main_scene_path_.clear();
}

std::string ProjectSettings::LocalizePath(std::string_view path) const {
    // Check if we have a special path (like res://) or a protocol identifier.
    auto p = path.find("://");
    bool found = false;
    if (p > 0) {
        found = true;
        for (int i = 0; i < p; i++) {
            if (!isalnum(path[i])) {
                found = false;
                break;
            }
        }
    }
    if (found) {
        return SimplifyPath(path);
    }

    auto simplify_path = SimplifyPath(ReplaceAll(path.data(), "\\", "/"));
    if (project_path_.empty()) {
        return simplify_path;
    }

    std::string project_path = SimplifyPath(ReplaceAll(project_path_, "\\", "/"));
    if (IsAbsolutePath(simplify_path)) {
        if (simplify_path == project_path) {
            return "res://";
        }

        const std::string project_prefix = project_path + "/";
        if (!simplify_path.starts_with(project_prefix)) {
            return simplify_path;
        }

        return "res://" + simplify_path.substr(project_prefix.size());
    }

    if (simplify_path.empty()) {
        return "res://";
    }

    return "res://" + simplify_path;
}

std::string ProjectSettings::GlobalizePath(std::string_view path) const {
    std::string path_copy = std::string(path);
    if (path_copy.starts_with("res://")) {
        if (!project_path_.empty()) {
            return ReplaceAll(path_copy, "res:/", project_path_);
        }
        return ReplaceAll(path_copy, "res://", "");
    }
    // TODO(wqq): add userdata path

    return path_copy;
}

bool ProjectSettings::SetMainScenePath(const std::string& main_scene_path) {
    if (project_path_.empty()) {
        LOG_ERROR("Cannot set main scene without an open project.");
        return false;
    }

    const std::string local_path = LocalizePath(main_scene_path);
    if (!local_path.starts_with("res://")) {
        LOG_ERROR("Main scene must be inside the current project: {}", main_scene_path);
        return false;
    }

    const std::string extension = std::filesystem::path(local_path).extension().string();
    if (extension != ".jscn") {
        LOG_ERROR("Main scene must be a .jscn file: {}", local_path);
        return false;
    }

    if (!std::filesystem::exists(GlobalizePath(local_path))) {
        LOG_ERROR("Main scene does not exist: {}", local_path);
        return false;
    }

    const std::string previous_main_scene_path = main_scene_path_;
    main_scene_path_ = local_path;
    if (!SaveProjectConfig()) {
        main_scene_path_ = previous_main_scene_path;
        return false;
    }
    return true;
}

bool ProjectSettings::SaveProjectConfig() const {
    if (project_path_.empty()) {
        return false;
    }

    Json json = Json::object();
    json["main_scene"] = main_scene_path_;

    const std::filesystem::path config_path = GetProjectConfigPath();
    std::ofstream output(config_path, std::ios::out | std::ios::trunc);
    if (!output.is_open()) {
        LOG_ERROR("Failed to write project config '{}'.", config_path.string());
        return false;
    }
    output << json.dump(4) << '\n';
    return true;
}

void ProjectSettings::LoadProjectConfig() {
    main_scene_path_.clear();

    if (project_path_.empty()) {
        return;
    }

    const std::filesystem::path config_path = GetProjectConfigPath();
    if (!std::filesystem::exists(config_path)) {
        return;
    }

    std::ifstream input(config_path);
    if (!input.is_open()) {
        LOG_ERROR("Failed to read project config '{}'.", config_path.string());
        return;
    }

    Json json;
    try {
        input >> json;
    } catch (const std::exception& error) {
        LOG_ERROR("Failed to parse project config '{}': {}", config_path.string(), error.what());
        return;
    }

    if (!json.is_object()) {
        LOG_ERROR("Project config '{}' must be a JSON object.", config_path.string());
        return;
    }

    if (!json.contains("main_scene") || json["main_scene"].is_null()) {
        return;
    }
    if (!json["main_scene"].is_string()) {
        LOG_ERROR("Project config '{}' has non-string main_scene.", config_path.string());
        return;
    }

    const std::string local_path = LocalizePath(json["main_scene"].get<std::string>());
    if (!local_path.starts_with("res://")) {
        LOG_ERROR("Ignoring main scene outside project in '{}': {}", config_path.string(), local_path);
        return;
    }
    main_scene_path_ = local_path;
}

std::string ProjectSettings::GetProjectConfigPath() const {
    if (project_path_.empty()) {
        return {};
    }
    return (std::filesystem::path(project_path_) / "project.gobot").string();
}

}
