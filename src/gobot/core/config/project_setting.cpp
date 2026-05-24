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

namespace {

constexpr const char* kEditorSceneViewsKey = "editor_scene_views";
constexpr const char* kEyeKey = "eye";
constexpr const char* kAtKey = "at";
constexpr const char* kUpKey = "up";

Json Vector3ToJson(const Vector3& vector) {
    return Json::array({vector.x(), vector.y(), vector.z()});
}

std::optional<Vector3> Vector3FromJson(const Json& json) {
    if (!json.is_array() || json.size() != 3) {
        return std::nullopt;
    }

    Vector3 vector{};
    for (int i = 0; i < 3; ++i) {
        if (!json[i].is_number()) {
            return std::nullopt;
        }
        vector[i] = static_cast<RealType>(json[i].get<double>());
    }
    return vector;
}

std::optional<EditorSceneViewState> SceneViewStateFromJson(const Json& json) {
    if (!json.is_object() || !json.contains(kEyeKey) ||
        !json.contains(kAtKey) || !json.contains(kUpKey)) {
        return std::nullopt;
    }

    std::optional<Vector3> eye = Vector3FromJson(json[kEyeKey]);
    std::optional<Vector3> at = Vector3FromJson(json[kAtKey]);
    std::optional<Vector3> up = Vector3FromJson(json[kUpKey]);
    if (!eye || !at || !up) {
        return std::nullopt;
    }

    return EditorSceneViewState{
            .eye = *eye,
            .at = *at,
            .up = *up,
    };
}

Json SceneViewStateToJson(const EditorSceneViewState& state) {
    return Json{
            {kEyeKey, Vector3ToJson(state.eye)},
            {kAtKey, Vector3ToJson(state.at)},
            {kUpKey, Vector3ToJson(state.up)},
    };
}

} // namespace

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
    editor_scene_view_states_.clear();
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

bool ProjectSettings::SetEditorSceneViewState(const std::string& scene_path,
                                              const EditorSceneViewState& state) {
    return SaveEditorSceneViewState(scene_path, state);
}

bool ProjectSettings::CacheEditorSceneViewState(const std::string& scene_path,
                                                const EditorSceneViewState& state) {
    if (project_path_.empty()) {
        return false;
    }

    const std::string local_path = LocalizePath(scene_path);
    if (local_path.empty() || !local_path.starts_with("res://")) {
        return false;
    }

    editor_scene_view_states_[local_path] = state;
    return true;
}

bool ProjectSettings::SaveEditorSceneViewState(const std::string& scene_path,
                                               const EditorSceneViewState& state) {
    if (!CacheEditorSceneViewState(scene_path, state)) {
        return false;
    }
    return SaveProjectConfig();
}

std::optional<EditorSceneViewState> ProjectSettings::GetEditorSceneViewState(const std::string& scene_path) const {
    const std::string local_path = LocalizePath(scene_path);
    auto it = editor_scene_view_states_.find(local_path);
    if (it == editor_scene_view_states_.end()) {
        return std::nullopt;
    }
    return it->second;
}

bool ProjectSettings::SaveProjectConfig() const {
    if (project_path_.empty()) {
        return false;
    }

    Json json = Json::object();
    json["main_scene"] = main_scene_path_;
    if (!editor_scene_view_states_.empty()) {
        Json scene_views = Json::object();
        for (const auto& [path, state] : editor_scene_view_states_) {
            scene_views[path] = SceneViewStateToJson(state);
        }
        json[kEditorSceneViewsKey] = scene_views;
    }

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
    editor_scene_view_states_.clear();

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

    if (json.contains(kEditorSceneViewsKey) && !json[kEditorSceneViewsKey].is_null()) {
        if (!json[kEditorSceneViewsKey].is_object()) {
            LOG_ERROR("Project config '{}' has non-object editor_scene_views.", config_path.string());
        } else {
            for (auto it = json[kEditorSceneViewsKey].begin(); it != json[kEditorSceneViewsKey].end(); ++it) {
                const std::string local_path = LocalizePath(it.key());
                if (!local_path.starts_with("res://")) {
                    LOG_ERROR("Ignoring editor scene view outside project in '{}': {}",
                              config_path.string(), local_path);
                    continue;
                }

                std::optional<EditorSceneViewState> state = SceneViewStateFromJson(it.value());
                if (!state) {
                    LOG_ERROR("Ignoring invalid editor scene view in '{}': {}",
                              config_path.string(), local_path);
                    continue;
                }
                editor_scene_view_states_[local_path] = *state;
            }
        }
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
