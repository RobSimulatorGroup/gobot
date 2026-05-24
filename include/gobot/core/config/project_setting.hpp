/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * This file is created by Qiqi Wu, 22-12-14
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "gobot/core/math/matrix.hpp"
#include "gobot/core/object.hpp"
#include "gobot/core/types.hpp"

#include <map>
#include <optional>
#include <string>
#include <string_view>

namespace gobot {

struct EditorSceneViewState {
    Vector3 eye{};
    Vector3 at{};
    Vector3 up{0.0, 0.0, 1.0};
};

class GOBOT_EXPORT ProjectSettings : public Object {
    GOBCLASS(ProjectSettings, Object)
public:
    static ProjectSettings* s_singleton;

    ProjectSettings();

    ~ProjectSettings() override;

    static ProjectSettings* GetInstance();

    [[nodiscard]] std::string LocalizePath(std::string_view path) const;

    [[nodiscard]] std::string GlobalizePath(std::string_view path) const;

    bool SetProjectPath(const std::string& project_path);

    void ClearProjectPath();

    [[nodiscard]] FORCE_INLINE const std::string& GetProjectPath() const { return project_path_; }

    [[nodiscard]] FORCE_INLINE const std::string& GetMainScenePath() const { return main_scene_path_; }

    bool SetMainScenePath(const std::string& main_scene_path);

    bool SetEditorSceneViewState(const std::string& scene_path, const EditorSceneViewState& state);

    bool CacheEditorSceneViewState(const std::string& scene_path, const EditorSceneViewState& state);

    bool SaveEditorSceneViewState(const std::string& scene_path, const EditorSceneViewState& state);

    [[nodiscard]] std::optional<EditorSceneViewState> GetEditorSceneViewState(const std::string& scene_path) const;

    bool SaveProjectConfig() const;

private:
    void LoadProjectConfig();

    [[nodiscard]] std::string GetProjectConfigPath() const;

    std::string project_path_;
    std::string main_scene_path_;
    std::map<std::string, EditorSceneViewState> editor_scene_view_states_;
};

}
