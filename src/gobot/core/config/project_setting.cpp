/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * This file is created by Qiqi Wu, 22-12-14
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/core/config/project_setting.hpp"
#include "gobot/core/string_utils.hpp"
#include "gobot/error_macros.hpp"

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
        return false;
    }
    // TODO(wqq): check if it is a project path(has a project file)
    return true;
}

void ProjectSettings::ClearProjectPath() {
    project_path_.clear();
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


}
