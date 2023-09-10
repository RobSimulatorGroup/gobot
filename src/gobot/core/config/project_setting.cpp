/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 22-12-14
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

std::string ProjectSettings::LocalizePath(std::string_view path) const {
    if (project_path_.empty() || ( IsAbsolutePath(path) && !path.starts_with(project_path_))) {
        return SimplifyPath(path);
    }

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
        return std::string(path.substr(0, p + 3)) + std::filesystem::weakly_canonical(path.substr(p + 3)).string();
    }

    auto simplify_path = SimplifyPath(ReplaceAll(path.data(), "\\", "/"));
    if (std::filesystem::exists(simplify_path)) {
        std::filesystem::current_path(simplify_path);
        auto cwd = std::filesystem::current_path().string();
        cwd.append("/");
        auto temp_project_path = project_path_ + "/";
        if (!cwd.starts_with(temp_project_path)) {
            return std::string(path);
        }
        return ReplaceAll(cwd, temp_project_path, "res://");
    } else {
        auto sep = simplify_path.find_last_of("/");
        if (sep == -1) {
            return "res://" + simplify_path;
        }

        std::string parent = simplify_path.substr(0, sep);
        std::string plocal = LocalizePath(parent);
        if (plocal.empty()) {
            return "";
        }
        // Only strip the starting '/' from 'path' if its parent ('plocal') ends with '/'
        if (plocal[plocal.length() - 1] == '/') {
            sep += 1;
        }
        return plocal + std::string(path.substr(sep, path.size() - sep));
    }
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
