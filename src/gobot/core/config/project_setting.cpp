/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 22-12-14
*/

#include "gobot/core/config/project_setting.hpp"
#include "gobot/core/string_utils.hpp"
#include "gobot/error_macros.hpp"
#include <QDir>

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

void ProjectSettings::SetProjectPath(const String& project_path) {
    project_path_ = QDir::cleanPath(project_path);
}

String ProjectSettings::LocalizePath(const String &path) const {
    if (project_path_.isEmpty() || ( IsAbsolutePath(path) && !path.startsWith(project_path_))) {
        return SimplifyPath(path);
    }

    // Check if we have a special path (like res://) or a protocol identifier.
    int p = path.indexOf("://");
    bool found = false;
    if (p > 0) {
        found = true;
        for (int i = 0; i < p; i++) {
            if (!path[i].isLetterOrNumber()) {
                found = false;
                break;
            }
        }
    }
    if (found) {
        return path.left(p + 3) + QDir::cleanPath(path.mid(p + 3));
    }

    auto simplify_path = SimplifyPath(String(path).replace("\\", "/"));
    if (QDir::setCurrent(simplify_path)) {
        String cwd = QDir::currentPath();
        cwd.append("/");
        auto temp_project_path = project_path_ + "/";
        if (!cwd.startsWith(temp_project_path)) {
            return path;
        }
        return cwd.replace(temp_project_path, "res://");
    } else {
        int sep = simplify_path.lastIndexOf("/");
        if (sep == -1) {
            return "res://" + simplify_path;
        }

        String parent = simplify_path.left(sep);
        String plocal = LocalizePath(parent);
        if (plocal.isEmpty()) {
            return "";
        }
        // Only strip the starting '/' from 'path' if its parent ('plocal') ends with '/'
        if (plocal[plocal.length() - 1] == '/') {
            sep += 1;
        }
        return plocal + path.mid(sep, path.size() - sep);
    }
}

String ProjectSettings::GlobalizePath(const String &path) const {
    String path_copy = path;
    if (path_copy.startsWith("res://")) {
        if (!project_path_.isEmpty()) {
            return path_copy.replace("res:/", project_path_);
        }
        return path_copy.replace("res://", "");
    }
    // TODO(wqq): add userdata path

    return path_copy;
}


}
