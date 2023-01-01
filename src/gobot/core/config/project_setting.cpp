/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 22-12-14
*/

#include "gobot/core/config/project_setting.hpp"
#include <QDir>

namespace gobot {

ProjectSettings::ProjectSettings() {

}

ProjectSettings& ProjectSettings::GetInstance() {
    static ProjectSettings project_settings;
    return project_settings;
}

void ProjectSettings::SetProjectPath(const String& project_path) {
    project_path_ = project_path;
}

String ProjectSettings::LocalizePath(const String &path) const {
    if (project_path_.isEmpty() || ( QDir::isAbsolutePath(path) && !path.startsWith(project_path_))) {
        return QDir::cleanPath(path);
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
        return QDir::cleanPath(path);
    }

    String clean_path = QDir::cleanPath(path);
    QDir dir(clean_path);
    if (dir.exists()) {
        if (!clean_path.startsWith(project_path_)) {
            return path;
        }
        return clean_path.replace(project_path_, "res://");
    } else {
        int sep = clean_path.lastIndexOf("/");
        if (sep == -1) {
            return "res://" + clean_path;
        }

        String parent = clean_path.left(sep);
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


}
