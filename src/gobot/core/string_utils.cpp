/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 22-12-14
*/

#include "gobot/core/string_utils.hpp"
#include "gobot/core/config/project_setting.hpp"

namespace gobot {

String GetFileExtension(const String& path) {
    return QFileInfo(path).completeSuffix();
}

bool IsNetworkSharePath(const String& path) {
    return path.startsWith("//") || path.startsWith("\\\\");
}


bool IsAbsolutePath(const String& path){
    if (path.length() > 1) {
        return (path[0] == '/' || path[0] == '\\' || path.indexOf(":/") != -1 || path.indexOf(":\\") != -1);
    } else if ((path.length()) == 1) {
        return (path[0] == '/' || path[0] == '\\');
    } else {
        return false;
    }
}

bool IsRelativePath(const String& path) {
    return !IsAbsolutePath(path);
}

String SimplifyPath(const String& path) {
    String s = path;
    String drive;

    // Check if we have a special path (like res://) or a protocol identifier.
    int p = s.indexOf("://");
    bool found = false;
    if (p > 0) {
        bool only_chars = true;
        for (int i = 0; i < p; i++) {
            if (!s[i].isLetter()) {
                only_chars = false;
                break;
            }
        }
        if (only_chars) {
            found = true;
            drive = s.mid(0, p + 3);
            s = s.mid(p + 3);
        }
    }
    if (!found) {
        if (IsNetworkSharePath(path)) {
            // Network path, beginning with // or \\.
            drive = s.mid(0, 2);
            s = s.mid(2);
        } else if (s.startsWith("/") || s.startsWith("\\")) {
            // Absolute path.
            drive = s.mid(0, 1);
            s = s.mid(1);
        } else {
            // Windows-style drive path, like C:/ or C:\.
            p = s.indexOf(":/");
            if (p == -1) {
                p = s.indexOf(":\\");
            }
            if (p != -1 && p < s.indexOf("/")) {
                drive = s.mid(0, p + 2);
                s = s.mid(p + 2);
            }
        }
    }

    s = s.replace("\\", "/");
    while (true) { // in case of using 2 or more slash
        String compare = s.replace("//", "/");
        if (s == compare) {
            break;
        } else {
            s = compare;
        }
    }
    auto dirs = s.split("/", Qt::SkipEmptyParts);

    for (int i = 0; i < dirs.size(); i++) {
        String d = dirs[i];
        if (d == ".") {
            dirs.removeAt(i);
            i--;
        } else if (d == "..") {
            if (i == 0) {
                dirs.removeAt(i);
                i--;
            } else {
                dirs.removeAt(i);
                dirs.removeAt(i - 1);
                i -= 2;
            }
        }
    }

    s = "";

    for (int i = 0; i < dirs.size(); i++) {
        if (i > 0) {
            s += "/";
        }
        s += dirs[i];
    }

    return drive + s;
}

String ValidateLocalPath(const String& path) {
    if (IsRelativePath(path)) {
        return "res://" + path;
    } else {
        return ProjectSettings::GetSingleton().LocalizePath(path);
    }
}

// If the string is a valid file path, returns the base directory name
// "/path/to/file.txt" =>  "/path/to"
String GetBaseDir(const String& path) {
    int end = 0;

    // URL scheme style base.
    int basepos = path.indexOf("://");;
    if (basepos != -1) {
        end = basepos + 3;
    }

    // Windows top level directory base.
    if (end == 0) {
        basepos = path.indexOf(":/");
        if (basepos == -1) {
            basepos = path.indexOf(":\\");
        }
        if (basepos != -1) {
            end = basepos + 2;
        }
    }

    // Windows UNC network share path.
    if (end == 0) {
        if (IsNetworkSharePath(path)) {
            basepos = path.indexOf("/", 2);
            if (basepos == -1) {
                basepos = path.indexOf("\\", 2);
            }
            int servpos = path.indexOf("/", basepos + 1);
            if (servpos == -1) {
                servpos = path.indexOf("\\", basepos + 1);
            }
            if (servpos != -1) {
                end = servpos + 1;
            }
        }
    }

    // Unix root directory base.
    if (end == 0) {
        if (path.startsWith("/")) {
            end = 1;
        }
    }

    String rs;
    String base;
    if (end != 0) {
        rs = path.mid(end);
        base = path.mid(0, end);
    } else {
        rs = path;
    }

    int sep = std::max(rs.lastIndexOf("/"), rs.lastIndexOf("\\"));
    if (sep == -1) {
        return base;
    }

    return base + rs.mid(0, sep);
}

String PathJoin(const String &base, const String &file) {
    if (base.isEmpty()) {
        return file;
    }
    if (base.endsWith('/')  || (file.size() > 0 && file.startsWith('/'))) {
        return base + file;
    }
    return base + "/" + file;
}

}