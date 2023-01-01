/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 22-12-14
*/

#include "gobot/core/string_utils.hpp"

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

}