/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 22-12-14
*/

#include "gobot/core/string_utils.hpp"

#include <filesystem>
#include "gobot/log.hpp"
#include "gobot/core/config/project_setting.hpp"

namespace gobot {

std::vector<std::string> Split(std::string s, std::string delimiter, StringSplitBehavior split_behavior) {
    size_t pos_start = 0, pos_end, delim_len = delimiter.length();
    std::string token;
    std::vector<std::string> res;

    while ((pos_end = s.find(delimiter, pos_start)) != std::string::npos) {
        token = s.substr(pos_start, pos_end - pos_start);
        pos_start = pos_end + delim_len;
        if (split_behavior == StringSplitBehavior::SkipEmptyParts && token.empty())
            continue;
        res.push_back(token);
    }

    auto last = s.substr(pos_start);
    if (split_behavior == StringSplitBehavior::SkipEmptyParts && last.empty()) {
        return res;
    }
    res.push_back(last);
    return res;
}

std::string ToLower(std::string_view str) {
    std::string res;
    std::transform(str.begin(), str.end(), res.begin(), ::toupper);
    return res;
}

std::string ToUpper(std::string_view str) {
    std::string res;
    std::transform(str.begin(), str.end(), res.begin(), ::tolower);
    return res;
}

std::string ReplaceAll(std::string str, const std::string& from, const std::string& to) {
    size_t start_pos = 0;
    while((start_pos = str.find(from, start_pos)) != std::string::npos) {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length(); // Handles case where 'to' is a substring of 'from'
    }
    return str;
}

bool Replace(std::string& str, const std::string& from, const std::string& to) {
    size_t start_pos = str.find(from);
    if(start_pos == std::string::npos)
        return false;
    str.replace(start_pos, from.length(), to);
    return true;
}

std::string GetFileExtension(std::string_view path) {
    return std::filesystem::path(path).extension().string().substr(1);
}

bool IsNetworkSharePath(std::string_view path) {
    return path.starts_with("//") || path.starts_with("\\\\");
}


bool IsAbsolutePath(std::string_view path){
    if (path.length() > 1) {
        return (path[0] == '/' || path[0] == '\\' ||
                path.find(":/") != std::string_view::npos ||
                path.find(":\\") != std::string_view::npos);
    } else if ((path.length()) == 1) {
        return (path[0] == '/' || path[0] == '\\');
    } else {
        return false;
    }
}

bool IsRelativePath(std::string_view path) {
    return !IsAbsolutePath(path);
}

std::string SimplifyPath(std::string_view path) {
    std::string s = std::string(path);
    std::string drive;

    // Check if we have a special path (like res://) or a protocol identifier.
    auto p = s.find("://");
    bool found = false;
    if (p > 0) {
        bool only_chars = true;
        for (int i = 0; i < p; i++) {
            if (!isalpha(s[i])) {
                only_chars = false;
                break;
            }
        }
        if (only_chars) {
            found = true;
            drive = s.substr(0, p + 3);
            s = s.substr(p + 3);
        }
    }
    if (!found) {
        if (IsNetworkSharePath(path)) {
            // Network path, beginning with // or \\.
            drive = s.substr(0, 2);
            s = s.substr(2);
        } else if (s.starts_with("/") || s.starts_with("\\")) {
            // Absolute path.
            drive = s.substr(0, 1);
            s = s.substr(1);
        } else {
            // Windows-style drive path, like C:/ or C:\.
            p = s.find(":/");
            if (p == -1) {
                p = s.find(":\\");
            }
            if (p != -1 && p < s.find('/')) {
                drive = s.substr(0, p + 2);
                s = s.substr(p + 2);
            }
        }
    }

    Replace(s, "\\", "/");
    while (true) { // in case of using 2 or more slash
        std::string compare = ReplaceAll(s, "//", "/");
        if (s == compare) {
            break;
        } else {
            s = compare;
        }
    }
    auto dirs = Split(s, "/", StringSplitBehavior::SkipEmptyParts);

    for (int i = 0; i < dirs.size(); i++) {
        std::string d = dirs[i];
        if (d == ".") {
            dirs.erase(dirs.begin() + i);
            i--;
        } else if (d == "..") {
            if (i == 0) {
                dirs.erase(dirs.begin() + i);
                i--;
            } else {
                dirs.erase(dirs.begin() + i);
                dirs.erase(dirs.begin() + i - 1);
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

std::string ValidateLocalPath(const std::string& path) {
    if (IsRelativePath(path)) {
        return "res://" + path;
    } else {
        return ProjectSettings::GetInstance()->LocalizePath(path);
    }
}

// If the string is a valid file path, returns the base directory name
// "/path/to/file.txt" =>  "/path/to"
std::string GetBaseDir(std::string_view path) {
    int end = 0;

    // URL scheme style base.
    int basepos = path.find("://");;
    if (basepos != std::string_view::npos) {
        end = basepos + 3;
    }

    // Windows top level directory base.
    if (end == 0) {
        basepos = path.find(":/");
        if (basepos == -1) {
            basepos = path.find(":\\");
        }
        if (basepos != -1) {
            end = basepos + 2;
        }
    }

    // Windows UNC network share path.
    if (end == 0) {
        if (IsNetworkSharePath(path)) {
            basepos = path.find("/", 2);
            if (basepos == -1) {
                basepos = path.find("\\", 2);
            }
            int servpos = path.find("/", basepos + 1);
            if (servpos == -1) {
                servpos = path.find("\\", basepos + 1);
            }
            if (servpos != -1) {
                end = servpos + 1;
            }
        }
    }

    // Unix root directory base.
    if (end == 0) {
        if (path.starts_with("/")) {
            end = 1;
        }
    }

    std::string rs;
    std::string base;
    if (end != 0) {
        rs = path.substr(end);
        base = path.substr(0, end);
    } else {
        rs = path;
    }

    int sep = std::max(int(rs.find_last_of('/')), int(rs.find_last_of('\\')));
    if (sep == -1) {
        return base;
    }

    return base + rs.substr(0, sep);
}

std::string PathJoin(std::string_view base, std::string_view file) {
    if (base.empty()) {
        return std::string(file);
    }
    if (base.ends_with('/')  || (file.size() > 0 && file.starts_with('/'))) {
        return std::string(base) + std::string(file);
    }
    return std::string(base) + "/" + std::string(file);
}

}