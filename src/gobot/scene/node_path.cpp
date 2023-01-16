/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 22-11-6
*/


#include "gobot/scene/node_path.hpp"

namespace gobot {

template <typename T>
using Vector = QVector<T>;
static constexpr Qt::SplitBehaviorFlags FlagSkipEmptyParts = Qt::SkipEmptyParts;

NodePath::NodePath(const std::vector<String>& path, bool absolute) {
    subpath_ = std::vector<String>();
    absolute_ = absolute;

    if (path.empty()) {
        path_ = std::vector<String>();
        valid_ = false;
        return;
    }

    path_ = path;
    valid_ = true;
}

NodePath::NodePath(const std::vector<String> &path, const std::vector<String> &subpath, bool absolute) {
    absolute_ = absolute;

    if (path.empty() || subpath.empty()) {
        path_ = std::vector<String>();
        subpath_ = std::vector<String>();
        return;
    }

    path_ = path;
    subpath_ = subpath;
    valid_ = true;

}

NodePath::NodePath(const NodePath &path) {
    if (!path.valid_) {
        path_ = std::vector<String>();
        subpath_ = std::vector<String>();
        return;
    }

    path_ = path.path_;
    subpath_ = path.subpath_;
    absolute_ = path.absolute_;
}

NodePath::NodePath(const String& path) {
    bool is_valid = true;
    if (!path.length()) {
        LOG_ERROR("Invalid NodePath {}.", path);
        is_valid = false;
    }

    String raw_path = path;
    bool is_absolute = (raw_path.front() == u'/');
    Vector<String> path_list;
    Vector<String> subpath_list;

    int subpath_pos = path.indexOf(u':');
    if (is_valid && subpath_pos == 0) {
        LOG_ERROR("Invalid NodePath {}.", path);
        is_valid = false;
    }

    if (is_valid && subpath_pos > 0) {
        path_list = raw_path.split(u':', FlagSkipEmptyParts).toVector();
        if (!path_list.empty()) {
            subpath_list = Vector<String>(path_list.begin() + 1, path_list.end());
            raw_path = path_list.front();
        }
    } else {// subpath_pos == -1, no subpath exists
        if (raw_path.isEmpty()) {
            LOG_ERROR("Invalid NodePath {}.", path);
            is_valid = false;
        }
    }

    if (is_valid) {
        if (is_absolute) {
            raw_path.remove(0, 1);
        }
        path_list = raw_path.split(u'/', FlagSkipEmptyParts).toVector();
        if (path_list.empty())
            path_list.push_back(raw_path);

        path_ = std::vector<String>(path_list.begin(), path_list.end());
        subpath_ = std::vector<String>(subpath_list.begin(), subpath_list.end());
    } else {
        path_ = std::vector<String>();
        subpath_ = std::vector<String>();
    }
    valid_ = is_valid;
    absolute_ = is_absolute;
}

bool NodePath::IsAbsolute() const {
    return absolute_;
}

ulong NodePath::GetNameCount() const {
    if (!valid_) {
        return 0;
    }
    return path_.size();
}

String NodePath::GetName(int idx) const {
    if (idx < 0 || idx >= path_.size()) {
        LOG_ERROR("Invalid index {}.", idx);
        return {};
    }
    if (!valid_ || path_.empty()) {
        LOG_ERROR("Void NodePath {}.");
        return {};
    }
    return path_[idx];
}

ulong NodePath::GetSubNameCount() const {
    if (!valid_) {
        return 0;
    }
    return subpath_.size();
}

String NodePath::GetSubName(int idx) const {
    if (idx < 0 || idx >= subpath_.size()) {
        LOG_ERROR("Invalid index {}.", idx);
        return {};
    }
    if (!valid_ || subpath_.empty()) {
        LOG_ERROR("Void NodePath {}.");
        return {};
    }
    return subpath_[idx];
}

std::vector<String> NodePath::GetNames() const {
    if (valid_) {
        return path_;
    }
    return {};
}

std::vector<String> NodePath::GetSubNames() const {
    if (valid_) {
        return subpath_;
    }
    return {};
}


}