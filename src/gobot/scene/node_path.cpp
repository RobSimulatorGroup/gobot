/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 22-11-6
 * This file is modified by Zikun Yu, 23-1-15
*/


#include "gobot/scene/node_path.hpp"

namespace gobot {

template <typename T>
using Vector = QVector<T>;
static constexpr Qt::SplitBehaviorFlags FlagSkipEmptyParts = Qt::SkipEmptyParts;

NodePath::NodePath(const std::vector<String> &path, bool absolute) {
    data_.path_ = path;
    data_.absolute_ = absolute;
}

NodePath::NodePath(const std::vector<String> &path, const std::vector<String> &subpath, bool absolute) {
    data_.path_ = path;
    data_.subpath_ = subpath;
    data_.absolute_ = absolute;
}

NodePath::NodePath(const String &path) {
    if (!path.length()) {
        LOG_ERROR("Invalid NodePath {}.", path);
        return;
    }

    String raw_path = path;
    bool is_absolute = (raw_path.front() == u'/');
    Vector<String> path_list;
    Vector<String> subpath_list;
    int subpath_pos = path.indexOf(u':');

    if (subpath_pos >= 0) {
        subpath_list = raw_path.split(u':', FlagSkipEmptyParts).toVector();
        if (subpath_pos > 0) {
            raw_path = subpath_list.front();
            subpath_list.pop_front();
        } else {
            raw_path = String("");
        }
    } else {
        subpath_list = Vector<String>();
    }

    if (raw_path.isEmpty()) {
        path_list = Vector<String>();
    } else {
        if (is_absolute) raw_path.remove(0, 1);
        path_list = raw_path.split(u'/', FlagSkipEmptyParts).toVector();
        if (path_list.isEmpty())
            path_list = Vector<String>(raw_path.size(), raw_path);
    }

    data_.absolute_ = is_absolute;
    data_.path_ = std::vector<String>(path_list.begin(), path_list.end());
    data_.subpath_ = std::vector<String>(subpath_list.begin(), subpath_list.end());
}

bool NodePath::IsAbsolute() const {
    return data_.absolute_;
}

ulong NodePath::GetNameCount() const {
    return data_.path_.size();
}

String NodePath::GetName(int idx) const {
    if (idx < 0 || idx >= data_.path_.size()) {
        LOG_ERROR("Invalid index {}.", idx);
        return {};
    }
    if (data_.path_.empty()) {
        LOG_ERROR("Empty NodePath {}.");
        return {};
    }
    return data_.path_[idx];
}

ulong NodePath::GetSubNameCount() const {
    return data_.subpath_.size();
}

String NodePath::GetSubName(int idx) const {
    if (idx < 0 || idx >= data_.subpath_.size()) {
        LOG_ERROR("Invalid index {}.", idx);
        return {};
    }
    if (data_.subpath_.empty()) {
        LOG_ERROR("Void NodePath.");
        return {};
    }
    return data_.subpath_[idx];
}

std::vector<String> NodePath::GetNames() const {
    return data_.path_;
}

std::vector<String> NodePath::GetSubNames() const {
    return data_.subpath_;
}

String NodePath::GetConcatenatedNames() const {
    if (IsEmpty()) LOG_ERROR("Empty NodePath.");

    if (data_.concatenated_path_.isEmpty()) {
        String concatenated;
        std::vector<String> path = data_.path_;

        if (data_.absolute_) concatenated += "/";
        for (int i = 0; i < path.size(); ++i) {
            concatenated += i == 0 ? path[i] : "/" + path[i];
        }
        data_.concatenated_path_ = concatenated;
    }

    return data_.concatenated_path_;
}

String NodePath::GetConcatenatedSubNames() const {
    if (IsEmpty()) LOG_ERROR("Empty NodePath.");

    if (data_.concatenated_subpath_.isEmpty()) {
        String concatenated;
        std::vector<String> subpath = data_.subpath_;
        for (int i = 0; i < subpath.size(); ++i) {
            concatenated += i == 0 ? subpath[i] : ":" + subpath[i];
        }
        data_.concatenated_subpath_ = concatenated;
    }

    return data_.concatenated_subpath_;
}

NodePath NodePath::GetAsPropertyPath() const {
    if (IsEmpty() || data_.path_.empty()) return *this;

    std::vector<String> new_path = data_.subpath_;
    String initial_subname = data_.path_[0];
    for (int i = 1; i < data_.path_.size(); ++i) {
        initial_subname += "/" + data_.path_[i];
    }
    new_path.insert(new_path.begin(), initial_subname);

    return {std::vector<String>(), new_path, false};
}

NodePath::operator String() const {
    if (IsEmpty()) return {};

    String ret;
    if (data_.absolute_) ret = "/";

    for (int i = 0; i < data_.path_.size(); ++i) {
        if (i > 0) ret += "/";
        ret += data_.path_[i];
    }

    for (const auto & str : data_.subpath_) {
        ret += ":" + str;
    }

    return ret;
}

bool NodePath::IsEmpty() const {
    return data_.path_.empty() && data_.subpath_.empty();
}

bool NodePath::operator==(const NodePath &path) const {
    if (IsEmpty() != path.IsEmpty()) return false;

    if (data_.absolute_ != path.data_.absolute_) return false;

    if (data_.path_.size() != path.data_.path_.size()) return false;

    if (data_.subpath_.size() != path.data_.subpath_.size()) return false;

    if (!data_.path_.empty()) {
        for (int i = 0; i < data_.path_.size(); ++i) {
            if (data_.path_[i] != path.data_.path_[i]) return false;
        }
    }

    if (!data_.subpath_.empty()) {
        for (int i = 0; i < data_.subpath_.size(); ++i) {
            if (data_.subpath_[i] != path.data_.subpath_[i]) return false;
        }
    }

    return true;
}

bool NodePath::operator!=(const NodePath &path) const {
    return !(*this == path);
}

void NodePath::Simplify() {
    if (IsEmpty()) return;

    for (int i = 0; i < data_.path_.size(); ++i) {
        if (data_.path_.size() == 1) break;

        if (data_.path_[i] == ".") {
            data_.path_.erase(data_.path_.begin() + i);
            i--;
        } else if (i > 0 && data_.path_[i] == ".." && data_.path_[i - 1] != "." && data_.path_[i - 1] != "..") {
            // remove path_[i - 1] and path_[i]
            data_.path_.erase(data_.path_.begin() + i - 1, data_.path_.begin() + i + 1);
            i -= 2;
            if (data_.path_.empty()) {
                data_.path_.emplace_back(".");
                break;
            }
        }
    }
}

NodePath NodePath::Simplified() const {
    NodePath np = *this;
    np.Simplify();

    return np;
}

}