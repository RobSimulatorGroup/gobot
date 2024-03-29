/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 22-11-6
 * This file is modified by Zikun Yu, 23-1-15
*/

#include "gobot/scene/node_path.hpp"
#include "gobot/core/registration.hpp"
#include "gobot/core/string_utils.hpp"

namespace gobot {

NodePath::NodePath(const std::vector<std::string> &path, bool absolute) {
    data_.path = path;
    data_.absolute = absolute;
}

NodePath::NodePath(const std::vector<std::string> &path, const std::vector<std::string> &subpath, bool absolute) {
    data_.path = path;
    data_.subpath = subpath;
    data_.absolute = absolute;
}

NodePath::NodePath(const std::string &path) {
    if (!path.length()) return;

    std::string raw_path = path;
    bool is_absolute = (raw_path.front() == '/');
    std::vector<std::string> path_list;
    std::vector<std::string> subpath_list;
    int subpath_pos = path.find(':');

    if (subpath_pos >= 0) {
        subpath_list = Split(raw_path, ":", StringSplitBehavior::SkipEmptyParts);
        if (subpath_pos > 0) {
            raw_path = subpath_list.front();
            subpath_list.erase(subpath_list.begin());
        } else {
            raw_path = "";
        }
    } else {
        subpath_list = std::vector<std::string>();
    }

    if (raw_path.empty()) {
        path_list = std::vector<std::string>();
    } else {
        if (is_absolute) raw_path.erase(0, 1);
        path_list = Split(raw_path, "/", StringSplitBehavior::SkipEmptyParts);
        if (path_list.empty())
            path_list = std::vector<std::string>(raw_path.size(), raw_path);
    }

    data_.absolute = is_absolute;
    data_.path = std::move(std::vector<std::string>(path_list.begin(), path_list.end()));
    data_.subpath = std::move(std::vector<std::string>(subpath_list.begin(), subpath_list.end()));
}

bool NodePath::IsAbsolute() const {
    return data_.absolute;
}

std::size_t NodePath::GetNameCount() const {
    return data_.path.size();
}

std::string NodePath::GetName(int idx) const {
    ERR_FAIL_COND_V(data_.path.empty(), std::string());
    ERR_FAIL_INDEX_V(idx, data_.path.size(), std::string());
    return data_.path[idx];
}

std::size_t NodePath::GetSubNameCount() const {
    return data_.subpath.size();
}

std::string NodePath::GetSubName(int idx) const {
    ERR_FAIL_COND_V(data_.subpath.empty(), std::string());
    ERR_FAIL_INDEX_V(idx, data_.subpath.size(), std::string());
    return data_.subpath[idx];
}

std::vector<std::string> NodePath::GetNames() const {
    return data_.path;
}

std::vector<std::string> NodePath::GetSubNames() const {
    return data_.subpath;
}

std::string NodePath::GetConcatenatedNames() const {
    ERR_FAIL_COND_V(IsEmpty(), std::string());

    if (data_.concatenated_path.empty()) {
        std::string concatenated;
        std::vector<std::string> path = data_.path;

        if (data_.absolute) concatenated += "/";
        for (int i = 0; i < path.size(); ++i) {
            concatenated += i == 0 ? path[i] : "/" + path[i];
        }
        data_.concatenated_path = concatenated;
    }

    return data_.concatenated_path;
}

std::string NodePath::GetConcatenatedSubNames() const {
    ERR_FAIL_COND_V(IsEmpty(), std::string());

    if (data_.concatenated_subpath.empty()) {
        std::string concatenated;
        std::vector<std::string> subpath = data_.subpath;
        for (int i = 0; i < subpath.size(); ++i) {
            concatenated += i == 0 ? subpath[i] : ":" + subpath[i];
        }
        data_.concatenated_subpath = concatenated;
    }

    return data_.concatenated_subpath;
}

NodePath NodePath::GetAsPropertyPath() const {
    if (IsEmpty() || data_.path.empty()) return *this;

    std::vector<std::string> new_path = data_.subpath;
    std::string initial_subname = data_.path[0];
    for (int i = 1; i < data_.path.size(); ++i) {
        initial_subname += "/" + data_.path[i];
    }
    new_path.insert(new_path.begin(), initial_subname);

    return {std::vector<std::string>(), new_path, false};
}

NodePath::operator std::string() const {
    if (IsEmpty()) return {};

    std::string ret;
    if (data_.absolute) ret = "/";

    for (int i = 0; i < data_.path.size(); ++i) {
        if (i > 0) ret += "/";
        ret += data_.path[i];
    }

    for (const auto & str : data_.subpath) {
        ret += ":" + str;
    }

    return ret;
}

bool NodePath::IsEmpty() const {
    return data_.path.empty() && data_.subpath.empty();
}

bool NodePath::operator!=(const NodePath &path) const {
    return !(*this == path);
}

void NodePath::Simplify() {
    if (IsEmpty()) return;

    for (int i = 0; i < data_.path.size(); ++i) {
        if (data_.path.size() == 1) break;

        if (data_.path[i] == ".") {
            data_.path.erase(data_.path.begin() + i);
            i--;
        } else if (i > 0 && data_.path[i] == ".." && data_.path[i - 1] != "." && data_.path[i - 1] != "..") {
            // remove path_[i - 1] and path_[i]
            data_.path.erase(data_.path.begin() + i - 1, data_.path.begin() + i + 1);
            i -= 2;
            if (data_.path.empty()) {
                data_.path.emplace_back(".");
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

void NodePath::SetStrData(const std::string& str) {
    *this = NodePath(str);
}

std::string NodePath::GetStrData() const {
    return this->operator std::string();
}


} // End of namespace gobot

GOBOT_REGISTRATION {

    Class_<NodePath>("NodePath")
            .constructor()(CtorAsObject)

            .property_readonly("is_absolute", &NodePath::IsAbsolute)
            .property_readonly("is_empty", &NodePath::IsEmpty)
            .property_readonly("name_count", &NodePath::GetNameCount)
            .property_readonly("subname_count", &NodePath::GetSubNameCount)
            .property_readonly("name_list", &NodePath::GetNames)
            .property_readonly("subname_list", &NodePath::GetSubNames)
            .property_readonly("name_path", &NodePath::GetConcatenatedNames)
            .property_readonly("subname_path", &NodePath::GetConcatenatedSubNames)
            .property_readonly("to_property_path", &NodePath::GetAsPropertyPath)
            .property_readonly("simplified", &NodePath::Simplified)
            .property_readonly("to_string", &NodePath::operator std::string)

            .property("str_data", &NodePath::GetStrData, &NodePath::SetStrData)

            .method("get_name", &NodePath::GetName)
            .method("get_subname", &NodePath::GetSubName)
            .method("simplify", &NodePath::Simplify)
            ;
}
