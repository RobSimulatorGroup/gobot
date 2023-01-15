/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 22-11-6
*/


#include "gobot/scene/node_path.hpp"

namespace gobot {

static constexpr Qt::SplitBehaviorFlags FlagSkipEmptyParts = Qt::SkipEmptyParts;

NodePath::NodePath(const String &path) {
    if (!path.length()) return;

    String raw_path = path;
    std::vector<String> path_list;

    int subpath_pos = path.indexOf(u':');
    if (subpath_pos == 0) {
        // exception: invalid path
    }
    if (subpath_pos > 0) {
        path_list = raw_path.split(u':', FlagSkipEmptyParts).toVector().toStdVector();
        if (!path_list.empty()) {
            this->subpath_ = std::vector(path_list.begin() + 1, path_list.end());
            raw_path = path_list.front();
        }
    } else {// subpath_pos == -1, no subpath exists
        if (raw_path.isEmpty()) {
            // exception: invvalid path
        }
        this->subpath_ = std::vector<String>();
    }

    bool absolute = (raw_path.front() == u'/');
    if (absolute) {
        raw_path.remove(0, 1);
    }
    this->path_ = raw_path.split(u'/', FlagSkipEmptyParts).toVector().toStdVector();
    if (this->path_.empty())
        this->path_.push_back(raw_path);
}

NodePath::NodePath() {
}

bool NodePath::IsAbsolute() const {
    return absolute_;
}

}