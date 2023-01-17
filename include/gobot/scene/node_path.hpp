/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 22-11-6
 * This file is modified by Zikun Yu, 23-1-15
*/

#pragma once

#include "gobot/log.hpp"
#include "gobot/core/types.hpp"
#include <vector>
#include <QStringList>


namespace gobot {

class NodePath {
public:
    NodePath(const std::vector<String> &path, bool absolute);
    NodePath(const std::vector<String> &path, const std::vector<String> &subpath, bool absolute);
    NodePath(const NodePath &path) = default;
    explicit NodePath(const String &path);
    NodePath() = default;
    ~NodePath() = default;

    [[nodiscard]] bool IsAbsolute() const;
    [[nodiscard]] ulong GetNameCount() const;
    [[nodiscard]] String GetName(int idx) const;
    [[nodiscard]] ulong GetSubNameCount() const;
    [[nodiscard]] String GetSubName(int idx) const;
    [[nodiscard]] std::vector<String> GetNames() const;
    [[nodiscard]] std::vector<String> GetSubNames() const;
    [[nodiscard]] String GetConcatenatedNames() const;
    [[nodiscard]] String GetConcatenatedSubNames() const;

    [[nodiscard]] NodePath GetAsPropertyPath() const;
    explicit operator String() const;
    [[nodiscard]] bool IsEmpty() const;

    bool operator==(const NodePath &path) const;
    bool operator!=(const NodePath &path) const;
    NodePath& operator=(const NodePath &path) = default;

    void Simplify();
    [[nodiscard]] NodePath Simplified() const;

private:
    struct Data {
        std::vector<String> path_ = std::vector<String>();
        std::vector<String> subpath_ = std::vector<String>();
        String concatenated_path_ = String();
        String concatenated_subpath_ = String();
        bool absolute_ = false;
    };

    mutable Data data_;
};

}