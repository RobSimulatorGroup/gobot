/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 22-11-6
 * This file is modified by Zikun Yu, 23-1-15
*/

#pragma once

#include "gobot/error_macros.hpp"
#include "gobot/core/types.hpp"
#include "gobot/core/macros.hpp"
#include "gobot/log.hpp"
#include <vector>
#include <QStringList>


namespace gobot {

class GOBOT_EXPORT NodePath {
/**
 * @brief pre-parsed scene tree path.
 *  A pre-parsed relative or absolute path in a scene tree, for use with Node.get_node and similar functions.
 *  It can reference a node, a resource within a node, or a property of a node or resource.
 *  For example, "Path2D/PathFollow2D/Sprite2D:texture:size" would refer to the size property of the texture resource
 *  on the node named "Sprite2D" which is a child of the other named nodes in the path.
 *
 *  A NodePath is composed of a list of slash-separated node names (like a filesystem path) and
 *  an optional colon-separated list of "subnames" which can be resources or properties.
 */
public:
    NodePath(const std::vector<String> &path, bool absolute);
    NodePath(const std::vector<String> &path, const std::vector<String> &subpath, bool absolute);
    NodePath(const NodePath &path) = default;
    explicit NodePath(const String &path);
    NodePath() = default;
    ~NodePath() = default;

    /**
     * @brief Absolute node paths can be used to access the root node ("/root") or autoloads
     *  (e.g. "/global" if a "global" autoload was registered).
     *
     * @returns true if the node path is absolute (as opposed to relative), which means that it starts with a
     *  slash character (/).
     */
    [[nodiscard]] bool IsAbsolute() const;

    /**
     * @brief Gets the number of node names which make up the path. Subnames (see GetSubNameCount) are not included.
     *  For example, "Path2D/PathFollow2D/Sprite2D" has 3 names.
     *
     * @returns number of node names.
     */
    [[nodiscard]] std::size_t GetNameCount() const;

    /**
     * @brief Gets the node name indicated by idx (0 to GetNameCount - 1).
     * @params
     *  idx[int]: Name index of list of names.
     *
     * @returns a String of indexed name or an empty String for an invalid index.
     */
    [[nodiscard]] String GetName(int idx) const;

    /**
     * @brief Gets the number of resource or property names ("subnames") in the path. Each subname is listed
     *  after a colon character (:) in the node path.
     *
     * @returns number of node subnames.
     */
    [[nodiscard]] std::size_t GetSubNameCount() const;

    /**
     * @brief Gets the resource or property name indicated by idx (0 to GetSubnameCount).
     * @params
     *  idx[int]: Name index of list of subnames.
     *
     * @returns a String of indexed subname or an empty String for an invalid index.
     */
    [[nodiscard]] String GetSubName(int idx) const;

    /**
     * @brief Gets the list of node names which make up the path.
     *
     * @returns a list (std::vector<String>) of node names.
     */
    [[nodiscard]] std::vector<String> GetNames() const;

    /**
     * @brief Gets the list of resource or property names in the path.
     *
     * @returns a list (std::vector<String>) of subnames.
     */
    [[nodiscard]] std::vector<String> GetSubNames() const;

    /**
     * @brief Gets all paths concatenated with a slash character (/) as separator without subnames.
     *
     * @returns a String of path.
     */
    [[nodiscard]] String GetConcatenatedNames() const;

    /**
    * @brief Gets all subnames concatenated with a colon character (:) as separator,
    *  i.e. the right side of the first colon in a node path.
    *
    * @returns a String of resource or property path.
    */
    [[nodiscard]] String GetConcatenatedSubNames() const;

    /**
     * @brief Gets a node path with a colon character (:) prepended,
     *  transforming it to a pure property path with no node name (defaults to resolving from the current node).
     *
     * @returns a String of property path.
     */
    [[nodiscard]] NodePath GetAsPropertyPath() const;

    explicit operator String() const;

    /**
     * @brief The node path is empty if both node names and subnames are empty.
     *
     * @returns true if node path is empty, otherwise false.
     */
    [[nodiscard]] bool IsEmpty() const;

    bool operator==(const NodePath &path) const = default;
    bool operator!=(const NodePath &path) const;
    NodePath& operator=(const NodePath &path) = default;

    /**
     * @brief Simplifies a node path with "." or ".." noting current node name or last node name respectively.
     *  This removes these two separators and replaces the original node path with a pure path.
     */
    void Simplify();

    /**
     * @brief Gets a simplified node path for the original path.
     *
     * @returns a new NodePath and preserves the original one.
     */
    [[nodiscard]] NodePath Simplified() const;

private:
    // For rttr
    void SetStrData(const String& str);

    String GetStrData() const;

    struct Data {
        std::vector<String> path = std::vector<String>();
        std::vector<String> subpath = std::vector<String>();
        String concatenated_path = String();
        String concatenated_subpath = String();
        bool absolute = false;

        bool operator==(const Data &data) const = default;
    };

    mutable Data data_;

    static constexpr Qt::SplitBehaviorFlags s_split_behavior_flags = Qt::SkipEmptyParts;

    GOBOT_REGISTRATION_FRIEND
};

}

template<>
struct fmt::formatter<gobot::NodePath> : fmt::formatter<std::string>
{
    static auto format(const gobot::NodePath& node_path, format_context &ctx) -> decltype(ctx.out())
    {
        return fmt::formatter<gobot::String>::format(node_path.operator gobot::String(), ctx);
    }
};