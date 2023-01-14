/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 22-11-6
*/

#pragma once

#include "gobot/core/object.hpp"
#include "gobot/error_marcos.hpp"

namespace gobot {

class NodePath;
class SceneTree;

class Node : public Object {
    GOBCLASS(Node, Object)
public:
    Node();

    ~Node();

    Node(const String& name);

    String GetName() const;

    void SetName(const String & name);

    void AddChild(Node *child);

    void AddSibling(Node *sibling);

    void RemoveChild(Node *child);

    Node* GetChild(int p_index) const;

    bool HasNode(const NodePath& path) const;

    // https://docs.godotengine.org/zh_CN/stable/tutorials/scripting/scene_unique_nodes.html
    Node* GetNode(const NodePath &path) const;

    Node* GetNodeOrNull(const NodePath &path) const;

    Node* GetParent() const;

    void PrintTree();

    void PrintTreePretty();

    FORCE_INLINE SceneTree* GetTree() const {
        ERR_FAIL_COND_V(!tree_, nullptr);
        return tree_;
    }

    FORCE_INLINE bool IsInsideTree() const { return inside_tree_; }

    bool IsAncestorOf(const Node *node) const;

    NodePath GetPath() const;

    NodePath GetPathTo(const Node *node) const;

    Node* FindCommonParentWith(const Node* node) const;

protected:
    void AddChildNoCheck(Node *child, const String& name);

    void SetNameNoCheck(const String& name);

    void SetOwnerNoCheck(Node *owner);

    void Notification(NotificationType notification);

private:
    friend class SceneTree;

    void PropagateReverseNotification(int p_notification);

    void PropagateEnterTree();

    void PropagateReady();

    void PropagateExitTree();

    void PropagateAfterExitTree();

    friend class SceneTree;

    void SetTree(SceneTree* tree);

    Node *GetChildByName(const String& name) const;

    void ValidateChildName(Node* child);

    void GenerateSerialChildName(const Node* child, String&name) const;

private:
    String name_;
    std::vector<Node*> children_node_;
    mutable NodePath *path_cache_ = nullptr;

    bool inside_tree_ = false;
    SceneTree* tree_ = nullptr;

    Node* parent_ = nullptr;
};

}