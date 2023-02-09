/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 22-11-6
 * This file is modified by Zikun Yu, 23-1-20
*/

#pragma once

#include "gobot/core/object.hpp"
#include "gobot/scene/scene_tree.hpp"
#include "gobot/scene/node_path.hpp"
#include "gobot/error_macros.hpp"

namespace gobot {

class NodePath;
class SceneTree;

class GOBOT_EXPORT Node : public Object {
    GOBCLASS(Node, Object)

public:
    Node() = default;

    ~Node() override;

    String GetName() const;
    void SetName(const String &p_name);

    void AddChild(Node *child);
    void AddSibling(Node *sibling);
    void RemoveChild(Node *child);

    std::size_t GetChildCount() const;
    Node* GetChild(int index) const;
    bool HasNode(const NodePath& path) const;
    Node* GetNode(const NodePath &path) const;
    Node* GetNodeOrNull(const NodePath &path) const;

    virtual void Reparent(Node *parent, bool keep_global_transform = true);
    Node* GetParent() const;

    FORCE_INLINE SceneTree* GetTree() const {
        ERR_FAIL_COND_V(!tree_, nullptr);
        return tree_;
    }

    FORCE_INLINE bool IsInsideTree() const { return inside_tree_; }

    bool IsAncestorOf(const Node *node) const;

    NodePath GetPath() const;
    NodePath GetPathTo(const Node *node) const;
    Node* FindCommonParentWith(const Node* node) const;

    void MoveChild(Node *child, int index);

    std::vector<Node *>::iterator GetIterator() const;
    int GetIndex() const;

//    void PrintTreePretty();
//    void PrintTree();

protected:
    void NotificationCallBack(NotificationType notification);

    virtual void AddChildNotify(Node *child);
    virtual void RemoveChildNotify(Node *child);
    virtual void MoveChildNotify(Node *child);

    void AddChildNoCheck(Node *child, const String& name);
    void SetNameNoCheck(const String& name);

private:
    Node *parent_ = nullptr;
    std::vector<Node *> children_;

    String name_;
    SceneTree *tree_ = nullptr;
    bool inside_tree_ = false;
    mutable NodePath path_cache_;

//    void PrintTreePretty(const String &prefix, bool last);
//    void PrintTree(const Node *node);

    Node *GetChildByName(const String& name) const;

    friend class SceneTree;

    void SetTree(SceneTree *tree);

    void ValidateChildName(Node* child);

    String ValidateNodeName(const String &p_name) const;

    const String invalid_node_name_characters = ". : @ / \" ";

//    void GenerateSerialChildName(const Node* child, String&name) const;

    void PropagateNotification(NotificationType p_notification);
    void PropagateReverseNotification(NotificationType p_notification);
    void PropagateReady();
    void PropagateEnterTree();
    void PropagateExitTree();
    void PropagateAfterExitTree();
};

} // End of namespace gobot