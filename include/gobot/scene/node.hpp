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
/**
 * @brief Node as a building block, can be assigned as the child of another node,
 *  resulting in a tree arrangement. A given node can contain any number of nodes
 *  as children with the requirement that all siblings should have unique names.
 */

public:
    Node() = default;

    ~Node() override;

    /**
     * @brief This name is unique among the siblings (other child node from the same
     *  parent). When set to an existing name, the node will be automatically renamed.
     *  Note: Auto-generated names might include "@" character, which is reserved for
     *  unique names when using AddChild. When setting the name manually, any "@"
     *  will be removed.
     *
     * @return the name of the node.
     */
    String GetName() const;

    /**
     * @brief See GetName.
     *
     * @param p_name[String]: Given name.
     */
    void SetName(const String &p_name);

    /**
     * @brief Adds a child node. Nodes can have any number of children, but every child
     *  must have a unique name in a branch. Child nodes are automatically deleted when
     *  the parent node is deleted, so an entire scene can be removed by deleting its
     *  topmost node (root).
     *
     *  If force_readable_name is true, it improves the readability of the added node.
     *  If not named, the node is renamed to its type, and if it shares name with a
     *  sibling, a number is suffixed more appropriately. This operation is very slow.
     *  As such, it is recommended leaving this to false, which assigns a dummy name
     *  featuring @ in both situations.
     *
     *  Note: If the child node already has a parent, the function will fail. Use RemoveChild
     *  first to remove the node from its current parent. If you need the child node to be
     *  added below a specific node in the list of children, use AddSibling instead of
     *  this method.
     *
     * @param child[Node*]: Child node pointer.
     * @param force_readable_name[bool]: Generate a human readable name for child node.
     */
    void AddChild(Node *child, bool force_readable_name = false);

    /**
     * @brief Adds a sibling to current node's parent, at the same level as that node,
     *  right below it.
     *
     *  If force_readable_name is true, it improves the readability of the added node.
     *  If not named, the node is renamed to its type, and if it shares name with a
     *  sibling, a number is suffixed more appropriately. This operation is very slow.
     *  As such, it is recommended leaving this to false, which assigns a dummy name
     *  featuring @ in both situations.
     *
     *  Use AddChild instead of this method if you don't need the child node to be added
     *  below a specific node in the list of children.
     *
     * @param sibling[Node*]: Sibling node pointer.
     * @param force_readable_name[bool]: Generate a human readable name for child node.
     */
    void AddSibling(Node *sibling, bool force_readable_name = false);

    /**
     * @biref See AddChild.
     *
     * @param child[Node*]: Child node pointer.
     */
    void RemoveChild(Node *child);

    /**
     * @brief Removes a child node from its parent's list of children. The node is NOT
     *  deleted and must be deleted manually.
     *
     * @return the number of child nodes.
     */
    std::size_t GetChildCount() const;

    /**
     * @brief See AddChild. This method is often used for iterating all children of a node.
     *  To query the index of a node, see GetIterator or GetIndex.
     *
     * @param index[int]: Index of child in the children list.
     *
     * @return a child node by its index (See GetChildCount).
     */
    Node* GetChild(int index) const;

    /**
     * @brief See NodePath.
     *
     * @param path[NodePath]: Path points to the node.
     *
     * @return true if the node that the path points to exists.
     */
    bool HasNode(const NodePath& path) const;

    /**
     * @brief Fetches a node. The node path can be either a relative path (from the current node)
     *  or an absolute path (in the scene tree) to a node. If the path does not exist, nullptr is
     *  returned and an error is logged. Attempts to access methods on the return value will result
     *  in an "Attempts to call <method> on a null instance." error.
     *  Note: Fetching absolute paths only works when the node is inside the scene tree (See IsInsideTree).
     *
     * @param path[NodePath]: Path points to the node.
     *
     * @return the node pointer that the path points to if it exists, otherwise nullptr with an error logged.
     */
    Node* GetNode(const NodePath &path) const;

    /**
     * @brief See GetNode. Dose not log an error if path does not point to a valid node.
     *
     * @param path[NodePath]: Path points to the node.
     *
     * @return the node pointer that the path points to if it exists, otherwise nullptr.
     */
    Node* GetNodeOrNull(const NodePath &path) const;

    /**
     * @brief The node sets the given parent node in replace of the original parent which is required.
     *
     * @param parent[Node*]: New parent node pointer.
     * @param keep_global_transform[bool]: Keep the global transform.
     */
    virtual void Reparent(Node *parent, bool keep_global_transform = true);

    /**
     * @brief See GetNode.
     *
     * @return the parent node of the current node, or null if the node lacks a parent.
     */
    Node* GetParent() const;

    /**
     * @brief See SceneTree.
     *
     * @return pointer to the scene tree that the node attached to.
     */
    FORCE_INLINE SceneTree* GetTree() const {
        ERR_FAIL_COND_V(!tree_, nullptr);
        return tree_;
    }

    /**
     * @brief The node is inside a tree that is notified EnterTree (See SetTree), and outside the tree that is
     *  notified ExitTree.
     *
     * @return true if it is inside a tree, otherwise false.
     */
    FORCE_INLINE bool IsInsideTree() const { return inside_tree_; }

    /**
     * @brief Ancestor of a node locates at higher layer in the same tree and can be traced in the ascending order.
     *
     * @param node[Node*]: Pointer to a node
     *
     * @return true if this node is an ancestor of the given node, otherwise false.
     */
    bool IsAncestorOf(const Node *node) const;

    /**
     * @brief This only works if the current node is inside the tree (See IsInsideTree).
     *
     * @return the absolute path of the current node.
     */
    NodePath GetPath() const;

    /**
     * @brief This only works if both nodes are in the same tree.

     * @param node[Node*]: Pointer to a node that path points to.
     *
     * @return the relative path from this node to the specified node.
     */
    NodePath GetPathTo(const Node *node) const;

    /**
     * @brief Common parent exists for two sibling nodes.
     *
     * @param node[Node*]: Pointer to a node that could be a sibling.
     *
     * @return pinter to the sibling node if common parent exists, otherwise nullptr.
     */
    Node* FindCommonParentWith(const Node* node) const;

    /**
     * @brief Move a child to another position among the children list by index. This works only if
     *  the given child is one of the children.
     *
     * @param child[Node*]: Pointer to a child node in the same tree.
     * @param index[int]: Index of the moving child.
     */
    void MoveChild(Node *child, int index);

    /**
     * @brief Get the STL iterator of the node in the children list.
     *
     * @return std::vector<Node*>::iterator.
     */
    std::vector<Node *>::iterator GetIterator() const;

    /**
     * @brief Get the position of the node in the children list.
     *
     * @return a position.
     */
    int GetIndex() const;

    /**
     * @brief Similar to PrintTree, this prints the tree to stdout. This version displays a more graphical
     *  representation similar to what is displayed in the scene inspector. It is useful for inspecting
     *  larger trees.
     */
    void PrintTreePretty();

    /**
     * @brief Print the tree to stdout. Used mainly for debugging purposes. This version displays the path
     *  relative to the current node, and is good for copy/pasting into the GetNode function.
     */
    void PrintTree();

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

    void PrintTreePretty(const String &prefix, bool last);
    void PrintTree(const Node *node);

    Node *GetChildByName(const String& name) const;

    friend class SceneTree;

    void SetTree(SceneTree *tree);

    void ValidateChildName(Node* child, bool force_human_readable);

    String ValidateNodeName(const String &p_name) const;

    const String invalid_node_name_characters = ". : @ / \" ";

    void GenerateSerialChildName(const Node* child, String &name) const;

    void PropagateNotification(NotificationType p_notification);
    void PropagateReverseNotification(NotificationType p_notification);
    void PropagateReady();
    void PropagateEnterTree();
    void PropagateExitTree();
    void PropagateAfterExitTree();
};

} // End of namespace gobot