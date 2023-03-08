/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 22-11-20
 * This file is modified by Zikun Yu, 23-1-20
*/

#include "gobot/scene/node.hpp"

#include <ranges>
#include "gobot/core/registration.hpp"

namespace gobot {

void Node::NotificationCallBack(NotificationType notification) {
    switch (notification) {

        case NotificationType::EnterTree: {
            // TODO: ERR_FAIL_COND(!ViewPort);
            ERR_FAIL_COND(!GetTree());

            GetTree()->node_count_++;
        } break;

        case NotificationType::ExitTree: {
            // TODO: ERR_FAIL_COND(!ViewPort);
            ERR_FAIL_COND(!GetTree());

            GetTree()->node_count_--;

            if (!path_cache_.IsEmpty()) {
                path_cache_ = NodePath();
            }
        } break;

        case NotificationType::PathRenamed: {
            if (!path_cache_.IsEmpty()) {
                path_cache_ = NodePath();
            }
        } break;

        case NotificationType::Ready: {
            // TODO: Call ready
        } break;

        case NotificationType::PreDelete: {
            if (parent_) {
                parent_->RemoveChild(this);
            }

            // Kill children as cleanly as possible
            for (auto & child : children_) {
                Delete(child);
            }
        } break;
    }
}

void Node::AddChildNotify(Node *child) {

}

void Node::RemoveChildNotify(Node *child) {

}

void Node::MoveChildNotify(Node *child) {

}

String Node::GetName() const {
    return name_;
}

void Node::SetNameNoCheck(const String &name) {
    name_ = name;
}

void Node::SetName(const String &p_name) {
    String name = ValidateNodeName(p_name);
    ERR_FAIL_COND(name.isEmpty());

    SetNameNoCheck(name);
    if (parent_) {
        parent_->ValidateChildName(this, true);
    }

    PropagateNotification(NotificationType::PathRenamed);

    if (IsInsideTree()) {
        // TODO: Signal nodeRenamed to this node
        Q_EMIT tree_->nodeRenamed(this);
        Q_EMIT tree_->treeChanged();
    }
}

void Node::ValidateChildName(Node* child, bool force_human_readable) {
    /* Make sure the name is unique */

    if (force_human_readable) {
        // This approach to autoset node names is human-readable but very slow

        String name = child->name_;
        GenerateSerialChildName(child, name);
        child->name_ = name;
    } else {
        //This approach to autoset node names is fast but not as readable
        // It's the default and reserves the '@' character for unique names.

        bool unique = true;

        if (child->name_ == String()) {
            // new unique name must be assigned
            unique = false;
        } else {
            // check if exists
            for (const auto & ith_child : children_) {
                if (ith_child == child)
                    continue;
                if (ith_child->name_ == child->name_) {
                    unique = false;
                    break;
                }
            }
        }

        if (!unique) {
            // Temporarily set as default
            String name = "@" + String(child->GetName()) + "@" + String::number(children_.size());
            child->name_ = name;
        }
    }
}

String Node::ValidateNodeName(const String &p_name) const {
    auto chars = invalid_node_name_characters.split(" ");
    String name = p_name;
    for (const auto & c : chars) {
        name = name.replace(c, "");
    }

    return name;
}

String IncreaseNumericString(const String &s) {
    String res = s;
    bool carry = res.length() > 0;

    for (int i = res.length() - 1; i >= 0; i --) {
        if (!carry) {
            break;
        }
        QChar n = s[i];
        if (n == '9') {
            res[i] = '0';
        } else {
            res[i] = s[i].unicode() + 1;
            carry = false;
        }
    }

    if (carry) {
        res = "1" + res;
    }

    return res;
}

void Node::GenerateSerialChildName(const Node *child, String &name) const {
    if (name == String()) {
        // No name and a new name is needed, create one.
        name = String::fromStdString(child->GetClassStringName().data());
    }

    // Quickly test if proposed name exists
    bool exists = false;
    for (const auto & c : children_) {
        if (c == child) continue; // Exclude self in renaming if it's already a child

        if (c->name_ == name) {
            exists = true;
            break;
        }
    }

    if (!exists) {
        return; // If it does not exist, it does not need validation
    }

    // Extract trailing number
    String name_string = name;
    String nums;
    for (const auto & i : std::ranges::reverse_view(name_string)) {
        if (i.isDigit()) {
            nums = i + nums;
        } else {
            break;
        }
    }

    // Extract base name
    name_string = name_string.mid(0, name_string.length() - nums.length());

    for (;;) {
        String attempt = name_string + nums;
        exists = false;

        for (const auto & i : children_) {
            if (i == child) continue;
            if (i->name_ == attempt) {
                exists = true;
                break;
            }
        }

        if (!exists) {
            name = attempt;
            return;
        } else {
            if (nums.length() == 0) {
                // Name was undecorated to skip to 2 for a more natural result
                nums = "2";
            } else {
                nums = IncreaseNumericString(nums);
            }
        }
    }
}

void Node::AddChildNoCheck(Node *child, const String &name) {
    child->name_ = name;
    children_.push_back(child);
    child->parent_ = this;

    child->Notification(NotificationType::Parented);

    if (tree_) {
        child->SetTree(tree_);
    }

    AddChildNotify(child);
}

void Node::AddChild(Node *child, bool force_readable_name) {
    ERR_FAIL_NULL(child);
    ERR_FAIL_COND_MSG(child == this, fmt::format("Can't add child '%s'", child->GetName()));
    ERR_FAIL_COND_MSG(child == parent_, fmt::format("Can't add child '%s' to '%s', already has a parent {}.",
                                                    child->GetName(), GetName(), child->parent_->GetName()));

    ValidateChildName(child, force_readable_name);
    AddChildNoCheck(child, child->name_);
}

void Node::AddSibling(Node *sibling, bool force_readable_name) {
    ERR_FAIL_NULL(sibling);
    ERR_FAIL_NULL(parent_);
    ERR_FAIL_COND_MSG(sibling == this, fmt::format("Can't add sibling '%s' to itself.", sibling->GetName()));

    parent_->AddChild(sibling, force_readable_name);
    parent_->MoveChild(sibling, GetIndex() + 1);
}

void Node::RemoveChild(Node *child) {
    ERR_FAIL_NULL(child);

    auto it = std::find(children_.begin(), children_.end(), child);
    ERR_FAIL_COND_MSG(it == children_.end(),
                      fmt::format("Cannot remove child '%s' as it is not a child of this node.", child->GetName()));

    child->SetTree(nullptr);
    RemoveChildNotify(child);
    child->Notification(NotificationType::Unparented);

    children_.erase(std::remove(children_.begin(), children_.end(), child), children_.end());

    child->parent_ = nullptr;
    if (inside_tree_) {
        child->PropagateAfterExitTree();
    }
}

std::size_t Node::GetChildCount() const {
    return children_.size();
}

Node* Node::GetChild(int index) const {
    ERR_FAIL_INDEX_V(index, children_.size(), nullptr);
    return children_[index];
}

Node* Node::GetChildByName(const String &name) const {
    for (const auto & child : children_) {
        if (child->name_ == name)
            return child;
    }

    return nullptr;
}

Node* Node::GetNodeOrNull(const NodePath &path) const {
    if (path.IsEmpty()) return nullptr;

    ERR_FAIL_COND_V_MSG(!inside_tree_ && path.IsAbsolute(), nullptr,
                      "Can't use GetNode() with absolute paths from outside the active scene tree");

    NodePath full_path = path.Simplified();
    Node *current = nullptr;
    Node *root = nullptr;
    if (!full_path.IsAbsolute()) {
        current = const_cast<Node *>(this);
    } else {
        root = const_cast<Node *>(this);
        while (root->parent_) {
            root = root->parent_;
        }
    }

    for (const auto & name : full_path.GetNames()) {
        if (current == nullptr) {
            current = root;
            continue;
        }

        auto it = std::find_if(current->children_.begin(), current->children_.end(),
                               [&](const Node *child) { return child->name_ == name; });
        if (it == current->children_.end()) {
            return nullptr;
        } else {
            current = *it;
        }
    }

    return current;
}

Node* Node::GetNode(const NodePath &path) const {
    Node *node = GetNodeOrNull(path);

    if (node == nullptr) [[unlikely]] {
        String desc;
        if (IsInsideTree()) {
            desc = GetPath().operator String();
        } else {
            desc = GetName();
            if (desc.isEmpty()) {
                desc = String::fromStdString(GetClassStringName().data());
            }
        }

        if (path.IsAbsolute()) {
            ERR_FAIL_V_MSG(nullptr,
                           fmt::format("Node not found: {} (absolute path attempted from {}).", path, desc));
        } else {
            ERR_FAIL_V_MSG(nullptr,
                           fmt::format("Node not found: {} (relative to {}).", path, desc));
        }
    }

    return node;
}

void Node::SetTree(SceneTree *tree) {
    SceneTree *tree_changed_a = nullptr;
    SceneTree *tree_changed_b = nullptr;

    if (tree_) {
        PropagateExitTree();
        tree_changed_a = tree_;
    }
    tree_ = tree;

    if (tree_) {
        PropagateEnterTree();
        if (!parent_) {
            PropagateReady();
        }
        tree_changed_b = tree_;
    }

    if (tree_changed_a) {
        Q_EMIT tree_changed_a->treeChanged();
    }
    if (tree_changed_b) {
        Q_EMIT tree_changed_b->treeChanged();
    }
}

bool Node::HasNode(const NodePath &path) const {
    return GetNodeOrNull(path) != nullptr;
}

void Node::Reparent(Node *parent) {
    ERR_FAIL_NULL(parent);
    ERR_FAIL_NULL_MSG(parent_, "Node needs a parent to be reparented.");

    if (parent == parent_) return;

    parent_->RemoveChild(this);
    parent->AddChild(this);
}

Node* Node::GetParent() const {
    return parent_;
}

bool Node::IsAncestorOf(const Node *node) const {
    ERR_FAIL_NULL_V(node, false);
    Node *p = node->parent_;
    while (p) {
        if (p == this) {
            return true;
        }
        p = p->parent_;
    }

    return false;
}

NodePath Node::GetPath() const {
    ERR_FAIL_COND_V_MSG(!IsInsideTree(), NodePath(), "Cannot get path of this node as it is not in a scene tree.");

    if (!path_cache_.IsEmpty()) {
        return path_cache_;
    }

    const Node *n = this;
    std::vector<String> path;
    while (n) {
        path.push_back(n->GetName());
        n = n->parent_;
    }
    std::reverse(path.begin(), path.end());
    path_cache_ = NodePath(path, true);

    return path_cache_;
}

NodePath Node::GetPathTo(const Node *node) const {
    ERR_FAIL_NULL_V(node, NodePath());

    if (this == node) return NodePath(".");

    const Node *common_parent = FindCommonParentWith(node);
    ERR_FAIL_COND_V(!common_parent, NodePath());

    std::vector<String> path;
    String up = String("..");
    const Node *n = node;
    while (n != common_parent) {
        path.push_back(n->GetName());
        n = n->parent_;
    }
    n = this;
    while (n != common_parent) {
        path.push_back(up);
        n = n->parent_;
    }
    std::reverse(path.begin(), path.end());

    return {path, false};
}

Node* Node::FindCommonParentWith(const Node *node) const {
    if (node == this) {
        return const_cast<Node *>(node);
    }

    std::unordered_set<const Node *> visited;
    const Node *n = this;
    while (n) {
        visited.insert(n);
        n = n->parent_;
    }

    const Node *common_parent = node;
    while (common_parent) {
        if (visited.find(common_parent) != visited.end()) {
            break;
        }
        common_parent = common_parent->parent_;
    }

    if (!common_parent) {
        return nullptr;
    }

    return const_cast<Node *>(common_parent);
}

void Node::PrintTreePretty() {
    PrintTreePretty("", true);
}

void Node::PrintTree() {
    PrintTree(this);
}

void Node::PrintTreePretty(const String &prefix, bool last) {
    String new_prefix = last ? String(" ┖╴ ") : String(" ┠╴ ");
    std::cout << prefix.toStdString() << new_prefix.toStdString() << GetName().toStdString() << std::endl;
    for (const auto & child : children_) {
        new_prefix = last ? String("   ") : String(" ┃ ");
        child->PrintTreePretty(prefix + new_prefix, child == children_.back());
    }
}

void Node::PrintTree(const Node *node) {
    std::cout << node->GetPathTo(this).operator String().toStdString() << std::endl;
    for (const auto & child : children_) {
        child->PrintTree(node);
    }
}

void Node::PropagateNotification(NotificationType p_notification) {
    Notification(p_notification);

    for (const auto & child : children_) {
        child->PropagateNotification(p_notification);
    }
}

void Node::PropagateReverseNotification(NotificationType p_notification) {
    for (int i = children_.size() - 1; i >= 0; --i) {
        children_[i]->PropagateReverseNotification(p_notification);
    }

    Notification(p_notification, true);
}

void Node::PropagateReady() {
    for (const auto & child : children_) {
        child->PropagateReady();
    }

    Notification(NotificationType::Ready);

    // TODO: Signal ready to SceneStringNames
}

void Node::PropagateEnterTree() {
    // This needs to happen to all children before any EnterTree

    if (parent_) {
        tree_ = parent_->tree_;
    }

    // TODO: Set ViewPort

    inside_tree_ = true;

    Notification(NotificationType::EnterTree);

    // TODO: Signal enter tree to SceneStringNames
    Q_EMIT tree_->nodeAdded(this);

    for (const auto & child : children_) {
        if (!child->IsInsideTree()) {
            child->PropagateEnterTree();
        }
    }
}

void Node::PropagateExitTree() {
    for (int i = children_.size() - 1; i >= 0; --i) {
        children_[i]->PropagateExitTree();
    }

    // TODO: Signal exit tree to SceneStringNames

    Notification(NotificationType::ExitTree, true);
    if (tree_) {
        Q_EMIT tree_->nodeRemoved(this);
    }

    if (parent_) {
        // TODO: parent signal exit tree
    }

    // TODO: clear viewport

    if (tree_) {
        Q_EMIT tree_->treeChanged();
    }

    inside_tree_ = false;
    tree_ = nullptr;
}

void Node::PropagateAfterExitTree() {
    for (int i = children_.size() - 1; i >= 0; --i) {
        children_[i]->PropagateAfterExitTree();
    }

    // TODO: Signal TreeExited tree to SceneStringNames
}

void Node::MoveChild(Node *child, int index) {
    ERR_FAIL_NULL(child);
    ERR_FAIL_COND_MSG(child->parent_ != this, "Moving child is not a child of this node.");
    ERR_FAIL_INDEX_MSG(index, children_.size(), fmt::format("Invalid new child index {}.", index));

    auto old_index = child->GetIndex();

    if (old_index > index) {
        auto move_from = children_.rend() - old_index - 1;
        auto move_to = children_.rend() - index;
        // right rotate elements from index to old_index
        std::rotate(move_from, children_.rend() - old_index, move_to);
        for (auto i = move_from; i != move_to; ++ i) {
            (*i)->Notification(NotificationType::MovedInParent);
        }
    } else {
        auto move_from = children_.begin() + old_index;
        auto move_to = children_.begin() + index + 1;
        // left rotate elements from old_index to index
        std::rotate(move_from, children_.begin() + old_index + 1, move_to);
        for (auto i = move_from; i != move_to; ++ i) {
            (*i)->Notification(NotificationType::MovedInParent);
        }
    }

    if (tree_)
        Q_EMIT tree_->treeChanged();

    MoveChildNotify(child);
}

std::vector<Node *>::iterator Node::GetIterator() const {
    ERR_FAIL_NULL_V(parent_, std::vector<Node *>::iterator());

    return std::find(parent_->children_.begin(), parent_->children_.end(), this);
}

int Node::GetIndex() const {
    auto it = GetIterator();
    ERR_FAIL_COND_V(it == std::vector<Node *>::iterator(), -1);

    return static_cast<int>(it - parent_->children_.begin());
}

Node::~Node() {
    children_.clear();

    ERR_FAIL_COND(parent_);
    ERR_FAIL_COND(!children_.empty());
}

void Node::SetProcess(bool process) {
    // TODO
}

double Node::GetProcessDeltaTime() const {
    if (tree_) {
        return tree_->GetProcessTime();
    } else {
        return 0;
    }
}

bool Node::IsProcessing() const {
    // TODO
    return false;
}

void Node::SetPhysicsProcess(bool process) {
    // TODO
}

double Node::GetPhysicsProcessDeltaTime() const {
    if (tree_) {
        return tree_->GetPhysicsProcessTime();
    } else {
        return 0;
    }
}

bool Node::IsPhysicsProcessing() const {
    // TODO
    return false;
}

} // End of namespace gobot

GOBOT_REGISTRATION {

    Class_<Node>("Node")
            .constructor()(CtorAsRawPtr)

            .property_readonly("child_count", &Node::GetChildCount)
            .property_readonly("inside_tree", &Node::IsInsideTree)
            .property_readonly("index", &Node::GetIndex)

            .property("name", &Node::GetName, &Node::SetName)

            .method("has_node", &Node::HasNode)
            .method("add_child", &Node::AddChild)
            .method("add_sibling", &Node::AddSibling)
            .method("remove_child", &Node::RemoveChild)
            .method("get_child", &Node::GetChild)
            .method("get_node", &Node::GetNode)
            .method("get_node_or_null", &Node::GetNodeOrNull)
            .method("reparent", &Node::Reparent)
            .method("get_parent", &Node::GetParent)
            .method("get_tree", &Node::GetTree)
            .method("is_ancestor_of", &Node::IsAncestorOf)
            .method("get_path", &Node::GetPath)
            .method("get_path_to", &Node::GetPathTo)
            .method("move_child", &Node::MoveChild)
            ;
};