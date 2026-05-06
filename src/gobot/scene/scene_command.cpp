#include "gobot/scene/scene_command.hpp"

#include <utility>

#include "gobot/log.hpp"
#include "gobot/scene/node.hpp"

namespace gobot {
namespace {

Node* ResolveNode(ObjectID id, const std::string& command_name, const std::string& role) {
    auto* node = Object::PointerCastTo<Node>(ObjectDB::GetInstance(id));
    if (node == nullptr) {
        LOG_ERROR("{} failed: {} node id {} is no longer valid.",
                  command_name,
                  role,
                  static_cast<std::uint64_t>(id));
    }
    return node;
}

} // namespace

class CompoundSceneCommand : public SceneCommand {
public:
    CompoundSceneCommand(std::string name, std::vector<std::unique_ptr<SceneCommand>> commands)
        : name_(std::move(name)),
          commands_(std::move(commands)) {
    }

    bool Do() override {
        std::size_t executed = 0;
        for (; executed < commands_.size(); ++executed) {
            if (!commands_[executed]->Do()) {
                while (executed > 0) {
                    --executed;
                    commands_[executed]->Undo();
                }
                return false;
            }
        }
        return true;
    }

    bool Undo() override {
        for (std::size_t i = commands_.size(); i > 0; --i) {
            if (!commands_[i - 1]->Undo()) {
                return false;
            }
        }
        return true;
    }

    std::string GetName() const override {
        return name_;
    }

private:
    std::string name_;
    std::vector<std::unique_ptr<SceneCommand>> commands_;
};

bool SceneCommand::MergeWith(const SceneCommand&) {
    return false;
}

bool SceneCommandStack::Execute(std::unique_ptr<SceneCommand> command) {
    if (!command) {
        return false;
    }

    if (!command->Do()) {
        return false;
    }

    if (IsInTransaction()) {
        transaction_commands_.push_back(std::move(command));
        ++version_;
        return true;
    }

    return PushExecutedCommand(std::move(command));
}

bool SceneCommandStack::PushExecutedCommand(std::unique_ptr<SceneCommand> command) {
    if (index_ < commands_.size()) {
        commands_.erase(commands_.begin() + static_cast<std::ptrdiff_t>(index_), commands_.end());
    }

    if (index_ > 0 && commands_[index_ - 1]->MergeWith(*command)) {
        ++version_;
        return true;
    }

    commands_.push_back(std::move(command));
    index_ = commands_.size();
    ++version_;
    return true;
}

bool SceneCommandStack::Undo() {
    if (!CanUndo()) {
        return false;
    }

    SceneCommand* command = commands_[index_ - 1].get();
    if (!command->Undo()) {
        return false;
    }

    --index_;
    ++version_;
    return true;
}

bool SceneCommandStack::Redo() {
    if (!CanRedo()) {
        return false;
    }

    SceneCommand* command = commands_[index_].get();
    if (!command->Do()) {
        return false;
    }

    ++index_;
    ++version_;
    return true;
}

void SceneCommandStack::Clear() {
    commands_.clear();
    transaction_commands_.clear();
    transaction_name_.clear();
    index_ = 0;
    clean_index_ = 0;
    ++version_;
}

void SceneCommandStack::MarkClean() {
    clean_index_ = index_;
    ++version_;
}

bool SceneCommandStack::BeginTransaction(std::string name) {
    if (IsInTransaction()) {
        LOG_ERROR("Cannot begin scene transaction '{}': transaction '{}' is already active.",
                  name,
                  transaction_name_);
        return false;
    }

    transaction_name_ = name.empty() ? "Scene Transaction" : std::move(name);
    transaction_commands_.clear();
    return true;
}

bool SceneCommandStack::CommitTransaction() {
    if (!IsInTransaction()) {
        return false;
    }

    std::vector<std::unique_ptr<SceneCommand>> commands = std::move(transaction_commands_);
    std::string name = std::move(transaction_name_);
    transaction_commands_.clear();
    transaction_name_.clear();

    if (commands.empty()) {
        ++version_;
        return true;
    }

    return PushExecutedCommand(std::make_unique<CompoundSceneCommand>(std::move(name), std::move(commands)));
}

bool SceneCommandStack::CancelTransaction() {
    if (!IsInTransaction()) {
        return false;
    }

    bool success = true;
    for (std::size_t i = transaction_commands_.size(); i > 0; --i) {
        if (!transaction_commands_[i - 1]->Undo()) {
            success = false;
        }
    }

    transaction_commands_.clear();
    transaction_name_.clear();
    ++version_;
    return success;
}

bool SceneCommandStack::IsInTransaction() const {
    return !transaction_name_.empty();
}

bool SceneCommandStack::CanUndo() const {
    return !IsInTransaction() && index_ > 0;
}

bool SceneCommandStack::CanRedo() const {
    return !IsInTransaction() && index_ < commands_.size();
}

bool SceneCommandStack::IsDirty() const {
    return IsInTransaction() || index_ != clean_index_;
}

std::size_t SceneCommandStack::GetVersion() const {
    return version_;
}

std::string SceneCommandStack::GetUndoName() const {
    return CanUndo() ? commands_[index_ - 1]->GetName() : std::string();
}

std::string SceneCommandStack::GetRedoName() const {
    return CanRedo() ? commands_[index_]->GetName() : std::string();
}

RenameNodeCommand::RenameNodeCommand(ObjectID node_id, std::string new_name)
    : node_id_(node_id),
      new_name_(std::move(new_name)) {
}

bool RenameNodeCommand::Do() {
    Node* node = ResolveNode(node_id_, GetName(), "target");
    if (node == nullptr) {
        return false;
    }
    if (!captured_old_name_) {
        old_name_ = node->GetName();
        captured_old_name_ = true;
    }
    node->SetName(new_name_);
    return true;
}

bool RenameNodeCommand::Undo() {
    Node* node = ResolveNode(node_id_, GetName(), "target");
    if (node == nullptr || !captured_old_name_) {
        return false;
    }
    node->SetName(old_name_);
    return true;
}

std::string RenameNodeCommand::GetName() const {
    return "Rename Node";
}

bool RenameNodeCommand::MergeWith(const SceneCommand& next) {
    const auto* rename = dynamic_cast<const RenameNodeCommand*>(&next);
    if (rename == nullptr || rename->node_id_ != node_id_) {
        return false;
    }
    new_name_ = rename->new_name_;
    return true;
}

SetNodePropertyCommand::SetNodePropertyCommand(ObjectID node_id,
                                               std::string property_name,
                                               Variant new_value)
    : node_id_(node_id),
      property_name_(std::move(property_name)),
      new_value_(std::move(new_value)) {
}

bool SetNodePropertyCommand::Do() {
    Node* node = ResolveNode(node_id_, GetName(), "target");
    if (node == nullptr) {
        return false;
    }
    if (!captured_old_value_) {
        old_value_ = node->Get(property_name_);
        if (!old_value_.is_valid()) {
            LOG_ERROR("{} failed: property '{}' does not exist on node '{}'.",
                      GetName(),
                      property_name_,
                      node->GetName());
            return false;
        }
        captured_old_value_ = true;
    }
    if (!node->Set(property_name_, new_value_)) {
        LOG_ERROR("{} failed: cannot set property '{}' on node '{}'.",
                  GetName(),
                  property_name_,
                  node->GetName());
        return false;
    }
    return true;
}

bool SetNodePropertyCommand::Undo() {
    Node* node = ResolveNode(node_id_, GetName(), "target");
    if (node == nullptr || !captured_old_value_) {
        return false;
    }
    if (!node->Set(property_name_, old_value_)) {
        LOG_ERROR("{} undo failed: cannot restore property '{}' on node '{}'.",
                  GetName(),
                  property_name_,
                  node->GetName());
        return false;
    }
    return true;
}

std::string SetNodePropertyCommand::GetName() const {
    return "Set Node Property";
}

bool SetNodePropertyCommand::MergeWith(const SceneCommand& next) {
    const auto* set_property = dynamic_cast<const SetNodePropertyCommand*>(&next);
    if (set_property == nullptr ||
        set_property->node_id_ != node_id_ ||
        set_property->property_name_ != property_name_) {
        return false;
    }
    new_value_ = set_property->new_value_;
    return true;
}

AddChildNodeCommand::AddChildNodeCommand(ObjectID parent_id, ObjectID child_id, bool force_readable_name)
    : parent_id_(parent_id),
      child_id_(child_id),
      force_readable_name_(force_readable_name) {
}

bool AddChildNodeCommand::Do() {
    Node* parent = ResolveNode(parent_id_, GetName(), "parent");
    Node* child = ResolveNode(child_id_, GetName(), "child");
    if (parent == nullptr || child == nullptr) {
        return false;
    }
    if (child->GetParent() != nullptr) {
        LOG_ERROR("{} failed: child '{}' already has parent '{}'.",
                  GetName(),
                  child->GetName(),
                  child->GetParent()->GetName());
        return false;
    }
    parent->AddChild(child, force_readable_name_);
    return true;
}

bool AddChildNodeCommand::Undo() {
    Node* parent = ResolveNode(parent_id_, GetName(), "parent");
    Node* child = ResolveNode(child_id_, GetName(), "child");
    if (parent == nullptr || child == nullptr) {
        return false;
    }
    if (child->GetParent() != parent) {
        LOG_ERROR("{} undo failed: child '{}' is no longer parented to '{}'.",
                  GetName(),
                  child->GetName(),
                  parent->GetName());
        return false;
    }
    parent->RemoveChild(child);
    return true;
}

std::string AddChildNodeCommand::GetName() const {
    return "Add Child Node";
}

RemoveChildNodeCommand::RemoveChildNodeCommand(ObjectID parent_id, ObjectID child_id, bool delete_child)
    : parent_id_(parent_id),
      child_id_(child_id),
      detached_child_id_(delete_child ? ObjectID() : child_id),
      delete_child_(delete_child) {
}

bool RemoveChildNodeCommand::Do() {
    Node* parent = ResolveNode(parent_id_, GetName(), "parent");
    Node* child = ResolveNode(delete_child_ ? child_id_ : detached_child_id_, GetName(), "child");
    if (parent == nullptr || child == nullptr) {
        return false;
    }
    if (child->GetParent() != parent) {
        LOG_ERROR("{} failed: child '{}' is not parented to '{}'.",
                  GetName(),
                  child->GetName(),
                  parent->GetName());
        return false;
    }

    if (delete_child_) {
        deleted_child_snapshot_ = MakeRef<PackedScene>();
        if (!deleted_child_snapshot_->Pack(child)) {
            LOG_ERROR("{} failed: cannot snapshot child subtree '{}'.", GetName(), child->GetName());
            deleted_child_snapshot_ = {};
            return false;
        }
    }

    parent->RemoveChild(child);
    if (delete_child_) {
        Object::Delete(child);
        child_id_ = ObjectID();
        detached_child_id_ = ObjectID();
    } else {
        detached_child_id_ = child->GetInstanceId();
    }
    return true;
}

bool RemoveChildNodeCommand::Undo() {
    Node* parent = ResolveNode(parent_id_, GetName(), "parent");
    if (parent == nullptr) {
        return false;
    }

    if (!delete_child_) {
        Node* child = ResolveNode(detached_child_id_, GetName(), "child");
        if (child == nullptr || child->GetParent() != nullptr) {
            return false;
        }
        parent->AddChild(child);
        return true;
    }

    if (!deleted_child_snapshot_.IsValid()) {
        LOG_ERROR("{} undo failed: deleted child snapshot is not available.", GetName());
        return false;
    }

    Node* restored_child = deleted_child_snapshot_->Instantiate();
    if (restored_child == nullptr) {
        LOG_ERROR("{} undo failed: cannot instantiate deleted child snapshot.", GetName());
        return false;
    }
    parent->AddChild(restored_child);
    child_id_ = restored_child->GetInstanceId();
    return true;
}

std::string RemoveChildNodeCommand::GetName() const {
    return delete_child_ ? "Delete Child Node" : "Detach Child Node";
}

ReparentNodeCommand::ReparentNodeCommand(ObjectID node_id, ObjectID new_parent_id)
    : node_id_(node_id),
      new_parent_id_(new_parent_id) {
}

bool ReparentNodeCommand::Do() {
    Node* node = ResolveNode(node_id_, GetName(), "target");
    Node* new_parent = ResolveNode(new_parent_id_, GetName(), "new parent");
    if (node == nullptr || new_parent == nullptr) {
        return false;
    }

    if (!captured_old_parent_) {
        Node* old_parent = node->GetParent();
        if (old_parent == nullptr) {
            LOG_ERROR("{} failed: target node '{}' has no parent.", GetName(), node->GetName());
            return false;
        }
        old_parent_id_ = old_parent->GetInstanceId();
        captured_old_parent_ = true;
    }

    if (node->GetParent() == new_parent) {
        return true;
    }
    node->Reparent(new_parent);
    return true;
}

bool ReparentNodeCommand::Undo() {
    Node* node = ResolveNode(node_id_, GetName(), "target");
    Node* old_parent = ResolveNode(old_parent_id_, GetName(), "old parent");
    if (node == nullptr || old_parent == nullptr || !captured_old_parent_) {
        return false;
    }
    if (node->GetParent() == old_parent) {
        return true;
    }
    node->Reparent(old_parent);
    return true;
}

std::string ReparentNodeCommand::GetName() const {
    return "Reparent Node";
}

} // namespace gobot
