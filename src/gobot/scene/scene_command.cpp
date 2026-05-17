#include "gobot/scene/scene_command.hpp"

#include <filesystem>
#include <fstream>
#include <utility>

#include "gobot/core/config/project_setting.hpp"
#include "gobot/core/io/resource.hpp"
#include "gobot/core/io/resource_loader.hpp"
#include "gobot/core/string_utils.hpp"
#include "gobot/core/types.hpp"
#include "gobot/log.hpp"
#include "gobot/scene/node.hpp"
#include "gobot/scene/node_3d.hpp"

namespace gobot {
namespace {

Object* ResolveObject(ObjectID id, const std::string& command_name, const std::string& role) {
    auto* object = ObjectDB::GetInstance(id);
    if (object == nullptr) {
        LOG_ERROR("{} failed: {} object id {} is no longer valid.",
                  command_name,
                  role,
                  static_cast<std::uint64_t>(id));
    }
    return object;
}

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

    bool AffectsSceneDirtyState() const override {
        for (const auto& command : commands_) {
            if (command != nullptr && command->AffectsSceneDirtyState()) {
                return true;
            }
        }
        return false;
    }

private:
    std::string name_;
    std::vector<std::unique_ptr<SceneCommand>> commands_;
};

bool SceneCommand::MergeWith(const SceneCommand&) {
    return false;
}

bool SceneCommand::AffectsSceneDirtyState() const {
    return true;
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
    if (IsInTransaction()) {
        for (const auto& command : transaction_commands_) {
            if (command != nullptr && command->AffectsSceneDirtyState()) {
                return true;
            }
        }
    }

    if (index_ == clean_index_) {
        return false;
    }

    if (index_ > clean_index_) {
        return HasDirtyingCommandBetween(clean_index_, index_);
    }

    return HasDirtyingCommandBetween(index_, clean_index_);
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

bool SceneCommandStack::HasDirtyingCommandBetween(std::size_t first, std::size_t last) const {
    last = std::min(last, commands_.size());
    for (std::size_t i = first; i < last; ++i) {
        if (commands_[i] != nullptr && commands_[i]->AffectsSceneDirtyState()) {
            return true;
        }
    }
    return false;
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

SetObjectPropertyCommand::SetObjectPropertyCommand(ObjectID object_id,
                                                   std::string property_name,
                                                   Variant new_value)
    : object_id_(object_id),
      property_name_(std::move(property_name)),
      new_value_(std::move(new_value)) {
}

bool SetObjectPropertyCommand::Do() {
    Object* object = ResolveObject(object_id_, GetName(), "target");
    if (object == nullptr) {
        return false;
    }

    if (!captured_old_value_) {
        old_value_ = object->Get(property_name_);
        if (!old_value_.is_valid()) {
            LOG_ERROR("{} failed: property '{}' does not exist on object '{}'.",
                      GetName(),
                      property_name_,
                      object->GetClassStringName());
            return false;
        }
        captured_old_value_ = true;
    }

    if (!object->Set(property_name_, new_value_)) {
        LOG_ERROR("{} failed: cannot set property '{}' on object '{}'.",
                  GetName(),
                  property_name_,
                  object->GetClassStringName());
        return false;
    }
    return true;
}

bool SetObjectPropertyCommand::Undo() {
    Object* object = ResolveObject(object_id_, GetName(), "target");
    if (object == nullptr || !captured_old_value_) {
        return false;
    }

    if (!object->Set(property_name_, old_value_)) {
        LOG_ERROR("{} undo failed: cannot restore property '{}' on object '{}'.",
                  GetName(),
                  property_name_,
                  object->GetClassStringName());
        return false;
    }
    return true;
}

std::string SetObjectPropertyCommand::GetName() const {
    return "Set Object Property";
}

bool SetObjectPropertyCommand::MergeWith(const SceneCommand& next) {
    const auto* set_property = dynamic_cast<const SetObjectPropertyCommand*>(&next);
    if (set_property == nullptr ||
        set_property->object_id_ != object_id_ ||
        set_property->property_name_ != property_name_) {
        return false;
    }
    new_value_ = set_property->new_value_;
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

AddPackedSceneChildCommand::AddPackedSceneChildCommand(ObjectID parent_id,
                                                       Ref<PackedScene> packed_scene,
                                                       bool force_readable_name,
                                                       bool mark_scene_instance)
    : parent_id_(parent_id),
      packed_scene_(std::move(packed_scene)),
      force_readable_name_(force_readable_name),
      mark_scene_instance_(mark_scene_instance) {
}

bool AddPackedSceneChildCommand::Do() {
    Node* parent = ResolveNode(parent_id_, GetName(), "parent");
    if (parent == nullptr || !packed_scene_.IsValid()) {
        return false;
    }

    Node* child = nullptr;
    if (child_id_.IsValid()) {
        child = ResolveNode(child_id_, GetName(), "child");
    } else {
        child = packed_scene_->Instantiate();
        if (child == nullptr) {
            LOG_ERROR("{} failed: cannot instantiate packed scene child.", GetName());
            return false;
        }
        child_id_ = child->GetInstanceId();
        if (mark_scene_instance_) {
            child->SetSceneInstance(packed_scene_);
        }
    }

    if (child == nullptr || child->GetParent() != nullptr) {
        LOG_ERROR("{} failed: child is unavailable or already parented.", GetName());
        return false;
    }

    parent->AddChild(child, force_readable_name_);
    return true;
}

bool AddPackedSceneChildCommand::Undo() {
    Node* parent = ResolveNode(parent_id_, GetName(), "parent");
    Node* child = ResolveNode(child_id_, GetName(), "child");
    if (parent == nullptr || child == nullptr || child->GetParent() != parent) {
        return false;
    }
    parent->RemoveChild(child);
    return true;
}

std::string AddPackedSceneChildCommand::GetName() const {
    return "Add Packed Scene Child";
}

ObjectID AddPackedSceneChildCommand::GetChildId() const {
    return child_id_;
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

SetNode3DTransformCommand::SetNode3DTransformCommand(ObjectID node_id, Affine3 new_transform, bool global)
    : node_id_(node_id),
      new_transform_(std::move(new_transform)),
      global_(global) {
}

bool SetNode3DTransformCommand::Do() {
    auto* node = Object::PointerCastTo<Node3D>(ResolveNode(node_id_, GetName(), "target"));
    if (node == nullptr) {
        return false;
    }
    if (!captured_old_transform_) {
        old_transform_ = global_ ? node->GetGlobalTransform() : node->GetTransform();
        captured_old_transform_ = true;
    }
    if (global_) {
        node->SetGlobalTransform(new_transform_);
    } else {
        node->SetTransform(new_transform_);
    }
    return true;
}

bool SetNode3DTransformCommand::Undo() {
    auto* node = Object::PointerCastTo<Node3D>(ResolveNode(node_id_, GetName(), "target"));
    if (node == nullptr || !captured_old_transform_) {
        return false;
    }
    if (global_) {
        node->SetGlobalTransform(old_transform_);
    } else {
        node->SetTransform(old_transform_);
    }
    return true;
}

std::string SetNode3DTransformCommand::GetName() const {
    return "Set Node3D Transform";
}

bool SetNode3DTransformCommand::MergeWith(const SceneCommand& next) {
    const auto* set_transform = dynamic_cast<const SetNode3DTransformCommand*>(&next);
    if (set_transform == nullptr ||
        set_transform->node_id_ != node_id_ ||
        set_transform->global_ != global_) {
        return false;
    }
    new_transform_ = set_transform->new_transform_;
    return true;
}

RenameResourceFileCommand::RenameResourceFileCommand(std::string old_path, std::string new_path)
    : old_path_(std::move(old_path)),
      new_path_(std::move(new_path)) {
}

RenameResourceFileCommand::RenameResourceFileCommand(std::string old_path,
                                                     std::string new_path,
                                                     ObjectID scene_root_id)
    : old_path_(std::move(old_path)),
      new_path_(std::move(new_path)),
      scene_root_id_(scene_root_id) {
}

bool RenameResourceFileCommand::Do() {
    return Rename(old_path_, new_path_, "Rename Resource File");
}

bool RenameResourceFileCommand::Undo() {
    return Rename(new_path_, old_path_, "Undo Rename Resource File");
}

std::string RenameResourceFileCommand::GetName() const {
    return "Rename Resource File";
}

bool RenameResourceFileCommand::AffectsSceneDirtyState() const {
    return false;
}

bool RenameResourceFileCommand::Rename(const std::string& from_path,
                                       const std::string& to_path,
                                       const char* phase) {
    if (from_path.empty() || to_path.empty()) {
        LOG_ERROR("{} failed: empty resource path.", phase);
        return false;
    }
    if (from_path == to_path) {
        return true;
    }

    auto* settings = ProjectSettings::GetInstance();
    const std::filesystem::path from_global = settings->GlobalizePath(from_path);
    const std::filesystem::path to_global = settings->GlobalizePath(to_path);
    if (!std::filesystem::exists(from_global)) {
        LOG_ERROR("{} failed: source resource does not exist '{}'.", phase, from_path);
        return false;
    }
    if (std::filesystem::exists(to_global)) {
        LOG_ERROR("{} failed: target resource already exists '{}'.", phase, to_path);
        return false;
    }

    std::error_code error;
    std::filesystem::rename(from_global, to_global, error);
    if (error) {
        LOG_ERROR("{} failed: cannot rename '{}' to '{}': {}",
                  phase,
                  from_path,
                  to_path,
                  error.message());
        return false;
    }

    Ref<Resource> cached_resource = ResourceCache::GetRef(from_path);
    if (cached_resource.IsValid()) {
        cached_resource->SetPath(to_path, true);
    }

    UpdateOpenSceneReferences(from_path, to_path);
    UpdateProjectSceneFileReferences(from_path, to_path);

    return true;
}

void RenameResourceFileCommand::UpdateOpenSceneReferences(const std::string& from_path,
                                                          const std::string& to_path) const {
    if (scene_root_id_ == ObjectID{}) {
        return;
    }

    Node* root = ResolveNode(scene_root_id_, GetName(), "scene root");
    if (root == nullptr) {
        return;
    }

    Ref<Resource> replacement = ResourceCache::GetRef(to_path);
    auto update_node = [&](Node* node, auto&& update_node_ref) -> void {
        if (node == nullptr) {
            return;
        }

        Ref<PythonScript> script = node->GetScript();
        if (script.IsValid() && script->GetPath() == from_path) {
            Ref<PythonScript> replacement_script = dynamic_pointer_cast<PythonScript>(replacement);
            if (!replacement_script.IsValid()) {
                replacement_script = dynamic_pointer_cast<PythonScript>(
                        ResourceLoader::Load(to_path, "PythonScript", ResourceFormatLoader::CacheMode::Reuse));
            }
            if (replacement_script.IsValid()) {
                node->SetScript(replacement_script);
            }
        }

        Ref<PackedScene> scene_instance = node->GetSceneInstance();
        if (scene_instance.IsValid() && scene_instance->GetPath() == from_path) {
            Ref<PackedScene> replacement_scene = dynamic_pointer_cast<PackedScene>(replacement);
            if (!replacement_scene.IsValid()) {
                replacement_scene = dynamic_pointer_cast<PackedScene>(
                        ResourceLoader::Load(to_path, "PackedScene", ResourceFormatLoader::CacheMode::Reuse));
            }
            if (replacement_scene.IsValid()) {
                node->SetSceneInstance(replacement_scene);
            }
        }

        for (std::size_t i = 0; i < node->GetChildCount(); ++i) {
            update_node_ref(node->GetChild(static_cast<int>(i)), update_node_ref);
        }
    };

    update_node(root, update_node);
}

void RenameResourceFileCommand::UpdateProjectSceneFileReferences(const std::string& from_path,
                                                                 const std::string& to_path) const {
    auto* settings = ProjectSettings::GetInstance();
    if (settings == nullptr || settings->GetProjectPath().empty()) {
        return;
    }

    const std::filesystem::path project_root = settings->GetProjectPath();
    std::error_code error;
    if (!std::filesystem::exists(project_root, error)) {
        return;
    }

    for (std::filesystem::recursive_directory_iterator it(project_root, error), end; it != end; it.increment(error)) {
        if (error) {
            LOG_WARN("Skipping project resource reference scan entry: {}", error.message());
            error.clear();
            continue;
        }
        if (!it->is_regular_file(error) || ToLower(it->path().extension().string()) != ".jscn") {
            continue;
        }

        std::ifstream input(it->path());
        if (!input.is_open()) {
            LOG_WARN("Cannot scan scene references in '{}': failed to open file.", it->path().string());
            continue;
        }

        Json scene_json;
        try {
            input >> scene_json;
        } catch (const std::exception& e) {
            LOG_WARN("Cannot scan scene references in '{}': {}", it->path().string(), e.what());
            continue;
        }

        Json* ext_resources = scene_json.contains("__EXT_RESOURCES__") ? &scene_json["__EXT_RESOURCES__"] : nullptr;
        if (ext_resources == nullptr || !ext_resources->is_array()) {
            continue;
        }

        bool changed = false;
        const std::string scene_local_path = settings->LocalizePath(it->path().string());
        for (auto& ext_resource : *ext_resources) {
            if (!ext_resource.is_object() || !ext_resource.contains("__PATH__") || !ext_resource["__PATH__"].is_string()) {
                continue;
            }

            std::string resource_path = ext_resource["__PATH__"];
            std::string resolved_path = resource_path;
            if (resolved_path.find("://") == std::string::npos && IsRelativePath(resolved_path)) {
                resolved_path = settings->LocalizePath(PathJoin(GetBaseDir(scene_local_path), resolved_path));
            }

            if (resolved_path == from_path) {
                ext_resource["__PATH__"] = to_path;
                changed = true;
            }
        }

        if (!changed) {
            continue;
        }

        std::ofstream output(it->path(), std::ios::out | std::ios::trunc);
        if (!output.is_open()) {
            LOG_ERROR("Cannot update scene resource references in '{}'.", it->path().string());
            continue;
        }
        output << scene_json.dump(4);
        if (!output.good()) {
            LOG_ERROR("Failed to write updated scene resource references in '{}'.", it->path().string());
            continue;
        }
        LOG_INFO("Updated scene resource references in '{}': {} -> {}",
                 scene_local_path,
                 from_path,
                 to_path);
    }
}

} // namespace gobot
