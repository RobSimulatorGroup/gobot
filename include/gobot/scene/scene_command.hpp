#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "gobot/core/object.hpp"
#include "gobot/core/math/geometry.hpp"
#include "gobot/scene/resources/packed_scene.hpp"

namespace gobot {

class Node;

class GOBOT_EXPORT SceneCommand {
public:
    virtual ~SceneCommand() = default;

    virtual bool Do() = 0;
    virtual bool Undo() = 0;
    virtual std::string GetName() const = 0;
    virtual bool MergeWith(const SceneCommand& next);
};

class GOBOT_EXPORT SceneCommandStack {
public:
    bool Execute(std::unique_ptr<SceneCommand> command);
    bool Undo();
    bool Redo();

    void Clear();
    void MarkClean();
    bool BeginTransaction(std::string name);
    bool CommitTransaction();
    bool CancelTransaction();
    [[nodiscard]] bool IsInTransaction() const;

    [[nodiscard]] bool CanUndo() const;
    [[nodiscard]] bool CanRedo() const;
    [[nodiscard]] bool IsDirty() const;
    [[nodiscard]] std::size_t GetVersion() const;
    [[nodiscard]] std::string GetUndoName() const;
    [[nodiscard]] std::string GetRedoName() const;

private:
    bool PushExecutedCommand(std::unique_ptr<SceneCommand> command);

    std::vector<std::unique_ptr<SceneCommand>> commands_;
    std::size_t index_{0};
    std::size_t clean_index_{0};
    std::size_t version_{0};
    std::vector<std::unique_ptr<SceneCommand>> transaction_commands_;
    std::string transaction_name_;
};

class GOBOT_EXPORT RenameNodeCommand : public SceneCommand {
public:
    RenameNodeCommand(ObjectID node_id, std::string new_name);

    bool Do() override;
    bool Undo() override;
    std::string GetName() const override;
    bool MergeWith(const SceneCommand& next) override;

private:
    ObjectID node_id_;
    std::string old_name_;
    std::string new_name_;
    bool captured_old_name_{false};
};

class GOBOT_EXPORT SetNodePropertyCommand : public SceneCommand {
public:
    SetNodePropertyCommand(ObjectID node_id, std::string property_name, Variant new_value);

    bool Do() override;
    bool Undo() override;
    std::string GetName() const override;
    bool MergeWith(const SceneCommand& next) override;

private:
    ObjectID node_id_;
    std::string property_name_;
    Variant old_value_;
    Variant new_value_;
    bool captured_old_value_{false};
};

class GOBOT_EXPORT AddChildNodeCommand : public SceneCommand {
public:
    AddChildNodeCommand(ObjectID parent_id, ObjectID child_id, bool force_readable_name);

    bool Do() override;
    bool Undo() override;
    std::string GetName() const override;

private:
    ObjectID parent_id_;
    ObjectID child_id_;
    bool force_readable_name_{false};
};

class GOBOT_EXPORT RemoveChildNodeCommand : public SceneCommand {
public:
    RemoveChildNodeCommand(ObjectID parent_id, ObjectID child_id, bool delete_child);

    bool Do() override;
    bool Undo() override;
    std::string GetName() const override;

private:
    ObjectID parent_id_;
    ObjectID child_id_;
    ObjectID detached_child_id_;
    Ref<PackedScene> deleted_child_snapshot_;
    bool delete_child_{false};
};

class GOBOT_EXPORT AddPackedSceneChildCommand : public SceneCommand {
public:
    AddPackedSceneChildCommand(ObjectID parent_id,
                               Ref<PackedScene> packed_scene,
                               bool force_readable_name,
                               bool mark_scene_instance);

    bool Do() override;
    bool Undo() override;
    std::string GetName() const override;
    [[nodiscard]] ObjectID GetChildId() const;

private:
    ObjectID parent_id_;
    ObjectID child_id_;
    Ref<PackedScene> packed_scene_;
    bool force_readable_name_{false};
    bool mark_scene_instance_{false};
};

class GOBOT_EXPORT ReparentNodeCommand : public SceneCommand {
public:
    ReparentNodeCommand(ObjectID node_id, ObjectID new_parent_id);

    bool Do() override;
    bool Undo() override;
    std::string GetName() const override;

private:
    ObjectID node_id_;
    ObjectID old_parent_id_;
    ObjectID new_parent_id_;
    bool captured_old_parent_{false};
};

class GOBOT_EXPORT SetNode3DTransformCommand : public SceneCommand {
public:
    SetNode3DTransformCommand(ObjectID node_id, Affine3 new_transform, bool global);

    bool Do() override;
    bool Undo() override;
    std::string GetName() const override;
    bool MergeWith(const SceneCommand& next) override;

private:
    ObjectID node_id_;
    Affine3 old_transform_{Affine3::Identity()};
    Affine3 new_transform_{Affine3::Identity()};
    bool global_{false};
    bool captured_old_transform_{false};
};

} // namespace gobot
