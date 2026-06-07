#include "gobot/main/runtime_scene_owner.hpp"

#include "gobot/core/object.hpp"
#include "gobot/scene/node.hpp"
#include "gobot/scene/scene_tree.hpp"
#include "gobot/scene/window.hpp"

namespace gobot {

RuntimeSceneOwner::~RuntimeSceneOwner() {
    Clear();
}

void RuntimeSceneOwner::Attach(Node* scene_root) {
    if (scene_root == nullptr || scene_root->IsInsideTree()) {
        return;
    }

    Clear();
    runtime_tree_ = Object::New<SceneTree>(false);
    scene_root_ = scene_root;
    runtime_tree_->Initialize();
    runtime_tree_->GetRoot()->AddChild(scene_root_, false);
}

void RuntimeSceneOwner::Clear() {
    if (runtime_tree_ == nullptr) {
        scene_root_ = nullptr;
        return;
    }

    Node* tree_root = runtime_tree_->GetRoot();
    if (scene_root_ != nullptr && scene_root_->GetParent() == tree_root) {
        tree_root->RemoveChild(scene_root_);
    }
    runtime_tree_->Finalize();
    Object::Delete(runtime_tree_);
    runtime_tree_ = nullptr;
    scene_root_ = nullptr;
}

} // namespace gobot
