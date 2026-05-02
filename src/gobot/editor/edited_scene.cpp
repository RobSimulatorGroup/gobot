#include "gobot/editor/edited_scene.hpp"

#include "gobot/core/io/resource_loader.hpp"
#include "gobot/core/io/resource_saver.hpp"
#include "gobot/core/registration.hpp"
#include "gobot/log.hpp"
#include "gobot/scene/node_3d.hpp"
#include "gobot/scene/resources/packed_scene.hpp"
#include "gobot/scene/robot_3d.hpp"

namespace gobot {

EditedScene::EditedScene() {
    SetName("EditedScene");

    NewScene();
}

bool EditedScene::NewScene() {
    auto* root = Object::New<Node3D>();
    root->SetName("Scene");
    return SetRoot(root);
}

Ref<PackedScene> EditedScene::Pack() const {
    if (root_ == nullptr) {
        return {};
    }

    Ref<PackedScene> packed_scene = MakeRef<PackedScene>();
    if (!packed_scene->Pack(root_)) {
        return {};
    }

    return packed_scene;
}

bool EditedScene::SaveToPath(const std::string& path) const {
    Ref<PackedScene> packed_scene = Pack();
    if (!packed_scene.IsValid()) {
        return false;
    }

    USING_ENUM_BITWISE_OPERATORS;
    return ResourceSaver::Save(packed_scene, path,
                               ResourceSaverFlags::ReplaceSubResourcePaths |
                               ResourceSaverFlags::ChangePath);
}

bool EditedScene::LoadFromPath(const std::string& path) {
    return LoadFromPath(path, false);
}

bool EditedScene::LoadFromPath(const std::string& path, bool default_robot_motion_mode) {
    Ref<Resource> resource = ResourceLoader::Load(path, "PackedScene", ResourceFormatLoader::CacheMode::Ignore);
    Ref<PackedScene> packed_scene = dynamic_pointer_cast<PackedScene>(resource);
    if (!packed_scene.IsValid()) {
        LOG_ERROR("EditedScene failed to load '{}': ResourceLoader did not return a PackedScene.", path);
        return false;
    }

    Node* instance = packed_scene->Instantiate();
    auto* node_3d = Object::PointerCastTo<Node3D>(instance);
    if (node_3d == nullptr) {
        LOG_ERROR("EditedScene failed to load '{}': PackedScene instantiate did not return a Node3D root.", path);
        if (instance != nullptr) {
            Object::Delete(instance);
        }
        return false;
    }

    if (!SetRoot(node_3d)) {
        LOG_ERROR("EditedScene failed to load '{}': cannot attach instantiated root '{}'.",
                  path, node_3d->GetName());
        return false;
    }

    if (default_robot_motion_mode) {
        if (auto* robot = Object::PointerCastTo<Robot3D>(root_)) {
            robot->SetMode(RobotMode::Motion);
            LOG_INFO("EditedScene loaded '{}' and set root Robot3D '{}' to Motion mode.",
                     path, robot->GetName());
        } else {
            LOG_ERROR("EditedScene loaded '{}' with default robot motion requested, but root '{}' is not Robot3D.",
                      path, root_->GetName());
        }
    }

    return true;
}

Node3D* EditedScene::AddSceneFromPath(const std::string& path) {
    if (root_ == nullptr) {
        LOG_ERROR("EditedScene cannot add '{}': scene root is null.", path);
        return nullptr;
    }

    Ref<Resource> resource = ResourceLoader::Load(path, "PackedScene", ResourceFormatLoader::CacheMode::Reuse);
    Ref<PackedScene> packed_scene = dynamic_pointer_cast<PackedScene>(resource);
    if (!packed_scene.IsValid()) {
        LOG_ERROR("EditedScene cannot add '{}': ResourceLoader did not return a PackedScene.", path);
        return nullptr;
    }
    if (packed_scene->GetPath().empty()) {
        packed_scene->SetPath(path, false);
    }

    Node* instance = packed_scene->Instantiate();
    auto* node_3d = Object::PointerCastTo<Node3D>(instance);
    if (node_3d == nullptr) {
        LOG_ERROR("EditedScene cannot add '{}': PackedScene instantiate did not return a Node3D root.", path);
        if (instance != nullptr) {
            Object::Delete(instance);
        }
        return nullptr;
    }

    node_3d->SetSceneInstance(packed_scene);
    root_->AddChild(node_3d, true);
    LOG_INFO("Added scene instance '{}' as child '{}' under '{}'.", path, node_3d->GetName(), root_->GetName());
    return node_3d;
}

bool EditedScene::SetRoot(Node3D* root) {
    if (root == nullptr || root == root_) {
        return root != nullptr;
    }

    if (root->GetParent() != nullptr) {
        return false;
    }

    if (root_ != nullptr) {
        Object::Delete(root_);
    }

    root_ = root;
    if (root_->GetName().empty()) {
        root_->SetName("Scene");
    }
    AddChild(root_, true);
    return true;
}

}

GOBOT_REGISTRATION {
    Class_<EditedScene>("EditedScene")
        .constructor()(CtorAsRawPtr)
        .method("pack", &EditedScene::Pack)
        .method("save_to_path", &EditedScene::SaveToPath)
        .method("load_from_path", static_cast<bool (EditedScene::*)(const std::string&)>(&EditedScene::LoadFromPath));
};
