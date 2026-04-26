#include "gobot/editor/edited_scene.hpp"

#include "gobot/core/io/resource_loader.hpp"
#include "gobot/core/io/resource_saver.hpp"
#include "gobot/core/registration.hpp"
#include "gobot/scene/node_3d.hpp"
#include "gobot/scene/resources/packed_scene.hpp"

namespace gobot {

EditedScene::EditedScene() {
    SetName("EditedScene");

    root_ = Object::New<Node3D>();
    root_->SetName("Scene");
    AddChild(root_, true);
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
    Ref<Resource> resource = ResourceLoader::Load(path, "PackedScene", ResourceFormatLoader::CacheMode::Ignore);
    Ref<PackedScene> packed_scene = dynamic_pointer_cast<PackedScene>(resource);
    if (!packed_scene.IsValid()) {
        return false;
    }

    Node* instance = packed_scene->Instantiate();
    auto* node_3d = Object::PointerCastTo<Node3D>(instance);
    if (node_3d == nullptr) {
        if (instance != nullptr) {
            Object::Delete(instance);
        }
        return false;
    }

    return SetRoot(node_3d);
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
        .method("load_from_path", &EditedScene::LoadFromPath);
};
