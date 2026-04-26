#include "gobot/editor/edited_scene.hpp"

#include "gobot/core/registration.hpp"
#include "gobot/scene/node_3d.hpp"

namespace gobot {

EditedScene::EditedScene() {
    SetName("EditedScene");

    root_ = Object::New<Node3D>();
    root_->SetName("Scene");
    AddChild(root_, true);
}

}

GOBOT_REGISTRATION {
    Class_<EditedScene>("EditedScene")
        .constructor()(CtorAsRawPtr);
};
