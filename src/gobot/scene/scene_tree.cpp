/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 22-11-20
 * This file is modified by Zikun Yu, 23-1-20
*/


#include "gobot/scene/scene_tree.hpp"
#include "gobot/scene/window.hpp"
#include "gobot/error_macros.hpp"
#include "gobot/core/events/event.hpp"

namespace gobot {

SceneTree *SceneTree::s_singleton = nullptr;

void SceneTree::Initialize()
{
    ERR_FAIL_COND(!root_);
    root_->SetTree(this);
    MainLoop::Initialize();
}

SceneTree* SceneTree::GetInstance() {
    ERR_FAIL_COND_V_MSG(s_singleton == nullptr, nullptr, "Must call this after initialize SceneTree");
    return s_singleton;
}

void SceneTree::Finalize()
{
    MainLoop::Finalize();

    if (root_) {
        root_->SetTree(nullptr);
        root_->PropagateAfterExitTree();
        Node::Delete(root_);
        root_ = nullptr;
    }
}

int SceneTree::GetNodeCount() const {
    return node_count_;
}

SceneTree::SceneTree(bool p_init_window) {
    if (s_singleton == nullptr) {
        s_singleton = this;
    }

    root_ = Node::New<Window>(p_init_window);
    root_->SetName("root");

    Event::Subscribe(EventType::WindowClose, [this](const Event& event){
        this->quit_ = true;
    });
}

void SceneTree::OnWindowClose() {
    quit_ = true;
}

bool SceneTree::PhysicsProcess(double time) {

    return quit_;
}

bool SceneTree::Process(double time) {
    process_time_ = time;

    MainLoop::Process(time);

    // TODO(wqq): Do we need group
    root_->PropagateNotification(NotificationType::PhysicsProcess);
    root_->PropagateNotification(NotificationType::Process);

    return quit_;
}


void SceneTree::PullEvent() {
    root_->PullEvent();
}


SceneTree::~SceneTree() {
    if (root_) {
        root_->SetTree(nullptr);
        root_->PropagateAfterExitTree();
        Node::Delete(root_);
        root_ = nullptr;
    }

    if (s_singleton == this) {
        s_singleton = nullptr;
    }
}

} // End of namespace gobot
