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

namespace gobot {

SceneTree *SceneTree::s_singleton = nullptr;

void SceneTree::Initialize()
{
    ERR_FAIL_COND(!root);
    root->SetTree(this);
    MainLoop::Initialize();
}

SceneTree* SceneTree::GetInstance() {
    ERR_FAIL_COND_V_MSG(s_singleton == nullptr, nullptr, "Must call this after initialize SceneTree");
    return s_singleton;
}

void SceneTree::Finalize()
{
    MainLoop::Finalize();

    if (root) {
        root->SetTree(nullptr);
        root->PropagateAfterExitTree();
        Node::Delete(root);
    }
}

int SceneTree::GetNodeCount() const {
    return node_count;
}

SceneTree::SceneTree() {
    if (s_singleton == nullptr) {
        s_singleton = this;
    }

    root = Node::New<Window>();
    root->SetName("root");

    Object::connect(root, &Window::windowCloseRequested, this, &SceneTree::OnWindowClose);
    Object::connect(root, &Window::windowResizeRequested, this, &SceneTree::OnWindowResize);
}

void SceneTree::OnWindowResize(WindowResizeEvent& e)
{
    if (e.GetWidth() == 0 || e.GetHeight() == 0) {
        return;
    }
//    Renderer::OnWindowResize(e.GetWidth(), e.GetHeight());
}

void SceneTree::OnWindowClose() {
    quit_ = true;
}


bool SceneTree::PhysicsProcess(double time) {

    return quit_;
}

bool SceneTree::Process(double time) {

    return quit_;
}


void SceneTree::PullEvent() {
    root->PullEvent();
}


SceneTree::~SceneTree() {
    if (root) {
        root->SetTree(nullptr);
        root->PropagateAfterExitTree();
        Node::Delete(root);
    }

    if (s_singleton == this) {
        s_singleton = nullptr;
    }
}

} // End of namespace gobot
