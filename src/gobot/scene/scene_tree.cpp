/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * This file is created by Qiqi Wu, 22-11-20
 * This file is modified by Zikun Yu, 23-1-20
 * SPDX-License-Identifier: Apache-2.0
 */


#include "gobot/scene/scene_tree.hpp"
#include "gobot/scene/window.hpp"
#include "gobot/error_macros.hpp"
#include "gobot/core/events/event.hpp"
#include "gobot/simulation/simulation_server.hpp"

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
        this->RequestQuit();
    });
}

void SceneTree::OnWindowClose() {
    RequestQuit();
}

void SceneTree::RequestQuit() {
    quit_requested_ = true;
}

void SceneTree::ConfirmQuit() {
    quit_requested_ = false;
    quit_ = true;
}

void SceneTree::CancelQuit() {
    quit_requested_ = false;
}

bool SceneTree::PhysicsProcess(double time) {
    physics_process_time_ = time;

    // Physics notifications run before the world step so scripts can update controls for this tick.
    root_->PropagateNotification(NotificationType::PhysicsProcess);

    if (SimulationServer::HasInstance()) {
        SimulationServer::GetInstance()->Step(static_cast<RealType>(time));
    }

    return quit_;
}

bool SceneTree::Process(double time) {
    process_time_ = time;

    MainLoop::Process(time);

    // TODO(wqq): Do we need group
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
