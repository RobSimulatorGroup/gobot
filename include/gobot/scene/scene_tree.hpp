/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * This file is created by Qiqi Wu, 22-11-20
 * This file is modified by Zikun Yu, 23-1-20
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "gobot/core/object.hpp"
#include "gobot/scene/node.hpp"
#include "gobot/core/os/main_loop.hpp"
#include "gobot/core/events/window_event.hpp"

namespace gobot {

class Window;
class Node;

class GOBOT_EXPORT SceneTree : public MainLoop {
    GOBCLASS(SceneTree, MainLoop)
public:
    SceneTree(bool p_init_window = true);

    ~SceneTree() override;

    [[nodiscard]] FORCE_INLINE Window* GetRoot() const { return root_; }

    [[nodiscard]] int GetNodeCount() const;

    static SceneTree* GetInstance();

    void Initialize() override;

    void Finalize() override;

    bool PhysicsProcess(double time) override;

    bool Process(double time) override;

    void PullEvent() override;

    [[nodiscard]] FORCE_INLINE double GetPhysicsProcessTime() const { return physics_process_time_; }

    [[nodiscard]] FORCE_INLINE double GetProcessTime() const { return process_time_; }

private:
    void OnWindowClose();

private:
    friend class Node;
    static SceneTree *s_singleton;

    bool quit_ = false;

    Window *root_ = nullptr;
    int node_count_ = 0;

    double physics_process_time_ = 0.0;
    double process_time_ = 0.0;
};

}
