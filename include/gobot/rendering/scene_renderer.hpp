/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * This file is created by Qiqi Wu, 23-3-19
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "gobot/core/rid.hpp"

namespace gobot {

class Texture;
class Camera3D;
class SceneTree;

class SceneRenderer {
public:
    SceneRenderer();

    ~SceneRenderer();

    void SetRenderTarget(const RID& texture_rid);

    void OnRenderer(const SceneTree* scene_tree);

    void FinalPass();

    void GridPass();

    void DebugPass();

private:
    Camera3D* camera_ = nullptr;

    RID view_texture_{};
    RID view_frame_buffer_{};
};


}
