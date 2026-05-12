/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * This file is created by Qiqi Wu, 23-6-11
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

class SDL_Window;

namespace gobot {

class ImGuiRenderer {
public:
    virtual ~ImGuiRenderer() = default;

    virtual void Init(SDL_Window* window) = 0;

    virtual void Shutdown() { }

    virtual void NewFrame() = 0;

    virtual void Render() = 0;

    virtual void Clear() { }

    virtual void RebuildFontTexture() = 0;
};


}
