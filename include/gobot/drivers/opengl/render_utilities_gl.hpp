/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * This file is created by Qiqi Wu, 23-6-28
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "gobot/rendering/render_utilities.hpp"

namespace gobot::opengl {

class GLRendererUtilities : public RendererUtilities {
public:
    GLRendererUtilities();

    ~GLRendererUtilities();

    bool Free(RID p_rid) override;

    static RendererUtilities* GetInstance() { return s_singleton; }

private:
    static RendererUtilities *s_singleton;
};



}