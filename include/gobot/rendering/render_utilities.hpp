/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * This file is created by Qiqi Wu, 23-6-28
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "gobot/core/rid.hpp"

namespace gobot {

class RendererUtilities {
public:
    virtual ~RendererUtilities() {}

    virtual bool Free(RID p_rid) = 0;


};



}