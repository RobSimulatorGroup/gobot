/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * This file is created by Qiqi Wu, 23-1-13
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "gobot/core/io/resource.hpp"

namespace gobot {


// Shape3D is for collision checker
class GOBOT_EXPORT Shape3D : public Resource {
    GOBCLASS(Shape3D, Resource)
public:
    Shape3D();

    ~Shape3D();

};


}
