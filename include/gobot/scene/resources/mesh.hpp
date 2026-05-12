/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * This file is created by Qiqi Wu, 22-12-21
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "gobot/core/io/resource.hpp"

namespace gobot {

class GOBOT_EXPORT Mesh : public Resource {
    GOBCLASS(Mesh, Resource);
public:
    Mesh();

};

}
