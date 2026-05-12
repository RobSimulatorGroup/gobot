/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * This file is created by Qiqi Wu, 22-12-21
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/scene/resources/mesh.hpp"
#include "gobot/core/registration.hpp"

namespace gobot {

Mesh::Mesh() {

}

}

GOBOT_REGISTRATION {
    Class_<Mesh>("Mesh")
        .constructor()(CtorAsRawPtr);

    gobot::Type::register_wrapper_converter_for_base_classes<Ref<Mesh>, Ref<Resource>>();

};


