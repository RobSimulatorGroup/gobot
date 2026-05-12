/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * This file is created by Qiqi Wu, 23-6-11
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace gobot {

enum class RendererType {
    None,
    OpenGL46,
    OpenGLES32,
    Vulkan
};

enum class AttachmentAccess
{
    Read,      //!< Read
    Write,     //!< Write
    ReadWrite, //!< Read and write

    Count
};


}
