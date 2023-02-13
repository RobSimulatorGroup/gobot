/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-12
*/

#pragma once

#include "gobot/drivers/opengl/gl.hpp"

#if GL_DEBUG
#define GLCall(x)                          \
    GLClearError();                        \
    x;                                     \
    if(!GLLogCall(#x, __FILE__, __LINE__)) \
        LUMOS_BREAK();
#else
#define GLCall(x) x
#endif