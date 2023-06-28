/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-6-28
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