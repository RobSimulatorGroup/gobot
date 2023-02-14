/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-10
*/

#pragma once

#include "gobot/core/object.hpp"

namespace gobot {

class GOBOT_EXPORT MainLoop : public Object {
    GOBCLASS(MainLoop, Object)
public:
    MainLoop() = default;

    ~MainLoop() override = default;

    virtual void Initialize();

    virtual bool PhysicsProcess(double time);

    virtual bool Process(double time);

    virtual void Finalize();
};



}