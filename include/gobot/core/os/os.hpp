/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-10
*/

#pragma once

#include <gobot_export.h>
#include "gobot/core/os/main_loop.hpp"

namespace gobot {

class MainLoop;

class GOBOT_EXPORT OS {
public:
    OS();

    virtual ~OS();

    static OS* GetInstance() {
        return s_singleton;
    }

    virtual MainLoop* GetMainLoop() const;

    virtual void SetMainLoop(MainLoop* main_loop);

    virtual void DeleteMainLoop();

protected:
    friend class Main;
    static OS* s_singleton;

    MainLoop* main_loop_ = nullptr;
};


}
