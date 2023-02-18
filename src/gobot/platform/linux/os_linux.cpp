/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-18
*/

#include "gobot/platform/linux/os_linux.hpp"
#include "gobot/scene/window.hpp"
#include "gobot/main/main.hpp"
#include "gobot/log.hpp"

namespace gobot {

void LinuxOS::Run()
{
    if (!main_loop_) {
        return;
    }

    main_loop_->Initialize();

    while (true) {
        OS::GetInstance()->GetMainLoop()->PullEvent();
        if (Main::Iteration()) {
            break;
        }
    }

    main_loop_->Finalize();
}



}