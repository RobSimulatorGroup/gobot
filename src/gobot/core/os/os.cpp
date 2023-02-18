/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-10
*/

#include "gobot/core/os/os.hpp"

namespace gobot {

OS::OS()
{

}

OS::~OS()
{

}

MainLoop* OS::GetMainLoop() const
{
    return main_loop_;
}

void OS::SetMainLoop(MainLoop* main_loop)
{
    main_loop_ = main_loop;
}

void OS::DeleteMainLoop()
{
    if (main_loop_) {
        Object::Delete(main_loop_);
    }
    main_loop_ = nullptr;
}


}
