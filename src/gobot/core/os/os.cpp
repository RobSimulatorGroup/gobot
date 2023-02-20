/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-10
*/

#include "gobot/core/os/os.hpp"
#include "gobot/error_macros.hpp"
#include <chrono>

namespace gobot {

OS* OS::s_singleton = nullptr;

OS::OS() {
    s_singleton = this;
}

OS* OS::GetInstance() {
    ERR_FAIL_COND_V_MSG(s_singleton == nullptr, nullptr, "Must call this after initialize OS");
    return s_singleton;
}

OS::~OS() {
    s_singleton = nullptr;
}

MainLoop* OS::GetMainLoop() const {
    return main_loop_;
}

void OS::SetMainLoop(MainLoop* main_loop) {
    main_loop_ = main_loop;
}

void OS::DeleteMainLoop() {
    if (main_loop_) {
        Object::Delete(main_loop_);
    }
    main_loop_ = nullptr;
}


}
