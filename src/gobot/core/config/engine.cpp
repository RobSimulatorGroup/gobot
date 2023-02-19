/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-18
*/

#include "gobot/core/config/engine.hpp"
#include "gobot/error_macros.hpp"

namespace gobot {

Engine* Engine::s_singleton = nullptr;


Engine::Engine() {
    s_singleton = this;
}

Engine::~Engine() {
    s_singleton = nullptr;
}

VersionInfo Engine::GetVersionInfo() const {
    // TODO(wqq)
    return {};
}


Engine *Engine::GetInstance() {
    ERR_FAIL_COND_V_MSG(s_singleton == nullptr, nullptr, "Must call this after initialize Engine");
    return s_singleton;
}

}
