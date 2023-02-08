/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-8
*/

#include "gobot/core/os/window.hpp"
#include "gobot/error_macros.hpp"

namespace gobot {

Window* (*Window::CreateFunc)(const WindowDesc&) = nullptr;

Window* Window::Create(const WindowDesc& windowDesc) {
    ERR_FAIL_COND_V(CreateFunc != nullptr, nullptr);
    return CreateFunc(windowDesc);
}

}