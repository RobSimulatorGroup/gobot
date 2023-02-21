/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-20
*/

#include "gobot/render/swap_chain.hpp"
#include "gobot/error_macros.hpp"

namespace gobot {

SwapChain* (*SwapChain::CreateFunc)(uint32_t, uint32_t) = nullptr;

SwapChain* SwapChain::Create(uint32_t width, uint32_t height)
{
    CRASH_COND_MSG(CreateFunc == nullptr, "No SwapChain Create Function");
    return CreateFunc(width, height);
}

}
