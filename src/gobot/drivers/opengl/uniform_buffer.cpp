/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-20
*/


#include "gobot/render/uniform_buffer.hpp"
#include "gobot/error_macros.hpp"

namespace gobot {

UniformBuffer* (*UniformBuffer::CreateFunc)()                          = nullptr;
UniformBuffer* (*UniformBuffer::CreateDataFunc)(uint32_t, const void*) = nullptr;

UniformBuffer* UniformBuffer::Create()
{
    CRASH_COND_MSG(CreateFunc == nullptr, "No UniformBuffer Create Function");
    return CreateFunc();
}

UniformBuffer* UniformBuffer::Create(uint32_t size, const void* data)
{
    CRASH_COND_MSG(CreateFunc == nullptr, "No UniformBuffer Create Function");
    return CreateDataFunc(size, data);
}

}