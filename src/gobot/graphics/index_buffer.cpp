/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-20
*/

#include "gobot/graphics/index_buffer.hpp"
#include "gobot/log.hpp"
#include "gobot/error_macros.hpp"

namespace gobot {

IndexBuffer* (*IndexBuffer::Create16Func)(uint16_t*, uint32_t, BufferUsage) = nullptr;
IndexBuffer* (*IndexBuffer::CreateFunc)(uint32_t*, uint32_t, BufferUsage)   = nullptr;

IndexBuffer* IndexBuffer::Create(uint16_t* data, uint32_t count, BufferUsage bufferUsage)
{
    CRASH_COND_MSG(CreateFunc == nullptr, "No IndexBuffer Create Function");
    return Create16Func(data, count, bufferUsage);
}

IndexBuffer* IndexBuffer::Create(uint32_t* data, uint32_t count, BufferUsage bufferUsage)
{
    CRASH_COND_MSG(CreateFunc == nullptr, "No IndexBuffer Create Function");
    return CreateFunc(data, count, bufferUsage);
}


}