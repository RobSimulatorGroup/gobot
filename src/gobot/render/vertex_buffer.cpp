/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-20
*/

#include "gobot/render/vertex_buffer.hpp"
#include "gobot/error_macros.hpp"

namespace gobot {

VertexBuffer* (*VertexBuffer::CreateFunc)(const BufferUsage&) = nullptr;

VertexBuffer* VertexBuffer::Create(const BufferUsage& usage)
{
    CRASH_COND_MSG(CreateFunc == nullptr, "No VertexBuffer Create Function");
    return CreateFunc(usage);
}

}