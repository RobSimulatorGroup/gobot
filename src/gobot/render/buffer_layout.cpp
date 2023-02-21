/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-20
*/


#include "gobot/render/buffer_layout.hpp"

namespace gobot {

BufferLayout::BufferLayout()
        : m_Size(0)
{
}

void BufferLayout::Push(const String& name, RHIFormat format, uint32_t size, bool Normalised)
{
    m_Layout.push_back({name, format, m_Size, Normalised});
    m_Size += size;
}

template <>
void BufferLayout::Push<uint32_t>(const String& name, bool Normalised)
{
    Push(name, RHIFormat::R32_UInt, sizeof(uint32_t), Normalised);
}

template <>
void BufferLayout::Push<uint8_t>(const String& name, bool Normalised)
{
    Push(name, RHIFormat::R8_UInt, sizeof(uint8_t), Normalised);
}

template <>
void BufferLayout::Push<float>(const String& name, bool Normalised)
{
    Push(name, RHIFormat::R32_Float, sizeof(float), Normalised);
}

template <>
void BufferLayout::Push<Vector2f>(const String& name, bool Normalised)
{
    Push(name, RHIFormat::R32G32_Float, sizeof(float) * 2, Normalised);
}

template <>
void BufferLayout::Push<Vector3f>(const String& name, bool Normalised)
{
    Push(name, RHIFormat::R32G32B32_Float, sizeof(float) * 3, Normalised);
}

template <>
void BufferLayout::Push<Vector4f>(const String& name, bool Normalised)
{
    Push(name, RHIFormat::R32G32B32A32_Float, sizeof(float) * 4, Normalised);
}


}
