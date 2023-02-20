/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-20
*/

#pragma once

#include "gobot/graphics/index_buffer.hpp"

namespace gobot {

class GLIndexBuffer : public IndexBuffer
{
private:
    uint32_t m_Handle;
    uint32_t m_Count;
    BufferUsage m_Usage;
    bool m_Mapped = false;

public:
    GLIndexBuffer(uint16_t* data, uint32_t count, BufferUsage bufferUsage);
    GLIndexBuffer(uint32_t* data, uint32_t count, BufferUsage bufferUsage);
    ~GLIndexBuffer();

    void Bind(CommandBuffer* commandBuffer) const override;
    void Unbind() const override;
    uint32_t GetCount() const override;
    void SetCount(uint32_t m_index_count) override { m_Count = m_index_count; };

    void* GetPointerInternal() override;
    void ReleasePointer() override;

    static void MakeDefault();

protected:
    static IndexBuffer* CreateFuncGL(uint32_t* data, uint32_t count, BufferUsage bufferUsage);
    static IndexBuffer* CreateFunc16GL(uint16_t* data, uint32_t count, BufferUsage bufferUsage);
};

}