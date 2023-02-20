/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-20
*/

#pragma once
#include "gobot/graphics/vertex_buffer.hpp"

namespace gobot {

class GLVertexBuffer : public VertexBuffer
{
private:
    uint32_t m_Handle {};
    BufferUsage m_Usage;
    uint32_t m_Size;
    bool m_Mapped = false;

public:
    explicit GLVertexBuffer(BufferUsage usage);
    ~GLVertexBuffer();

    void Resize(uint32_t size) override;
    void SetData(uint32_t size, const void* data) override;
    void SetDataSub(uint32_t size, const void* data, uint32_t offset) override;

    void ReleasePointer() override;

    void Bind(CommandBuffer* commandBuffer, Pipeline* pipeline) override;
    void Unbind() override;
    uint32_t GetSize() override { return m_Size; }

    static void MakeDefault();

protected:
    static VertexBuffer* CreateFuncGL(const BufferUsage& usage);

protected:
    void* GetPointerInternal() override;
};

}