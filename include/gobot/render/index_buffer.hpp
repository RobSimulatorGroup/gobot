/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-20
*/

#pragma once

#include <gobot_export.h>

#include "definitions.hpp"

namespace gobot {

class GOBOT_EXPORT IndexBuffer
{
public:
    virtual ~IndexBuffer()                                          = default;
    virtual void Bind(CommandBuffer* commandBuffer = nullptr) const = 0;
    virtual void Unbind() const                                     = 0;

    virtual uint32_t GetCount() const = 0;
    virtual uint32_t GetSize() const { return 0; }
    virtual void SetCount(uint32_t m_index_count) = 0;
    virtual void ReleasePointer() {};

    template <typename T>
    T* GetPointer()
    {
        return static_cast<T*>(GetPointerInternal());
    }

public:
    static IndexBuffer* Create(uint16_t* data, uint32_t count, BufferUsage bufferUsage = BufferUsage::STATIC);
    static IndexBuffer* Create(uint32_t* data, uint32_t count, BufferUsage bufferUsage = BufferUsage::STATIC);

protected:
    virtual void* GetPointerInternal() { return nullptr; }

    static IndexBuffer* (*Create16Func)(uint16_t*, uint32_t, BufferUsage);
    static IndexBuffer* (*CreateFunc)(uint32_t*, uint32_t, BufferUsage);
};

}