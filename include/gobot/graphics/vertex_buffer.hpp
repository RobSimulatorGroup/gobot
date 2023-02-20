/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-20
*/

#pragma once

#include "definitions.hpp"

namespace gobot {

class VertexBuffer
{
        public:
        virtual ~VertexBuffer()                                                   = default;
        virtual void Resize(uint32_t size)                                        = 0;
        virtual void SetData(uint32_t size, const void* data)                     = 0;
        virtual void SetDataSub(uint32_t size, const void* data, uint32_t offset) = 0;
        virtual void ReleasePointer()                                             = 0;
        virtual void Bind(CommandBuffer* commandBuffer, Pipeline* pipeline)       = 0;
        virtual void Unbind()                                                     = 0;
        virtual uint32_t GetSize() { return 0; }

        template <typename T>
        T* GetPointer()
        {
            return static_cast<T*>(GetPointerInternal());
        }

        protected:
        static VertexBuffer* (*CreateFunc)(const BufferUsage&);
        virtual void* GetPointerInternal() = 0;

        public:
        static VertexBuffer* Create(const BufferUsage& usage = BufferUsage::STATIC);
};

}