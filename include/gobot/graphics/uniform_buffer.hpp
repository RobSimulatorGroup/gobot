/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-20
*/

#pragma once

#include <cstdint>
#include "gobot/core/ref_counted.hpp"

namespace gobot {

class UniformBuffer : public RefCounted
{
public:
    virtual ~UniformBuffer() = default;
    static UniformBuffer* Create();
    static UniformBuffer* Create(uint32_t size, const void* data);

    virtual void Init(uint32_t size, const void* data)                              = 0;
    virtual void SetData(const void* data)                                          = 0;
    virtual void SetData(uint32_t size, const void* data)                           = 0;
    virtual void SetDynamicData(uint32_t size, uint32_t typeSize, const void* data) = 0;

    virtual uint8_t* GetBuffer() const = 0;

protected:
    static UniformBuffer* (*CreateFunc)();
    static UniformBuffer* (*CreateDataFunc)(uint32_t, const void*);
};

}