/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-20
*/

#pragma once

#include "gobot/render/uniform_buffer.hpp"

namespace gobot {

class GLShader;

class GOBOT_EXPORT GLUniformBuffer : public UniformBuffer
{
public:
    GLUniformBuffer();
    ~GLUniformBuffer();

    void Init(uint32_t size, const void* data) override;
    void SetData(uint32_t size, const void* data) override;
    void SetDynamicData(uint32_t size, uint32_t typeSize, const void* data) override;

    void SetData(const void* data) override { SetData(m_Size, data); }

    void Bind(uint32_t slot, GLShader* shader, String & name);

    uint8_t* GetBuffer() const override
    {
        return m_Data;
    };

    uint32_t GetSize() const
    {
        return m_Size;
    }
    uint32_t GetTypeSize() const
    {
        return m_DynamicTypeSize;
    }
    bool GetDynamic() const
    {
        return m_Dynamic;
    }
    uint32_t GetHandle() const
    {
        return m_Handle;
    }

    static void MakeDefault();

protected:
    static UniformBuffer* CreateFuncGL();
    static UniformBuffer* CreateDataFuncGL(uint32_t, const void*);

private:
    uint8_t* m_Data            = nullptr;
    uint32_t m_Size            = 0;
    uint32_t m_DynamicTypeSize = 0;
    bool m_Dynamic             = false;
    uint32_t m_Handle;
    uint32_t m_Index;
};

}