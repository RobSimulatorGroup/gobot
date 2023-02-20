/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-20
*/

#pragma once

#include "gobot_export.h"
#include "gobot/core/types.hpp"
#include "definitions.hpp"
#include <Eigen/Dense>
#include <type_traits>

namespace gobot {

struct BufferElement {
    String name;
    RHIFormat format = RHIFormat::R32G32B32A32_Float;
    uint32_t offset  = 0;
    bool Normalised  = false;
};

class BufferLayout {
private:
    uint32_t m_Size;
    std::vector<BufferElement> m_Layout;

public:
    BufferLayout();

    template <typename T>
    void Push(const String& name, bool Normalised = false)
    {
    }

    inline const std::vector<BufferElement>& GetLayout() const
    {
        return m_Layout;
    }
    inline uint32_t GetStride() const
    {
        return m_Size;
    }

    private:
    void Push(const String& name, RHIFormat format, uint32_t size, bool Normalised);
};

template <>
void BufferLayout::Push<float>(const String& name, bool Normalised);

template <>
void BufferLayout::Push<uint32_t>(const String& name, bool Normalised);

template <>
void BufferLayout::Push<uint8_t>(const String& name, bool Normalised);

template <>
void BufferLayout::Push<Eigen::Vector2f>(const String& name, bool Normalised);

template <>
void BufferLayout::Push<Eigen::Vector3f>(const String& name, bool Normalised);

template <>
void BufferLayout::Push<Eigen::Vector4f>(const String& name, bool Normalised);


}
