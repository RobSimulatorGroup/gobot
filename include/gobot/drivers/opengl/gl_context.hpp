/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-8
*/

#pragma once

#include "gobot/graphics/RHI/graphics_context.hpp"
#include <gobot_export.h>

namespace gobot {

class GOBOT_EXPORT GLContext : public GraphicsContext
{
public:
    GLContext();

    ~GLContext();

    void Present() override;

    void Init() override {};

    size_t GetMinUniformBufferOffsetAlignment() const override { return 256; }

    bool FlipImGUITexture() const override { return true; }

    float GetGPUMemoryUsed() override { return 0.0f; }
    
    float GetTotalGPUMemory() override { return 0.0f; }

    void WaitIdle() const override { }

    void OnImGui() override;
};

}
