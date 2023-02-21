/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-12
*/

#pragma once

#include <cstdint>
#include <gobot_export.h>

namespace gobot {

class CommandBuffer;

class GOBOT_EXPORT IMGUIRenderer
{
public:
    static IMGUIRenderer* Create(std::uint32_t width, std::uint32_t height, bool clear_screen);

    virtual ~IMGUIRenderer() = default;

    virtual void Init() = 0;

    virtual void NewFrame() = 0;

    virtual void Render(CommandBuffer* commandBuffer) = 0;

    virtual void OnResize(uint32_t width, uint32_t height) = 0;

    virtual void Clear() { }

    virtual bool Implemented() const = 0;

    virtual void RebuildFontTexture() = 0;

protected:
//    static IMGUIRenderer* (*CreateFunc)(uint32_t, uint32_t, bool);
};

}
