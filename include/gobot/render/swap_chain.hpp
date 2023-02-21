/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-20
*/

#pragma once

#include "gobot/core/os/window.hpp"

namespace gobot {

class Texture;
class Framebuffer;
class RenderPass;
class CommandBuffer;
class GOBOT_EXPORT SwapChain
{
public:
    virtual ~SwapChain() = default;
    static SwapChain* Create(uint32_t width, uint32_t height);

    virtual bool Init(bool vsync, WindowInterface* window)    = 0;
    virtual bool Init(bool vsync)                    = 0;
    virtual Texture* GetCurrentImage()               = 0;
    virtual Texture* GetImage(uint32_t index)        = 0;
    virtual uint32_t GetCurrentBufferIndex() const   = 0;
    virtual uint32_t GetCurrentImageIndex() const    = 0;
    virtual size_t GetSwapChainBufferCount() const   = 0;
    virtual CommandBuffer* GetCurrentCommandBuffer() = 0;
    virtual void SetVSync(bool vsync)                = 0;

protected:
    static SwapChain* (*CreateFunc)(uint32_t, uint32_t);
};

}
