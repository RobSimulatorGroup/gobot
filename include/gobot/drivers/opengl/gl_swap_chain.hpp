/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-20
*/

#pragma once

#include "gobot/graphics/swap_chain.hpp"

namespace gobot {

class GLCommandBuffer;
class GLTexture2D;

class GLSwapChain : public SwapChain
{
public:
    GLSwapChain(uint32_t width, uint32_t height);
    ~GLSwapChain();
    bool Init(bool vsync, WindowInterface* window) override { return Init(vsync); };
    bool Init(bool vsync) override;

    Texture* GetCurrentImage() override;
    Texture* GetImage(uint32_t index) override { return nullptr; };
    uint32_t GetCurrentBufferIndex() const override;
    uint32_t GetCurrentImageIndex() const override { return 0; };
    CommandBuffer* GetCurrentCommandBuffer() override;
    void OnResize(uint32_t width, uint32_t height)
    {
        m_Width  = width;
        m_Height = height;
    }
    uint32_t GetWidth() const { return m_Width; }
    uint32_t GetHeight() const { return m_Height; }

    size_t GetSwapChainBufferCount() const override;
    void SetVSync(bool vsync) override { }

    static void MakeDefault();

protected:
    static SwapChain* CreateFuncGL(uint32_t width, uint32_t height);

private:
    std::vector<GLTexture2D*> swapChainBuffers;
    std::shared_ptr<GLCommandBuffer> MainCommandBuffer;
    uint32_t currentBuffer = 0;

    uint32_t m_Width;
    uint32_t m_Height;
};

}