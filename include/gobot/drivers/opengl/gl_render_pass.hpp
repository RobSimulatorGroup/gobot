/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-20
*/

#pragma once

#include "gobot/render/render_pass.hpp"

namespace gobot {

class GOBOT_EXPORT GLRenderPass : public RenderPass
{
public:
    GLRenderPass(const RenderPassDesc& renderPassDesc);

    ~GLRenderPass();

    bool Init(const RenderPassDesc& renderPassDesc);

    void BeginRenderpass(CommandBuffer* commandBuffer,
                         float* clearColour,
                         Framebuffer* frame,
                         SubPassContents contents,
                         uint32_t width,
                         uint32_t height) const override;

    void EndRenderpass(CommandBuffer* commandBuffer) override;

    int GetAttachmentCount() const override { return m_ClearCount; };

    static void MakeDefault();

protected:
    static RenderPass* CreateFuncGL(const RenderPassDesc& renderPassDesc);

private:
    bool m_Clear     = true;
    int m_ClearCount = 0;
};


}