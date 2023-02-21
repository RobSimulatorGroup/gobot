/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-20
*/

#pragma once

#include "gobot/render/command_buffer.hpp"

namespace gobot {


class GOBOT_EXPORT GLCommandBuffer : public CommandBuffer
{
public:
    GLCommandBuffer();
    ~GLCommandBuffer();

    bool Init(bool primary) override;
    void Unload() override;
    void BeginRecording() override;
    void BeginRecordingSecondary(RenderPass* renderPass, Framebuffer* framebuffer) override;
    void EndRecording() override;
    void ExecuteSecondary(CommandBuffer* primaryCmdBuffer) override;

    void BindPipeline(Pipeline* pipeline) override;
    void UnBindPipeline() override;

    void UpdateViewport(uint32_t width, uint32_t height, bool flipViewport) override {};
    static void MakeDefault();

protected:
    static CommandBuffer* CreateFuncGL();

private:
    bool primary;
    Pipeline* m_BoundPipeline = nullptr;
};

}