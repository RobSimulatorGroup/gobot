/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-20
*/

#pragma once

#include "gobot/render/pipeline.hpp"
#include "gobot/render/frame_buffer.hpp"
#include "gobot/render/buffer_layout.hpp"

namespace gobot {

class GLRenderPass;
class CommandBuffer;
class RenderPass;

class GOBOT_EXPORT GLPipeline : public Pipeline
{
public:
    GLPipeline(const PipelineDesc& pipelineDesc);

    ~GLPipeline();

    bool Init(const PipelineDesc& pipelineDesc);

    void Bind(CommandBuffer* commandBuffer, uint32_t layer) override;

    void End(CommandBuffer* commandBuffer) override;

    void ClearRenderTargets(CommandBuffer* commandBuffer) override;

    void BindVertexArray();

    void CreateFramebuffers();

    Shader* GetShader() const override { return m_Shader; }

    static void MakeDefault();

protected:
    static Pipeline* CreateFuncGL(const PipelineDesc& pipelineDesc);

private:
    Shader* m_Shader = nullptr;
    Ref<RenderPass> m_RenderPass;
    std::vector<Ref<Framebuffer>> m_Framebuffers;
    std::string pipelineName;
    bool m_TransparencyEnabled = false;
    uint32_t m_VertexArray     = -1;
    BufferLayout m_VertexBufferLayout;
    CullMode m_CullMode;
    BlendMode m_BlendMode;
    float m_LineWidth = 1.0f;
};

}
