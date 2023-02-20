/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-20
*/

#pragma once

#include "definitions.hpp"
#include "gobot/core/ref_counted.hpp"
#include <gobot_export.h>

namespace gobot {

struct PipelineDesc
{
    std::shared_ptr<Shader> shader;

    CullMode cullMode       = CullMode::BACK;
    PolygonMode polygonMode = PolygonMode::FILL;
    DrawType drawType       = DrawType::TRIANGLE;
    BlendMode blendMode     = BlendMode::None;

    bool transparencyEnabled = false;
    bool depthBiasEnabled    = false;
    bool swapchainTarget     = false;
    bool clearTargets        = false;

    std::array<Texture*, MAX_RENDER_TARGETS> colourTargets = {};

    Texture* cubeMapTarget        = nullptr;
    Texture* depthTarget          = nullptr;
    Texture* depthArrayTarget     = nullptr;
    float clearColour[4]          = { 0.2f, 0.2f, 0.2f, 1.0f };
    float lineWidth               = 1.0f;
    float depthBiasConstantFactor = 0.0f;
    float depthBiasSlopeFactor    = 0.0f;
    int cubeMapIndex              = 0;
    int mipIndex                  = -1;
};

class GOBOT_EXPORT Pipeline : public RefCounted
{
public:
    static Pipeline* Create(const PipelineDesc& pipelineDesc);
    static Ref<Pipeline> Get(const PipelineDesc& pipelineDesc);
    static void ClearCache();
    static void DeleteUnusedCache();

    virtual ~Pipeline() = default;

    virtual void Bind(CommandBuffer* commandBuffer, uint32_t layer = 0) = 0;
    virtual void End(CommandBuffer* commandBuffer) { }
    virtual void ClearRenderTargets(CommandBuffer* commandBuffer) { }
    virtual Shader* GetShader() const = 0;

    uint32_t GetWidth();
    uint32_t GetHeight();

protected:
    static Pipeline* (*CreateFunc)(const PipelineDesc&);
    PipelineDesc m_Description;
};

}