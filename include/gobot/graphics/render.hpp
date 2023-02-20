/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-20
*/

#pragma once

#include "gobot/core/types.hpp"
#include "gobot_export.h"

#include "gobot/graphics/definitions.hpp"
#include <Eigen/Dense>

namespace gobot {

struct RenderAPICapabilities
{
    QString Vendor;
    QString Renderer;
    QString Version;

    int MaxSamples                   = 0;
    float MaxAnisotropy              = 0.0f;
    int MaxTextureUnits              = 0;
    int UniformBufferOffsetAlignment = 0;
    bool WideLines                   = false;
    bool SupportCompute              = false;
};

class GOBOT_EXPORT Renderer {
public:
    Renderer()          = default;
    virtual ~Renderer() = default;

    static void Init(bool loadEmbeddedShaders = true);
    static void Release();
    void LoadEngineShaders(bool loadEmbeddedShaders);
    virtual void InitInternal()                            = 0;
    virtual void Begin()                                   = 0;
    virtual void OnResize(uint32_t width, uint32_t height) = 0;
    virtual void ClearRenderTarget(Texture* texture, CommandBuffer* commandBuffer, Color clear_color = (0.1f, 0.1f, 0.1f, 1.0f)) { }
    inline static Renderer* GetRenderer()
    {
        return s_Instance;
    }

    virtual void PresentInternal()                                                                                                                                                                            = 0;
    virtual void PresentInternal(CommandBuffer* commandBuffer)                                                                                                                                      = 0;
    virtual void BindDescriptorSetsInternal(Pipeline* pipeline, CommandBuffer* commandBuffer, uint32_t dynamicOffset, DescriptorSet** descriptorSets, uint32_t descriptorCount) = 0;

    virtual const std::string& GetTitleInternal() const                                                                             = 0;
    virtual void DrawIndexedInternal(CommandBuffer* commandBuffer, DrawType type, uint32_t count, uint32_t start) const             = 0;
    virtual void DrawInternal(CommandBuffer* commandBuffer, DrawType type, uint32_t count, DataType datayType, void* indices) const = 0;
    virtual void Dispatch(CommandBuffer* commandBuffer, uint32_t workGroupSizeX, uint32_t workGroupSizeY, uint32_t workGroupSizeZ) { }
    virtual void DrawSplashScreen(Texture* texture) { }
    virtual uint32_t GetGPUCount() const { return 1; }
    virtual bool SupportsCompute() { return false; }

    inline static void Present()
    {
        s_Instance->PresentInternal();
    }
    inline static void Present(CommandBuffer* commandBuffer)
    {
        s_Instance->PresentInternal(commandBuffer);
    }
    inline static void BindDescriptorSets(Pipeline* pipeline, CommandBuffer* commandBuffer, uint32_t dynamicOffset, DescriptorSet** descriptorSets, uint32_t descriptorCount)
    {
        s_Instance->BindDescriptorSetsInternal(pipeline, commandBuffer, dynamicOffset, descriptorSets, descriptorCount);
    }
    inline static void Draw(CommandBuffer* commandBuffer, DrawType type, uint32_t count, DataType datayType = DataType::UNSIGNED_INT, void* indices = nullptr)
    {
        s_Instance->DrawInternal(commandBuffer, type, count, datayType, indices);
    }
    inline static void DrawIndexed(CommandBuffer* commandBuffer, DrawType type, uint32_t count, uint32_t start = 0)
    {
        s_Instance->DrawIndexedInternal(commandBuffer, type, count, start);
    }
    inline static const std::string& GetTitle()
    {
        return s_Instance->GetTitleInternal();
    }

    static RenderAPICapabilities& GetCapabilities()
    {
        static RenderAPICapabilities capabilities;
        return capabilities;
    }

//    static GraphicsContext* GetGraphicsContext() { return Application::Get().GetWindow()->GetGraphicsContext(); }
//    static SwapChain* GetMainSwapChain() { return Application::Get().GetWindow()->GetSwapChain(); }
//    static void DrawMesh(CommandBuffer* commandBuffer, Graphics::Pipeline* pipeline, Graphics::Mesh* mesh);

protected:
    static Renderer* (*CreateFunc)();

    static Renderer* s_Instance;
};

}