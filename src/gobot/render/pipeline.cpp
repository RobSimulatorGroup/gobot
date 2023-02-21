/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-20
*/

#include "gobot/render/pipeline.hpp"
#include "gobot/log.hpp"
#include "gobot/error_macros.hpp"
#include "gobot/render/texture.hpp"

namespace gobot {

struct PipelineAsset
{
    Ref<Pipeline> pipeline;
    float timeSinceLastAccessed;
};
static std::unordered_map<std::size_t, PipelineAsset> m_PipelineCache;
static const float m_CacheLifeTime = 0.1f;

Pipeline* (*Pipeline::CreateFunc)(const PipelineDesc&) = nullptr;

Pipeline* Pipeline::Create(const PipelineDesc& pipelineDesc)
{
    CRASH_COND_MSG(CreateFunc == nullptr, "No Pipeline Create Function");
    return CreateFunc(pipelineDesc);
}

Ref<Pipeline> Pipeline::Get(const PipelineDesc& pipelineDesc)
{
    size_t hash = 0;
//    HashCombine(hash, pipelineDesc.shader.get(), pipelineDesc.cullMode, pipelineDesc.depthBiasEnabled, pipelineDesc.drawType, pipelineDesc.polygonMode, pipelineDesc.transparencyEnabled);

    for(auto texture : pipelineDesc.colourTargets)
    {
//        if(texture)
//        {
//            HashCombine(hash, texture, texture->GetWidth(), texture->GetHeight(), texture->GetImageHande(), texture->Handle);
//
//#ifdef LUMOS_RENDER_API_VULKAN
//
//            if(GraphicsContext::GetRenderAPI() == RenderAPI::VULKAN)
//                    {
//                        VkDescriptorImageInfo* imageHandle = (VkDescriptorImageInfo*)(texture->GetDescriptorInfo());
//                        HashCombine(hash, imageHandle->imageLayout, imageHandle->imageView, imageHandle->sampler);
//                    }
//#endif
//        }
    }

//    if(pipelineDesc.depthTarget)
//    {
//#ifdef LUMOS_RENDER_API_VULKAN
//
//        if(GraphicsContext::GetRenderAPI() == RenderAPI::VULKAN)
//                {
//                    VkDescriptorImageInfo* depthImageHandle = (VkDescriptorImageInfo*)(pipelineDesc.depthTarget->GetDescriptorInfo());
//                    HashCombine(hash, depthImageHandle->imageLayout, depthImageHandle->imageView, depthImageHandle->sampler);
//                }
//#endif
//        HashCombine(hash, pipelineDesc.depthTarget, pipelineDesc.depthTarget->GetWidth(), pipelineDesc.depthTarget->GetHeight(), pipelineDesc.depthTarget->GetImageHande(), pipelineDesc.depthTarget->Handle);
//    }
//
//    if(pipelineDesc.depthArrayTarget)
//    {
//        HashCombine(hash, pipelineDesc.depthArrayTarget, pipelineDesc.depthArrayTarget->GetImageHande());
//    }
//
//    HashCombine(hash, pipelineDesc.clearTargets);
//    HashCombine(hash, pipelineDesc.swapchainTarget);
//    HashCombine(hash, pipelineDesc.lineWidth);
//    HashCombine(hash, pipelineDesc.depthBiasConstantFactor);
//    HashCombine(hash, pipelineDesc.depthBiasSlopeFactor);
//    HashCombine(hash, pipelineDesc.cubeMapIndex);
//    HashCombine(hash, pipelineDesc.cubeMapTarget);
//    HashCombine(hash, pipelineDesc.mipIndex);

    if(pipelineDesc.swapchainTarget)
    {
        // Add one swapchain image to hash
//        auto texture = Renderer::GetMainSwapChain()->GetCurrentImage();
//        if(texture)
//        {
//            HashCombine(hash, texture->GetImageHande());
//        }
    }

    auto found = m_PipelineCache.find(hash);
    if(found != m_PipelineCache.end() && found->second.pipeline)
    {
//        found->second.timeSinceLastAccessed = (float)Engine::GetTimeStep().GetElapsedSeconds();
        return found->second.pipeline;
    }

    Ref<Pipeline> pipeline = Ref<Pipeline>(Create(pipelineDesc));
//    m_PipelineCache[hash]        = { pipeline, (float)Engine::GetTimeStep().GetElapsedSeconds() };
    return pipeline;
}

void Pipeline::ClearCache()
{
    m_PipelineCache.clear();
}

void Pipeline::DeleteUnusedCache()
{

    static std::size_t keysToDelete[256];
    std::size_t keysToDeleteCount = 0;

    for(auto&& [key, value] : m_PipelineCache)
    {
//        if(value.pipeline && value.pipeline->GetReferenceCount() == 1 && (Engine::GetTimeStep().GetElapsedSeconds() - value.timeSinceLastAccessed) > m_CacheLifeTime)
//        {
//            keysToDelete[keysToDeleteCount] = key;
//            keysToDeleteCount++;
//        }

        if(keysToDeleteCount >= 256)
            break;
    }

    for(std::size_t i = 0; i < keysToDeleteCount; i++)
    {
        m_PipelineCache[keysToDelete[i]].pipeline = nullptr;
        m_PipelineCache.erase(keysToDelete[i]);
    }
}

uint32_t Pipeline::GetWidth()
{
    if(m_Description.swapchainTarget)
    {
#ifdef LUMOS_RENDER_API_OPENGL
        if(GraphicsContext::GetRenderAPI() == RenderAPI::OPENGL)
                    return ((GLSwapChain*)Renderer::GetMainSwapChain())->GetWidth();
#endif
//        return Renderer::GetMainSwapChain()->GetCurrentImage()->GetWidth();
    }

    if(m_Description.colourTargets[0])
    {
        return m_Description.mipIndex > 0 ? m_Description.colourTargets[0]->GetWidth(m_Description.mipIndex) : m_Description.colourTargets[0]->GetWidth();
    }

    if(m_Description.depthTarget)
        return m_Description.depthTarget->GetWidth();

    if(m_Description.depthArrayTarget)
        return m_Description.depthArrayTarget->GetWidth();

    if(m_Description.cubeMapTarget)
        return m_Description.cubeMapTarget->GetWidth();

    LOG_WARN("Invalid pipeline width");

    return 0;
}

uint32_t Pipeline::GetHeight()
{
    if(m_Description.swapchainTarget)
    {
#ifdef LUMOS_RENDER_API_OPENGL
        if(GraphicsContext::GetRenderAPI() == RenderAPI::OPENGL)
                    return ((GLSwapChain*)Renderer::GetMainSwapChain())->GetHeight();
#endif
//        return Renderer::GetMainSwapChain()->GetCurrentImage()->GetHeight();
    }

    if(m_Description.colourTargets[0])
    {
        // TODO
        return m_Description.mipIndex > 0 ? m_Description.colourTargets[0]->GetHeight(m_Description.mipIndex) : m_Description.colourTargets[0]->GetHeight();
    }

    if(m_Description.depthTarget)
        return m_Description.depthTarget->GetHeight();

    if(m_Description.depthArrayTarget)
        return m_Description.depthArrayTarget->GetHeight();

    if(m_Description.cubeMapTarget)
        return m_Description.cubeMapTarget->GetHeight();

    LOG_WARN("Invalid pipeline height");

    return 0;
}


}