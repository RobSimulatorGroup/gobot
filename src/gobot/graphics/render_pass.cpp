/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-20
*/

#include "gobot/graphics/render_pass.hpp"
#include "gobot/log.hpp"
#include "gobot/error_macros.hpp"

namespace gobot {

static std::unordered_map<std::size_t, Ref<RenderPass>> m_RenderPassCache;

RenderPass::~RenderPass()                                    = default;
RenderPass* (*RenderPass::CreateFunc)(const RenderPassDesc&) = nullptr;

RenderPass* RenderPass::Create(const RenderPassDesc& renderPassDesc)
{
    CRASH_COND_MSG(CreateFunc == nullptr, "No RenderPass Create Function");
    return CreateFunc(renderPassDesc);
}

Ref<RenderPass> RenderPass::Get(const RenderPassDesc& renderPassDesc)
{
    size_t hash = 0;
//    HashCombine(hash, renderPassDesc.attachmentCount, renderPassDesc.clear);
//
//    for(uint32_t i = 0; i < renderPassDesc.attachmentCount; i++)
//    {
//        HashCombine(hash, renderPassDesc.attachmentTypes[i], renderPassDesc.attachments[i], renderPassDesc.cubeMapIndex, renderPassDesc.mipIndex);
//    }
//
//    auto found = m_RenderPassCache.find(hash);
//    if(found != m_RenderPassCache.end() && found->second)
//    {
//        return found->second;
//    }

    auto renderPass         = Ref<RenderPass>(Create(renderPassDesc));
    m_RenderPassCache[hash] = renderPass;
    return renderPass;
}

void RenderPass::ClearCache()
{
    m_RenderPassCache.clear();
}

void RenderPass::DeleteUnusedCache()
{
    static std::size_t keysToDelete[256];
    std::size_t keysToDeleteCount = 0;

    for(auto&& [key, value] : m_RenderPassCache)
    {
        if(value->GetReferenceCount() == 1)
        {
            keysToDelete[keysToDeleteCount] = key;
            keysToDeleteCount++;
        }
    }

    for(std::size_t i = 0; i < keysToDeleteCount; i++)
    {
        m_RenderPassCache[keysToDelete[i]] = nullptr;
        m_RenderPassCache.erase(keysToDelete[i]);
    }
}

}
