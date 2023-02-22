/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-20
*/

#pragma once

#include "gobot_export.h"
#include "definitions.hpp"
#include "gobot/core/ref_counted.hpp"

namespace gobot {

class GOBOT_EXPORT RenderPass : public RefCounted {
public:
    virtual ~RenderPass();
    static RenderPass* Create(const RenderPassDesc& renderPassDesc);

    static Ref<RenderPass> Get(const RenderPassDesc& renderPassDesc);

    static void ClearCache();

    static void DeleteUnusedCache();

    virtual void BeginRenderpass(CommandBuffer* commandBuffer,
                                 float* clearColour,
                                 Framebuffer* frame,
                                 SubPassContents contents,
                                 uint32_t width,
                                 uint32_t height) const = 0;

    virtual void EndRenderpass(CommandBuffer* commandBuffer) = 0;

    virtual int GetAttachmentCount() const = 0;

    protected:
    static RenderPass* (*CreateFunc)(const RenderPassDesc&);
};

}
