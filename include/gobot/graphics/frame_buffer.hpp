/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-20
*/

#pragma once

#include "gobot_export.h"
#include "definitions.hpp"
#include <Eigen/Dense>
#include "gobot/core/ref_counted.hpp"

namespace gobot {

enum class CubeFace
{
    PositiveX,
    NegativeX,
    PositiveY,
    NegativeY,
    PositiveZ,
    NegativeZ
};

struct FramebufferDesc
{
    uint32_t width;
    uint32_t height;
    uint32_t layer = 0;
    uint32_t attachmentCount;
    uint32_t msaaLevel;
    int mipIndex   = -1;
    bool screenFBO = false;
    Texture** attachments;
    TextureType* attachmentTypes;
    RenderPass* renderPass;
};

class GOBOT_EXPORT Framebuffer : public RefCounted
{
public:
    static Ref<Framebuffer> Get(const FramebufferDesc& framebufferDesc);

    static Framebuffer* Create(const FramebufferDesc& framebufferDesc);
    static void ClearCache();
    static void DeleteUnusedCache();

    virtual ~Framebuffer();

    virtual void Bind(uint32_t width, uint32_t height) const = 0;
    virtual void Bind() const                                = 0;
    virtual void UnBind() const                              = 0;
    virtual void Clear()                                     = 0;
    virtual void Validate() {};
    virtual void AddTextureAttachment(RHIFormat format, Texture* texture)                        = 0;
    virtual void AddCubeTextureAttachment(RHIFormat format, CubeFace face, TextureCube* texture) = 0;
    virtual void AddShadowAttachment(Texture* texture)                                           = 0;
    virtual void AddTextureLayer(int index, Texture* texture)                                    = 0;
    virtual void GenerateFramebuffer()                                                           = 0;

    virtual uint32_t GetWidth() const                    = 0;
    virtual uint32_t GetHeight() const                   = 0;
    virtual void SetClearColour(const Color& color) = 0;

protected:
    static Framebuffer* (*CreateFunc)(const FramebufferDesc&);
};

}