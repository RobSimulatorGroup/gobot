/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-20
*/

#pragma once

#include "gobot/graphics/frame_buffer.hpp"
#include "gobot/graphics/definitions.hpp"
#include "gobot/drivers/opengl/gl.hpp"
#include <Eigen/Dense>
#include "gobot_export.h"

namespace gobot {

class GOBOT_EXPORT GLFramebuffer : public Framebuffer
{
public:
    GLFramebuffer();
    GLFramebuffer(const FramebufferDesc& frameBufferDesc);
    ~GLFramebuffer();

    inline uint32_t GetFramebuffer() const { return m_Handle; }

    void GenerateFramebuffer() override;

    void Bind(uint32_t width, uint32_t height) const override;
    void Bind() const override;
    void UnBind() const override;
    void Clear() override { }
    uint32_t GetWidth() const override { return m_Width; }
    uint32_t GetHeight() const override { return m_Height; }

    GLenum GetAttachmentPoint(RHIFormat format);

    inline void SetClearColour(const Color& color) override { clear_color_ = color; }

    void AddTextureAttachment(RHIFormat format, Texture* texture) override;
    void AddCubeTextureAttachment(RHIFormat format, CubeFace face, TextureCube* texture) override;

    void AddShadowAttachment(Texture* texture) override;
    void AddTextureLayer(int index, Texture* texture) override;

    void Validate() override;

    static void MakeDefault();

protected:
    static Framebuffer* CreateFuncGL(const FramebufferDesc& frameBufferDesc);

private:
    uint32_t m_Handle;
    uint32_t m_Width, m_Height, m_ColourAttachmentCount;
    Color clear_color_;
    std::vector<GLenum> m_AttachmentData;
    bool m_ScreenFramebuffer = false;
};

}