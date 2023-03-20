/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-3-18
*/

#pragma once

#include "gobot/rendering/render_handle.hpp"
#include "gobot/scene/resources/texture.hpp"

namespace gobot {


struct FBOAttachment {
    /// Texture handle.
    Ref<Texture> texture;
    /// Mip level.
    std::uint16_t mip = 0;
    /// Cubemap side or depth layer/slice.
    std::uint16_t layer = 0;
};

class FrameBuffer {
    HANDLE_IMPL(FrameBufferHandle)
public:
    FrameBuffer() = default;

    FrameBuffer(std::uint16_t width, std::uint16_t height, TextureFormat format, TextureFlags texture_flags);

    FrameBuffer(const std::vector<Ref<Texture>>& textures);

    FrameBuffer(const std::vector<FBOAttachment>& textures);

    void Populate(const std::vector<FBOAttachment>& textures);

    const FBOAttachment& GetAttachment(std::uint32_t index = 0) const;

    const Ref<Texture>& GetTexture(std::uint32_t index = 0) const;

    std::size_t GetAttachmentCount() const;

    Vector2i GetSize() const;

private:
    /// Back buffer ratio if any.
    BackbufferRatio bbratio_ = BackbufferRatio::Equal;

    /// Size of the surface. If {0,0} then it is controlled by backbuffer ratio
    Vector2i cached_size_ = {0, 0};

    /// Texture attachments to the frame buffer
    std::vector<FBOAttachment> textures_;
};


}
