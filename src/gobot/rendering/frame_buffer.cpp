/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-3-18
*/

#include "gobot/rendering/frame_buffer.hpp"

namespace gobot {

FrameBuffer::FrameBuffer(std::uint16_t width, std::uint16_t height, TextureFormat format, TextureFlags texture_flags)
    : FrameBuffer(std::vector<Ref<Texture>>{MakeRef<Texture2D>(width, height, false, 1, format, texture_flags)})
{
}

FrameBuffer::FrameBuffer(const std::vector<Ref<Texture>>& textures) {
    std::vector<FBOAttachment> tex_descs;
    tex_descs.reserve(textures.size());
    for(auto& tex : textures)
    {
        FBOAttachment tex_desc;
        tex_desc.texture = tex;
        tex_descs.push_back(tex_desc);
    }

    Populate(tex_descs);
}

FrameBuffer::FrameBuffer(const std::vector<FBOAttachment>& textures) {
    Populate(textures);
}

void FrameBuffer::Populate(const std::vector<FBOAttachment>& textures)
{
    std::vector<Attachment> buffer;
    buffer.reserve(textures.size());

    Vector2i size = {0, 0};
    auto ratio = BackbufferRatio::Count;
    for(auto& tex : textures) {
        ratio = tex.texture->GetBackbufferRatio();
        size = {tex.texture->GetTextureInfo().width, tex.texture->GetTextureInfo().height};
        Attachment att;
        att.init(tex.texture->GetHandle(), bgfx::Access::Write, tex.layer, tex.mip);
        buffer.push_back(att);
    }
    textures_ = textures;

    handle_ = bgfx::createFrameBuffer(static_cast<std::uint8_t>(buffer.size()), &buffer[0], false);

    if(ratio == BackbufferRatio::Count) {
        bbratio_ = ratio;
        cached_size_ = size;
    } else {
        bbratio_ = ratio;
        cached_size_ = {0, 0};
    }
}

const FBOAttachment& FrameBuffer::GetAttachment(std::uint32_t index) const {
    return textures_[index];
}

const Ref<Texture>& FrameBuffer::GetTexture(std::uint32_t index) const {
    return GetAttachment(index).texture;
}

std::size_t FrameBuffer::GetAttachmentCount() const {
    return textures_.size();
}

Vector2i FrameBuffer::GetSize() const {
    if(bbratio_ == BackbufferRatio::Count) {
        return cached_size_;
    } // End if Absolute

    return Texture::GetSizeFromRatio(bbratio_);
    // End if Relative
}

}
