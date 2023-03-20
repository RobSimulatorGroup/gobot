/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-3-17
*/

#include "gobot/scene/resources/texture.hpp"
#include "bgfx/bgfx.h"

namespace gobot {

Texture::Texture(const TextureInfo& texture_info,
                 TextureFlags flags,
                 BackbufferRatio ratio)
        : info_(texture_info),
          flags_(flags),
          ratio_(ratio)
{
}

Texture::Texture(TextureFlags flags, BackbufferRatio ratio)
        : flags_(flags),
          ratio_(ratio)
{
}


Texture::~Texture() {
    DisposeHandle();
}

bool Texture::IsOriginBottomLeft() {
    return bgfx::getCaps()->originBottomLeft;
}

Vector2i Texture::GetSizeFromRatio(BackbufferRatio ratio) {
    auto stats = bgfx::getStats();
    auto width = stats->width;
    auto height = stats->height;
    switch(ratio) {
        case BackbufferRatio::Half:
            width /= 2;
            height /= 2;
            break;
        case BackbufferRatio::Quarter:
            width /= 4;
            height /= 4;
            break;
        case BackbufferRatio::Eighth:
            width /= 8;
            height /= 8;
            break;
        case BackbufferRatio::Sixteenth:
            width /= 16;
            height /= 16;
            break;
        case BackbufferRatio::Double:
            width *= 2;
            height *= 2;
            break;

        default:
            break;
    }

    return {std::max<int>(1, width), std::max<int>(1, height)};
}

Vector2i Texture::GetSize() const {
    if(ratio_ == BackbufferRatio::Count) {
        Vector2i size = {static_cast<int>(info_.width), static_cast<int>(info_.height)};
        return size;

    } // End if Absolute

    return GetSizeFromRatio(ratio_); // End if Relative
}

bool Texture::IsRenderTarget() const {
    USING_ENUM_BITWISE_OPERATORS;
    return 0 != (bool)(flags_ & TextureFlags::RT_MASK);
}


Texture2D::Texture2D(uint16_t width,
                     uint16_t height,
                     bool has_mips,
                     uint16_t num_layers,
                     TextureFormat format,
                     TextureFlags flags,
                     const RenderMemoryView* mem)
   : Texture(flags)
{

    handle_ = bgfx::createTexture2D(width, height, has_mips, num_layers,
                                    bgfx::TextureFormat::Enum(format),
                                    ENUM_UINT_CAST(flags), mem);
    bgfx::calcTextureSize(info_, width, height, 1, false, has_mips, num_layers, bgfx::TextureFormat::Enum(format));

}

void Texture2D::Resize(uint16_t width,
                       uint16_t height) {
    DisposeHandle();
    handle_ = bgfx::createTexture2D(width, height, info_.numMips, info_.numLayers,
                                    info_.format,
                                    ENUM_UINT_CAST(flags_));
    bgfx::calcTextureSize(info_, width, height, 1, false, info_.numMips, info_.numLayers, info_.format);
}

Texture3D::Texture3D(uint16_t width, uint16_t height, std::uint16_t depth, bool has_mips,
                     TextureFormat format, TextureFlags flags, const RenderMemoryView *mem)
    :  Texture(flags)
{
    handle_ = bgfx::createTexture3D(width, height, depth, has_mips,
                                    bgfx::TextureFormat::Enum(format),
                                    ENUM_UINT_CAST(flags), mem);
    bgfx::calcTextureSize(info_, width, height, depth, false, has_mips, 1, bgfx::TextureFormat::Enum(format));
}

TextureCube::TextureCube(std::uint16_t size, bool has_mips, std::uint16_t num_layers,
                         TextureFormat format, TextureFlags flags, const RenderMemoryView *mem)

    :  Texture(flags)
{
    handle_ = bgfx::createTextureCube(size, has_mips, num_layers,
                                    bgfx::TextureFormat::Enum(format),
                                    ENUM_UINT_CAST(flags), mem);
    bgfx::calcTextureSize(info_, size, size, size, false, has_mips, num_layers, bgfx::TextureFormat::Enum(format));
}


};


