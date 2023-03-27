/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-3-17
*/

#include "gobot/scene/resources/texture.hpp"
#include "gobot/rendering/render_server.hpp"
#include "bgfx/bgfx.h"

namespace gobot {

Texture::~Texture() {
    GET_RS()->FreeTexture(texture_rid_);
    texture_rid_ = RenderRID();
}

RenderRID Texture::GetRID() const {
    return texture_rid_;
}


Texture2D::Texture2D(uint16_t width,
                     uint16_t height,
                     bool has_mips,
                     uint16_t num_layers,
                     TextureFormat format,
                     TextureFlags flags,
                     const MemoryView* mem)
{
    texture_rid_ = GET_RS()->CreateTexture2D(width, height, has_mips, num_layers, format, flags, mem);
}

Texture3D::Texture3D(uint16_t width, uint16_t height, std::uint16_t depth, bool has_mips,
                     TextureFormat format, TextureFlags flags, const MemoryView *mem)
{
    texture_rid_ = GET_RS()->CreateTexture3D(width, height, depth, has_mips, format, flags, mem);
}

TextureCube::TextureCube(std::uint16_t size, bool has_mips, std::uint16_t num_layers,
                         TextureFormat format, TextureFlags flags, const MemoryView *mem)

{
    texture_rid_ = GET_RS()->CreateTextureCube(size, has_mips, num_layers, format, flags, mem);
}


};


