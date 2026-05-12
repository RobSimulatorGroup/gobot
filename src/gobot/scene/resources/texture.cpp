/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * This file is created by Qiqi Wu, 23-3-17
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/scene/resources/texture.hpp"
#include "gobot/rendering/render_server.hpp"

namespace gobot {

Texture::~Texture() {
    RS::GetInstance()->Free(texture_rid_);
    texture_rid_ = RID();
}

RID Texture::GetRID() const {
    return texture_rid_;
}


//Texture2D::Texture2D(uint16_t width,
//                     uint16_t height,
//                     bool has_mips,
//                     uint16_t num_layers,
//                     TextureFormat format)
//{
////    texture_rid_ = GET_RS()->CreateTexture2D(width, height, has_mips, num_layers, format, flags);
//}

Texture3D::Texture3D(uint16_t width, uint16_t height, std::uint16_t depth, bool has_mips,
                     TextureFormat format)
{
//    texture_rid_ = GET_RS()->CreateTexture3D(width, height, depth, has_mips, format, flags);
}

TextureCube::TextureCube(std::uint16_t size, bool has_mips, std::uint16_t num_layers,
                         TextureFormat format)

{
//    texture_rid_ = GET_RS()->CreateTextureCube(size, has_mips, num_layers, format, flags);
}



};


