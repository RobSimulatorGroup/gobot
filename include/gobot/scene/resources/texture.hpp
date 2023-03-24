/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-3-17
*/

#pragma once

#include "gobot/rendering/render_rid.hpp"
#include "gobot/core/math/matrix.hpp"
#include "gobot/core/io/resource.hpp"
#include "gobot/core/rid.h"
#include "gobot/rendering/render_types.hpp"

namespace gobot {

class Texture : public Resource {
    GOBCLASS(Texture, Resource)
public:

    Texture() = default;

    ~Texture();

    RenderRID GetRID() const;

protected:
    RenderRID texture_rid_{};
};

class Texture2D : public Texture {
    GOBCLASS(Texture2D, Texture)
public:
    Texture2D() = default;

    Texture2D(uint16_t width,
              uint16_t height,
              bool has_mips,
              uint16_t num_layers,
              TextureFormat format,
              TextureFlags flags,
              const MemoryView* mem = nullptr);
};

class Texture3D : public Texture {
    GOBCLASS(Texture3D, Texture)
public:
    Texture3D() = default;

    Texture3D(uint16_t width, uint16_t height, std::uint16_t depth, bool has_mips,
              TextureFormat format, TextureFlags flags, const MemoryView *mem = nullptr);
};


class TextureCube : public Texture {
    GOBCLASS(TextureCube, Texture)
public:
    TextureCube() = default;

    TextureCube(std::uint16_t size, bool has_mips, std::uint16_t num_layers,
                TextureFormat format, TextureFlags flags, const MemoryView *mem = nullptr);
};

}
