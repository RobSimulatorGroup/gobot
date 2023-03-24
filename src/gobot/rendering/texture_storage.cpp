/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-3-23
*/

#include "gobot/rendering/texture_storage.hpp"
#include "gobot/error_macros.hpp"
#include "gobot/rendering/render_server.hpp"

namespace gobot {

TextureStorage* TextureStorage::s_singleton = nullptr;

TextureStorage::TextureStorage() {
    s_singleton = this;
}

TextureStorage::~TextureStorage() {
    s_singleton = nullptr;

    for (const auto& rid: texture_owner_.GetOwnedList()) {
        texture_owner_.Free(rid);
    }
}

TextureStorage* TextureStorage::GetInstance() {
    ERR_FAIL_COND_V_MSG(s_singleton == nullptr, nullptr, "Must call this after initialize TextureStorage");
    return s_singleton;
}

void TextureStorage::CalculateTextureSize(TextureInfo& info,
                                          uint16_t width,
                                          uint16_t height,
                                          uint16_t depth,
                                          bool cube_map,
                                          bool has_mips,
                                          uint16_t num_layers,
                                          TextureFormat format) {
    bgfx::calcTextureSize(info, width, height, depth, cube_map, has_mips, num_layers, bgfx::TextureFormat::Enum(format));
}

TextureStorage::Texture* TextureStorage::GetTexture(RenderRID rid) {
    return texture_owner_.GetOrNull(rid);
}

bool TextureStorage::IsOriginBottomLeft() {
    return bgfx::getCaps()->originBottomLeft;
}

bool TextureStorage::IsRenderTarget(RenderRID rid) {
    USING_ENUM_BITWISE_OPERATORS;
    auto* texture = texture_owner_.GetOrNull(rid);
    ERR_FAIL_COND_V_MSG(texture == nullptr, false, "The input rid is not inside of owner");
    return 0 != (bool) (texture->creation_flags & TextureFlags::RT_MASK);
}


RenderRID TextureStorage::CreateTexture2D(uint16_t width,
                                          uint16_t height,
                                          bool has_mips,
                                          uint16_t num_layers,
                                          TextureFormat format,
                                          TextureFlags flags,
                                          const MemoryView* mem) {
    bgfx::TextureHandle handle = bgfx::createTexture2D(width, height, has_mips, num_layers,
                                                       bgfx::TextureFormat::Enum(format),
                                                       ENUM_UINT_CAST(flags), mem);
    auto rid = RenderRID::FromUint16(handle.idx);
    TextureInfo texture_info;
    CalculateTextureSize(texture_info, width, height, 1, false, has_mips, num_layers, format);
    texture_owner_.InitializeRID(rid, {flags, Texture2D, texture_info});
    return rid;
}

RenderRID TextureStorage::CreateTexture3D(uint16_t width,
                                          uint16_t height,
                                          uint16_t depth,
                                          bool has_mips,
                                          TextureFormat format,
                                          TextureFlags flags,
                                          const MemoryView* mem) {
    bgfx::TextureHandle handle = bgfx::createTexture3D(width, height, depth, has_mips,
                                                       bgfx::TextureFormat::Enum(format),
                                                       ENUM_UINT_CAST(flags), mem);
    auto rid = RenderRID::FromUint16(handle.idx);
    TextureInfo texture_info;
    CalculateTextureSize(texture_info, width, height, depth, false, has_mips, 1, format);
    texture_owner_.InitializeRID(rid, {flags, Texture3D, texture_info});
    return rid;
}

RenderRID TextureStorage::CreateTextureCube(uint16_t size,
                                            bool has_mips,
                                            uint16_t num_layers,
                                            TextureFormat format,
                                            TextureFlags flags,
                                            const MemoryView* mem) {
    bgfx::TextureHandle handle = bgfx::createTextureCube(size, has_mips, num_layers,
                                                       bgfx::TextureFormat::Enum(format),
                                                       ENUM_UINT_CAST(flags), mem);
    auto rid = RenderRID::FromUint16(handle.idx);
    TextureInfo texture_info;
    CalculateTextureSize(texture_info, size, size, size, false, has_mips, num_layers, format);
    texture_owner_.InitializeRID(rid, {flags, Texture3D, texture_info});
    return rid;
}

bool TextureStorage::Free(RenderRID rid) {
    if (texture_owner_.Owns(rid)) {
        texture_owner_.Free(rid);
        bgfx::destroy(bgfx::TextureHandle{rid.GetID()});
    }

    return false;
}


}