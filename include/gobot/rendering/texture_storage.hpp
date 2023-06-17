/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-3-23
*/

#pragma once

#include "gobot/core/hash_combine.hpp"
#include "gobot/core/rid.hpp"
#include "gobot/core/rid_owner.hpp"

namespace gobot {

class TextureStorage {
public:

    TextureStorage();

    virtual ~TextureStorage();

    static TextureStorage* GetInstance();

//    RID CreateTexture2D(uint16_t width,
//                        uint16_t height,
//                        bool has_mips,
//                        uint16_t num_layers,
//                        TextureFormat format,
//                        TextureFlags flags);
//
//    RID CreateTexture3D(uint16_t width,
//                        uint16_t height,
//                        uint16_t depth,
//                        bool has_mips,
//                        TextureFormat format,
//                        TextureFlags flags);
//
//    RID CreateTextureCube(uint16_t size,
//                          bool has_mips,
//                          uint16_t num_layers,
//                          TextureFormat format,
//                          TextureFlags flags);

    /// Calculate amount of memory required for texture.
    ///
    /// @param[out] info Resulting texture info structure. See: `TextureInfo`.
    /// @param[in] width Width.
    /// @param[in] height Height.
    /// @param[in] depth Depth dimension of volume texture.
    /// @param[in] cube_map Indicates that texture contains cubemap.
    /// @param[in] has_mips Indicates that texture contains full mip-map chain.
    /// @param[in] num_layers Number of layers in texture array.
    /// @param[in] format Texture format.
//    void CalculateTextureSize(TextureInfo& info,
//                              uint16_t width,
//                              uint16_t height,
//                              uint16_t depth,
//                              bool cube_map,
//                              bool has_mips,
//                              uint16_t num_layers,
//                              TextureFormat format);

//    Texture* GetTexture(RID rid);

    bool Free(RID rid);

private:
    static TextureStorage *s_singleton;

//    RID_Owner<Texture, true> texture_owner_{};
};

}

//namespace std
//{
//template <>
//struct hash<gobot::TextureInfo>
//{
//    std::size_t operator()(const gobot::TextureInfo& texture_info) const
//    {
//        size_t hash = 0;
//        gobot::HashCombine(hash, texture_info.format, texture_info.storageSize,
//                           texture_info.width, texture_info.height, texture_info.depth,
//                           texture_info.numLayers, texture_info.numMips, texture_info.bitsPerPixel, texture_info.cubeMap);
//        return hash;
//    }
//};
//}