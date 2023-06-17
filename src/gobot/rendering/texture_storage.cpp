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
}

TextureStorage* TextureStorage::GetInstance() {
    ERR_FAIL_COND_V_MSG(s_singleton == nullptr, nullptr, "Must call this after initialize TextureStorage");
    return s_singleton;
}

//RID TextureStorage::CreateTexture2D(uint16_t width,
//                                    uint16_t height,
//                                    bool has_mips,
//                                    uint16_t num_layers,
//                                    TextureFormat format) {
//    auto rid = RenderRID::FromUint16(handle.idx);
//    TextureInfo texture_info;
//    CalculateTextureSize(texture_info, width, height, 1, false, has_mips, num_layers, format);
//    texture_owner_.InitializeRID(rid, {flags, Texture2D, texture_info});
//    return rid;
//}
//
//RID TextureStorage::CreateTexture3D(uint16_t width,
//                                    uint16_t height,
//                                    uint16_t depth,
//                                    bool has_mips,
//                                    TextureFormat format,
//                                    TextureFlags flags) {
//    auto rid = RenderRID::FromUint16(handle.idx);
//    TextureInfo texture_info;
//    CalculateTextureSize(texture_info, width, height, depth, false, has_mips, 1, format);
//    texture_owner_.InitializeRID(rid, {flags, Texture3D, texture_info});
//    return rid;
//}
//
//RID TextureStorage::CreateTextureCube(uint16_t size,
//                                            bool has_mips,
//                                            uint16_t num_layers,
//                                            TextureFormat format,
//                                            TextureFlags flags,
//                                            const MemoryView* mem) {
//    auto rid = RenderRID::FromUint16(handle.idx);
//    TextureInfo texture_info;
//    CalculateTextureSize(texture_info, size, size, size, false, has_mips, num_layers, format);
//    texture_owner_.InitializeRID(rid, {flags, TextureCube, texture_info});
//    return rid;
//}

bool TextureStorage::Free(RID rid) {
//    if (texture_owner_.Owns(rid)) {
//        texture_owner_.Free(rid);
//        return true;
//    }

    return false;
}


}