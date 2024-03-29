/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-3-18
*/

#include "gobot/rendering/frame_buffer_cache.hpp"
#include "gobot/core/hash_combine.hpp"
#include "gobot/error_macros.hpp"
#include "gobot/rendering/rendering_server_globals.hpp"
#include "gobot/rendering/texture_storage.hpp"

namespace gobot {

FrameBufferCache* FrameBufferCache::s_singleton = nullptr;

FrameBufferCache::FrameBufferCache() {
    s_singleton = this;
}

FrameBufferCache::~FrameBufferCache() {
    s_singleton = nullptr;

//    for (const auto& rid: frame_buffer_owner_.GetOwnedList()) {
//        Free(rid);
//    }
}

//RID FrameBufferCache::GetCacheFromTextures(const std::vector<RenderRID>& textures) {
//    ERR_FAIL_COND_V_MSG(textures.empty(), {}, "Input textures is empty");
//
//    std::vector<Attachment> attachments;
//    for (const auto& texture_id: textures) {
//        Attachment attachment;
//        attachment.init(TextureHandle{texture_id.GetID()}, bgfx::Access::Write, 0, 1, 0, BGFX_RESOLVE_AUTO_GEN_MIPS);
//        attachments.emplace_back(attachment);
//    }
//
//    return GetCacheFromAttachment(attachments);
//}
//
//RenderRID FrameBufferCache::GetCacheFromAttachment(const std::vector<Attachment>& attachments) {
//    ERR_FAIL_COND_V_MSG(attachments.empty(), {}, "Input attachments is empty");
//
//    size_t hash = 0;
//    std::vector<Vector2i> sizes;
//    for (const auto& attachment : attachments ) {
//        auto* texture = RSG::texture_storage->GetTexture(RenderRID::FromUint16(attachments[0].handle.idx));
//        ERR_FAIL_COND_V_MSG(texture == nullptr, {}, "attachment's texture is invalid");
//        sizes.emplace_back(texture->texture_info.width, texture->texture_info.height);
//        HashCombine(hash, texture->texture_info);
//        HashCombine(hash, attachment);
//    }
//
//    auto it = frame_buffer_cache_.find(hash);
//    if (it != frame_buffer_cache_.end()) {
//        return it->second;
//    }
//
//    // create new frame buffer
//    auto handle = bgfx::createFrameBuffer(static_cast<std::uint8_t>(attachments.size()), &attachments[0], false);
//    auto rid = RenderRID::FromUint16(handle.idx);
//    frame_buffer_cache_.insert({hash, rid});
//
//    frame_buffer_owner_.InitializeRID(rid, {attachments, sizes, hash});
//    return rid;
//}

Vector2i FrameBufferCache::GetSize(const RID& frame_buffer_rid, std::size_t attachment_index) {
    auto* cache = frame_buffer_owner_.GetOrNull(frame_buffer_rid);
    ERR_FAIL_COND_V_MSG(cache == nullptr, {}, "The frame_buffer is not in frame_buffer_owner");
    ERR_FAIL_INDEX_V(attachment_index, cache->texture_sizes.size(), {});
    return cache->texture_sizes[attachment_index];
}


bool FrameBufferCache::Free(const RID& frame_buffer_rid) {
//    auto* cache = frame_buffer_owner_.GetOrNull(frame_buffer_rid);
//    if (cache) {
//        frame_buffer_cache_.erase(cache->hash);
//        frame_buffer_owner_.Erase(frame_buffer_rid);
//        bgfx::destroy(bgfx::FrameBufferHandle{frame_buffer_rid.GetID()});
//        return true;
//    }

    return false;
}


FrameBufferCache* FrameBufferCache::GetInstance() {
    ERR_FAIL_COND_V_MSG(s_singleton == nullptr, nullptr, "Must call this after initialize FrameBufferCache");
    return s_singleton;
}

}
