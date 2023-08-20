/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-3-18
*/

#pragma once

#include "gobot/core/rid.hpp"
#include "gobot/core/rid_owner.hpp"
#include "gobot/scene/resources/texture.hpp"
#include "gobot/core/hash_combine.hpp"
#include "gobot_export.h"

namespace gobot {

class GOBOT_EXPORT FrameBufferCache {
public:
    FrameBufferCache();

    virtual ~FrameBufferCache();

//    RID GetCache(const FramebufferDesc& framebuffer_desc);

    static FrameBufferCache* GetInstance();

    bool Free(const RID& frame_buffer_rid);

    Vector2i GetSize(const RID& frame_buffer_rid, std::size_t attachment_index = 0);

private:
    struct Cache {
//        FramebufferDesc framebuffer_desc;
        std::vector<Vector2i> texture_sizes;
        std::uint64_t hash;
    };

    static FrameBufferCache* s_singleton;

    // [hash_id, frame_buffer_id]
    std::unordered_map<std::uint64_t, RID> frame_buffer_cache_{};

    RID_Owner<Cache, true> frame_buffer_owner_{};

};

}

//namespace std
//{
//template <>
//struct hash<gobot::FramebufferDesc>
//{
//    std::size_t operator()(const gobot::rendering::FramebufferDesc& framebuffer_desc) const
//    {
//        size_t hash = 0;
//        gobot::HashCombine(hash, framebuffer_desc.width, framebuffer_desc.height, framebuffer_desc.layer,
//                           framebuffer_desc.attachment_count, framebuffer_desc.msaa_level, framebuffer_desc.screen_fbo);
//        return hash;
//    }
//};
//}