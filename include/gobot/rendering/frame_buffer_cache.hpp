/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-3-18
*/

#pragma once

#include "gobot/rendering/render_rid.hpp"
#include "gobot/scene/resources/texture.hpp"
#include "gobot/core/hash_combine.hpp"
#include "gobot/rendering/render_rid_owner.hpp"
#include "gobot_export.h"

namespace gobot {

class GOBOT_EXPORT FrameBufferCache {
public:
    struct Cache {
        std::vector<Attachment> attachments;
        Vector2i size;
    };

    FrameBufferCache();

    virtual ~FrameBufferCache();

    RenderRID GetCacheFromTextures(const std::vector<RenderRID>& textures);

    RenderRID GetCacheFromAttachment(const std::vector<Attachment>& attachments);

    static FrameBufferCache* GetInstance();

    bool Free(const RenderRID& frame_buffer_rid);

    Vector2i GetSize(const RenderRID& frame_buffer_rid);

private:
    static FrameBufferCache* s_singleton;

    // [hash_id, frame_buffer_id]
    std::unordered_map<std::uint64_t, RenderRID> frame_buffer_cache_{};
    RenderRID_Owner<Cache, true> frame_buffer_owner_{};

};

}

namespace std
{
template <>
struct hash<gobot::Attachment>
{
    std::size_t operator()(const gobot::Attachment& attachment) const
    {
        size_t hash = 0;
        gobot::HashCombine(hash, attachment.access, attachment.handle.idx, attachment.mip,
                           attachment.layer, attachment.numLayers, attachment.resolve);
        return hash;
    }
};
}