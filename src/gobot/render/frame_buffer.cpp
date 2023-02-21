/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-20
*/

#include "gobot/render/frame_buffer.hpp"
#include "gobot/core/hash_combine.hpp"
#include "gobot/render/texture.hpp"
#include "gobot/error_macros.hpp"

namespace gobot {

Framebuffer *(*Framebuffer::CreateFunc)(const FramebufferDesc &) = nullptr;

Framebuffer *Framebuffer::Create(const FramebufferDesc &framebufferDesc) {
    CRASH_COND_MSG(CreateFunc == nullptr, "No Framebuffer Create Function");
    return CreateFunc(framebufferDesc);
}

static std::unordered_map<std::size_t, Ref<Framebuffer>> framebuffer_cache;

Framebuffer::~Framebuffer()
{
}

Ref<Framebuffer> Framebuffer::Get(const FramebufferDesc &framebufferDesc) {
    size_t hash = 0;
    HashCombine(hash, framebufferDesc.attachmentCount, framebufferDesc.width, framebufferDesc.height,
                framebufferDesc.layer, framebufferDesc.renderPass, framebufferDesc.screenFBO);

    for (uint32_t i = 0; i < framebufferDesc.attachmentCount; i++) {
        HashCombine(hash, framebufferDesc.attachmentTypes[i]);

        if (framebufferDesc.attachments[i]) {
            HashCombine(hash, framebufferDesc.attachments[i]->GetImageHande());
        }
    }

    auto found = framebuffer_cache.find(hash);
    if (found != framebuffer_cache.end() && found->second) {
        return found->second;
    }

    auto framebuffer = Ref<Framebuffer>(Create(framebufferDesc));
    framebuffer_cache[hash] = framebuffer;
    return framebuffer;
}

void Framebuffer::ClearCache() {
    framebuffer_cache.clear();
}

void Framebuffer::DeleteUnusedCache() {
    static const size_t keyDeleteSize = 256;
    static std::size_t keysToDelete[keyDeleteSize];
    std::size_t keysToDeleteCount = 0;

    for (auto &&[key, value]: framebuffer_cache) {
        if (!value) {
            keysToDelete[keysToDeleteCount] = key;
            keysToDeleteCount++;
        }
        if (keysToDeleteCount >= keyDeleteSize)
            break;
    }

    for (std::size_t i = 0; i < keysToDeleteCount; i++) {
        framebuffer_cache[keysToDelete[i]] = nullptr;
        framebuffer_cache.erase(keysToDelete[i]);
    }
}

}