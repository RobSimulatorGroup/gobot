/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-3-24
*/

#pragma once

#include "gobot/rendering/render_rid.hpp"
#include "gobot/error_macros.hpp"
#include <unordered_map>

namespace gobot {

template <class T, bool THREAD_SAFE = false>
class RenderRID_Owner {
public:
    RenderRID_Owner() = default;

    virtual ~RenderRID_Owner() = default;

    void InitializeRID(RenderRID rid, const T &value) {
        if (!rid.IsValid()) {
            ERR_FAIL_MSG("Cannot InitializeRID because T");
        }

        if (THREAD_SAFE) {
            spin_lock_.lock();
        }

        auto it = owner_map_.find(rid);

        if (it != owner_map_.end()) {
            spin_lock_.unlock();
            ERR_FAIL_MSG("RenderRID is already inside of owner");
        }

        owner_map_.insert({rid, value});

        if (THREAD_SAFE) {
            spin_lock_.unlock();
        }
    }

    void SetDescription(const char *description) {
        description_ = description;
    }

    FORCE_INLINE T* GetOrNull(const RenderRID &rid) {
        if (!rid.IsValid()) {
            return nullptr;
        }

        if (THREAD_SAFE) {
            spin_lock_.lock();
        }

        auto it = owner_map_.find(rid);

        if (it == owner_map_.end()) {
            spin_lock_.unlock();
            ERR_FAIL_V_MSG(nullptr, "RenderRID is not inside of owner");
        }

        if (THREAD_SAFE) {
            spin_lock_.unlock();
        }
        return &(it->second);
    }

    FORCE_INLINE void Free(const RenderRID& rid) {
        if (!rid.IsValid()) {
            ERR_FAIL_MSG("Cannot Free because T");
        }

        if (THREAD_SAFE) {
            spin_lock_.lock();
        }

        auto it = owner_map_.find(rid);

        if (it == owner_map_.end()) {
            spin_lock_.unlock();
            ERR_FAIL_MSG("RenderRID is not inside of owner");
        }

        owner_map_.erase(it);

        if (THREAD_SAFE) {
            spin_lock_.unlock();
        }
    }

    FORCE_INLINE bool Owns(const RenderRID &rid) const {
        if (THREAD_SAFE) {
            spin_lock_.lock();
        }

        bool owned = owner_map_.contains(rid);

        if (THREAD_SAFE) {
            spin_lock_.unlock();
        }

        return owned;
    }

    std::vector<RenderRID> GetOwnedList() const {
        if (THREAD_SAFE) {
            spin_lock_.lock();
        }
        std::vector<RenderRID> owned_list;
        for (const auto& [rid, _]: owner_map_) {
            if (rid.IsValid()) {
                owned_list.emplace_back(rid);
            }
        }
        if (THREAD_SAFE) {
            spin_lock_.unlock();
        }
        return owned_list;
    }


private:
    std::unordered_map<RenderRID, T> owner_map_{};

    const char *description_ = nullptr;

    mutable SpinLock spin_lock_{};
};


}
