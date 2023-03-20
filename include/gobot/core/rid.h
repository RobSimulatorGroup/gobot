/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Zikun Yu, 23-3-20
*/

#pragma once

#include <cstdint>
#include "gobot/core/macros.hpp"

namespace gobot {

class RID {
    friend class RID_AllocBase;
    uint64_t id_ = 0;

public:
    ALWAYS_INLINE bool operator==(const RID &rid) const {
        return id_ == rid.id_;
    }

    ALWAYS_INLINE bool operator<(const RID &rid) const {
        return id_ < rid.id_;
    }

    ALWAYS_INLINE bool operator<=(const RID &rid) const {
        return id_ <= rid.id_;
    }

    ALWAYS_INLINE bool operator>(const RID &rid) const {
        return id_ > rid.id_;
    }

    ALWAYS_INLINE bool operator>=(const RID &rid) const {
        return id_ >= rid.id_;
    }

    ALWAYS_INLINE bool operator!=(const RID &rid) const {
        return id_ != rid.id_;
    }

    ALWAYS_INLINE bool IsValid() const { return id_ != 0; }

    ALWAYS_INLINE bool IsNull() const { return id_ == 0; }

    ALWAYS_INLINE uint32_t GetLocalIndex() const { return id_ & 0xFFFFFFFF; }

    static ALWAYS_INLINE RID FromUint64(uint64_t id) {
        RID rid;
        rid.id_ = id;
        return rid;
    }

    ALWAYS_INLINE uint64_t GetID() const { return id_; }

    ALWAYS_INLINE RID() = default;
};

} // End of namespace gobot