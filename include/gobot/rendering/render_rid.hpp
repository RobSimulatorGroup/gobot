/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-3-17
*/

#pragma once

#include "gobot/core/rid.h"
#include "gobot/core/object.hpp"

namespace gobot {

class RenderRID {
public:
    RenderRID() = default;

    static ALWAYS_INLINE RenderRID FromUint16(uint16_t id) {
        RenderRID rid;
        rid.id_ = id;
        return rid;
    }

    ALWAYS_INLINE bool operator==(const RenderRID &rid) const {
        return id_ == rid.id_;
    }

    operator void* () const {
        return reinterpret_cast<void*>(id_);
    }

    operator std::uint16_t () const {
        return id_;
    }

    [[nodiscard]] ALWAYS_INLINE std::uint16_t GetID() const { return id_; }

    ALWAYS_INLINE bool operator<(const RenderRID &rid) const {
        return id_ < rid.id_;
    }

    ALWAYS_INLINE bool operator<=(const RenderRID &rid) const {
        return id_ <= rid.id_;
    }

    ALWAYS_INLINE bool operator>(const RenderRID &rid) const {
        return id_ > rid.id_;
    }

    ALWAYS_INLINE bool operator>=(const RenderRID &rid) const {
        return id_ >= rid.id_;
    }

    ALWAYS_INLINE bool operator!=(const RenderRID &rid) const {
        return id_ != rid.id_;
    }

    [[nodiscard]] virtual ALWAYS_INLINE bool IsValid() const { return id_ != UINT16_MAX; }

    [[nodiscard]] virtual ALWAYS_INLINE bool IsNull() const { return id_ == UINT16_MAX; }

private:
    std::uint16_t id_{UINT16_MAX};
};

}

namespace std
{
template <>
struct hash<gobot::RenderRID>
{
    std::size_t operator()(const gobot::RenderRID& rid) const
    {
        return std::hash<std::uint64_t>()(rid.operator std::uint16_t());
    }
};
}
