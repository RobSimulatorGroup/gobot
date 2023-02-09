/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 22-12-9
*/

#pragma once

#include <cstdint>
#include "gobot/core/object_id.hpp"
#include "gobot/core/macros.hpp"

namespace gobot {

class ObjectID {
public:
    ALWAYS_INLINE ObjectID() {}

    ALWAYS_INLINE explicit ObjectID(const uint64_t id) { id_ = id; }

    ALWAYS_INLINE explicit ObjectID(const int64_t id) { id_ = id; }

    // The First bits of uint64_t marks it is a reference.
    [[nodiscard]] ALWAYS_INLINE bool IsRefCounted() const { return (id_ & (uint64_t(1) << 63)) != 0; }

    [[nodiscard]] ALWAYS_INLINE bool IsValid() const { return id_ != 0; }

    [[nodiscard]] ALWAYS_INLINE bool IsNull() const { return id_ == 0; }

    ALWAYS_INLINE operator uint64_t() const { return id_; }

    ALWAYS_INLINE operator int64_t() const { return id_; }

    ALWAYS_INLINE bool operator==(const ObjectID &id) const { return id_ == id.id_; }

    ALWAYS_INLINE bool operator!=(const ObjectID &id) const { return id_ != id.id_; }

    ALWAYS_INLINE bool operator<(const ObjectID &id) const { return id_ < id.id_; }

    ALWAYS_INLINE void operator=(int64_t p_int64) { id_ = p_int64; }

    ALWAYS_INLINE void operator=(uint64_t p_uint64) { id_ = p_uint64; }

private:
    std::uint64_t id_ = 0; // 0 is invalid
};

}

namespace std
{
template <>
struct hash<gobot::ObjectID>
{
    std::size_t operator()(const gobot::ObjectID& id) const
    {
        return std::hash<std::uint64_t>()(id.operator int64_t());
    }
};
}