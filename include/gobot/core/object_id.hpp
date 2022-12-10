/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 22-12-9
*/

#pragma once

#include <cstdint>
#include "gobot/core/object_id.hpp"
#include "gobot/core/marcos.hpp"

class ObjectID {
public:
    ALWAYS_INLINE ObjectID() {}

    ALWAYS_INLINE explicit ObjectID(const uint64_t p_id) { id = p_id; }
    ALWAYS_INLINE explicit ObjectID(const int64_t p_id) { id = p_id; }

    // The First bits of uint64_t marks it is a reference.
    [[nodiscard]] ALWAYS_INLINE bool IsRefCounted() const { return (id & (uint64_t(1) << 63)) != 0; }

    ALWAYS_INLINE bool IsValid() const { return id != 0; }
    ALWAYS_INLINE bool IsNull() const { return id == 0; }

    ALWAYS_INLINE operator uint64_t() const { return id; }
    ALWAYS_INLINE operator int64_t() const { return id; }

    ALWAYS_INLINE bool operator==(const ObjectID &p_id) const { return id == p_id.id; }
    ALWAYS_INLINE bool operator!=(const ObjectID &p_id) const { return id != p_id.id; }
    ALWAYS_INLINE bool operator<(const ObjectID &p_id) const { return id < p_id.id; }

    ALWAYS_INLINE void operator=(int64_t p_int64) { id = p_int64; }
    ALWAYS_INLINE void operator=(uint64_t p_uint64) { id = p_uint64; }

private:
    std::uint64_t id = 0; // 0 is invalid
};