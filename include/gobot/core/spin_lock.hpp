/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 22-12-10
*/

#pragma once

#include <atomic>
#include "gobot/core/macros.hpp"

namespace gobot {

class SpinLock {
    std::atomic_flag locked = ATOMIC_FLAG_INIT;

public:
    ALWAYS_INLINE void lock() {
        while (locked.test_and_set(std::memory_order_acquire)) {
            // Continue.
        }
    }
    ALWAYS_INLINE void unlock() {
        locked.clear(std::memory_order_release);
    }
};

}

