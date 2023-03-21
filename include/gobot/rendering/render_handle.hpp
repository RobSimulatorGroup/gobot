/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-3-17
*/

#pragma once

#include "gobot/core/macros.hpp"
#include "gobot/rendering/render_types.hpp"

namespace gobot {

#define HANDLE_IMPL(HandleType)                         \
public:                                                 \
    HandleType GetHandle() const { return handle_; }    \
                                                        \
    auto GetHandleID() const                            \
    {                                                   \
        return reinterpret_cast<void*>(handle_.idx);    \
    }                                                   \
                                                        \
    static HandleType InvalidHandle()                   \
    {                                                   \
        HandleType invalid = {bgfx::kInvalidHandle};    \
        return invalid;                                 \
    }                                                   \
                                                        \
    [[nodiscard]] FORCE_INLINE bool IsValid() const     \
    {                                                   \
        return bgfx::isValid(handle_);                  \
    }                                                   \
                                                        \
    void DisposeHandle()                                \
    {                                                   \
       if(IsValid()) {                                  \
           bgfx::destroy(handle_);                      \
       }                                                \
                                                        \
       handle_ = InvalidHandle();                       \
    }                                                   \
                                                        \
                                                        \
protected:                                              \
    HandleType handle_ = InvalidHandle();


}
