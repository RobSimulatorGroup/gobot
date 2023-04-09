/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Zikun Yu, 23-4-3
*/

#pragma once

#include "gobot/physics/physics_server_3d.h"
#include "gobot/physics/dart/dart_shape_3d.h"

#include <dart/dart.hpp>

namespace gobot {

using DartSkeleton = dart::dynamics::Skeleton;

class DartSkeleton3D {
public:
    DartSkeleton3D() = default;
    virtual ~DartSkeleton3D() = default;

    FORCE_INLINE void SetSelf(const RID &self) { self_ = self; }
    [[nodiscard]] FORCE_INLINE RID GetSelf() const { return self_; }



private:
    RID self_;
    DartSkeleton *skeleton_ = nullptr;
};

} // End of namespace gobot