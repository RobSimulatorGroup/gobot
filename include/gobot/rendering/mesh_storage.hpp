/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-4-21
*/

#pragma once

#include "gobot/core/rid.hpp"
#include "gobot/core/rid_owner.hpp"
#include "gobot/core/math/matrix.hpp"

namespace gobot {

class MeshStorage {
public:
    virtual ~MeshStorage() {}

    virtual RID MeshAllocate() = 0;

    virtual void MeshInitialize(const RID& p_rid) = 0;

    virtual void MeshSetBox(const RID& p_rid, const Vector3& size) = 0;

    virtual void MeshSetCylinder(const RID& p_rid, RealType radius, RealType height, int radial_segments) = 0;

    virtual void MeshSetSphere(const RID& p_rid, RealType radius, int radial_segments, int rings) = 0;

    virtual bool OwnsMesh(const RID& p_rid) const = 0;

    virtual void MeshFree(const RID& p_rid) = 0;
};

}
