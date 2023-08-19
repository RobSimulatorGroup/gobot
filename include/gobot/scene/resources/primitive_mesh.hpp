/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 22-12-21
*/

#pragma once

#include "gobot/scene/resources/mesh.hpp"
#include "gobot/scene/resources/material.hpp"
#include "gobot/core/rid.hpp"
#include "gobot/core/math/matrix.hpp"

namespace gobot {

class GOBOT_EXPORT PrimitiveMesh : public Mesh {
    GOBCLASS(PrimitiveMesh, Mesh)
public:
    PrimitiveMesh();

    ~PrimitiveMesh();

    void SetMaterial(const Ref<Material>& material);

    const Ref<Material>& GetMaterial() const;

private:
    RID mesh_;
    Ref<Material> material_{nullptr};
};


class GOBOT_EXPORT BoxMesh : public PrimitiveMesh {
    GOBCLASS(BoxMesh, PrimitiveMesh)
public:
    BoxMesh();

    void SetWidth(RealType p_width);

    RealType GetWidth() const;

    void SetSize(Vector3 size);

    const Vector3& GetSize() const;

private:
    Vector3 size_ = Vector3::Ones();
};

class GOBOT_EXPORT CylinderMesh : public PrimitiveMesh {
    GOBCLASS(CylinderMesh, PrimitiveMesh)
public:
    CylinderMesh();

private:

};

class GOBOT_EXPORT PlaneMesh : public PrimitiveMesh {
    GOBCLASS(PlaneMesh, PrimitiveMesh)
public:
    PlaneMesh();
};

class GOBOT_EXPORT SphereMesh : public PrimitiveMesh {
    GOBCLASS(SphereMesh, PrimitiveMesh)
public:
    SphereMesh();

};


class GOBOT_EXPORT CapsuleMesh : public PrimitiveMesh {
    GOBCLASS(CapsuleMesh, PrimitiveMesh)
public:

    CapsuleMesh();
};



} // end of namespace gobot
