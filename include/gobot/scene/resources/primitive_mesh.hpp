/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 22-12-21
*/

#pragma once

#include "gobot/scene/resources/mesh.hpp"
#include "gobot/scene/resources/material.hpp"

namespace gobot {

class GOBOT_API PrimitiveMesh : public Mesh {
    GOBCLASS(PrimitiveMesh, Mesh)
public:
    PrimitiveMesh();

    void SetMaterial(const Ref<Material>& material);

    Ref<Material> GetMaterial() const;

private:
    Ref<Material> material_{nullptr};
};


class GOBOT_API BoxMesh : public PrimitiveMesh {
    GOBCLASS(BoxMesh, PrimitiveMesh)
public:

    BoxMesh();

    void SetWidth(float width);

    float GetWidth() const;

private:
    float width_{0.5};
};

class GOBOT_API CylinderMesh : public PrimitiveMesh {
    GOBCLASS(CylinderMesh, PrimitiveMesh)
public:
    CylinderMesh();

private:

};

class GOBOT_API PlaneMesh : public PrimitiveMesh {
    GOBCLASS(PlaneMesh, PrimitiveMesh)
public:
    PlaneMesh();
};

class GOBOT_API SphereMesh : public PrimitiveMesh {
    GOBCLASS(SphereMesh, PrimitiveMesh)
public:
    SphereMesh();

};


class GOBOT_API CapsuleMesh : public PrimitiveMesh {
    GOBCLASS(CapsuleMesh, PrimitiveMesh)
public:

    CapsuleMesh();
};



} // end of namespace gobot
