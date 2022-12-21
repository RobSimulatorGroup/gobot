/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 22-12-21
*/

#pragma once

#include "gobot/scene/resources/mesh.hpp"

namespace gobot {

class PrimitiveMesh : public Mesh {
    GOBCLASS(PrimitiveMesh, Mesh)
public:

};


class BoxMesh : public PrimitiveMesh {
    GOBCLASS(BoxMesh, PrimitiveMesh)
public:

};

class CylinderMesh : public PrimitiveMesh {
    GOBCLASS(CylinderMesh, PrimitiveMesh)
public:

};

class PlaneMesh : public PrimitiveMesh {
    GOBCLASS(PlaneMesh, PrimitiveMesh)
public:

};

class SphereMesh : public PrimitiveMesh {
    GOBCLASS(SphereMesh, PrimitiveMesh)
public:

};


class CapsuleMesh : public PrimitiveMesh {
    GOBCLASS(CapsuleMesh, PrimitiveMesh)
public:

};



} // end of namespace gobot
