/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
*/

#pragma once

#include "gobot/core/color.hpp"
#include "gobot/core/math/geometry.hpp"
#include "gobot/core/math/matrix.hpp"
#include "gobot/core/rid.hpp"
#include "gobot/scene/resources/material.hpp"
#include "gobot/scene/resources/shape_3d.hpp"

#include "gobot_export.h"

#include <vector>

namespace gobot {

class Node;

struct VisualMeshRenderItem {
    RID mesh;
    Ref<Material> material;
    Matrix4 model = Matrix4::Identity();
    Color surface_color{0.66f, 0.78f, 0.95f, 1.0f};
    RealType metallic = 0.0f;
    RealType roughness = 0.5f;
    RealType specular = 0.5f;
};

struct CollisionDebugRenderItem {
    Ref<Shape3D> shape;
    Affine3 transform = Affine3::Identity();
};

struct SceneRenderItems {
    std::vector<VisualMeshRenderItem> visual_meshes;
    std::vector<CollisionDebugRenderItem> collision_shapes;
};

GOBOT_EXPORT SceneRenderItems CollectSceneRenderItems(const Node* scene_root);

}
