/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <vector>

#include "gobot/scene/node_3d.hpp"
#include "gobot/scene/resources/physics_material_3d.hpp"
#include "gobot/scene/resources/shape_3d.hpp"

namespace gobot {

class GOBOT_EXPORT CollisionShape3D : public Node3D {
    GOBCLASS(CollisionShape3D, Node3D)

public:
    CollisionShape3D() = default;

    void SetShape(const Ref<Shape3D>& shape);

    const Ref<Shape3D>& GetShape() const;

    void SetDisabled(bool disabled);

    bool IsDisabled() const;

    void SetPhysicsMaterial(const Ref<PhysicsMaterial3D>& material);
    const Ref<PhysicsMaterial3D>& GetPhysicsMaterial() const;

    void SetCollisionLayer(std::uint32_t layer);
    std::uint32_t GetCollisionLayer() const;

    void SetCollisionMask(std::uint32_t mask);
    std::uint32_t GetCollisionMask() const;

    void SetContactOffset(RealType offset);
    RealType GetContactOffset() const;

    void SetRestOffset(RealType offset);
    RealType GetRestOffset() const;

private:
    Ref<Shape3D> shape_{nullptr};
    Ref<PhysicsMaterial3D> physics_material_{nullptr};
    bool disabled_{false};
    std::uint32_t collision_layer_{1};
    std::uint32_t collision_mask_{1};
    RealType contact_offset_{0.0};
    RealType rest_offset_{0.0};
};

}
