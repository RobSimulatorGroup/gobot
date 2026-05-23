/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <vector>

#include "gobot/scene/node_3d.hpp"
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

    void SetFriction(const Vector3& friction);

    const Vector3& GetFriction() const;

    void SetContactType(int contype);

    int GetContactType() const;

    void SetContactAffinity(int conaffinity);

    int GetContactAffinity() const;

    void SetContactDimension(int condim);

    int GetContactDimension() const;

    void SetSolref(const Vector2& solref);

    const Vector2& GetSolref() const;

    void SetSolimp(const std::vector<RealType>& solimp);

    const std::vector<RealType>& GetSolimp() const;

    void SetMargin(RealType margin);

    RealType GetMargin() const;

    void SetGap(RealType gap);

    RealType GetGap() const;

private:
    Ref<Shape3D> shape_{nullptr};
    bool disabled_{false};
    Vector3 friction_{1.0, 0.005, 0.0001};
    int contype_{1};
    int conaffinity_{1};
    int condim_{3};
    Vector2 solref_{0.02, 1.0};
    std::vector<RealType> solimp_{0.9, 0.95, 0.001, 0.5, 2.0};
    RealType margin_{0.0};
    RealType gap_{0.0};
};

}
