/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Zikun Yu, 23-3-24
*/

#pragma once

#include "gobot/physics/physics_server_3d.h"

#include <dart/dart.hpp>

namespace gobot {

using DartShape = dart::dynamics::Shape;
using DartBoxShape = dart::dynamics::BoxShape;
using DartShapeNode = dart::dynamics::ShapeNode;

class DartShape3D;

class DartShapeOwner3D {
public:
    virtual void ShapeChanged() = 0;
    virtual void RemoveShape(DartShape3D *shape) = 0;

    virtual ~DartShapeOwner3D() = default;
};

class DartShape3D {

friend class DartBody3D;

public:
    DartShape3D() = default;
    virtual ~DartShape3D() {
        ERR_FAIL_COND(shape_node_);
    }

    FORCE_INLINE void SetSelf(const RID &self) { self_ = self; }
    [[nodiscard]] FORCE_INLINE RID GetSelf() const { return self_; }

    [[nodiscard]] virtual PhysicsServer3D::ShapeType GetType() const = 0;

    virtual void SetData(const Variant &data) = 0;
    [[nodiscard]] virtual Variant GetData() const = 0;

protected:
    std::shared_ptr<DartShape> shape_;
    DartShapeNode *shape_node_ = nullptr;

private:
    RID self_;
};

class DartBoxShape3D : public DartShape3D {
public:
    DartBoxShape3D() = default;

    [[nodiscard]] PhysicsServer3D::ShapeType GetType() const override { return PhysicsServer3D::ShapeType::Box; }

    void SetData(const Variant &data) override {
        const auto& extents = data.get_value<Vector3d>();
        dynamic_pointer_cast<DartBoxShape>(shape_)->setSize(extents);
    }

    [[nodiscard]] Variant GetData() const override {
        Vector3d extents = dynamic_pointer_cast<DartBoxShape>(shape_)->getSize();
        return extents;
    }
};

} // End of namespace gobot
