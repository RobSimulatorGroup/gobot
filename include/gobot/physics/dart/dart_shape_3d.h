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

using Shape3D = dart::dynamics::Shape;
using BoxShape3D = dart::dynamics::BoxShape;

class DartShape3D;

//class DartShapeOwner3D {
//public:
//    virtual void ShapeChanged() = 0;
//    virtual void RemoveShape(DartShape3D *shape) = 0;
//
//    virtual ~DartShapeOwner3D() = default;
//};

class DartShape3D {
public:
    DartShape3D() = default;
    virtual ~DartShape3D() = default;

    [[nodiscard]] virtual double GetVolume() const = 0;
    [[nodiscard]] virtual Matrix3d GetInertia(real_t mass) const = 0;

    FORCE_INLINE void SetSelf(const RID &self) { self_ = self; }
    [[nodiscard]] FORCE_INLINE RID GetSelf() const { return self_; }

    [[nodiscard]] virtual PhysicsServer3D::ShapeType GetType() const = 0;

    // todo: GetAABB/GetBoundingBox

//    FORCE_INLINE bool IsConfigured() const { return configured_; }

    virtual void SetData(const Variant &data) = 0;
    [[nodiscard]] virtual Variant GetData() const = 0;

    // todo: void AddOwner(DartShapeOwner3D *owner);
    // todo: void RemoveOwner(DartShapeOwner3D *owner);
    // todo: bool IsOwner(DartShapeOwner3D *owner) const;
    // todo: const std::unordered_map<std::size_t, DartShapeOwner3D *> &GetOwners() const;

protected:
//    void Configure(std::shared_ptr<Shape3D> *shape);
//    std::shared_ptr<Shape3D> shape_;

private:
    RID self_;
//    bool configured_ = false;
//    std::unordered_map<std::size_t, DartShapeOwner3D *> owners_;
};

class DartBoxShape3D : public DartShape3D {
public:
    explicit DartBoxShape3D(const Vector3d &extents = Vector3d::Identity());
    ~DartBoxShape3D() override = default;

    [[nodiscard]] FORCE_INLINE Vector3d GetExtents() const { return extents_; }

    [[nodiscard]] double GetVolume() const override;
    [[nodiscard]] Matrix3d GetInertia(real_t mass) const override;

    [[nodiscard]] PhysicsServer3D::ShapeType GetType() const override { return PhysicsServer3D::ShapeType::Box; }

    void SetData(const Variant &data) override;
    [[nodiscard]] Variant GetData() const override;

private:
    std::shared_ptr<BoxShape3D> shape_;
    Vector3d extents_;
};

} // End of namespace gobot
