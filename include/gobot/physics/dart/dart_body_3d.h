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

using DartBodyNode = dart::dynamics::BodyNode;
using CollisionAspect = dart::dynamics::CollisionAspect;
using DynamicsAspect = dart::dynamics::DynamicsAspect;

class DartShape3D;
class DartShapeOwner3D;

class DartBody3D : public DartShapeOwner3D {

public:
    DartBody3D() = default;
    ~DartBody3D() override = default;;

    FORCE_INLINE void SetSelf(const RID &self) { self_ = self; }
    [[nodiscard]] FORCE_INLINE RID GetSelf() const { return self_; }

    FORCE_INLINE void SetInstanceID(const ObjectID &instance_id) { instance_id_ = instance_id; }
    [[nodiscard]] FORCE_INLINE ObjectID GetInstanceID() const { return instance_id_; }

    void ShapeChanged() override {
        // todo: check collision
    }

    void AddShape(DartShape3D *shape, const Affine3 &transform = Affine3::Identity());
    // todo: move shape node from another body to this body
    // todo: void SetShape(DartShape3D *shape);
    void SetShapeTransform(std::size_t index, const Affine3 &transform = Affine3::Identity());
    [[nodiscard]] FORCE_INLINE std::size_t GetShapeCount() const { return shapes_.size(); }
    [[nodiscard]] FORCE_INLINE DartShape3D *GetShape(std::size_t index) const {
        // todo: CRASH_BAD_INDEX
        ERR_FAIL_INDEX_V(index, shapes_.size(), nullptr);
        return shapes_[index];
    }

    void RemoveShape(DartShape3D *shape) override;
    void RemoveShape(std::size_t index);

    void SetParam(PhysicsServer3D::BodyParameter param, const Variant &value);
    [[nodiscard]] Variant GetParam(PhysicsServer3D::BodyParameter param) const;

private:
    RID self_;
    ObjectID instance_id_;
    DartBodyNode *body_node_ = nullptr;
    std::vector<DartShape3D *> shapes_;
};

} // End of namespace gobot