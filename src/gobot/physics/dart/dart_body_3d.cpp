/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Zikun Yu, 23-4-3
*/

#include "gobot/physics/dart/dart_body_3d.h"

namespace gobot {

void DartBody3D::AddShape(DartShape3D *shape, const Affine3 &transform) {
    ERR_FAIL_NULL(body_node_);

    shape->shape_node_ = body_node_->createShapeNodeWith<CollisionAspect, DynamicsAspect>(shape->shape_);
    Isometry3 tfm;
    tfm.linear() = transform.linear();
    tfm.translation() = transform.translation();
    shape->shape_node_->setRelativeTransform(tfm.cast<double>());
    shapes_.push_back(shape);

    // todo: add owner for callback
}

void DartBody3D::SetShapeTransform(std::size_t index, const Affine3 &transform) {
    ERR_FAIL_INDEX(index, shapes_.size());
    ERR_FAIL_NULL(body_node_);

    Isometry3 tfm;
    tfm.linear() = transform.linear();
    tfm.translation() = transform.translation();
    shapes_[index]->shape_node_->setRelativeTransform(tfm.cast<double>());
}

void DartBody3D::RemoveShape(DartShape3D *shape) {
    // remove a shape, all the times it appears
    for (int i = 0; i < shapes_.size(); ++ i) {
        if (shapes_[i]->shape_node_->getShape() == shape->shape_) {
            RemoveShape(i);
            i --;
        }
    }
}

void DartBody3D::RemoveShape(std::size_t index) {
    ERR_FAIL_INDEX(index, shapes_.size());

    // todo: shape remove owner
    shapes_[index]->shape_node_->remove();
    shapes_[index]->shape_node_ = nullptr;
    shapes_.erase(std::remove(shapes_.begin(), shapes_.end(), shapes_[index]), shapes_.end());
}

} // End of namespace gobot