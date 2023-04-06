/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Zikun Yu, 23-3-24
*/

#include "gobot/physics/dart/dart_shape_3d.h"

namespace gobot {

//void DartShape3D::AddOwner(DartShapeOwner3D *owner) {
//    owner_ = owner;
//}
//
//void DartShape3D::RemoveOwner(DartShapeOwner3D *owner) {
//    owner_ = nullptr;
//}
//
//bool DartShape3D::IsOwner(DartShapeOwner3D *owner) const {
//    return owner == owner_;
//}
//
//DartShapeOwner3D *DartShape3D::GetOwner() const {
//    return owner_;
//}
//
//DartBoxShape3D::DartBoxShape3D(const Vector3d &extents) {
//    extents_ = extents;
//    shape_ = std::make_shared<BoxShape3D>(extents);
//}
//
//double DartBoxShape3D::GetVolume() const {
//    return shape_->getVolume();
//}
//
//Matrix3d DartBoxShape3D::GetInertia(real_t mass) const {
//    return shape_->computeInertia(mass);
//}
//
//void DartBoxShape3D::SetData(const Variant &data) {
//    extents_ = data.get_value<Vector3d>();
//    dynamic_pointer_cast<BoxShape3D>(shape_)->setSize(extents_);
//}
//
//Variant DartBoxShape3D::GetData() const {
//    return extents_;
//}

} // End of namespace gobot