/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-17
*/

#pragma once

#include <rttr/type.h>

#include <Eigen/Geometry>

namespace gobot {

namespace internal {
template <typename T>
class Quaternion : public Eigen::Quaternion<T> {
 public:
  using EigenBase = Eigen::Quaternion<T>;

  using EigenBase::EigenBase;

  T GetX() const { return this->x(); }

  T GetY() const { return this->y(); }

  T GetZ() const { return this->z(); }

  T GetW() const { return this->w(); }

  void SetX(T x) { this->x() = x; }

  void SetY(T y) { this->y() = y; }

  void SetZ(T z) { this->z() = z; }

  void SetW(T w) { this->w() = w; }
};

}  // end of namespace internal

using Quaterniond = internal::Quaternion<double>;
using Quaternionf = internal::Quaternion<float>;

using AngleAxisd = Eigen::AngleAxisd;
using AngleAxisf = Eigen::AngleAxisf;

using Affine2d = Eigen::Affine2d;
using Affine2f = Eigen::Affine2f;
using Affine3d = Eigen::Affine3d;
using Affine3f = Eigen::Affine3f;

using Projective2d = Eigen::Projective2d;
using Projective2f = Eigen::Projective2f;
using Projective3d = Eigen::Projective3d;
using Projective3f = Eigen::Projective3f;

#ifdef MATRIX_IS_DOUBLE
using Quaternion = Quaterniond;
using AngleAxis = AngleAxisd;
using Affine2 = Affine2d;
using Affine3 = Affine3d;
using Projective2 = Projective2d;
using Projective3 = Projective2d;
#else
using Quaternion = Quaternionf;
using AngleAxis = AngleAxisf;
using Affine2 = Affine2f;
using Affine3 = Affine3f;
using Projective2 = Projective2f;
using Projective3 = Projective2f;
#endif

}  // namespace gobot
