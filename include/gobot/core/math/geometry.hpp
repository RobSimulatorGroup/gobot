/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-17
*/

#pragma once

#include <rttr/type.h>

#include <Eigen/Geometry>

#include "gobot/core/math/math_defs.hpp"

namespace gobot {

namespace internal {
template <typename _Scalar>
class Quaternion : public Eigen::Quaternion<_Scalar> {
 public:
  using Base = Eigen::Quaternion<_Scalar>;

  using Base::Base;

  _Scalar GetX() const { return this->x(); }

  _Scalar GetY() const { return this->y(); }

  _Scalar GetZ() const { return this->z(); }

  _Scalar GetW() const { return this->w(); }

  void SetX(_Scalar x) { this->x() = x; }

  void SetY(_Scalar y) { this->y() = y; }

  void SetZ(_Scalar z) { this->z() = z; }

  void SetW(_Scalar w) { this->w() = w; }
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

using Quaternion = internal::Quaternion<real_t>;
using AngleAxis = Eigen::AngleAxis<real_t>;
using Affine2 = Eigen::Transform<real_t,2,Eigen::Affine>;
using Affine3 = Eigen::Transform<real_t,3,Eigen::Affine>;
using Projective2 = Eigen::Transform<real_t,2,Eigen::Projective>;
using Projective3 = Eigen::Transform<real_t,3,Eigen::Projective>;

}  // namespace gobot
