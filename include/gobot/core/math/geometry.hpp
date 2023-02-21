/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-17
 * This file is modified by Yingnan Wu, 23-2-19
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

template <typename _Scalar, int _Dim, int _Mode, int _Options = Eigen::AutoAlign>
class Transform: public Eigen::Transform<_Scalar, _Dim, _Mode, _Options> {
public:
    using Base = Eigen::Transform<_Scalar, _Dim, _Mode, _Options>;
    using Base::Base;

    Transform(const Eigen::Transform<_Scalar, _Dim, _Mode, _Options>& transform)
        : Eigen::Transform<_Scalar, _Dim, _Mode, _Options>(transform)
    {
    }

    MatrixData<_Scalar> GetMatrixData() const {
        auto self_view = this->matrix().reshaped();
        return {this->rows(), this->cols(),
                std::vector(self_view.begin(), self_view.end())};
    }

    void SetMatrixData(const MatrixData<_Scalar> &data) {
        if (data.rows != _Dim || data.cols != _Dim) [[unlikely]] {
            return;
        }

        if (data.rows * data.cols != data.storage.size()) [[unlikely]] {
            return;
        }

        auto self_view = this->matrix().reshaped();
        std::copy(data.storage.cbegin(), data.storage.cend(), self_view.begin());
    }

    Quaternion<_Scalar> GetQuaternion() const {
        return Quaternion<_Scalar>(this->rotation());
    }

    void SetQuaternion(const Quaternion<_Scalar>& quaternion) const {
        this->linear() = quaternion.toRotationMatrix();
    }

};

}  // end of namespace internal


using AngleAxisd = Eigen::AngleAxisd;
using AngleAxisf = Eigen::AngleAxisf;

using Quaterniond = internal::Quaternion<double>;
using Quaternionf = internal::Quaternion<float>;

using Isometry2d = internal::Transform<double, 2, Eigen::Isometry>;
using Isometry2f = internal::Transform<float, 2, Eigen::Isometry>;
using Isometry3d = internal::Transform<double, 3, Eigen::Isometry>;
using Isometry3f = internal::Transform<float, 3, Eigen::Isometry>;

using Affine2d = internal::Transform<double, 2, Eigen::Affine>;
using Affine2f = internal::Transform<float, 2, Eigen::Affine>;
using Affine3d = internal::Transform<double, 3, Eigen::Affine>;
using Affine3f = internal::Transform<float, 3, Eigen::Affine>;

using Projective2d = internal::Transform<double, 2, Eigen::Projective>;
using Projective2f = internal::Transform<float, 2, Eigen::Projective>;
using Projective3d = internal::Transform<double, 3, Eigen::Projective>;
using Projective3f = internal::Transform<float, 3, Eigen::Projective>;


using AngleAxis = Eigen::AngleAxis<real_t>;

using Quaternion = internal::Quaternion<real_t>;

using Isometry2 = internal::Transform<real_t, 2, Eigen::Isometry>;
using Isometry3 = internal::Transform<real_t, 3, Eigen::Isometry>;

using Affine2 = internal::Transform<real_t,2,Eigen::Affine>;
using Affine3 = internal::Transform<real_t,3,Eigen::Affine>;

using Projective2 = internal::Transform<real_t,2,Eigen::Projective>;
using Projective3 = internal::Transform<real_t,3,Eigen::Projective>;

}  // namespace gobot
