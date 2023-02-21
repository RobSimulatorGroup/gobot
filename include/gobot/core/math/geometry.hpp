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
#include "gobot/core/math/matrix.hpp"
#include "gobot/core/macros.hpp"

namespace gobot {

namespace internal {
template <typename Scalar>
class Quaternion : public Eigen::Quaternion<Scalar> {
public:
    using Base = Eigen::Quaternion<Scalar>;
    using Base::Base;

    Scalar GetX() const { return this->x(); }
    Scalar GetY() const { return this->y(); }
    Scalar GetZ() const { return this->z(); }
    Scalar GetW() const { return this->w(); }

    void SetX(Scalar x) { this->x() = x; }
    void SetY(Scalar y) { this->y() = y; }
    void SetZ(Scalar z) { this->z() = z; }
    void SetW(Scalar w) { this->w() = w; }
};


template <typename Scalar>
using EulerAngle = Matrix<Scalar, 3, 1>;

template <typename Scalar, int Dim, int Mode, int Options = Eigen::AutoAlign>
class Transform: public Eigen::Transform<Scalar, Dim, Mode, Options> {
public:
    using Base = Eigen::Transform<Scalar, Dim, Mode, Options>;
    using Base::Base;

    Transform(const Eigen::Transform<Scalar, Dim, Mode, Options>& transform)
        : Eigen::Transform<Scalar, Dim, Mode, Options>(transform)
    {
    }

    std::vector<Scalar> GetMatrixData() const {
        auto self_view = this->matrix().reshaped();
        return {std::vector(self_view.begin(), self_view.end())};
    }

    void SetMatrixData(const std::vector<Scalar> &data) {
        auto self_view = this->matrix().reshaped();
        if (self_view.size() != data.size()) [[unlikely]] {
            return;
        }
        std::copy(data.cbegin(), data.cend(), self_view.begin());
    }

    Quaternion<Scalar> GetQuaternion() const {
        static_assert(Dim == 3 && Mode == Eigen::Isometry, "GetQuaternion can only called when Dim is 3");
        return Quaternion<Scalar>(this->rotation());
    }

    void SetQuaternion(const Quaternion<Scalar>& quaternion) {
        static_assert(Dim == 3 && Mode == Eigen::Isometry, "SetQuaternion can only called when Dim is 3");
        this->linear() = quaternion.normalized().toRotationMatrix();
    }

    Scalar GetAngle() const {
        static_assert(Dim == 2 && Mode == Eigen::Isometry, "GetAngle can only called when Dim is 2");
        return Eigen::Rotation2D(this->linear()).angle();
    }

    void SetAngle(const Scalar& angle) {
        static_assert(Dim == 2 && Mode == Eigen::Isometry, "SetAngle can only called when Dim is 2");
        this->linear() = Eigen::Rotation2D(angle).toRotationMatrix();
    }

    EulerAngle<Scalar> GetEulerAngle(EulerOrder euler_order) const {
        static_assert(Dim == 3 && Mode == Eigen::Isometry, "GetEulerAngle can only called when Dim is 3");
        switch (euler_order) {
            case EulerOrder::RXYZ:
                return this->rotation().eulerAngles(0, 1, 2);
            case EulerOrder::RYZX:
                return this->rotation().eulerAngles(1, 2, 0);
            case EulerOrder::RZYX:
                return this->rotation().eulerAngles(2, 1, 0);
            case EulerOrder::RXZY:
                return this->rotation().eulerAngles(0, 2, 1);
            case EulerOrder::RXZX:
                return this->rotation().eulerAngles(0, 2, 0);
            case EulerOrder::RYXZ:
                return this->rotation().eulerAngles(1, 0, 2);
            case EulerOrder::SXYZ: {
                auto euler_angle = this->rotation().eulerAngles(2, 1, 0);
                std::swap(euler_angle[0], euler_angle[2]);
                return euler_angle;
            }
            case EulerOrder::SYZX: {
                auto euler_angle = this->rotation().eulerAngles(0, 2, 1);
                std::swap(euler_angle[0], euler_angle[2]);
                return euler_angle;
            }
            case EulerOrder::SZYX: {
                auto euler_angle = this->rotation().eulerAngles(0, 1, 2);
                std::swap(euler_angle[0], euler_angle[2]);
                return euler_angle;
            }
            case EulerOrder::SXZY: {
                auto euler_angle = this->rotation().eulerAngles(1, 2, 0);
                std::swap(euler_angle[0], euler_angle[2]);
                return euler_angle;
            }
            case EulerOrder::SXZX: {
                auto euler_angle = this->rotation().eulerAngles(0, 2, 0);
                std::swap(euler_angle[0], euler_angle[2]);
                return euler_angle;
            }
            case EulerOrder::SYXZ: {
                auto euler_angle = this->rotation().eulerAngles(2, 0, 1);
                std::swap(euler_angle[0], euler_angle[2]);
                return euler_angle;
            }
        }
        // never go here
        return {};
    }

    void SetEulerAngle(const EulerAngle<Scalar>& angles , EulerOrder euler_order) {
        static_assert(Dim == 3 && Mode == Eigen::Isometry, "SetEulerAngle can only called when Dim is 3");
        static auto unit_x = Matrix<Scalar, 3, 1>::UnitX();
        static auto unit_y = Matrix<Scalar, 3, 1>::UnitY();
        static auto unit_z = Matrix<Scalar, 3, 1>::UnitZ();
        switch (euler_order) {
            case EulerOrder::RXYZ:
                this->linear() = (Eigen::AngleAxis<Scalar>(angles.x(), unit_x) *
                        Eigen::AngleAxis<Scalar>(angles.y(), unit_y) *
                        Eigen::AngleAxis<Scalar>(angles.z(), unit_z)).toRotationMatrix();
                break;
            case EulerOrder::RYZX:
                this->linear() = Eigen::AngleAxis<Scalar>(angles.x(), unit_y) *
                        Eigen::AngleAxis<Scalar>(angles.y(), unit_z) *
                        Eigen::AngleAxis<Scalar>(angles.z(), unit_x).toRotationMatrix();
                break;
            case EulerOrder::RZYX:
                this->linear() = Eigen::AngleAxis<Scalar>(angles.x(), unit_z) *
                        Eigen::AngleAxis<Scalar>(angles.y(), unit_y) *
                        Eigen::AngleAxis<Scalar>(angles.z(), unit_x).toRotationMatrix();
                break;
            case EulerOrder::RXZY:
                this->linear() = Eigen::AngleAxis<Scalar>(angles.x(), unit_x) *
                        Eigen::AngleAxis<Scalar>(angles.y(), unit_z) *
                        Eigen::AngleAxis<Scalar>(angles.z(), unit_y).toRotationMatrix();
                break;
            case EulerOrder::RXZX:
                this->linear() = Eigen::AngleAxis<Scalar>(angles.x(), unit_x) *
                        Eigen::AngleAxis<Scalar>(angles.y(), unit_z) *
                        Eigen::AngleAxis<Scalar>(angles.z(), unit_z).toRotationMatrix();
                break;
            case EulerOrder::RYXZ:
                this->linear() = Eigen::AngleAxis<Scalar>(angles.x(), unit_y) *
                        Eigen::AngleAxis<Scalar>(angles.y(), unit_x) *
                        Eigen::AngleAxis<Scalar>(angles.z(), unit_z).toRotationMatrix();
                break;
            case EulerOrder::SXYZ:
                this->linear() = Eigen::AngleAxis<Scalar>(angles.z(), unit_z) *
                        Eigen::AngleAxis<Scalar>(angles.y(), unit_y) *
                        Eigen::AngleAxis<Scalar>(angles.x(), unit_x).toRotationMatrix();
                break;
            case EulerOrder::SYZX:
                this->linear() = Eigen::AngleAxis<Scalar>(angles.z(), unit_x) *
                        Eigen::AngleAxis<Scalar>(angles.y(), unit_z) *
                        Eigen::AngleAxis<Scalar>(angles.x(), unit_y).toRotationMatrix();
                break;
            case EulerOrder::SZYX:
                this->linear() = Eigen::AngleAxis<Scalar>(angles.z(), unit_x) *
                        Eigen::AngleAxis<Scalar>(angles.y(), unit_y) *
                        Eigen::AngleAxis<Scalar>(angles.x(), unit_z).toRotationMatrix();
                break;
            case EulerOrder::SXZY:
                this->linear() = Eigen::AngleAxis<Scalar>(angles.z(), unit_y) *
                        Eigen::AngleAxis<Scalar>(angles.y(), unit_z) *
                        Eigen::AngleAxis<Scalar>(angles.x(), unit_x).toRotationMatrix();
                break;
            case EulerOrder::SXZX:
                this->linear() = Eigen::AngleAxis<Scalar>(angles.z(), unit_x) *
                        Eigen::AngleAxis<Scalar>(angles.y(), unit_z) *
                        Eigen::AngleAxis<Scalar>(angles.x(), unit_x).toRotationMatrix();
            case EulerOrder::SYXZ:
                this->linear() = Eigen::AngleAxis<Scalar>(angles.z(), unit_z) *
                        Eigen::AngleAxis<Scalar>(angles.y(), unit_x) *
                        Eigen::AngleAxis<Scalar>(angles.x(), unit_y).toRotationMatrix();
                break;
        }
    }
};

}  // end of namespace internal


using AngleAxisd = Eigen::AngleAxisd;
using AngleAxisf = Eigen::AngleAxisf;

using Quaterniond = internal::Quaternion<double>;
using Quaternionf = internal::Quaternion<float>;

using EulerAngled = internal::EulerAngle<double>;
using EulerAnglef = internal::EulerAngle<float>;

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

using EulerAngle = internal::EulerAngle<real_t>;

using Isometry2 = internal::Transform<real_t, 2, Eigen::Isometry>;
using Isometry3 = internal::Transform<real_t, 3, Eigen::Isometry>;

using Affine2 = internal::Transform<real_t, 2, Eigen::Affine>;
using Affine3 = internal::Transform<real_t, 3, Eigen::Affine>;

using Projective2 = internal::Transform<real_t, 2, Eigen::Projective>;
using Projective3 = internal::Transform<real_t, 3, Eigen::Projective>;

}  // namespace gobot

namespace rttr::detail {
// Remark: If we didn't explict list all types, the clion will raise a error, but the build(gcc-10) is fine.
#define GOBOT_GEOMETRY_MAKE_FIXED_RTTR(Name, Type)                         \
template<>                                                                 \
struct template_type_trait<gobot::Name> : std::true_type {                 \
    static std::vector<::rttr::type> get_template_arguments() {            \
        return {::rttr::type::get<Type>() }; }                             \
};

GOBOT_GEOMETRY_MAKE_FIXED_RTTR(Isometry2d, double);
GOBOT_GEOMETRY_MAKE_FIXED_RTTR(Isometry3d, double);
GOBOT_GEOMETRY_MAKE_FIXED_RTTR(Isometry2f, float);
GOBOT_GEOMETRY_MAKE_FIXED_RTTR(Isometry3f, float);

GOBOT_GEOMETRY_MAKE_FIXED_RTTR(Affine2d , double);
GOBOT_GEOMETRY_MAKE_FIXED_RTTR(Affine3d, double);
GOBOT_GEOMETRY_MAKE_FIXED_RTTR(Affine2f, float);
GOBOT_GEOMETRY_MAKE_FIXED_RTTR(Affine3f, float);

GOBOT_GEOMETRY_MAKE_FIXED_RTTR(Projective2d , double);
GOBOT_GEOMETRY_MAKE_FIXED_RTTR(Projective3d, double);
GOBOT_GEOMETRY_MAKE_FIXED_RTTR(Projective2f, float);
GOBOT_GEOMETRY_MAKE_FIXED_RTTR(Projective3f, float);

}  // end of namespace rttr::detail
