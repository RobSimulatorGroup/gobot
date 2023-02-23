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
        static_assert(Dim == 3 && (Mode == Eigen::Isometry | Mode == Eigen::Affine),
                "GetQuaternion can only called when Dim is 3");

        return Quaternion<Scalar>(this->linear());
    }

    void SetQuaternion(const Quaternion<Scalar>& quaternion) {
        static_assert(Dim == 3 && (Mode == Eigen::Isometry || Mode == Eigen::Affine),
                "SetQuaternion can only called when Dim is 3");

        this->linear() = quaternion.normalized().toRotationMatrix();
    }

    void SetQuaternionScaled(const Quaternion<Scalar> &quaternion, const Vector3 &scale) {
        static_assert(Dim == 3 && Mode == Eigen::Affine, "GetEulerAngleNormalized can only called when Dim is 3");

        this->SetQuaternion(quaternion);
        this->linear() *= Eigen::Scaling(scale.x(), scale.y(), scale.z());
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
        static_assert(Dim == 3 && (Mode == Eigen::Isometry || Mode == Eigen::Affine),
                "GetEulerAngle can only called when Dim is 3");

        switch (euler_order) {
            case EulerOrder::RXYZ:
                return this->linear().eulerAngles(0, 1, 2);
            case EulerOrder::RYZX:
                return this->linear().eulerAngles(1, 2, 0);
            case EulerOrder::RZYX:
                return this->linear().eulerAngles(2, 1, 0);
            case EulerOrder::RXZY:
                return this->linear().eulerAngles(0, 2, 1);
            case EulerOrder::RXZX:
                return this->linear().eulerAngles(0, 2, 0);
            case EulerOrder::RYXZ:
                return this->linear().eulerAngles(1, 0, 2);
            case EulerOrder::SXYZ: {
                auto euler_angle = this->linear().eulerAngles(2, 1, 0);
                std::swap(euler_angle[0], euler_angle[2]);
                return euler_angle;
            }
            case EulerOrder::SYZX: {
                auto euler_angle = this->linear().eulerAngles(0, 2, 1);
                std::swap(euler_angle[0], euler_angle[2]);
                return euler_angle;
            }
            case EulerOrder::SZYX: {
                auto euler_angle = this->linear().eulerAngles(0, 1, 2);
                std::swap(euler_angle[0], euler_angle[2]);
                return euler_angle;
            }
            case EulerOrder::SXZY: {
                auto euler_angle = this->linear().eulerAngles(1, 2, 0);
                std::swap(euler_angle[0], euler_angle[2]);
                return euler_angle;
            }
            case EulerOrder::SXZX: {
                auto euler_angle = this->linear().eulerAngles(0, 2, 0);
                std::swap(euler_angle[0], euler_angle[2]);
                return euler_angle;
            }
            case EulerOrder::SYXZ: {
                auto euler_angle = this->linear().eulerAngles(2, 0, 1);
                std::swap(euler_angle[0], euler_angle[2]);
                return euler_angle;
            }
        }
        // never go here
        return {};
    }

    void SetEulerAngle(const EulerAngle<Scalar>& angles, EulerOrder euler_order) {
        static_assert(Dim == 3 && (Mode == Eigen::Isometry || Eigen::Affine),
                "SetEulerAngle can only called when Dim is 3");

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

    void SetEulerAngleScaled(const EulerAngle<Scalar> &angles, const Vector3 &scale, EulerOrder euler_order) {
        static_assert(Dim == 3 && Mode == Eigen::Affine, "GetEulerAngleNormalized can only called when Dim is 3");

        this->SetEulerAngle(angles, euler_order);
        this->linear() *= Eigen::Scaling(scale.x(), scale.y(), scale.z());
    }

    /**
     * @brief Assuming for an Affine transform, M = R.S (R is rotation and S is scaling).
     *  The method is used to extract the scaling factors with their signs as a vector, e.g <s1, s2, s3>,
     *  which resembles the scaling S,
     *                      [ s1,  0,  0 ]
     *                      [  0, s2,  0 ]
     *                      [  0,  0, s3 ]
     *  Note:
     *  With a transformation matrix M (Isometry/Affine),
     *                      [ a, b, c, d ]
     *                      [ e, f, g, h ]
     *                      [ i, j, k, l ]
     *                      [ 0, 0, 0, 1 ]
     *  The scaling vector s becomes,
     *                      s1 = || <a, e, i> ||
     *                      s2 = || <b, f, j> ||
     *                      s3 = || <c, g, k> ||
     *
     * @return a scaling vector depending on the dimension.
     */
    [[nodiscard]] Matrix<Scalar, Dim, 1> GetScale() const {
        static_assert(Mode == Eigen::Isometry || Mode == Eigen::Affine,
                "GetScaleAbs works for Isometry and Affine");

        int sign = Sign(this->linear().determinant());
        if (Mode == Eigen::Isometry) {
            return sign * Matrix<Scalar, Dim, 1>::Ones();
        }

        Matrix<Scalar, Dim, 1> v;
        for (auto col = 0; col < Dim; ++ col) {
            v[col] = this->linear().template block<Dim, 1>(0, col).norm();
        }

        return sign * v;
    }

    /**
     * @brief Gram-Schmidt Process. The linear part is changed.
     *  Note:
     *  With a 3x3 matrix, the process could be as follows,
     *              x = col(0), y = col(1), z = col(2)
     *              step1. x.normalize()
     *              step2. y = y - x * (x.dot(y)), y.normalize()
     *              step3. z = z - (x * (x.dot(z))) - (y * (y.dot(z))), z.normalize()
     */
    void Orthonormalize() {
        this->linear().col(0).normalize();
        for (auto i = 1; i < Dim; ++ i) {
            for (int j = 0; j < i; ++ j) {
                this->linear().col(i) -=
                        this->linear().col(j) * (this->linear().col(j)).dot(this->linear().col(i));
            }
            this->linear().col(i).normalize();
        }
    }

    /**
     * @brief See Orthonormalize. The original transform is reserved.
     *
     * @return an orthonormalized transform depending on the dimension.
     */
    Transform Orthonormalized() const {
        Transform tf = *this;
        tf.Orthonormalize();
        return tf;
    }

    void Orthogonalize() {
        static_assert(Mode == Eigen::Isometry || Mode == Eigen::Affine,
                      "Orthogonalize works for Isometry and Affine");

        Matrix<Scalar, Dim, 1> s = GetScale();
        Orthonormalize();
        this->scale(s);
    }

    Transform Orthogonalized() const {
        Transform tf = *this;
        tf.Orthogonalize();
        return tf;
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
