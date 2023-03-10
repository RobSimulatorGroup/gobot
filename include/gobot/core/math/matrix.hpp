/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Yingnan Wu<wuyingnan@users.noreply.github.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license
 * document, but changing it is not allowed. This version of the GNU Lesser
 * General Public License incorporates the terms and conditions of version 3 of
 * the GNU General Public License. This file is created by Yingnan Wu, 23-2-10
 * This file is modified by Qiqi Wu, 23-2-17
 */

#pragma once

#include <rttr/type.h>

#include <Eigen/Dense>

#include "gobot/core/math/math_defs.hpp"

namespace gobot {

namespace internal {

template <typename _Scalar, int _Rows, int _Cols,
          int _Options = Eigen::AutoAlign |
                         ((_Rows == 1 && _Cols != 1) ? Eigen::RowMajor
                          : (_Cols == 1 && _Rows != 1)
                              ? Eigen::ColMajor
                              : EIGEN_DEFAULT_MATRIX_STORAGE_ORDER_OPTION),
          int _MaxRows = _Rows, int _MaxCols = _Cols>
class Matrix : public Eigen::Matrix<_Scalar, _Rows, _Cols> {
 public:
  using BaseType =
      Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>;
  using BaseType::BaseType;

  MatrixData<_Scalar> GetMatrixData() const {
      // https://stackoverflow.com/questions/22881768/eigen-convert-matrix-to-vector
    auto self_view = this->reshaped();
    if constexpr (_Rows != Eigen::Dynamic && _Cols != Eigen::Dynamic) {
      return {_Rows, _Cols, std::vector(self_view.begin(), self_view.end())};
    } else {
      return {this->rows(), this->cols(),
              std::vector(self_view.begin(), self_view.end())};
    };
  }

  void SetMatrixData(const MatrixData<_Scalar> &data) {
    if constexpr (_Rows != Eigen::Dynamic) {
      if (data.rows != _Rows) [[unlikely]] {
        return;
      }
    }

    if constexpr (_Cols != Eigen::Dynamic) {
      if (data.cols != _Cols) [[unlikely]] {
        return;
      }
    }

    if (data.rows * data.cols != data.storage.size()) [[unlikely]] {
      return;
    }

    if constexpr (_Rows == Eigen::Dynamic || _Cols == Eigen::Dynamic) {
      this->resize(data.rows, data.cols);
    }

    auto self_view = this->reshaped();
    std::copy(data.storage.cbegin(), data.storage.cend(), self_view.begin());
  }

    // eye is the position of the camera's viewpoint, and center is where you are looking at (a position)
    static Matrix<_Scalar, 4, 4>
            LookAt(const Matrix<_Scalar, 3, 1> &eye,
                   const Matrix<_Scalar, 3, 1> &at,
                   const Matrix<_Scalar, 3, 1> &up = { 0.0f, 1.0f, 0.0f },
                   Handedness handedness = Handedness::Left) {
        static_assert(_Cols ==4 && _Rows ==4, "The Look at matrix must a 4*4 matrix");

        Matrix<_Scalar, 3, 1> dir = Handedness::Right == handedness
                                    ? eye - at : at - eye;
        dir.normalize();

        Matrix<_Scalar, 3, 1> right = up.cross(dir);
        right.normalize();

        Matrix<_Scalar, 3, 1> new_up = dir.cross(right);

        Matrix<_Scalar, 4, 4> result = Matrix<_Scalar, 4, 4>::Zero();
        result(0, 0) = right.x();
        result(1, 0) = new_up.x();
        result(2, 0) = dir.x();
        result(0, 1) = right.y();
        result(1, 1) = new_up.y();
        result(2, 1) = dir.y();
        result(0, 2) = right.z();
        result(1, 2) = new_up.z();
        result(2, 2) = dir.z();
        result(0, 3) = -right.dot(eye);
        result(1, 3) = -new_up.dot(eye);
        result(2, 3) = -dir.dot(eye);
        result(3, 3) = 1.0;
        return result;
    }

    static Matrix<_Scalar, 4, 4> Ortho(_Scalar left, _Scalar right,
                                       _Scalar bottom, _Scalar top,
                                       _Scalar near, _Scalar far,
                                       Handedness handedness = Handedness::Left) {
        static_assert(_Cols ==4 && _Rows ==4, "The Look at matrix must a 4*4 matrix");

        _Scalar rl = 1 / (right - left),
                tb = 1 / (top - bottom),
                fn = 1 / (far - near);

        Matrix<_Scalar, 4, 4> result = Matrix<_Scalar, 4, 4>::Zero();

        result(0, 0) = 2 * rl;
        result(1, 1) = 2 * tb;
        result(2, 2) = Handedness::Right == handedness ? -2 * fn : 2 * fn;

        result(0, 3) = -(right + left) * rl;
        result(1, 3) = -(top + bottom) * tb;
        result(2, 3) = -(far + near) * fn;
        result(3, 3) = 1.0;

        return result;
    }

    // fovy is degree
    static Matrix<_Scalar, 4, 4> Perspective(_Scalar fovy, _Scalar aspect, _Scalar near, _Scalar far,
                                             Handedness handedness = Handedness::Left) {
        static_assert(_Cols ==4 && _Rows ==4, "The Look at matrix must a 4*4 matrix");
        _Scalar recip = 1 / (near - far);
        _Scalar c     = 1 / std::tan(.5f * DEG_TO_RAD(fovy));

        Matrix<_Scalar, 4, 4> trafo = Matrix<_Scalar, 4, 4>::Zero();
        trafo(0, 0) = c / aspect;
        trafo(1, 1) = c;
        trafo(2, 2) = (Handedness::Right == handedness) ? (near + far) * recip :
                                                          -(near + far) * recip;

        trafo(2, 3) = 2.f * near * far * recip;                           // a[14]
        trafo(3, 2) = (Handedness::Right == handedness) ? -1.f : 1.0f;   // a[11]

        return trafo;
    }


};

};  // end of namespace internal

#define GOBOT_MATRIX_MAKE_TYPEDEFS(Type, TypeSuffix, Size, SizeSuffix)       \
  using Matrix##SizeSuffix##TypeSuffix = internal::Matrix<Type, Size, Size>; \
  using Vector##SizeSuffix##TypeSuffix = internal::Matrix<Type, Size, 1>;    \
  using RowVector##SizeSuffix##TypeSuffix = internal::Matrix<Type, 1, Size>;

#define GOBOT_MATRIX_MAKE_FIXED_TYPEDEFS(Type, TypeSuffix, Size) \
  using Matrix##Size##X##TypeSuffix =                            \
      internal::Matrix<Type, Size, Eigen::Dynamic>;              \
  using Matrix##X##Size##TypeSuffix =                            \
      internal::Matrix<Type, Eigen::Dynamic, Size>;

#define GOBOT_MAKE_TYPEDEFS_ALL_SIZES(Type, TypeSuffix)           \
  GOBOT_MATRIX_MAKE_TYPEDEFS(Type, TypeSuffix, 2, 2)              \
  GOBOT_MATRIX_MAKE_TYPEDEFS(Type, TypeSuffix, 3, 3)              \
  GOBOT_MATRIX_MAKE_TYPEDEFS(Type, TypeSuffix, 4, 4)              \
  GOBOT_MATRIX_MAKE_TYPEDEFS(Type, TypeSuffix, Eigen::Dynamic, X) \
  GOBOT_MATRIX_MAKE_FIXED_TYPEDEFS(Type, TypeSuffix, 2)           \
  GOBOT_MATRIX_MAKE_FIXED_TYPEDEFS(Type, TypeSuffix, 3)           \
  GOBOT_MATRIX_MAKE_FIXED_TYPEDEFS(Type, TypeSuffix, 4)           \
  GOBOT_MATRIX_MAKE_FIXED_TYPEDEFS(Type, TypeSuffix, 5)           \
  GOBOT_MATRIX_MAKE_FIXED_TYPEDEFS(Type, TypeSuffix, 6)           \
  GOBOT_MATRIX_MAKE_FIXED_TYPEDEFS(Type, TypeSuffix, 7)

GOBOT_MAKE_TYPEDEFS_ALL_SIZES(int, i)
GOBOT_MAKE_TYPEDEFS_ALL_SIZES(float, f)
GOBOT_MAKE_TYPEDEFS_ALL_SIZES(double, d)

#undef GOBOT_MATRIX_MAKE_TYPEDEFS
#undef GOBOT_MATRIX_MAKE_FIXED_TYPEDEFS
#undef GOBOT_MAKE_TYPEDEFS_ALL_SIZES

#define GOBOT_MATRIX_MAKE_DEFAULT(Size, SizeSuffix)                \
  using Matrix##SizeSuffix = internal::Matrix<real_t, Size, Size>; \
  using Vector##SizeSuffix = internal::Matrix<real_t, Size, 1>;    \
  using RowVector##SizeSuffix = internal::Matrix<real_t, 1, Size>;

#define GOBOT_MATRIX_MAKE_FIXED_DEFAULT(Size)                             \
  using Matrix##Size##X = internal::Matrix<real_t, Size, Eigen::Dynamic>; \
  using Matrix##X##Size = internal::Matrix<real_t, Eigen::Dynamic, Size>;

GOBOT_MATRIX_MAKE_DEFAULT(2, 2)
GOBOT_MATRIX_MAKE_DEFAULT(3, 3)
GOBOT_MATRIX_MAKE_DEFAULT(4, 4)
GOBOT_MATRIX_MAKE_DEFAULT(Eigen::Dynamic, X)
GOBOT_MATRIX_MAKE_FIXED_DEFAULT(2)
GOBOT_MATRIX_MAKE_FIXED_DEFAULT(3)
GOBOT_MATRIX_MAKE_FIXED_DEFAULT(4)
GOBOT_MATRIX_MAKE_FIXED_DEFAULT(5)
GOBOT_MATRIX_MAKE_FIXED_DEFAULT(6)
GOBOT_MATRIX_MAKE_FIXED_DEFAULT(7)

};  // namespace gobot

namespace rttr::detail {

// Remark: If we didn't explict list all types, the clion will raise a error, but the build(gcc-10) is fine.
#define GOBOT_MATRIX_MAKE_RTTR(Type, TypeSuffix, Size, SizeSuffix)                                    \
template<>                                                                                            \
struct template_type_trait<gobot::Matrix##SizeSuffix##TypeSuffix> : std::true_type {                  \
    static std::vector<::rttr::type> get_template_arguments() {                                       \
        return {::rttr::type::get<Type>(), ::rttr::type::get<int>(), ::rttr::type::get<int>() }; }    \
};                                                                                                    \
template<>                                                                                            \
struct template_type_trait<gobot::Vector##SizeSuffix##TypeSuffix> : std::true_type {                  \
    static std::vector<::rttr::type> get_template_arguments() {                                       \
        return {::rttr::type::get<Type>(), ::rttr::type::get<int>(), ::rttr::type::get<int>() }; }    \
};                                                                                                    \
template<>                                                                                            \
struct template_type_trait<gobot::RowVector##SizeSuffix##TypeSuffix> : std::true_type {               \
    static std::vector<::rttr::type> get_template_arguments() {                                       \
        return {::rttr::type::get<Type>(), ::rttr::type::get<int>(), ::rttr::type::get<int>() }; }    \
};


#define GOBOT_MATRIX_MAKE_FIXED_RTTR(Type, TypeSuffix, Size)                                          \
template<>                                                                                            \
struct template_type_trait<gobot::Matrix##Size##X##TypeSuffix> : std::true_type {                     \
    static std::vector<::rttr::type> get_template_arguments() {                                       \
        return {::rttr::type::get<Type>(), ::rttr::type::get<int>(), ::rttr::type::get<int>() }; }    \
};                                                                                                    \
template<>                                                                                            \
struct template_type_trait<gobot::Matrix##X##Size##TypeSuffix> : std::true_type {                     \
    static std::vector<::rttr::type> get_template_arguments() {                                       \
        return {::rttr::type::get<Type>(), ::rttr::type::get<int>(), ::rttr::type::get<int>() }; }    \
};

#define GOBOT_MAKE_RTTR_ALL_SIZES(Type, TypeSuffix)             \
GOBOT_MATRIX_MAKE_RTTR(Type, TypeSuffix, 2, 2)                  \
GOBOT_MATRIX_MAKE_RTTR(Type, TypeSuffix, 3, 3)                  \
GOBOT_MATRIX_MAKE_RTTR(Type, TypeSuffix, 4, 4)                  \
GOBOT_MATRIX_MAKE_RTTR(Type, TypeSuffix, Eigen::Dynamic, X)     \
GOBOT_MATRIX_MAKE_FIXED_RTTR(Type, TypeSuffix, 2)               \
GOBOT_MATRIX_MAKE_FIXED_RTTR(Type, TypeSuffix, 3)               \
GOBOT_MATRIX_MAKE_FIXED_RTTR(Type, TypeSuffix, 4)               \
GOBOT_MATRIX_MAKE_FIXED_RTTR(Type, TypeSuffix, 5)               \
GOBOT_MATRIX_MAKE_FIXED_RTTR(Type, TypeSuffix, 6)               \
GOBOT_MATRIX_MAKE_FIXED_RTTR(Type, TypeSuffix, 7)

GOBOT_MAKE_RTTR_ALL_SIZES(int,    i)
GOBOT_MAKE_RTTR_ALL_SIZES(float,  f)
GOBOT_MAKE_RTTR_ALL_SIZES(double, d)

}  // end of namespace rttr::detail

namespace Eigen::internal {

template <typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows,
          int _MaxCols>
class traits<gobot::internal::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows,
                                     _MaxCols>>
    : public Eigen::internal::traits<
          Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>> {
};

};  // namespace Eigen::internal
