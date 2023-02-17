/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Yingnan Wu<wuyingnan@users.noreply.github.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license
 * document, but changing it is not allowed. This version of the GNU Lesser
 * General Public License incorporates the terms and conditions of version 3 of
 * the GNU General Public License. This file is created by Yingnan Wu, 23-2-10
 */

#pragma once

#include <Eigen/Dense>
#include <rttr/type.h>

#include "rttr/detail/comparison/compare_less.h"
#include "rttr/detail/comparison/compare_equal.h"

#include "gobot/core/math/math_defs.hpp"

namespace gobot {

template <typename Scalar>
struct MatrixData {
  int rows;
  int cols;
  std::vector<Scalar> storage;
};

namespace internal {

template<typename _Scalar, int _Rows, int _Cols,
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
            if (data.rows != _Rows) {
                return;
            }
        }

        if constexpr (_Cols != Eigen::Dynamic) {
            if (data.cols != _Cols) {
                return;
            }
        }

        if (data.rows * data.cols != data.storage.size()) {
            return;
        }

        if constexpr (_Rows == Eigen::Dynamic || _Cols == Eigen::Dynamic) {
            this->resize(data.rows, data.cols);
        }

        auto self_view = this->reshaped();
        std::copy(data.storage.cbegin(), data.storage.cend(), self_view.begin());
    }
};

} // end of namespace internal

#define GOBOT_MATRIX_MAKE_TYPEDEFS(Type, TypeSuffix, Size, SizeSuffix)   \
  using Matrix##SizeSuffix##TypeSuffix    = internal::Matrix<Type, Size, Size>;    \
  using Vector##SizeSuffix##TypeSuffix    = internal::Matrix<Type, Size, 1>;       \
  using RowVector##SizeSuffix##TypeSuffix = internal::Matrix<Type, 1, Size>;

#define GOBOT_MATRIX_MAKE_FIXED_TYPEDEFS(Type, TypeSuffix, Size)           \
  using Matrix##Size##X##TypeSuffix = internal::Matrix<Type, Size, Eigen::Dynamic>;  \
  using Matrix##X##Size##TypeSuffix = internal::Matrix<Type, Eigen::Dynamic, Size>;




#define GOBOT_MAKE_TYPEDEFS_ALL_SIZES(Type, TypeSuffix)             \
GOBOT_MATRIX_MAKE_TYPEDEFS(Type, TypeSuffix, 2, 2)                  \
GOBOT_MATRIX_MAKE_TYPEDEFS(Type, TypeSuffix, 3, 3)                  \
GOBOT_MATRIX_MAKE_TYPEDEFS(Type, TypeSuffix, 4, 4)                  \
GOBOT_MATRIX_MAKE_TYPEDEFS(Type, TypeSuffix, Eigen::Dynamic, X)     \
GOBOT_MATRIX_MAKE_FIXED_TYPEDEFS(Type, TypeSuffix, 2)               \
GOBOT_MATRIX_MAKE_FIXED_TYPEDEFS(Type, TypeSuffix, 3)               \
GOBOT_MATRIX_MAKE_FIXED_TYPEDEFS(Type, TypeSuffix, 4)               \
GOBOT_MATRIX_MAKE_FIXED_TYPEDEFS(Type, TypeSuffix, 5)               \
GOBOT_MATRIX_MAKE_FIXED_TYPEDEFS(Type, TypeSuffix, 6)               \
GOBOT_MATRIX_MAKE_FIXED_TYPEDEFS(Type, TypeSuffix, 7)

GOBOT_MAKE_TYPEDEFS_ALL_SIZES(int,                  i)
GOBOT_MAKE_TYPEDEFS_ALL_SIZES(float,                f)
GOBOT_MAKE_TYPEDEFS_ALL_SIZES(double,               d)

#undef GOBOT_MATRIX_MAKE_TYPEDEFS
#undef GOBOT_MATRIX_MAKE_FIXED_TYPEDEFS
#undef GOBOT_MAKE_TYPEDEFS_ALL_SIZES

#define GOBOT_MATRIX_MAKE_DEFAULT(Type, Size, SizeSuffix)              \
  using Matrix##SizeSuffix    = internal::Matrix<Type, Size, Size>;    \
  using Vector##SizeSuffix    = internal::Matrix<Type, Size, 1>;       \
  using RowVector##SizeSuffix = internal::Matrix<Type, 1, Size>;

#define GOBOT_MATRIX_MAKE_FIXED_DEFAULT(Type, Size)                      \
  using Matrix##Size##X = internal::Matrix<Type, Size, Eigen::Dynamic>;  \
  using Matrix##X##Size = internal::Matrix<Type, Eigen::Dynamic, Size>;

#define GOBOT_MAKE_DEFAULT_ALL_SIZES(Type)                         \
GOBOT_MATRIX_MAKE_DEFAULT(Type, 2, 2)                              \
GOBOT_MATRIX_MAKE_DEFAULT(Type, 3, 3)                              \
GOBOT_MATRIX_MAKE_DEFAULT(Type, 4, 4)                              \
GOBOT_MATRIX_MAKE_DEFAULT(Type, Eigen::Dynamic, X)                 \
GOBOT_MATRIX_MAKE_FIXED_DEFAULT(Type, 2)                           \
GOBOT_MATRIX_MAKE_FIXED_DEFAULT(Type, 3)                           \
GOBOT_MATRIX_MAKE_FIXED_DEFAULT(Type, 4)                           \
GOBOT_MATRIX_MAKE_FIXED_DEFAULT(Type, 5)                           \
GOBOT_MATRIX_MAKE_FIXED_DEFAULT(Type, 6)                           \
GOBOT_MATRIX_MAKE_FIXED_DEFAULT(Type, 7)

#ifdef MATRIX_IS_DOUBLE
GOBOT_MAKE_DEFAULT_ALL_SIZES(double)
#else
GOBOT_MAKE_DEFAULT_ALL_SIZES(float)
#endif

};  // namespace gobot

namespace rttr::detail {

template<typename Scalar, int Rows, int Cols>
struct template_type_trait<gobot::internal::Matrix<Scalar, Rows, Cols>> : std::true_type {
static std::vector<::rttr::type> get_template_arguments() {
    return {::rttr::type::get<Scalar>(), ::rttr::type::get<int>(), ::rttr::type::get<int>() }; }
};

} // end of namespace rttr::detail

namespace Eigen::internal {

template <typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows,
          int _MaxCols>
class traits<gobot::internal::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>>
    : public Eigen::internal::traits<
          Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>> {
};


};  // namespace Eigen::internal

// rttr::registration
