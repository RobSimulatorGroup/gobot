/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Yingnan Wu<wuyingnan@users.noreply.github.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license
 * document, but changing it is not allowed. This version of the GNU Lesser
 * General Public License incorporates the terms and conditions of version 3 of
 * the GNU General Public License. This file is created by Yingnan Wu, 23-2-10
 */

#pragma once

#include <Eigen/Dense>

namespace gobot {

template <typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows,
          int _MaxCols>
class Matrix : public Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows,
                                    _MaxCols> {
 public:
  constexpr size_t GetCols() const {
    if (std::is_constant_evaluated()) {
      return _Cols;
    } else {
      return this->cols();
    }
  }

  constexpr size_t GetRows() const {
    if (std::is_constant_evaluated()) {
      return _Rows;
    } else {
      return this->rows();
    }
  }
};


template <typename _Scalar, int _Rows, int _Cols,
          int _Options = Eigen::AutoAlign |
                         ((_Rows == 1 && _Cols != 1) ? Eigen::RowMajor
                          : (_Cols == 1 && _Rows != 1)
                              ? Eigen::ColMajor
                              : EIGEN_DEFAULT_MATRIX_STORAGE_ORDER_OPTION),
          int _MaxRows = _Rows, int _MaxCols = _Cols>
class Matrix;


#define GOBOT_MATRIX_MAKE_TYPEDEFS(Size, SizeSuffix)   \
  template <typename Type>                             \
  using Matrix##SizeSuffix = Matrix<Type, Size, Size>; \
                                                       \
  template <typename Type>                             \
  using Vector##SizeSuffix = Matrix<Type, Size, 1>;    \
                                                       \
  template <typename Type>                             \
  using RowVector##SizeSuffix = Matrix<Type, 1, Size>;

#define GOBOT_MATRIX_MAKE_FIXED_TYPEDEFS(Size)         \
  template <typename Type>                             \
  using Matrix##Size##X = Matrix<Type, Size, Dynamic>; \
                                                       \
  template <typename Type>                             \
  using Matrix##X##Size = Matrix<Type, Dynamic, Size>;

GOBOT_MATRIX_MAKE_TYPEDEFS(2, 2)
GOBOT_MATRIX_MAKE_TYPEDEFS(3, 3)
GOBOT_MATRIX_MAKE_TYPEDEFS(4, 4)
GOBOT_MATRIX_MAKE_TYPEDEFS(5, 5)
GOBOT_MATRIX_MAKE_TYPEDEFS(6, 6)
GOBOT_MATRIX_MAKE_TYPEDEFS(7, 7)
GOBOT_MATRIX_MAKE_TYPEDEFS(Eigen::Dynamic, X)
//GOBOT_MATRIX_MAKE_FIXED_TYPEDEFS(2)
//GOBOT_MATRIX_MAKE_FIXED_TYPEDEFS(3)
//GOBOT_MATRIX_MAKE_FIXED_TYPEDEFS(4)

#undef GOBOT_MATRIX_MAKE_TYPEDEFS
#undef GOBOT_MATRIX_MAKE_FIXED_TYPEDEFS

template <typename Type, int Size>
using Vector = Matrix<Type, Size, 1>;

template <typename Type, int Size>
using RowVector = Matrix<Type, 1, Size>;

};  // namespace gobot

namespace Eigen::internal {

template <typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows,
          int _MaxCols>
class traits<gobot::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>>
    : public Eigen::internal::traits<
          Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>> {
};
};  // namespace Eigen::internal

// rttr::registration
