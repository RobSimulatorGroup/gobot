/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Yingnan Wu<wuyingnan@users.noreply.github.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license
 * document, but changing it is not allowed. This version of the GNU Lesser
 * General Public License incorporates the terms and conditions of version 3 of
 * the GNU General Public License. This file is created by Yingnan Wu, 23-2-10
 */

#pragma once

#include <Eigen/Dense>

#include "gobot/core/math/math_defs.hpp"

namespace gobot {

template <typename _Scalar = real_t>
struct MatrixData {
  int rows;
  int cols;
  std::vector<_Scalar> storage;
};

template <typename _Scalar, int _Rows, int _Cols,
          int _Options = Eigen::AutoAlign |
                         ((_Rows == 1 && _Cols != 1) ? Eigen::RowMajor
                          : (_Cols == 1 && _Rows != 1)
                              ? Eigen::ColMajor
                              : EIGEN_DEFAULT_MATRIX_STORAGE_ORDER_OPTION),
          int _MaxRows = _Rows, int _MaxCols = _Cols>
class Matrix : public Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows,
                                    _MaxCols> {
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

  void SetMatrixData(const MatrixData<_Scalar>& data) {
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

#define GOBOT_MATRIX_MAKE_TYPEDEFS(Size, SizeSuffix)   \
  template <typename Type = real_t>                    \
  using Matrix##SizeSuffix = Matrix<Type, Size, Size>; \
                                                       \
  template <typename Type = real_t>                    \
  using Vector##SizeSuffix = Matrix<Type, Size, 1>;    \
                                                       \
  template <typename Type = real_t>                    \
  using RowVector##SizeSuffix = Matrix<Type, 1, Size>;

#define GOBOT_MATRIX_MAKE_FIXED_TYPEDEFS(Size)                \
  template <typename Type = real_t>                           \
  using Matrix##Size##X = Matrix<Type, Size, Eigen::Dynamic>; \
                                                              \
  template <typename Type = real_t>                           \
  using Matrix##X##Size = Matrix<Type, Eigen::Dynamic, Size>;

GOBOT_MATRIX_MAKE_TYPEDEFS(2, 2)
GOBOT_MATRIX_MAKE_TYPEDEFS(3, 3)
GOBOT_MATRIX_MAKE_TYPEDEFS(4, 4)
GOBOT_MATRIX_MAKE_TYPEDEFS(5, 5)
GOBOT_MATRIX_MAKE_TYPEDEFS(6, 6)
GOBOT_MATRIX_MAKE_TYPEDEFS(7, 7)
GOBOT_MATRIX_MAKE_TYPEDEFS(Eigen::Dynamic, X)
GOBOT_MATRIX_MAKE_FIXED_TYPEDEFS(2)
GOBOT_MATRIX_MAKE_FIXED_TYPEDEFS(3)
GOBOT_MATRIX_MAKE_FIXED_TYPEDEFS(4)

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
