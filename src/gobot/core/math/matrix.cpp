/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Yingnan Wu<wuyingnan@users.noreply.github.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license
 * document, but changing it is not allowed. This version of the GNU Lesser
 * General Public License incorporates the terms and conditions of version 3 of
 * the GNU General Public License. This file is created by Yingnan Wu, 23-2-10
 */

#include "gobot/core/math/matrix.hpp"
#include "gobot/core/registration.hpp"

GOBOT_REGISTRATION {
  Class_<MatrixData<int>>("MatrixDatai")
      .constructor()(CtorAsObject)
      .property("rows", &MatrixData<int>::rows)
      .property("cols", &MatrixData<int>::cols)
      .property("storage", &MatrixData<int>::storage);

    Class_<MatrixData<double>>("MatrixDatad")
            .constructor()(CtorAsObject)
            .property("rows", &MatrixData<double>::rows)
            .property("cols", &MatrixData<double>::cols)
            .property("storage", &MatrixData<double>::storage);

    Class_<MatrixData<float>>("MatrixDataf")
            .constructor()(CtorAsObject)
            .property("rows", &MatrixData<float>::rows)
            .property("cols", &MatrixData<float>::cols)
            .property("storage", &MatrixData<float>::storage);

  Class_<Matrix3i>("Matrix3i")
      .constructor()(CtorAsObject)
      .property("matrix_data", &Matrix3i::GetMatrixData,
                &Matrix3i::SetMatrixData);

};
