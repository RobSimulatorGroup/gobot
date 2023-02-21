/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Yingnan Wu<wuyingnan@users.noreply.github.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license
 * document, but changing it is not allowed. This version of the GNU Lesser
 * General Public License incorporates the terms and conditions of version 3 of
 * the GNU General Public License. This file is created by Yingnan Wu, 23-2-10
 */

#include "gobot/core/math/matrix.hpp"
#include "gobot/core/macros.hpp"
#include "gobot/core/registration.hpp"

#define GOBOT_MATRIX_MAKE_RTTR_REGISTRATION(Type, TypeSuffix, Size, SizeSuffix)                   \
    Class_<Matrix##SizeSuffix##TypeSuffix>(GOB_STRINGIFY(Matrix##SizeSuffix##TypeSuffix))         \
        .constructor()(CtorAsObject)                                                              \
        .property("matrix_data", &Matrix##SizeSuffix##TypeSuffix::GetMatrixData,                  \
                                 &Matrix##SizeSuffix##TypeSuffix::SetMatrixData);                 \
    Class_<Vector##SizeSuffix##TypeSuffix>(GOB_STRINGIFY(Vector##SizeSuffix##TypeSuffix))         \
        .constructor()(CtorAsObject)                                                              \
        .property("matrix_data", &Vector##SizeSuffix##TypeSuffix::GetMatrixData,                  \
                                 &Vector##SizeSuffix##TypeSuffix::SetMatrixData);                 \
    Class_<RowVector##SizeSuffix##TypeSuffix>(GOB_STRINGIFY(RowVector##SizeSuffix##TypeSuffix))   \
        .constructor()(CtorAsObject)                                                              \
        .property("matrix_data", &RowVector##SizeSuffix##TypeSuffix::GetMatrixData,               \
                                 &RowVector##SizeSuffix##TypeSuffix::SetMatrixData);


#define GOBOT_MATRIX_MAKE_FIXED_RTTR_REGISTRATION(Type, TypeSuffix, Size)                         \
     Class_<Matrix##Size##X##TypeSuffix>(GOB_STRINGIFY(Matrix##Size##X##TypeSuffix))              \
        .constructor()(CtorAsObject)                                                              \
        .property("matrix_data", &Matrix##Size##X##TypeSuffix::GetMatrixData,                     \
                                 &Matrix##Size##X##TypeSuffix::SetMatrixData);                    \
     Class_<Matrix##X##Size##TypeSuffix>(GOB_STRINGIFY(Matrix##X##Size##TypeSuffix))              \
        .constructor()(CtorAsObject)                                                              \
        .property("matrix_data", &Matrix##X##Size##TypeSuffix::GetMatrixData,                     \
                                 &Matrix##X##Size##TypeSuffix::SetMatrixData);


#define GOBOT_MAKE_RTTR_REGISTRATION_ALL_SIZES(Type, TypeSuffix)             \
GOBOT_MATRIX_MAKE_RTTR_REGISTRATION(Type, TypeSuffix, 2, 2)                  \
GOBOT_MATRIX_MAKE_RTTR_REGISTRATION(Type, TypeSuffix, 3, 3)                  \
GOBOT_MATRIX_MAKE_RTTR_REGISTRATION(Type, TypeSuffix, 4, 4)                  \
GOBOT_MATRIX_MAKE_RTTR_REGISTRATION(Type, TypeSuffix, Eigen::Dynamic, X)     \
GOBOT_MATRIX_MAKE_FIXED_RTTR_REGISTRATION(Type, TypeSuffix, 2)               \
GOBOT_MATRIX_MAKE_FIXED_RTTR_REGISTRATION(Type, TypeSuffix, 3)               \
GOBOT_MATRIX_MAKE_FIXED_RTTR_REGISTRATION(Type, TypeSuffix, 4)               \
GOBOT_MATRIX_MAKE_FIXED_RTTR_REGISTRATION(Type, TypeSuffix, 5)               \
GOBOT_MATRIX_MAKE_FIXED_RTTR_REGISTRATION(Type, TypeSuffix, 6)               \
GOBOT_MATRIX_MAKE_FIXED_RTTR_REGISTRATION(Type, TypeSuffix, 7)

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


    GOBOT_MAKE_RTTR_REGISTRATION_ALL_SIZES(int,    i)
    GOBOT_MAKE_RTTR_REGISTRATION_ALL_SIZES(float,  f)
    GOBOT_MAKE_RTTR_REGISTRATION_ALL_SIZES(double, d)

};
