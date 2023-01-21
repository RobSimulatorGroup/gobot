/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 2021/4/7
*/

#pragma once

#include <Eigen/Dense>
#include "rttr/type.h"


//template<typename Scalar, int _Rows, int _Cols,
//        int _Options = Eigen::AutoAlign |
//                       ( (_Rows==1 && _Cols!=1) ? Eigen::RowMajor
//                                                : (_Cols==1 && _Rows!=1) ? Eigen::ColMajor
//                                                                         : EIGEN_DEFAULT_MATRIX_STORAGE_ORDER_OPTION ),
//        int _MaxRows = _Rows,
//        int _MaxCols = _Cols>
//class Eigen::Matrix;


namespace rttr::detail {

template<typename Scalar, int Rows, int Cols>
struct template_type_trait<Eigen::Matrix<Scalar, Rows, Cols>> : std::true_type {
    static std::vector<::rttr::type> get_template_arguments() {
        return {::rttr::type::get<Eigen::Matrix<Scalar, Rows, Cols>>()}; }
};


}

namespace gobot {


}