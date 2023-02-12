/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Yingnan Wu<wuyingnan@users.noreply.github.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license
 * document, but changing it is not allowed. This version of the GNU Lesser
 * General Public License incorporates the terms and conditions of version 3 of
 * the GNU General Public License. This file is created by Yingnan Wu, 23-2-10
 */

#include <iostream>
#include "gobot/core/math/matrix.hpp"
#include "gobot/core/registration.hpp"

GOBOT_REGISTRATION {
  Matrix3<> a;
  a = Matrix3<float>();
  //Class_<Matrix3d>("Matrix3d").constructor()(CtorAsObject);
  //    //.property("resource_name", &MatrixXd::GetName, &MatrixXd::SetName)
  //    //.property("resource_path", &MatrixXd::GetPath,
  //    //          &MatrixXd::SetPathNotTakeOver);
};
