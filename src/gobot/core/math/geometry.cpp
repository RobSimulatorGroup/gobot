/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * This file is created by Qiqi Wu, 23-2-17
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/core/math/geometry.hpp"
#include "gobot/core/registration.hpp"
#include "gobot/core/macros.hpp"

#define GOBOT_GEOMETRY_MAKE_RTTR_REGISTRATION(Name)                                               \
    Class_<Name>(GOB_STRINGIFY(Name))                                                             \
        .constructor()(CtorAsObject)                                                              \
        .property("matrix_data", &Name::GetMatrixData,                                            \
                                 &Name::SetMatrixData);


GOBOT_REGISTRATION {
    Class_<Quaterniond>("Quaterniond")
            .constructor()(CtorAsObject)
            .property("x", &Quaterniond::GetX, &Quaterniond::SetX)
            .property("y", &Quaterniond::GetY, &Quaterniond::SetY)
            .property("z", &Quaterniond::GetZ, &Quaterniond::SetZ)
            .property("w", &Quaterniond::GetW, &Quaterniond::SetW);

    GOBOT_GEOMETRY_MAKE_RTTR_REGISTRATION(Isometry2f)
    GOBOT_GEOMETRY_MAKE_RTTR_REGISTRATION(Isometry2d)
    GOBOT_GEOMETRY_MAKE_RTTR_REGISTRATION(Isometry3f)
    GOBOT_GEOMETRY_MAKE_RTTR_REGISTRATION(Isometry3d)

    GOBOT_GEOMETRY_MAKE_RTTR_REGISTRATION(Affine2f)
    GOBOT_GEOMETRY_MAKE_RTTR_REGISTRATION(Affine2d)
    GOBOT_GEOMETRY_MAKE_RTTR_REGISTRATION(Affine3f)
    GOBOT_GEOMETRY_MAKE_RTTR_REGISTRATION(Affine3d)

    GOBOT_GEOMETRY_MAKE_RTTR_REGISTRATION(Projective2f)
    GOBOT_GEOMETRY_MAKE_RTTR_REGISTRATION(Projective2d)
    GOBOT_GEOMETRY_MAKE_RTTR_REGISTRATION(Projective3f)
    GOBOT_GEOMETRY_MAKE_RTTR_REGISTRATION(Projective2d)

};