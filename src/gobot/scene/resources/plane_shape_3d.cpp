/* SPDX-License-Identifier: Apache-2.0 */
#include "gobot/scene/resources/plane_shape_3d.hpp"
#include "gobot/core/registration.hpp"

GOBOT_REGISTRATION {
    Class_<gobot::PlaneShape3D>("PlaneShape3D").constructor()(CtorAsRawPtr);
    gobot::Type::register_wrapper_converter_for_base_classes<
            gobot::Ref<gobot::PlaneShape3D>, gobot::Ref<gobot::Shape3D>>();
}
