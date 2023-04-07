/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-4-4
*/


#include "gobot/scene/test_property_node.hpp"
#include "gobot/core/registration.hpp"

namespace gobot {

void TestPropertyNode::SetPropertyUsageFlags(PropertyUsageFlags flags) {
    property_usage_flags_ = flags;
}

PropertyUsageFlags TestPropertyNode::GetPropertyUsageFlags() {
    return property_usage_flags_;
}

void TestPropertyNode::SetColor(const Color& p_color) {
    color = p_color;
}

const Color& TestPropertyNode::GetColor() const {
    return color;
}

}

GOBOT_REGISTRATION {
    Class_<TestPropertyNode>("TestPropertyNode")
            .constructor()(CtorAsRawPtr)
            .property("property_usage_flags",
                      &TestPropertyNode::GetPropertyUsageFlags,
                      &TestPropertyNode::SetPropertyUsageFlags)(
                    AddMetaPropertyInfo(PropertyInfo().SetHint(PropertyHint::Flags)) )
            .property("property_hint", &TestPropertyNode::property_hint)
            .property("string", &TestPropertyNode::string)
            .property("boolean", &TestPropertyNode::boolean_)
            .property("vector2f", &TestPropertyNode::vector2f)
            .property("vector3f", &TestPropertyNode::vector3f)
            .property("vector4f", &TestPropertyNode::vector4f)
            .property("quaternionf", &TestPropertyNode::quaternionf)
            .property("euler_angle", &TestPropertyNode::euler_angle)
            .property("matrix2f", &TestPropertyNode::matrix2f)
            .property("matrix3f", &TestPropertyNode::matrix3f)
            .property("color", &TestPropertyNode::GetColor, &TestPropertyNode::SetColor)
            .property("multiline_text", &TestPropertyNode::multiline_text)(
                    AddMetaPropertyInfo(PropertyInfo().SetHint(PropertyHint::MultilineText)) )
            .property("uint8", &TestPropertyNode::uint8_);


};