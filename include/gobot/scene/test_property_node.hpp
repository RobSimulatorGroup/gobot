/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-4-4
*/

#pragma once

#include "gobot/scene/node_3d.hpp"
#include "gobot/core/color.hpp"
#include "gobot/core/math/matrix.hpp"

namespace gobot {

class TestPropertyNode : public Node3D {
    GOBCLASS(TestPropertyNode, Node3D)
public:

    TestPropertyNode() = default;

    void SetPropertyUsageFlags(PropertyUsageFlags flags);

    PropertyUsageFlags GetPropertyUsageFlags();

    void SetColor(const Color& p_color);

    const Color& GetColor() const;

private:
    GOBOT_REGISTRATION_FRIEND

    bool boolean_{false};

    uint8_t uint8_{10};
    std::int64_t int64_{10};

    PropertyUsageFlags property_usage_flags_{PropertyUsageFlags::Default};

    PropertyHint property_hint{PropertyHint::Dir};
    String string{};
    String multiline_text{};

    Color color{1.0f, 0.0f, 0.0f, 1.0f};

    Vector2f vector2f{0.0f, 0.0f};

    Vector3f vector3f{0.0f, 0.0f, 0.0f};

    Vector4f vector4f{0.0f, 0.0f, 0.0f, 0.0f};

    Quaternionf quaternionf{0.0f, 0.0f, 0.0f, 1.0f};

    EulerAngle euler_angle{0.0f, 90.0f, 0.0f};

    Matrix2f matrix2f{Matrix2f::Zero()};

    Matrix3f matrix3f{Matrix3f::Zero()};

    Isometry2f isometry2f{Isometry2f::Identity()};

    Isometry3f isometry3f{Isometry3f::Identity()};
};

}