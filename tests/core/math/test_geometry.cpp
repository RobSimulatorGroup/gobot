/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Yingnan Wu<wuyingnan@users.noreply.github.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license
 * document, but changing it is not allowed. This version of the GNU Lesser
 * General Public License incorporates the terms and conditions of version 3 of
 * the GNU General Public License. This file is created by Yingnan Wu, 23-2-10
 */

#include <gtest/gtest.h>

#include <gobot/log.hpp>
#include <gobot/core/math/geometry.hpp>
#include <gobot/core/io/variant_serializer.hpp>

TEST(TestGeometry, test_setter_getter) {
    using namespace gobot;
    Isometry3 isometry = Isometry3::Identity();
    auto data = isometry.GetMatrixData();
    Isometry3 test;
    test.SetMatrixData(data);
    ASSERT_EQ(isometry.matrix(), test.matrix());
}


TEST(TestGeometry, test_matrix_data_registration) {
    using namespace gobot;
    Isometry3 isometry{Isometry3::Identity()};
    auto json = gobot::VariantSerializer::VariantToJson(isometry);

    Variant test_variant((Isometry3()));
    ASSERT_TRUE(gobot::VariantSerializer::JsonToVariant(test_variant, json));
    ASSERT_EQ(test_variant.get_value<Isometry3>().matrix(), isometry.matrix());
}

TEST(TestGeometry, test_isometry) {
    using namespace gobot;
    Isometry3 isometry{Isometry3::Identity()};
    auto quaternion = isometry.GetQuaternion();
    auto euler_angle = isometry.GetEulerAngle(EulerOrder::RXYZ);
    ASSERT_FLOAT_EQ(quaternion.GetX(), 0.0);
    ASSERT_FLOAT_EQ(quaternion.GetY(), 0.0);
    ASSERT_FLOAT_EQ(quaternion.GetZ(), 0.0);
    ASSERT_FLOAT_EQ(quaternion.GetW(), 1.0);
    ASSERT_FLOAT_EQ(euler_angle.x(), 0.0);
    ASSERT_FLOAT_EQ(euler_angle.y(), 0.0);
    ASSERT_FLOAT_EQ(euler_angle.z(), 0.0);

    isometry.SetEulerAngle(Vector3{0, 0, Math_PI * 0.5}, EulerOrder::RXYZ);
    quaternion = isometry.GetQuaternion();
    euler_angle = isometry.GetEulerAngle(EulerOrder::RXYZ);
    ASSERT_FLOAT_EQ(quaternion.GetX(), 0.0);
    ASSERT_FLOAT_EQ(quaternion.GetY(), 0.0);
    ASSERT_FLOAT_EQ(quaternion.GetZ(), 0.70710671);
    ASSERT_FLOAT_EQ(quaternion.GetW(), 0.70710671);
    ASSERT_FLOAT_EQ(euler_angle.x(), 0.0);
    ASSERT_FLOAT_EQ(euler_angle.y(), 0.0);
    ASSERT_FLOAT_EQ(euler_angle.z(), Math_PI * 0.5);

    isometry.SetEulerAngle(Vector3{Math_PI * 0.5, 0.0, 0.0}, EulerOrder::RXYZ);
    quaternion = isometry.GetQuaternion();
    euler_angle = isometry.GetEulerAngle(EulerOrder::RXYZ);
    ASSERT_FLOAT_EQ(quaternion.GetX(), 0.70710671);
    ASSERT_FLOAT_EQ(quaternion.GetY(), 0.0);
    ASSERT_FLOAT_EQ(quaternion.GetZ(), 0.0);
    ASSERT_FLOAT_EQ(quaternion.GetW(), 0.70710671);
    ASSERT_FLOAT_EQ(euler_angle.x(), Math_PI * 0.5);
    ASSERT_FLOAT_EQ(euler_angle.y(), 0.0);
    ASSERT_FLOAT_EQ(euler_angle.z(), 0.0);

    isometry.SetEulerAngle(Vector3{Math_PI * 0.25, Math_PI * 0.25, Math_PI * 0.25}, EulerOrder::RXYZ);
    quaternion = isometry.GetQuaternion();
    euler_angle = isometry.GetEulerAngle(EulerOrder::RXYZ);
    ASSERT_FLOAT_EQ(quaternion.GetX(), 0.46193981);
    ASSERT_FLOAT_EQ(quaternion.GetY(), 0.19134171);
    ASSERT_FLOAT_EQ(quaternion.GetZ(), 0.46193981);
    ASSERT_FLOAT_EQ(quaternion.GetW(), 0.73253775);
    ASSERT_FLOAT_EQ(euler_angle.x(), Math_PI * 0.25);
    ASSERT_FLOAT_EQ(euler_angle.y(), Math_PI * 0.25);
    ASSERT_FLOAT_EQ(euler_angle.z(), Math_PI * 0.25);

    isometry.SetEulerAngle(Vector3{Math_PI * 0.25, -Math_PI * 0.25, Math_PI * 0.1}, EulerOrder::RZYX);
    euler_angle = isometry.GetEulerAngle(EulerOrder::RZYX);
    ASSERT_FLOAT_EQ(euler_angle.x(), Math_PI * 0.25);
    ASSERT_FLOAT_EQ(euler_angle.y(), -Math_PI * 0.25);
    ASSERT_FLOAT_EQ(euler_angle.z(), Math_PI * 0.1);

    isometry.SetEulerAngle(Vector3{Math_PI * 0.25, -Math_PI * 0.25, Math_PI * 0.1}, EulerOrder::SXYZ);
    euler_angle = isometry.GetEulerAngle(EulerOrder::SXYZ);
    ASSERT_FLOAT_EQ(euler_angle.x(), Math_PI * 0.25);
    ASSERT_FLOAT_EQ(euler_angle.y(), -Math_PI * 0.25);
    ASSERT_FLOAT_EQ(euler_angle.z(), Math_PI * 0.1);

    isometry.SetEulerAngle(Vector3{Math_PI * 0.25, -Math_PI * 0.25, Math_PI * 0.1}, EulerOrder::SZYX);
    euler_angle = isometry.GetEulerAngle(EulerOrder::SZYX);
    ASSERT_FLOAT_EQ(euler_angle.x(), Math_PI * 0.25);
    ASSERT_FLOAT_EQ(euler_angle.y(), -Math_PI * 0.25);
    ASSERT_FLOAT_EQ(euler_angle.z(), Math_PI * 0.1);
}
