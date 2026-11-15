/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Yingnan Wu<wuyingnan@users.noreply.github.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <gobot/core/math/geometry.hpp>
#include <gobot/core/math/mesh_validation.hpp>
#include <limits>
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

    ASSERT_EQ(isometry.GetScale(), Vector3::Ones());
}

TEST(TestGeometry, test_affine) {
    using namespace gobot;
    Affine3 affine{Affine3::Identity()};

    ASSERT_TRUE(affine.GetScale().isApprox(Vector3::Ones(), CMP_EPSILON));

    Vector3 ratio{2.0, 2.0, 2.0};
    affine.scale(ratio);
    ASSERT_TRUE(affine.GetScale().isApprox(ratio, CMP_EPSILON));

    ASSERT_TRUE(affine.Orthonormalized().linear().isApprox(Matrix3::Identity(), CMP_EPSILON));
    // The original transform is already orthogonal
    ASSERT_TRUE(affine.Orthogonalized().isApprox(affine, CMP_EPSILON));

    // Uniformly scale should not change the orientation.
    auto euler_angle = Vector3{Math_PI * 0.25, -Math_PI * 0.25, Math_PI * 0.1};
    affine.SetEulerAngleScaled(euler_angle, ratio, EulerOrder::SZYX);
    ASSERT_FLOAT_EQ(affine.GetEulerAngle(EulerOrder::SZYX).x(), euler_angle.x());
    ASSERT_FLOAT_EQ(affine.GetEulerAngle(EulerOrder::SZYX).y(), euler_angle.y());
    ASSERT_FLOAT_EQ(affine.GetEulerAngle(EulerOrder::SZYX).z(), euler_angle.z());
}

TEST(TestGeometry, triangle_mesh_accepts_slender_nonzero_area_faces) {
    const std::vector<std::uint32_t> indices{0, 1, 2};
    std::string error;
    for (const gobot::RealType scale : {0.001f, 1.0f, 1000.0f}) {
        const std::vector<gobot::Vector3> vertices{
                {0.0, 0.0, 0.0}, {scale, 0.0, 0.0}, {scale, scale * 1.0e-6f, 0.0}};
        EXPECT_TRUE(gobot::ValidateTriangleMesh(vertices, indices, &error)) << error;
    }

    // A valid narrow face from the official Newton G1 torso collision mesh.
    const std::vector<gobot::Vector3> torso_vertices{
            {0.030211931094527245, -0.06183788552880287, 0.010300304740667343},
            {0.003611930413171649, -0.06184277683496475, 0.010300111025571823},
            {0.002011930337175727, -0.06184307858347893, 0.010300134308636189}};
    EXPECT_TRUE(gobot::ValidateTriangleMesh(torso_vertices, indices, &error)) << error;
}

TEST(TestGeometry, triangle_mesh_rejects_degenerate_and_nonfinite_faces) {
    const std::vector<std::uint32_t> indices{0, 1, 2};
    std::vector<gobot::Vector3> vertices{{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {2.0, 0.0, 0.0}};
    std::string error;
    EXPECT_FALSE(gobot::ValidateTriangleMesh(vertices, indices, &error));
    EXPECT_NE(error.find("non-degenerate"), std::string::npos);
    vertices[2] = vertices[1];
    EXPECT_FALSE(gobot::ValidateTriangleMesh(vertices, indices, &error));
    for (const gobot::RealType value : {std::numeric_limits<gobot::RealType>::infinity(),
                                       std::numeric_limits<gobot::RealType>::quiet_NaN()}) {
        vertices[2] = {0.0, value, 0.0};
        EXPECT_FALSE(gobot::ValidateTriangleMesh(vertices, indices, &error));
        EXPECT_NE(error.find("not finite"), std::string::npos);
    }
}
