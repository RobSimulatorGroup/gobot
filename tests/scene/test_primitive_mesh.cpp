/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 22-11-27
*/

#include <gtest/gtest.h>

#include <memory>

#include <gobot/log.hpp>
#include <gobot/scene/resources/material.hpp>
#include <gobot/scene/resources/mesh.hpp>
#include <gobot/scene/resources/primitive_mesh.hpp>
#include "gobot/rendering/render_server.hpp"

TEST(TestBoxMesh, test_create) {
    auto render_server = std::make_unique<gobot::RenderServer>();

    gobot::Variant variant = new gobot::BoxMesh();
    auto *r = variant.convert<gobot::Resource*>();
    gobot::Ref<gobot::Resource> a = gobot::Ref<gobot::Resource>(r);
    auto box_mesh = a.DynamicPointerCast<gobot::BoxMesh>();

    box_mesh->SetMaterial(new gobot::PBRMaterial3D());
    auto material = box_mesh->Get("material");
    gobot::Instance instance(material);
    ASSERT_TRUE(material.is_valid());
}

TEST(TestBoxMesh, test_cast) {
    auto render_server = std::make_unique<gobot::RenderServer>();

    gobot::Variant variant =  gobot::Ref<gobot::Resource>(new gobot::PBRMaterial3D());
    ASSERT_TRUE(variant.can_convert<gobot::Ref<gobot::Material>>());
    ASSERT_TRUE(variant.can_convert<gobot::Ref<gobot::Resource>>());
}

TEST(TestCylinderMesh, test_properties_and_cast) {
    auto render_server = std::make_unique<gobot::RenderServer>();

    auto cylinder_mesh = gobot::MakeRef<gobot::CylinderMesh>();
    cylinder_mesh->SetRadius(0.75f);
    cylinder_mesh->SetHeight(2.5f);
    cylinder_mesh->SetRadialSegments(4);

    EXPECT_FLOAT_EQ(cylinder_mesh->GetRadius(), 0.75f);
    EXPECT_FLOAT_EQ(cylinder_mesh->GetHeight(), 2.5f);
    EXPECT_EQ(cylinder_mesh->GetRadialSegments(), 4);

    gobot::Variant variant = cylinder_mesh;
    ASSERT_TRUE(variant.can_convert<gobot::Ref<gobot::PrimitiveMesh>>());
}

TEST(TestSphereMesh, test_properties_and_cast) {
    auto render_server = std::make_unique<gobot::RenderServer>();

    auto sphere_mesh = gobot::MakeRef<gobot::SphereMesh>();
    sphere_mesh->SetRadius(1.25f);
    sphere_mesh->SetRadialSegments(6);
    sphere_mesh->SetRings(5);

    EXPECT_FLOAT_EQ(sphere_mesh->GetRadius(), 1.25f);
    EXPECT_EQ(sphere_mesh->GetRadialSegments(), 6);
    EXPECT_EQ(sphere_mesh->GetRings(), 5);

    gobot::Variant variant = sphere_mesh;
    ASSERT_TRUE(variant.can_convert<gobot::Ref<gobot::PrimitiveMesh>>());
}
