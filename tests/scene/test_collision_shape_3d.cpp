/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <gobot/scene/collision_shape_3d.hpp>
#include <gobot/scene/resources/box_shape_3d.hpp>
#include <gobot/scene/resources/cylinder_shape_3d.hpp>
#include <gobot/scene/resources/sphere_shape_3d.hpp>

TEST(TestCollisionShape3D, stores_shape_resource_and_transform) {
    auto* collision_shape = gobot::Object::New<gobot::CollisionShape3D>();
    gobot::Ref<gobot::CylinderShape3D> cylinder_shape = gobot::MakeRef<gobot::CylinderShape3D>();
    cylinder_shape->SetRadius(0.25f);
    cylinder_shape->SetHeight(2.0f);

    collision_shape->SetShape(cylinder_shape);
    collision_shape->SetPosition({1.0f, 2.0f, 3.0f});
    collision_shape->SetDisabled(true);

    gobot::Ref<gobot::CylinderShape3D> stored_shape =
            gobot::dynamic_pointer_cast<gobot::CylinderShape3D>(collision_shape->GetShape());
    ASSERT_TRUE(stored_shape.IsValid());
    EXPECT_FLOAT_EQ(stored_shape->GetRadius(), 0.25f);
    EXPECT_FLOAT_EQ(stored_shape->GetHeight(), 2.0f);
    EXPECT_TRUE(collision_shape->GetPosition().isApprox(gobot::Vector3(1.0f, 2.0f, 3.0f), CMP_EPSILON));
    EXPECT_TRUE(collision_shape->IsDisabled());

    gobot::Object::Delete(collision_shape);
}

TEST(TestCollisionShape3D, reflected_properties_are_available) {
    auto shape = gobot::Type::get<gobot::CollisionShape3D>().get_property("shape");
    auto disabled = gobot::Type::get<gobot::CollisionShape3D>().get_property("disabled");
    auto position = gobot::Type::get<gobot::CollisionShape3D>().get_property("position");

    EXPECT_TRUE(shape.is_valid());
    EXPECT_TRUE(disabled.is_valid());
    EXPECT_TRUE(position.is_valid());
}

TEST(TestCollisionShape3D, built_in_shape_resources_keep_dimensions) {
    gobot::Ref<gobot::BoxShape3D> box = gobot::MakeRef<gobot::BoxShape3D>();
    box->SetSize({1.0f, 2.0f, 3.0f});
    EXPECT_TRUE(box->GetSize().isApprox(gobot::Vector3(1.0f, 2.0f, 3.0f), CMP_EPSILON));

    gobot::Ref<gobot::SphereShape3D> sphere = gobot::MakeRef<gobot::SphereShape3D>();
    sphere->SetRadius(1.25f);
    EXPECT_FLOAT_EQ(sphere->GetRadius(), 1.25f);

    gobot::Ref<gobot::CylinderShape3D> cylinder = gobot::MakeRef<gobot::CylinderShape3D>();
    cylinder->SetRadius(0.75f);
    cylinder->SetHeight(2.5f);
    EXPECT_FLOAT_EQ(cylinder->GetRadius(), 0.75f);
    EXPECT_FLOAT_EQ(cylinder->GetHeight(), 2.5f);
}
