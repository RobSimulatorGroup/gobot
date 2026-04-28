#include <gtest/gtest.h>

#include <algorithm>
#include <memory>

#include <gobot/rendering/render_server.hpp>
#include <gobot/scene/resources/box_shape_3d.hpp>
#include <gobot/scene/resources/cylinder_shape_3d.hpp>
#include <gobot/scene/resources/material.hpp>
#include <gobot/scene/resources/primitive_mesh.hpp>
#include <gobot/scene/resources/resource_creation_registry.hpp>
#include <gobot/scene/resources/shape_3d.hpp>
#include <gobot/scene/resources/sphere_shape_3d.hpp>

namespace {

bool ContainsResourceType(const std::vector<const gobot::ResourceCreationEntry*>& entries,
                          std::string_view id) {
    return std::any_of(entries.begin(), entries.end(),
                       [id](const gobot::ResourceCreationEntry* entry) {
                           return entry->id == id;
                       });
}

}

TEST(TestResourceCreationRegistry, filters_resource_types_by_ref_property_type) {
    const auto mesh_types = gobot::ResourceCreationRegistry::GetCreatableTypesForProperty(
            gobot::Type::get<gobot::Ref<gobot::Mesh>>());
    EXPECT_TRUE(ContainsResourceType(mesh_types, "BoxMesh"));
    EXPECT_TRUE(ContainsResourceType(mesh_types, "CylinderMesh"));
    EXPECT_TRUE(ContainsResourceType(mesh_types, "SphereMesh"));
    EXPECT_FALSE(ContainsResourceType(mesh_types, "PBRMaterial3D"));

    const auto material_types = gobot::ResourceCreationRegistry::GetCreatableTypesForProperty(
            gobot::Type::get<gobot::Ref<gobot::Material>>());
    EXPECT_TRUE(ContainsResourceType(material_types, "PBRMaterial3D"));
    EXPECT_FALSE(ContainsResourceType(material_types, "BoxMesh"));

    const auto shape_types = gobot::ResourceCreationRegistry::GetCreatableTypesForProperty(
            gobot::Type::get<gobot::Ref<gobot::Shape3D>>());
    EXPECT_TRUE(ContainsResourceType(shape_types, "BoxShape3D"));
    EXPECT_TRUE(ContainsResourceType(shape_types, "CylinderShape3D"));
    EXPECT_TRUE(ContainsResourceType(shape_types, "SphereShape3D"));
    EXPECT_FALSE(ContainsResourceType(shape_types, "BoxMesh"));
}

TEST(TestResourceCreationRegistry, created_variant_converts_to_requested_ref_base_type) {
    auto render_server = std::make_unique<gobot::RenderServer>();

    gobot::Variant box_mesh = gobot::ResourceCreationRegistry::CreateResourceVariant(
            "BoxMesh",
            gobot::Type::get<gobot::Ref<gobot::Mesh>>());
    ASSERT_TRUE(box_mesh.is_valid());
    EXPECT_EQ(box_mesh.get_type(), gobot::Type::get<gobot::Ref<gobot::Mesh>>());

    gobot::Variant cylinder_mesh = gobot::ResourceCreationRegistry::CreateResourceVariant(
            "CylinderMesh",
            gobot::Type::get<gobot::Ref<gobot::Mesh>>());
    ASSERT_TRUE(cylinder_mesh.is_valid());
    EXPECT_EQ(cylinder_mesh.get_type(), gobot::Type::get<gobot::Ref<gobot::Mesh>>());

    gobot::Variant sphere_mesh = gobot::ResourceCreationRegistry::CreateResourceVariant(
            "SphereMesh",
            gobot::Type::get<gobot::Ref<gobot::Mesh>>());
    ASSERT_TRUE(sphere_mesh.is_valid());
    EXPECT_EQ(sphere_mesh.get_type(), gobot::Type::get<gobot::Ref<gobot::Mesh>>());

    gobot::Variant material = gobot::ResourceCreationRegistry::CreateResourceVariant(
            "PBRMaterial3D",
            gobot::Type::get<gobot::Ref<gobot::Material>>());
    ASSERT_TRUE(material.is_valid());
    EXPECT_EQ(material.get_type(), gobot::Type::get<gobot::Ref<gobot::Material>>());

    gobot::Variant box_shape = gobot::ResourceCreationRegistry::CreateResourceVariant(
            "BoxShape3D",
            gobot::Type::get<gobot::Ref<gobot::Shape3D>>());
    ASSERT_TRUE(box_shape.is_valid());
    EXPECT_EQ(box_shape.get_type(), gobot::Type::get<gobot::Ref<gobot::Shape3D>>());

    gobot::Variant cylinder_shape = gobot::ResourceCreationRegistry::CreateResourceVariant(
            "CylinderShape3D",
            gobot::Type::get<gobot::Ref<gobot::Shape3D>>());
    ASSERT_TRUE(cylinder_shape.is_valid());
    EXPECT_EQ(cylinder_shape.get_type(), gobot::Type::get<gobot::Ref<gobot::Shape3D>>());

    gobot::Variant sphere_shape = gobot::ResourceCreationRegistry::CreateResourceVariant(
            "SphereShape3D",
            gobot::Type::get<gobot::Ref<gobot::Shape3D>>());
    ASSERT_TRUE(sphere_shape.is_valid());
    EXPECT_EQ(sphere_shape.get_type(), gobot::Type::get<gobot::Ref<gobot::Shape3D>>());
}
