#include <gtest/gtest.h>

#include <filesystem>
#include <memory>
#include <algorithm>

#include <gobot/core/io/resource_format_mesh.hpp>
#include <gobot/rendering/render_server.hpp>
#include <gobot/scene/resources/array_mesh.hpp>

TEST(TestResourceFormatMesh, recognizes_common_mesh_extensions_for_mesh_resources) {
    gobot::ResourceFormatLoaderMesh loader;
    EXPECT_TRUE(loader.HandlesType("Mesh"));
    EXPECT_TRUE(loader.RecognizePath("robot_visual.dae", "Mesh"));
    EXPECT_TRUE(loader.RecognizePath("robot_visual.obj", "ArrayMesh"));
    EXPECT_FALSE(loader.RecognizePath("robot.urdf", "Mesh"));
}

TEST(TestResourceFormatMesh, imports_triangle_mesh_when_assimp_is_available) {
#ifndef GOBOT_HAS_ASSIMP
    GTEST_SKIP() << "Assimp support is not enabled.";
#else
    auto render_server = std::make_unique<gobot::RenderServer>();

    const std::filesystem::path fixture_path =
            std::filesystem::current_path() / "tests/fixtures/mesh/triangle.ply";

    gobot::ResourceFormatLoaderMesh loader;
    gobot::Ref<gobot::Resource> resource =
            loader.Load(fixture_path.string(), fixture_path.string(), gobot::ResourceFormatLoader::CacheMode::Ignore);
    gobot::Ref<gobot::ArrayMesh> mesh = gobot::dynamic_pointer_cast<gobot::ArrayMesh>(resource);

    ASSERT_TRUE(mesh.IsValid());
    EXPECT_EQ(mesh->GetVertices().size(), 3);
    EXPECT_EQ(mesh->GetIndices().size(), 3);
#endif
}

TEST(TestResourceFormatMesh, applies_assimp_node_transforms) {
#ifndef GOBOT_HAS_ASSIMP
    GTEST_SKIP() << "Assimp support is not enabled.";
#else
    auto render_server = std::make_unique<gobot::RenderServer>();

    const std::filesystem::path fixture_path =
            std::filesystem::current_path() / "tests/fixtures/mesh/translated_triangle.dae";

    gobot::ResourceFormatLoaderMesh loader;
    gobot::Ref<gobot::Resource> resource =
            loader.Load(fixture_path.string(), fixture_path.string(), gobot::ResourceFormatLoader::CacheMode::Ignore);
    gobot::Ref<gobot::ArrayMesh> mesh = gobot::dynamic_pointer_cast<gobot::ArrayMesh>(resource);

    ASSERT_TRUE(mesh.IsValid());
    ASSERT_EQ(mesh->GetVertices().size(), 3);

    float max_component = 0.0f;
    for (const gobot::Vector3& vertex : mesh->GetVertices()) {
        max_component = std::max(max_component, static_cast<float>(vertex.cwiseAbs().maxCoeff()));
    }
    EXPECT_GT(max_component, 1.5f);
#endif
}

TEST(TestResourceFormatMesh, preserves_collada_z_up_node_axis_transform) {
#ifndef GOBOT_HAS_ASSIMP
    GTEST_SKIP() << "Assimp support is not enabled.";
#else
    auto render_server = std::make_unique<gobot::RenderServer>();

    const std::filesystem::path fixture_path =
            std::filesystem::current_path() / "tests/fixtures/mesh/blender_z_up_axis_triangle.dae";

    gobot::ResourceFormatLoaderMesh loader;
    gobot::Ref<gobot::Resource> resource =
            loader.Load(fixture_path.string(), fixture_path.string(), gobot::ResourceFormatLoader::CacheMode::Ignore);
    gobot::Ref<gobot::ArrayMesh> mesh = gobot::dynamic_pointer_cast<gobot::ArrayMesh>(resource);

    ASSERT_TRUE(mesh.IsValid());
    ASSERT_EQ(mesh->GetVertices().size(), 3);

    float max_abs_y = 0.0f;
    float min_z = 0.0f;
    for (const gobot::Vector3& vertex : mesh->GetVertices()) {
        max_abs_y = std::max(max_abs_y, static_cast<float>(std::abs(vertex.y())));
        min_z = std::min(min_z, static_cast<float>(vertex.z()));
    }

    EXPECT_LT(max_abs_y, 1e-4f);
    EXPECT_LT(min_z, -1.5f);
#endif
}
