#include <gtest/gtest.h>

#include <filesystem>
#include <memory>

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
