#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <memory>
#include <algorithm>

#include <gobot/core/io/resource_format_mesh.hpp>
#include <gobot/core/io/resource_saver.hpp>
#include <gobot/core/config/project_setting.hpp>
#include <gobot/rendering/render_server.hpp>
#include <gobot/scene/resources/array_mesh.hpp>

namespace {

gobot::Ref<gobot::ArrayMesh> CreatePersistenceMesh() {
    auto mesh = gobot::MakeRef<gobot::ArrayMesh>();
    mesh->SetSurface(
            {
                    // 0.02f starts with byte 0x0a in little-endian encoding.
                    // This catches binary PLY readers that over-consume LF bytes
                    // after the end_header line.
                    {0.02f, 0.0f, 0.0f},
                    {1.0f, 0.0f, 0.0f},
                    {1.0f, 1.0f, 0.0f},
                    {0.0f, 1.0f, 0.0f},
            },
            {0, 1, 2, 0, 2, 3},
            {
                    {0.0f, 1.0f, 0.0f},
                    {0.0f, 1.0f, 0.0f},
                    {0.0f, 1.0f, 0.0f},
                    {0.0f, 1.0f, 0.0f},
            },
            {
                    {1.0f, 0.0f, 0.0f, 1.0f},
                    {0.0f, 1.0f, 0.0f, 0.75f},
                    {0.0f, 0.0f, 1.0f, 0.5f},
                    {0.25f, 0.5f, 0.75f, 0.25f},
            },
            {},
            {
                    {0.0f, 0.0f},
                    {1.0f, 0.0f},
                    {1.0f, 1.0f},
                    {0.0f, 1.0f},
            });
    return mesh;
}

} // namespace

TEST(TestResourceFormatMesh, recognizes_common_mesh_extensions_for_mesh_resources) {
    gobot::ResourceFormatLoaderMesh loader;
    EXPECT_TRUE(loader.HandlesType("Mesh"));
    EXPECT_TRUE(loader.RecognizePath("robot_visual.dae", "Mesh"));
    EXPECT_TRUE(loader.RecognizePath("robot_visual.obj", "ArrayMesh"));
    EXPECT_FALSE(loader.RecognizePath("robot.urdf", "Mesh"));
}

TEST(TestResourceFormatMesh, ply_saver_recognizes_only_array_mesh_ply_paths) {
    gobot::ResourceFormatSaverPLY saver;
    const gobot::Ref<gobot::ArrayMesh> mesh = CreatePersistenceMesh();

    EXPECT_TRUE(saver.Recognize(mesh));
    EXPECT_TRUE(saver.RecognizePath(mesh, "generated_mesh.ply"));
    EXPECT_FALSE(saver.RecognizePath(mesh, "generated_mesh.jres"));
}

TEST(TestResourceFormatMesh, saves_binary_ply_and_round_trips_vertex_attributes) {
    const std::filesystem::path output_root = "/tmp/gobot-ply-saver-test";
    std::filesystem::remove_all(output_root);
    std::filesystem::create_directories(output_root);

    gobot::ProjectSettings project_settings;
    ASSERT_TRUE(project_settings.SetProjectPath(output_root.string()));
    const gobot::Ref<gobot::ResourceFormatSaverPLY> saver =
            gobot::MakeRef<gobot::ResourceFormatSaverPLY>();
    gobot::ResourceSaver::AddResourceFormatSaver(saver, true);

    const gobot::Ref<gobot::ArrayMesh> source = CreatePersistenceMesh();
    ASSERT_TRUE(gobot::ResourceSaver::Save(source, "res://nested/generated_mesh.ply"));
    gobot::ResourceSaver::RemoveResourceFormatSaver(saver);

    const std::filesystem::path output_path = output_root / "nested/generated_mesh.ply";
    std::ifstream header_stream(output_path, std::ios::binary);
    ASSERT_TRUE(header_stream.is_open());
    std::string header(512, '\0');
    header_stream.read(header.data(), static_cast<std::streamsize>(header.size()));
    header.resize(static_cast<std::size_t>(header_stream.gcount()));
    EXPECT_TRUE(header.starts_with("ply\nformat binary_little_endian 1.0\n"));
    EXPECT_NE(header.find("property float nx\n"), std::string::npos);
    EXPECT_NE(header.find("property uchar gobot_padding\n"), std::string::npos);
    EXPECT_NE(header.find("property float texture_u\n"), std::string::npos);
    EXPECT_NE(header.find("property uchar alpha\n"), std::string::npos);

#ifndef GOBOT_HAS_ASSIMP
    GTEST_SKIP() << "Assimp support is not enabled; binary PLY output was still verified.";
#else
    auto render_server = std::make_unique<gobot::RenderServer>();
    gobot::ResourceFormatLoaderMesh loader;
    const gobot::Ref<gobot::Resource> loaded_resource = loader.Load(
            "res://nested/generated_mesh.ply",
            "res://nested/generated_mesh.ply",
            gobot::ResourceFormatLoader::CacheMode::Ignore);
    const gobot::Ref<gobot::ArrayMesh> loaded =
            gobot::dynamic_pointer_cast<gobot::ArrayMesh>(loaded_resource);
    ASSERT_TRUE(loaded.IsValid());

    const gobot::MeshSurfaceList surfaces = loaded->GetSurfaces();
    ASSERT_EQ(surfaces.size(), 1);
    const gobot::MeshSurfaceData& surface = surfaces.front();
    ASSERT_EQ(surface.vertices.size(), 4);
    ASSERT_EQ(surface.indices.size(), 6);
    ASSERT_EQ(surface.normals.size(), 4);
    ASSERT_EQ(surface.uv0.size(), 4);
    ASSERT_EQ(surface.colors.size(), 4);
    EXPECT_FALSE(surface.material.IsValid());
    EXPECT_NEAR(surface.vertices.front().x(), 0.02, 1e-6);
    for (const gobot::Vector3& vertex : surface.vertices) {
        EXPECT_TRUE(vertex.allFinite());
    }

    for (const gobot::Vector3& normal : surface.normals) {
        EXPECT_NEAR(normal.x(), 0.0, 1e-6);
        EXPECT_NEAR(normal.y(), 1.0, 1e-6);
        EXPECT_NEAR(normal.z(), 0.0, 1e-6);
    }
    EXPECT_NEAR(surface.uv0[2].x(), 1.0, 1e-6);
    EXPECT_NEAR(surface.uv0[2].y(), 1.0, 1e-6);
    EXPECT_NEAR(surface.colors[3].red(), 0.25, 1.0 / 255.0 + 1e-6);
    EXPECT_NEAR(surface.colors[3].green(), 0.5, 1.0 / 255.0 + 1e-6);
    EXPECT_NEAR(surface.colors[3].blue(), 0.75, 1.0 / 255.0 + 1e-6);
    EXPECT_NEAR(surface.colors[3].alpha(), 0.25, 1.0 / 255.0 + 1e-6);
#endif
}

TEST(TestResourceFormatMesh, ply_saver_rejects_multiple_surfaces) {
    gobot::ProjectSettings project_settings;
    const std::filesystem::path output_root = "/tmp/gobot-ply-saver-multi-surface-test";
    std::filesystem::create_directories(output_root);
    ASSERT_TRUE(project_settings.SetProjectPath(output_root.string()));

    const gobot::Ref<gobot::ArrayMesh> mesh = CreatePersistenceMesh();
    gobot::MeshSurfaceList surfaces = mesh->GetSurfaces();
    surfaces.push_back(surfaces.front());
    mesh->SetSurfaces(std::move(surfaces));

    gobot::ResourceFormatSaverPLY saver;
    EXPECT_FALSE(saver.Save(mesh, "res://multiple.ply"));
}

TEST(TestResourceFormatMesh, imports_triangle_mesh_when_assimp_is_available) {
#ifndef GOBOT_HAS_ASSIMP
    GTEST_SKIP() << "Assimp support is not enabled.";
#else
    gobot::ProjectSettings project_settings;
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
    ASSERT_EQ(mesh->GetSurfaces().size(), 1);
    EXPECT_EQ(mesh->GetSurfaces().front().normals.size(), 3);
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

TEST(TestResourceFormatMesh, preserves_multiple_surfaces_uv_tangents_and_material_textures) {
#ifndef GOBOT_HAS_ASSIMP
    GTEST_SKIP() << "Assimp support is not enabled.";
#else
    gobot::ProjectSettings project_settings;
    auto render_server = std::make_unique<gobot::RenderServer>();
    const std::filesystem::path fixture_path =
            std::filesystem::current_path() / "tests/fixtures/mesh/two_material.obj";

    gobot::ResourceFormatLoaderMesh loader;
    const auto resource = loader.Load(
            fixture_path.string(), fixture_path.string(), gobot::ResourceFormatLoader::CacheMode::Ignore);
    const auto mesh = gobot::dynamic_pointer_cast<gobot::ArrayMesh>(resource);
    ASSERT_TRUE(mesh.IsValid());

    const gobot::MeshSurfaceList surfaces = mesh->GetSurfaces();
    ASSERT_EQ(surfaces.size(), 2);
    bool found_texture = false;
    for (const gobot::MeshSurfaceData& surface : surfaces) {
        EXPECT_EQ(surface.vertices.size(), 3);
        EXPECT_EQ(surface.uv0.size(), surface.vertices.size());
        EXPECT_EQ(surface.tangents.size(), surface.vertices.size());
        const auto material = gobot::dynamic_pointer_cast<gobot::PBRMaterial3D>(surface.material);
        ASSERT_TRUE(material.IsValid());
        if (material->GetAlbedoTexture().IsValid()) {
            found_texture = true;
            ASSERT_TRUE(material->GetAlbedoTexture()->GetImage().IsValid());
            EXPECT_GT(material->GetAlbedoTexture()->GetImage()->GetWidth(), 0);
            EXPECT_GT(material->GetAlbedoTexture()->GetImage()->GetHeight(), 0);
        }
    }
    EXPECT_TRUE(found_texture);
#endif
}
