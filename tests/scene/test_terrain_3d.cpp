#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <iterator>

#include <gobot/core/io/resource_loader.hpp>
#include <gobot/core/io/resource_format_scene.hpp>
#include <gobot/core/config/project_setting.hpp>
#include <gobot/scene/resources/packed_scene.hpp>
#include <gobot/scene/resources/terrain_generator_config.hpp>
#include <gobot/scene/terrain_generator.hpp>
#include <gobot/scene/terrain_3d.hpp>

namespace {

class TestTerrain3D : public testing::Test {
protected:
    static void SetUpTestSuite() {
        static gobot::Ref<gobot::ResourceFormatLoaderScene> resource_loader_scene;
        static gobot::Ref<gobot::ResourceFormatSaverScene> resource_saver_scene;

        resource_saver_scene = gobot::MakeRef<gobot::ResourceFormatSaverScene>();
        gobot::ResourceSaver::AddResourceFormatSaver(resource_saver_scene, true);

        resource_loader_scene = gobot::MakeRef<gobot::ResourceFormatLoaderScene>();
        gobot::ResourceLoader::AddResourceFormatLoader(resource_loader_scene, true);
    }

    void SetUp() override {
        setenv("HOME", "/tmp/gobot-test-home", 1);
        std::filesystem::create_directories("/tmp/gobot_terrain_test_project");
        project_settings.SetProjectPath("/tmp/gobot_terrain_test_project");
    }

    gobot::ProjectSettings project_settings;
};

} // namespace

TEST_F(TestTerrain3D, generator_config_is_reflectively_constructible) {
    const gobot::Type type = gobot::Type::get_by_name("TerrainGeneratorConfig");
    ASSERT_TRUE(type.is_valid());

    gobot::Variant instance = type.create();
    bool success = false;
    gobot::Resource* resource = instance.convert<gobot::Resource*>(&success);
    ASSERT_TRUE(success);
    ASSERT_NE(resource, nullptr);
    gobot::Ref<gobot::Resource> owner(resource);
}

TEST_F(TestTerrain3D, generates_versioned_rough_terrain_deterministically) {
    const gobot::Ref<gobot::TerrainGeneratorConfig> config =
            gobot::MakeRoughTerrainGeneratorConfig();
    gobot::GeneratedTerrainData first;
    gobot::GeneratedTerrainData second;
    std::string error;

    ASSERT_TRUE(gobot::GenerateTerrain(*config.Get(), &first, &error)) << error;
    ASSERT_TRUE(gobot::GenerateTerrain(*config.Get(), &second, &error)) << error;
    EXPECT_EQ(first.boxes.size(), 514);
    EXPECT_EQ(first.heightfields.size(), 40);
    EXPECT_EQ(first.spawn_origins.size(), 70);
    ASSERT_EQ(first.heightfields.size(), second.heightfields.size());
    EXPECT_EQ(first.heightfields.front().heights, second.heightfields.front().heights);
    EXPECT_EQ(first.spawn_origins, second.spawn_origins);
}

TEST_F(TestTerrain3D, generator_resource_round_trips_without_baking_geometry) {
    gobot::Ref<gobot::TerrainGeneratorConfig> config =
            gobot::MakeRoughTerrainGeneratorConfig();
    USING_ENUM_BITWISE_OPERATORS;
    ASSERT_TRUE(gobot::ResourceSaver::Save(
            config,
            "res://rough_terrain.jres",
            gobot::ResourceSaverFlags::ChangePath));
    gobot::Ref<gobot::Resource> loaded_resource =
            gobot::ResourceLoader::Load("res://rough_terrain.jres",
                                        "TerrainGeneratorConfig",
                                        gobot::ResourceFormatLoader::CacheMode::Ignore);
    gobot::Ref<gobot::TerrainGeneratorConfig> loaded_config =
            gobot::dynamic_pointer_cast<gobot::TerrainGeneratorConfig>(loaded_resource);
    ASSERT_TRUE(loaded_config.IsValid());
    EXPECT_EQ(loaded_config->GetSchemaVersion(), 1);
    EXPECT_EQ(loaded_config->GetSubTerrains().size(), 7);
    config->SetPath("res://rough_terrain.jres", true);

    auto* terrain = gobot::Object::New<gobot::Terrain3D>();
    terrain->SetName("terrain");
    terrain->SetGeneratorConfig(config);
    ASSERT_EQ(terrain->GetBoxes().size(), 514);

    gobot::Ref<gobot::PackedScene> packed = gobot::MakeRef<gobot::PackedScene>();
    ASSERT_TRUE(packed->Pack(terrain));
    ASSERT_TRUE(gobot::ResourceSaver::Save(packed, "res://procedural_terrain.jscn"));

    const std::filesystem::path scene_path =
            "/tmp/gobot_terrain_test_project/procedural_terrain.jscn";
    std::ifstream input(scene_path);
    const std::string scene_text((std::istreambuf_iterator<char>(input)),
                                 std::istreambuf_iterator<char>());
    EXPECT_LT(scene_text.size(), 10'000);
    EXPECT_EQ(scene_text.find("\"heights\""), std::string::npos);
    EXPECT_NE(scene_text.find("rough_terrain.jres"), std::string::npos);

    gobot::Ref<gobot::Resource> loaded_scene_resource =
            gobot::ResourceLoader::Load("res://procedural_terrain.jscn",
                                        "PackedScene",
                                        gobot::ResourceFormatLoader::CacheMode::Ignore);
    gobot::Ref<gobot::PackedScene> loaded_scene =
            gobot::dynamic_pointer_cast<gobot::PackedScene>(loaded_scene_resource);
    ASSERT_TRUE(loaded_scene.IsValid());
    gobot::Node* instance = loaded_scene->Instantiate();
    auto* loaded_terrain = gobot::Object::PointerCastTo<gobot::Terrain3D>(instance);
    ASSERT_NE(loaded_terrain, nullptr);
    EXPECT_EQ(loaded_terrain->GetBoxes().size(), 514);
    EXPECT_EQ(loaded_terrain->GetHeightFields().size(), 40);
    EXPECT_EQ(loaded_terrain->GetSpawnOrigins().size(), 70);

    gobot::Object::Delete(instance);
    gobot::Object::Delete(terrain);
}

TEST_F(TestTerrain3D, stores_primitives_heightfields_and_spawn_origins) {
    auto* terrain = gobot::Object::New<gobot::Terrain3D>();
    terrain->SetName("terrain");
    gobot::TerrainBox box;
    box.center = {0.0, 0.0, -0.5};
    box.size = {2.0, 2.0, 1.0};
    box.rotation_degrees = {0.0, 0.0, 15.0};
    box.color = {0.2f, 0.3f, 0.4f, 1.0f};
    terrain->AddBox(box);

    gobot::TerrainHeightField heightfield;
    heightfield.center = {1.0, 0.0, 0.0};
    heightfield.size = {2.0, 2.0};
    heightfield.rows = 2;
    heightfield.cols = 3;
    heightfield.heights = {0.0, 0.1, 0.0, 0.2, 0.3, 0.1};
    heightfield.normalized_elevation = {0.0, 0.33, 0.0, 0.66, 1.0, 0.33};
    heightfield.base_thickness = 0.2;
    heightfield.z_offset = -0.1;
    terrain->AddHeightField(heightfield);
    gobot::TerrainMeshPatch mesh_patch;
    mesh_patch.center = {0.0, 1.0, 0.0};
    mesh_patch.vertices = {{-0.5, -0.5, 0.0}, {0.5, -0.5, 0.0}, {0.5, 0.5, 0.0}, {-0.5, 0.5, 0.0}};
    mesh_patch.indices = {0, 1, 2, 0, 2, 3};
    mesh_patch.color = {0.8f, 0.2f, 0.1f, 1.0f};
    terrain->AddMeshPatch(mesh_patch);
    terrain->SetSpawnOrigins({{0.0, 0.0, 0.0}, {1.0, 1.0, 0.3}});

    ASSERT_EQ(terrain->GetBoxes().size(), 1);
    ASSERT_EQ(terrain->GetHeightFields().size(), 1);
    ASSERT_EQ(terrain->GetMeshPatches().size(), 1);
    EXPECT_TRUE(terrain->GetBoxes()[0].size.isApprox(gobot::Vector3(2.0, 2.0, 1.0), CMP_EPSILON));
    EXPECT_FLOAT_EQ(terrain->GetBoxes()[0].color.blue(), 0.4f);
    EXPECT_EQ(terrain->GetHeightFields()[0].heights.size(), 6);
    EXPECT_EQ(terrain->GetHeightFields()[0].normalized_elevation.size(), 6);
    EXPECT_FLOAT_EQ(terrain->GetHeightFields()[0].z_offset, -0.1);
    EXPECT_EQ(terrain->GetSpawnOrigins().size(), 2);

    gobot::Ref<gobot::ArrayMesh> mesh = terrain->GetRenderMesh();
    ASSERT_TRUE(mesh.IsValid());
    EXPECT_GT(mesh->GetVertices().size(), 0);
    EXPECT_GT(mesh->GetIndices().size(), 0);
    EXPECT_EQ(mesh->GetNormals().size(), mesh->GetVertices().size());

    gobot::Object::Delete(terrain);
}

TEST_F(TestTerrain3D, render_mesh_uses_authored_colors) {
    auto* terrain = gobot::Object::New<gobot::Terrain3D>();
    terrain->SetColorMode(gobot::TerrainColorMode::Palette);

    gobot::TerrainBox box;
    box.center = {0.0, 0.0, -0.5};
    box.size = {1.0, 1.0, 1.0};
    box.color = {0.2f, 0.4f, 0.6f, 1.0f};
    terrain->AddBox(box);

    gobot::Ref<gobot::ArrayMesh> mesh = terrain->GetRenderMesh();
    ASSERT_TRUE(mesh.IsValid());
    ASSERT_EQ(mesh->GetColors().size(), mesh->GetVertices().size());
    ASSERT_GT(mesh->GetColors().size(), 0);
    EXPECT_FLOAT_EQ(mesh->GetColors()[0].red(), 0.2f);
    EXPECT_FLOAT_EQ(mesh->GetColors()[0].green(), 0.4f);
    EXPECT_FLOAT_EQ(mesh->GetColors()[0].blue(), 0.6f);

    gobot::Object::Delete(terrain);
}

TEST_F(TestTerrain3D, array_mesh_stores_optional_vertex_colors) {
    gobot::Ref<gobot::ArrayMesh> mesh = gobot::MakeRef<gobot::ArrayMesh>();
    mesh->SetSurface(
            {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}},
            {0, 1, 2},
            {{0.0, 0.0, 1.0}, {0.0, 0.0, 1.0}, {0.0, 0.0, 1.0}},
            {{0.1f, 0.2f, 0.3f, 1.0f}, {0.4f, 0.5f, 0.6f, 1.0f}, {0.7f, 0.8f, 0.9f, 1.0f}});

    ASSERT_EQ(mesh->GetColors().size(), mesh->GetVertices().size());
    EXPECT_FLOAT_EQ(mesh->GetColors()[0].red(), 0.1f);
    EXPECT_FLOAT_EQ(mesh->GetColors()[2].blue(), 0.9f);

    mesh->SetSurface({{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}},
                     {0, 1, 2},
                     {},
                     {{1.0f, 1.0f, 1.0f, 1.0f}});
    EXPECT_TRUE(mesh->GetColors().empty());
}

TEST_F(TestTerrain3D, height_ramp_render_mesh_generates_vertex_colors) {
    auto* terrain = gobot::Object::New<gobot::Terrain3D>();
    terrain->SetHeightLowColor({0.0f, 0.2f, 0.0f, 1.0f});
    terrain->SetHeightHighColor({1.0f, 0.8f, 0.2f, 1.0f});

    gobot::TerrainHeightField heightfield;
    heightfield.center = {0.0, 0.0, 0.0};
    heightfield.size = {1.0, 1.0};
    heightfield.rows = 2;
    heightfield.cols = 2;
    heightfield.heights = {0.0, 0.0, 1.0, 1.0};
    terrain->AddHeightField(heightfield);

    gobot::Ref<gobot::ArrayMesh> mesh = terrain->GetRenderMesh();
    ASSERT_TRUE(mesh.IsValid());
    ASSERT_EQ(mesh->GetColors().size(), mesh->GetVertices().size());
    ASSERT_GE(mesh->GetColors().size(), 4);
    EXPECT_LT(mesh->GetColors()[0].red(), mesh->GetColors()[2].red());
    EXPECT_LT(mesh->GetColors()[0].green(), mesh->GetColors()[2].green());

    terrain->SetColorMode(gobot::TerrainColorMode::SurfaceColor);
    mesh = terrain->GetRenderMesh();
    ASSERT_TRUE(mesh.IsValid());
    EXPECT_TRUE(mesh->GetColors().empty());

    gobot::Object::Delete(terrain);
}

TEST_F(TestTerrain3D, round_trips_through_packed_scene) {
    auto* terrain = gobot::Object::New<gobot::Terrain3D>();
    terrain->SetName("terrain");
    terrain->SetSurfaceColor({0.2f, 0.4f, 0.6f, 1.0f});
    terrain->SetColorMode(gobot::TerrainColorMode::Palette);
    terrain->SetHeightLowColor({0.1f, 0.2f, 0.3f, 1.0f});
    terrain->SetHeightHighColor({0.8f, 0.7f, 0.2f, 1.0f});
    terrain->SetHeightRangeMin(-0.25);
    terrain->SetHeightRangeMax(0.75);
    terrain->AddBox({0.0, 0.0, -0.5}, {3.0, 2.0, 1.0}, {0.0, 0.0, 10.0});

    gobot::TerrainHeightField heightfield;
    heightfield.center = {0.0, 0.0, 0.0};
    heightfield.size = {2.0, 2.0};
    heightfield.rows = 2;
    heightfield.cols = 2;
    heightfield.heights = {0.0, 0.1, 0.2, 0.3};
    heightfield.normalized_elevation = {0.0, 0.33, 0.66, 1.0};
    heightfield.z_offset = -0.1;
    terrain->AddHeightField(heightfield);
    gobot::TerrainMeshPatch mesh_patch;
    mesh_patch.center = {1.0, 0.0, 0.1};
    mesh_patch.vertices = {{0.0, 0.0, 0.0}, {0.5, 0.0, 0.0}, {0.0, 0.5, 0.0}};
    mesh_patch.indices = {0, 1, 2};
    mesh_patch.color = {0.5f, 0.6f, 0.7f, 1.0f};
    terrain->AddMeshPatch(mesh_patch);
    terrain->SetSpawnOrigins({{0.0, 0.0, 0.3}});

    gobot::Ref<gobot::PackedScene> packed = gobot::MakeRef<gobot::PackedScene>();
    ASSERT_TRUE(packed->Pack(terrain));
    ASSERT_TRUE(gobot::ResourceSaver::Save(packed, "res://gobot_terrain_roundtrip.jscn"));

    gobot::Ref<gobot::Resource> loaded =
            gobot::ResourceLoader::Load("res://gobot_terrain_roundtrip.jscn",
                                        "PackedScene",
                                        gobot::ResourceFormatLoader::CacheMode::Ignore);
    auto loaded_scene = gobot::dynamic_pointer_cast<gobot::PackedScene>(loaded);
    ASSERT_TRUE(loaded_scene.IsValid());
    gobot::Node* instance = loaded_scene->Instantiate();
    ASSERT_NE(instance, nullptr);

    auto* loaded_terrain = gobot::Object::PointerCastTo<gobot::Terrain3D>(instance);
    ASSERT_NE(loaded_terrain, nullptr);
    ASSERT_EQ(loaded_terrain->GetBoxes().size(), 1);
    ASSERT_EQ(loaded_terrain->GetHeightFields().size(), 1);
    ASSERT_EQ(loaded_terrain->GetMeshPatches().size(), 1);
    ASSERT_EQ(loaded_terrain->GetSpawnOrigins().size(), 1);
    EXPECT_TRUE(loaded_terrain->GetBoxes()[0].rotation_degrees.isApprox(
            gobot::Vector3(0.0, 0.0, 10.0), CMP_EPSILON));
    EXPECT_EQ(loaded_terrain->GetHeightFields()[0].heights.size(), 4);
    EXPECT_EQ(loaded_terrain->GetHeightFields()[0].normalized_elevation.size(), 4);
    EXPECT_FLOAT_EQ(loaded_terrain->GetHeightFields()[0].z_offset, -0.1);
    EXPECT_FLOAT_EQ(loaded_terrain->GetMeshPatches()[0].color.green(), 0.6f);
    EXPECT_FLOAT_EQ(loaded_terrain->GetSurfaceColor().blue(), 0.6f);
    EXPECT_EQ(loaded_terrain->GetColorMode(), gobot::TerrainColorMode::Palette);
    EXPECT_FLOAT_EQ(loaded_terrain->GetHeightLowColor().green(), 0.2f);
    EXPECT_FLOAT_EQ(loaded_terrain->GetHeightHighColor().red(), 0.8f);
    EXPECT_FLOAT_EQ(loaded_terrain->GetHeightRangeMin(), -0.25);
    EXPECT_FLOAT_EQ(loaded_terrain->GetHeightRangeMax(), 0.75);

    gobot::Object::Delete(instance);
    gobot::Object::Delete(terrain);
}
