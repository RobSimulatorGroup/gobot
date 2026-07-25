#include <gtest/gtest.h>

#include <gobot/core/config/project_setting.hpp>
#include <gobot/core/io/resource_format_gaussian_splat.hpp>
#include <gobot/core/io/resource_format_scene.hpp>
#include <gobot/rendering/scene_render_items.hpp>
#include <gobot/scene/gaussian_splat_3d.hpp>
#include <gobot/scene/mesh_instance_3d.hpp>
#include <gobot/scene/node_3d.hpp>
#include <gobot/scene/resources/gaussian_splat.hpp>
#include <gobot/scene/resources/packed_scene.hpp>
#include <gobot/scene/resources/primitive_mesh.hpp>

#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>

namespace {

void WriteAsciiPly(const std::filesystem::path& path, bool include_opacity = true) {
    std::ofstream stream(path);
    stream << "ply\n"
              "format ascii 1.0\n"
              "element vertex 1\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n";
    if (include_opacity) stream << "property float opacity\n";
    stream << "property float scale_0\nproperty float scale_1\nproperty float scale_2\n"
              "property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n"
              "end_header\n"
              "1 2 3 0.1 0.2 0.3 ";
    if (include_opacity) stream << "0 ";
    stream << "0 0 0 2 0 0 0\n";
}

template <typename T>
void WriteBinary(std::ofstream& stream, T value) {
    stream.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

void WriteBinaryDegreeOnePly(const std::filesystem::path& path) {
    std::ofstream stream(path, std::ios::binary);
    stream << "ply\nformat binary_little_endian 1.0\nelement vertex 1\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n";
    for (int i = 0; i < 9; ++i) stream << "property float f_rest_" << i << "\n";
    stream << "property float opacity\n"
              "property float scale_0\nproperty float scale_1\nproperty float scale_2\n"
              "property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n"
              "end_header\n";
    const std::array<float, 3> mean{4.0f, 5.0f, 6.0f};
    const std::array<float, 3> dc{0.25f, 0.5f, 0.75f};
    const std::array<float, 9> rest{1, 2, 3, 4, 5, 6, 7, 8, 9};
    for (float value : mean) WriteBinary(stream, value);
    for (float value : dc) WriteBinary(stream, value);
    for (float value : rest) WriteBinary(stream, value);
    WriteBinary(stream, std::log(3.0f));
    WriteBinary(stream, std::log(2.0f));
    WriteBinary(stream, std::log(3.0f));
    WriteBinary(stream, std::log(4.0f));
    WriteBinary(stream, 1.0f);
    WriteBinary(stream, 0.0f);
    WriteBinary(stream, 0.0f);
    WriteBinary(stream, 0.0f);
}

gobot::Node* FindNode(gobot::Node* node, const std::string& name) {
    if (node == nullptr || node->GetName() == name) return node;
    for (std::size_t i = 0; i < node->GetChildCount(); ++i) {
        if (gobot::Node* found = FindNode(node->GetChild(static_cast<int>(i)), name)) return found;
    }
    return nullptr;
}

class GaussianSplatLoaderTest : public testing::Test {
protected:
    void SetUp() override {
        root_ = "/tmp/gobot-gaussian-splat-tests";
        std::filesystem::remove_all(root_);
        std::filesystem::create_directories(root_);
        ASSERT_TRUE(settings_.SetProjectPath(root_.string()));
        scene_loader_ = gobot::MakeRef<gobot::ResourceFormatLoaderScene>();
        scene_saver_ = gobot::MakeRef<gobot::ResourceFormatSaverScene>();
        gaussian_loader_ = gobot::MakeRef<gobot::ResourceFormatLoaderGaussianSplat>();
        gobot::ResourceLoader::AddResourceFormatLoader(scene_loader_, true);
        gobot::ResourceLoader::AddResourceFormatLoader(gaussian_loader_, true);
        gobot::ResourceSaver::AddResourceFormatSaver(scene_saver_, true);
    }

    void TearDown() override {
        gobot::ResourceSaver::RemoveResourceFormatSaver(scene_saver_);
        gobot::ResourceLoader::RemoveResourceFormatLoader(gaussian_loader_);
        gobot::ResourceLoader::RemoveResourceFormatLoader(scene_loader_);
        settings_.ClearProjectPath();
        std::filesystem::remove_all(root_);
    }

    void WriteProxyScene() {
        auto* root = gobot::Object::New<gobot::Node3D>();
        root->SetName("ProxySource");
        auto* mesh = gobot::Object::New<gobot::MeshInstance3D>();
        mesh->SetName("ProxyMesh");
        mesh->SetMesh(gobot::MakeRef<gobot::BoxMesh>());
        root->AddChild(mesh, true);
        gobot::Ref<gobot::PackedScene> packed = gobot::MakeRef<gobot::PackedScene>();
        ASSERT_TRUE(packed->Pack(root));
        gobot::Object::Delete(root);
        ASSERT_TRUE(gobot::ResourceSaver::Save(packed, "res://proxy.jscn"));
    }

    gobot::ProjectSettings settings_;
    std::filesystem::path root_;
    gobot::Ref<gobot::ResourceFormatLoaderScene> scene_loader_;
    gobot::Ref<gobot::ResourceFormatSaverScene> scene_saver_;
    gobot::Ref<gobot::ResourceFormatLoaderGaussianSplat> gaussian_loader_;
};

} // namespace

TEST_F(GaussianSplatLoaderTest, parses_ascii_activated_values_and_normalized_quaternion) {
    const auto path = root_ / "scene.ply";
    WriteAsciiPly(path);
    gobot::Ref<gobot::GaussianSplatResource> resource = gobot::MakeRef<gobot::GaussianSplatResource>();
    std::string error;
    ASSERT_TRUE(resource->LoadPly(path.string(), &error)) << error;
    const auto data = resource->GetData();
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(data->count, 1u);
    EXPECT_EQ(data->sh_degree, 0);
    EXPECT_FLOAT_EQ(data->rotations_wxyz[0], 1.0f);
    EXPECT_FLOAT_EQ(data->scales[0], 1.0f);
    EXPECT_FLOAT_EQ(data->opacities[0], 0.5f);
    EXPECT_FLOAT_EQ(data->sh_coefficients[2], 0.3f);
    EXPECT_TRUE(data->bounds.GetMin().isApprox(gobot::Vector3{-2, -1, 0}, 1e-6));
    EXPECT_TRUE(data->bounds.GetMax().isApprox(gobot::Vector3{4, 5, 6}, 1e-6));
}

TEST_F(GaussianSplatLoaderTest, parses_binary_little_endian_degree_one_channel_major_sh) {
    const auto path = root_ / "binary.ply";
    WriteBinaryDegreeOnePly(path);
    gobot::Ref<gobot::GaussianSplatResource> resource = gobot::MakeRef<gobot::GaussianSplatResource>();
    std::string error;
    ASSERT_TRUE(resource->LoadPly(path.string(), &error)) << error;
    const auto data = resource->GetData();
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(data->sh_degree, 1);
    EXPECT_NEAR(data->opacities[0], 0.75f, 1e-6f);
    EXPECT_NEAR(data->scales[2], 4.0f, 1e-6f);
    EXPECT_FLOAT_EQ(data->sh_coefficients[3], 1.0f);
    EXPECT_FLOAT_EQ(data->sh_coefficients[4], 4.0f);
    EXPECT_FLOAT_EQ(data->sh_coefficients[5], 7.0f);
}

TEST_F(GaussianSplatLoaderTest, rejects_missing_required_property) {
    const auto path = root_ / "invalid.ply";
    WriteAsciiPly(path, false);
    gobot::Ref<gobot::GaussianSplatResource> resource = gobot::MakeRef<gobot::GaussianSplatResource>();
    std::string error;
    EXPECT_FALSE(resource->LoadPly(path.string(), &error));
    EXPECT_NE(error.find("opacity"), std::string::npos);
}

TEST_F(GaussianSplatLoaderTest, rejects_binary_big_endian_data) {
    const auto path = root_ / "big_endian.ply";
    std::ofstream(path, std::ios::binary)
            << "ply\nformat binary_big_endian 1.0\nelement vertex 1\nend_header\n";
    gobot::Ref<gobot::GaussianSplatResource> resource =
            gobot::MakeRef<gobot::GaussianSplatResource>();
    std::string error;
    EXPECT_FALSE(resource->LoadPly(path.string(), &error));
    EXPECT_NE(error.find("binary_big_endian"), std::string::npos);
}

TEST_F(GaussianSplatLoaderTest, rejects_duplicate_elements_and_properties) {
    const auto path = root_ / "duplicate.ply";
    std::ofstream(path)
            << "ply\nformat ascii 1.0\nelement vertex 1\nelement vertex 1\nend_header\n";
    gobot::Ref<gobot::GaussianSplatResource> resource =
            gobot::MakeRef<gobot::GaussianSplatResource>();
    std::string error;
    EXPECT_FALSE(resource->LoadPly(path.string(), &error));
    EXPECT_NE(error.find("duplicate element"), std::string::npos);

    std::ofstream(path)
            << "ply\nformat ascii 1.0\nelement vertex 1\n"
               "property float x\nproperty float x\nend_header\n";
    error.clear();
    EXPECT_FALSE(resource->LoadPly(path.string(), &error));
    EXPECT_NE(error.find("duplicate property"), std::string::npos);
}

TEST_F(GaussianSplatLoaderTest, rejects_manifest_field_type_errors_without_throwing) {
    WriteAsciiPly(root_ / "environment.ply");
    const auto path = root_ / "invalid.gsplat";
    std::ofstream(path)
            << R"({
  "__VERSION__": "1",
  "__TYPE__": 3,
  "ply": "environment.ply",
  "meters_per_unit": "one"
})";

    gobot::Ref<gobot::Resource> loaded;
    EXPECT_NO_THROW(loaded = gaussian_loader_->Load(path.string()));
    EXPECT_FALSE(loaded.IsValid());

    std::ofstream(path)
            << R"({
  "__VERSION__": 1,
  "__TYPE__": "GaussianSplatScene",
  "ply": "environment.ply",
  "meters_per_unit": "one"
})";
    EXPECT_NO_THROW(loaded = gaussian_loader_->Load(path.string()));
    EXPECT_FALSE(loaded.IsValid());

    std::ofstream(path)
            << R"({
  "__VERSION__": 18446744073709551615,
  "__TYPE__": "GaussianSplatScene",
  "ply": "environment.ply"
})";
    EXPECT_NO_THROW(loaded = gaussian_loader_->Load(path.string()));
    EXPECT_FALSE(loaded.IsValid());
}

TEST_F(GaussianSplatLoaderTest, imports_manifest_proxy_and_snapshot_contract) {
    WriteAsciiPly(root_ / "environment.ply");
    WriteProxyScene();
    std::ofstream(root_ / "environment.gsplat")
            << R"({
  "__VERSION__": 1,
  "__TYPE__": "GaussianSplatScene",
  "ply": "environment.ply",
  "proxy_scene": "proxy.jscn",
  "meters_per_unit": 2.0,
  "source_to_gobot": [1,0,0,1, 0,1,0,2, 0,0,1,3, 0,0,0,1]
})";

    gobot::Ref<gobot::PackedScene> packed = gobot::dynamic_pointer_cast<gobot::PackedScene>(
            gaussian_loader_->Load((root_ / "environment.gsplat").string()));
    ASSERT_TRUE(packed.IsValid());
    gobot::Node* root = packed->Instantiate();
    ASSERT_NE(root, nullptr);
    auto* gaussian = gobot::Object::PointerCastTo<gobot::GaussianSplat3D>(
            FindNode(root, "GaussianEnvironment"));
    auto* proxy = gobot::Object::PointerCastTo<gobot::MeshInstance3D>(FindNode(root, "ProxyMesh"));
    ASSERT_NE(gaussian, nullptr);
    ASSERT_NE(proxy, nullptr);
    EXPECT_FALSE(proxy->IsVisibleInRgb());
    EXPECT_FALSE(proxy->IsCastShadow());
    auto* root_3d = gobot::Object::PointerCastTo<gobot::Node3D>(root);
    ASSERT_NE(root_3d, nullptr);
    EXPECT_TRUE(root_3d->GetPosition().isApprox(gobot::Vector3{2, 4, 6}, 1e-5));
    EXPECT_TRUE(root_3d->GetScale().isApprox(gobot::Vector3{2, 2, 2}, 1e-5));

    const gobot::RenderSceneSnapshot snapshot = gobot::CaptureRenderSceneSnapshot(root);
    ASSERT_EQ(snapshot.gaussian_splats.size(), 1u);
    EXPECT_TRUE(snapshot.gaussian_splat_error.empty());
    ASSERT_EQ(snapshot.visual_meshes.size(), 1u);
    EXPECT_FALSE(snapshot.visual_meshes[0].visible_in_rgb);
    EXPECT_FALSE(snapshot.visual_meshes[0].cast_shadow);
    gobot::Object::Delete(root);

    ASSERT_TRUE(gobot::ResourceSaver::Save(packed, "res://environment_roundtrip.jscn"));
    gobot::Ref<gobot::PackedScene> restored = gobot::dynamic_pointer_cast<gobot::PackedScene>(
            gobot::ResourceLoader::Load("res://environment_roundtrip.jscn",
                                        "PackedScene",
                                        gobot::ResourceFormatLoader::CacheMode::Ignore));
    ASSERT_TRUE(restored.IsValid());
    gobot::Node* restored_root = restored->Instantiate();
    ASSERT_NE(restored_root, nullptr);
    const gobot::RenderSceneSnapshot restored_snapshot =
            gobot::CaptureRenderSceneSnapshot(restored_root);
    ASSERT_EQ(restored_snapshot.gaussian_splats.size(), 1u);
    EXPECT_EQ(restored_snapshot.gaussian_splats[0].data->count, 1u);
    ASSERT_EQ(restored_snapshot.visual_meshes.size(), 1u);
    EXPECT_FALSE(restored_snapshot.visual_meshes[0].visible_in_rgb);
    gobot::Object::Delete(restored_root);
}

TEST_F(GaussianSplatLoaderTest, reports_multiple_enabled_gaussian_nodes) {
    WriteAsciiPly(root_ / "environment.ply");
    gobot::Ref<gobot::GaussianSplatResource> resource = gobot::MakeRef<gobot::GaussianSplatResource>();
    ASSERT_TRUE(resource->LoadPly((root_ / "environment.ply").string()));
    auto* root = gobot::Object::New<gobot::Node3D>();
    for (int i = 0; i < 2; ++i) {
        auto* gaussian = gobot::Object::New<gobot::GaussianSplat3D>();
        gaussian->SetName("Gaussian" + std::to_string(i));
        gaussian->SetSplat(resource);
        root->AddChild(gaussian, true);
    }
    auto* hidden = gobot::Object::New<gobot::GaussianSplat3D>();
    hidden->SetName("HiddenGaussian");
    hidden->SetSplat(resource);
    hidden->SetVisible(false);
    root->AddChild(hidden, true);
    const gobot::RenderSceneSnapshot snapshot = gobot::CaptureRenderSceneSnapshot(root);
    EXPECT_EQ(snapshot.gaussian_splats.size(), 2u);
    EXPECT_FALSE(snapshot.gaussian_splat_error.empty());
    gobot::Object::Delete(root);
}
