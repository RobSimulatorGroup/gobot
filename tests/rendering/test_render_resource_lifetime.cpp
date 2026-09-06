#include <gtest/gtest.h>

#include <gobot/core/io/image.hpp>
#include <gobot/rendering/headless_render_context.hpp>
#include <gobot/rendering/render_product.hpp>
#include <gobot/rendering/render_server.hpp>
#include <gobot/scene/camera_3d.hpp>
#include <gobot/scene/mesh_instance_3d.hpp>
#include <gobot/scene/resources/array_mesh.hpp>
#include <gobot/scene/resources/primitive_mesh.hpp>

#include <cstdlib>

namespace gobot {

TEST(RenderResourceLifetime, image_owns_current_snapshot_and_preserves_retained_versions) {
    auto image = MakeRef<Image>(2, 2, false, ImageFormat::RGBA8);
    auto first = image->GetStorageSnapshot();
    std::weak_ptr<const ImageStorageData> weak = first;
    first.reset();
    EXPECT_FALSE(weak.expired());
    auto retained = weak.lock();
    EXPECT_EQ(retained, image->GetStorageSnapshot());
    image->SetPixel(0, 0, Color{1.0f, 0.0f, 0.0f, 1.0f});
    const auto changed = image->GetStorageSnapshot();
    EXPECT_NE(retained, changed);
    EXPECT_EQ(retained->data[0], 0);
    EXPECT_EQ(changed->data[0], 255);
    image.Reset();
    EXPECT_EQ(changed->data[0], 255);
    retained.reset();
    EXPECT_TRUE(weak.expired());
}

TEST(RenderResourceLifetime, mesh_separates_topology_geometry_and_material_revisions) {
    auto mesh = MakeRef<ArrayMesh>();
    std::vector<Vector3> vertices{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
    mesh->SetSurface(vertices, {0, 1, 2});
    const auto retained = mesh->GetSurfaceData();
    const auto geometry = mesh->GetGeometryRevision();
    const auto topology = mesh->GetTopologyRevision();
    mesh->SetMaterial(MakeRef<PBRMaterial3D>());
    EXPECT_EQ(mesh->GetGeometryRevision(), geometry);
    EXPECT_EQ(mesh->GetTopologyRevision(), topology);
    EXPECT_FALSE(retained->front().material.IsValid());
    EXPECT_TRUE(mesh->GetSurfaceData()->front().material.IsValid());
    vertices[0].z() = 0.1;
    mesh->SetSurface(vertices, {0, 1, 2});
    EXPECT_GT(mesh->GetGeometryRevision(), geometry);
    EXPECT_EQ(mesh->GetTopologyRevision(), topology);
    EXPECT_EQ(retained->front().vertices[0].z(), 0);
    mesh->SetSurface(vertices, {0, 2, 1});
    EXPECT_GT(mesh->GetTopologyRevision(), topology);
}

TEST(RenderResourceLifetime, opengl_reuses_uploads_and_reclaims_inactive_resources) {
    if (std::getenv("GOBOT_RUN_RENDER_GPU_TEST") == nullptr) {
        GTEST_SKIP() << "Set GOBOT_RUN_RENDER_GPU_TEST=1 on an EGL runner.";
    }
    HeadlessRenderContext context;
    ASSERT_TRUE(context.Initialize()) << context.GetLastError();
    auto* renderer = RenderServer::GetInstance();
    Node3D root;
    auto* node = Object::New<MeshInstance3D>();
    auto mesh = MakeRef<ArrayMesh>();
    auto box = MakeRef<BoxMesh>();
    mesh->SetSurfaces(*box->GetSurfaceData());
    auto image = MakeRef<Image>(2, 2, false, ImageFormat::RGBA8);
    auto texture = MakeRef<Texture2D>(image);
    auto material = MakeRef<PBRMaterial3D>();
    material->SetAlbedoTexture(texture);
    node->SetMesh(mesh);
    node->SetMaterial(material);
    root.AddChild(node);
    Camera3D camera;
    camera.SetViewMatrix({0, -3, 1}, {0, 0, 0}, {0, 0, 1});
    RenderProductDesc desc;
    desc.width = 64;
    desc.height = 64;
    desc.device = RenderDevice::Cpu;
    RenderProduct product(desc);
    const auto capture = [&] {
        return product.Capture(CaptureRenderSceneSnapshot(&root), CaptureRenderViewSnapshot(camera));
    };
    ASSERT_NE(capture(), nullptr);
    const auto initial = renderer->GetSceneRendererStats().resources;
    ASSERT_EQ(initial.mesh_entries, 1);
    ASSERT_EQ(initial.texture_entries, 1);
    for (int frame = 0; frame < 10; ++frame) {
        camera.SetViewMatrix({0, -3, RealType(1 + frame * 0.01)}, {0, 0, 0}, {0, 0, 1});
        ASSERT_NE(capture(), nullptr);
    }
    const auto stationary = renderer->GetSceneRendererStats().resources;
    EXPECT_EQ(stationary.uploaded_bytes, initial.uploaded_bytes);
    for (int frame = 0; frame < 100; ++frame) {
        auto surfaces = mesh->GetSurfaces();
        surfaces[0].vertices[0].z() += 0.001;
        mesh->SetSurfaces(std::move(surfaces));
        ASSERT_NE(capture(), nullptr);
    }
    const auto deformed = renderer->GetSceneRendererStats().resources;
    EXPECT_EQ(deformed.geometry_uploads, initial.geometry_uploads + 100);
    EXPECT_EQ(deformed.index_uploads, initial.index_uploads);
    EXPECT_EQ(deformed.mesh_entries, 1);
    EXPECT_EQ(deformed.resident_bytes, initial.resident_bytes);
    texture->SetWrapU(TextureWrap::ClampToEdge);
    ASSERT_NE(capture(), nullptr);
    EXPECT_EQ(renderer->GetSceneRendererStats().resources.image_uploads, initial.image_uploads);
    image->SetPixel(0, 0, Color{1.0f, 0.0f, 0.0f, 1.0f});
    ASSERT_NE(capture(), nullptr);
    EXPECT_EQ(renderer->GetSceneRendererStats().resources.image_uploads, initial.image_uploads + 1);
    root.RemoveChild(node);
    Object::Delete(node);
    for (int frame = 0; frame < 4; ++frame) {
        ASSERT_NE(capture(), nullptr);
    }
    const auto cleared = renderer->GetSceneRendererStats().resources;
    EXPECT_EQ(cleared.mesh_entries, 0);
    EXPECT_EQ(cleared.texture_entries, 0);
    EXPECT_EQ(cleared.resident_bytes, 0);
}

} // namespace gobot
