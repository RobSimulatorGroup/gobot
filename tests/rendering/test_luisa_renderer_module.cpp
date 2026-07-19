#include "gobot/rendering/luisa_renderer_module_api.hpp"
#include "gobot/rendering/headless_render_context.hpp"
#include "gobot/rendering/render_server.hpp"
#include "gobot/scene/camera_3d.hpp"
#include "gobot/scene/mesh_instance_3d.hpp"
#include "gobot/scene/resources/material.hpp"
#include "gobot/scene/resources/primitive_mesh.hpp"
#include "gobot/scene/scene_tree.hpp"
#include "gobot/scene/window.hpp"

#include <algorithm>
#include <cstdlib>
#include <dlfcn.h>
#include <memory>
#include <string>

#include <gtest/gtest.h>

namespace gobot {

TEST(LuisaRendererModule, ExportsMatchingBackendNeutralAbi) {
    void* library = dlopen(GOBOT_TEST_LUISA_MODULE_PATH, RTLD_NOW | RTLD_LOCAL);
    ASSERT_NE(library, nullptr) << dlerror();

    auto get_api = reinterpret_cast<GetLuisaRendererModuleApi>(
            dlsym(library, "gobot_luisa_renderer_get_api"));
    ASSERT_NE(get_api, nullptr) << dlerror();

    const LuisaRendererModuleApi* api = get_api();
    ASSERT_NE(api, nullptr);
    EXPECT_EQ(api->abi_version, GOBOT_LUISA_RENDERER_ABI_VERSION);
    ASSERT_NE(api->create, nullptr);
    ASSERT_NE(api->destroy, nullptr);
    ASSERT_NE(api->capabilities, nullptr);
    ASSERT_NE(api->render, nullptr);
    ASSERT_NE(api->reset_accumulation, nullptr);

    const SceneRendererCapabilities capabilities = api->capabilities(nullptr);
    EXPECT_TRUE(capabilities.ray_tracing_available);
    EXPECT_TRUE(capabilities.realtime);
    EXPECT_TRUE(capabilities.progressive);
    EXPECT_TRUE(capabilities.direct_presentation_interop);
    EXPECT_EQ(capabilities.backend_name, "LuisaCompute CUDA");

    EXPECT_EQ(dlclose(library), 0);
}

TEST(LuisaRendererGpu, RendersProgressiveFramesAndResetsAccumulation) {
    const char* run_gpu_test = std::getenv("GOBOT_RUN_LUISA_GPU_TEST");
    if (run_gpu_test == nullptr || std::string{run_gpu_test} != "1") {
        GTEST_SKIP() << "Set GOBOT_RUN_LUISA_GPU_TEST=1 on a CUDA/OpenGL runner.";
    }

    if (std::getenv("GOBOT_LUISA_RENDERER_LIBRARY") == nullptr) {
        ASSERT_EQ(setenv("GOBOT_LUISA_RENDERER_LIBRARY", GOBOT_TEST_LUISA_MODULE_PATH, 0), 0);
    }

    HeadlessRenderContext context;
    ASSERT_TRUE(context.Initialize()) << context.GetLastError();
    RenderServer* render_server = RenderServer::GetInstance();
    ASSERT_NE(render_server, nullptr);

    auto delete_tree = [](SceneTree* tree) {
        if (tree != nullptr) {
            Object::Delete(tree);
        }
    };
    std::unique_ptr<SceneTree, decltype(delete_tree)> tree(
            Object::New<SceneTree>(false), delete_tree);
    tree->Initialize();

    auto material = MakeRef<PBRMaterial3D>();
    material->SetAlbedo({0.15f, 0.55f, 0.9f, 1.0f});
    material->SetRoughness(0.35);
    material->SetEmissive({0.08f, 0.02f, 0.01f, 1.0f});
    auto* mesh = Object::New<MeshInstance3D>();
    mesh->SetMesh(MakeRef<BoxMesh>());
    mesh->SetMaterial(material);
    tree->GetRoot()->AddChild(mesh, true);

    constexpr int width = 160;
    constexpr int height = 120;
    Camera3D camera;
    camera.SetAspect(static_cast<RealType>(width) / static_cast<RealType>(height));
    camera.SetPerspective(55.0, 0.05, 100.0);
    camera.SetViewMatrix({2.4, -3.0, 1.8}, {0.0, 0.0, 0.0}, {0.0, 0.0, 1.0});

    const RID viewport = render_server->ViewportCreate();
    render_server->ViewportSetSize(viewport, width, height);

    SceneRendererSettings settings;
    settings.mode = SceneRendererMode::ProgressivePathTracing;
    settings.samples_per_frame = 1;
    settings.max_accumulated_samples = 16;
    settings.max_bounces = 2;
    settings.denoise = false;
    settings.adaptive_quality = false;
    render_server->SetSceneRendererSettings(settings);

    const SceneRendererCapabilities capabilities =
            render_server->GetSceneRendererCapabilities();
    ASSERT_TRUE(capabilities.ray_tracing_available) << capabilities.status;
    EXPECT_EQ(capabilities.backend_name, "LuisaCompute CUDA");

    for (std::uint64_t expected_samples = 1; expected_samples <= 3; ++expected_samples) {
        render_server->RenderSceneToViewport(viewport, tree->GetRoot(), &camera);
        const SceneRendererStats stats = render_server->GetSceneRendererStats();
        ASSERT_EQ(stats.active_mode, SceneRendererMode::ProgressivePathTracing) << stats.status;
        EXPECT_EQ(stats.accumulated_samples, expected_samples);
    }

    const std::vector<std::uint8_t> pixels =
            render_server->ReadViewportRgbPixels(viewport, true);
    ASSERT_EQ(pixels.size(), static_cast<std::size_t>(width * height * 3));
    const auto [minimum, maximum] = std::minmax_element(pixels.begin(), pixels.end());
    ASSERT_NE(minimum, pixels.end());
    EXPECT_GT(*maximum, 16u);
    EXPECT_GT(static_cast<int>(*maximum) - static_cast<int>(*minimum), 8);

    camera.SetViewMatrix({2.8, -3.0, 2.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 1.0});
    render_server->RenderSceneToViewport(viewport, tree->GetRoot(), &camera);
    const SceneRendererStats reset_stats = render_server->GetSceneRendererStats();
    ASSERT_EQ(reset_stats.active_mode, SceneRendererMode::ProgressivePathTracing)
            << reset_stats.status;
    EXPECT_EQ(reset_stats.accumulated_samples, 1u);

    std::unique_ptr<SceneTree, decltype(delete_tree)> empty_tree(
            Object::New<SceneTree>(false), delete_tree);
    empty_tree->Initialize();
    render_server->RenderSceneToViewport(viewport, empty_tree->GetRoot(), &camera);
    const SceneRendererStats fallback_stats = render_server->GetSceneRendererStats();
    EXPECT_EQ(fallback_stats.active_mode, SceneRendererMode::Raster);
    EXPECT_NE(fallback_stats.status.find("Scene has no renderable mesh"), std::string::npos)
            << fallback_stats.status;

    render_server->Free(viewport);
}

} // namespace gobot
