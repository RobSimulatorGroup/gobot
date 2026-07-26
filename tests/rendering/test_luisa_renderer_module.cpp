#include "gobot/rendering/luisa_renderer_module_api.hpp"
#include "gobot/rendering/headless_render_context.hpp"
#include "gobot/rendering/render_product.hpp"
#include "gobot/rendering/render_server.hpp"
#include "gobot/scene/camera_3d.hpp"
#include "gobot/scene/environment_3d.hpp"
#include "gobot/scene/gaussian_splat_3d.hpp"
#include "gobot/scene/mesh_instance_3d.hpp"
#include "gobot/scene/resources/gaussian_splat.hpp"
#include "gobot/scene/resources/material.hpp"
#include "gobot/scene/resources/primitive_mesh.hpp"
#include "gobot/scene/node_3d.hpp"
#include "gobot/scene/scene_tree.hpp"
#include "gobot/scene/window.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
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
    ASSERT_NE(api->render_gaussian_background, nullptr);
    ASSERT_NE(api->reset_accumulation, nullptr);
    ASSERT_NE(api->capture_render_product, nullptr);
    ASSERT_NE(api->release_render_product, nullptr);
    ASSERT_NE(api->readback_render_product, nullptr);

    const SceneRendererCapabilities capabilities = api->capabilities(nullptr);
    EXPECT_TRUE(capabilities.ray_tracing_available);
    EXPECT_TRUE(capabilities.realtime);
    EXPECT_TRUE(capabilities.progressive);
    EXPECT_TRUE(capabilities.direct_presentation_interop);
    EXPECT_TRUE(capabilities.cuda_render_products);
    EXPECT_EQ(capabilities.gaussian_splatting,
              static_cast<bool>(GOBOT_TEST_HAS_GSPLAT_INFERENCE));
    EXPECT_EQ(capabilities.gaussian_splat_cuda,
              static_cast<bool>(GOBOT_TEST_HAS_GSPLAT_INFERENCE));
    EXPECT_EQ(capabilities.backend_name, "LuisaCompute CUDA");

    EXPECT_EQ(dlclose(library), 0);
}

TEST(LuisaRendererModule, RejectsMismatchedAbiAndUsesCpuRenderProduct) {
    ASSERT_EQ(setenv("GOBOT_LUISA_RENDERER_LIBRARY",
                     GOBOT_TEST_LUISA_BAD_ABI_MODULE_PATH,
                     1),
              0);
    HeadlessRenderContext context;
    ASSERT_TRUE(context.Initialize()) << context.GetLastError();
    RenderServer* render_server = RenderServer::GetInstance();
    ASSERT_NE(render_server, nullptr);

    const SceneRendererCapabilities capabilities =
            render_server->GetSceneRendererCapabilities();
    EXPECT_FALSE(capabilities.ray_tracing_available);
    EXPECT_FALSE(capabilities.cuda_render_products);
    EXPECT_NE(capabilities.status.find("ABI"), std::string::npos)
            << capabilities.status;

    auto* root = Object::New<Node3D>();
    auto* mesh = Object::New<MeshInstance3D>();
    mesh->SetName("box");
    mesh->SetMesh(MakeRef<BoxMesh>());
    root->AddChild(mesh);
    Camera3D camera;
    camera.SetViewMatrix({0.0, -3.0, 0.0},
                         {0.0, 0.0, 0.0},
                         {0.0, 0.0, 1.0});

    RenderProductDesc desc;
    desc.width = 16;
    desc.height = 12;
    desc.device = RenderDevice::Auto;
    RenderProduct product(desc);
    const std::shared_ptr<RenderFrame> frame = product.Capture(
            CaptureRenderSceneSnapshot(root), CaptureRenderViewSnapshot(camera));
    ASSERT_NE(frame, nullptr);
    EXPECT_EQ(product.GetDevice(), RenderDevice::Cpu);
    EXPECT_NE(frame->Get(RenderOutputType::Rgb), nullptr);
    Object::Delete(root);
}

TEST(LuisaRendererGpu, RendersCudaRenderProductAovs) {
    const char* run_gpu_test = std::getenv("GOBOT_RUN_LUISA_GPU_TEST");
    if (run_gpu_test == nullptr || std::string{run_gpu_test} != "1") {
        GTEST_SKIP() << "Set GOBOT_RUN_LUISA_GPU_TEST=1 on a CUDA runner.";
    }
    ASSERT_EQ(setenv("GOBOT_LUISA_RENDERER_LIBRARY", GOBOT_TEST_LUISA_MODULE_PATH, 1), 0);

    HeadlessRenderContext context;
    ASSERT_TRUE(context.Initialize()) << context.GetLastError();

    auto delete_node = [](Node3D* node) {
        if (node != nullptr) {
            Object::Delete(node);
        }
    };
    std::unique_ptr<Node3D, decltype(delete_node)> root(Object::New<Node3D>(), delete_node);
    root->SetName("World");
    root->SetSemanticLabel("crate");
    auto* mesh = Object::New<MeshInstance3D>();
    mesh->SetName("box");
    mesh->SetMesh(MakeRef<BoxMesh>());
    root->AddChild(mesh);

    constexpr int width = 65;
    constexpr int height = 49;
    Camera3D camera;
    camera.SetPerspective(55.0, 0.05, 20.0);
    camera.SetViewMatrix({0.0, -3.0, 0.0},
                         {0.0, 0.0, 0.0},
                         {0.0, 0.0, 1.0});
    RenderViewSnapshot view = CaptureRenderViewSnapshot(camera);
    view.camera.projection = Matrix4f::Perspective(
            55.0, static_cast<RealType>(width) / height, 0.05, 20.0);
    view.camera.view_projection = view.camera.projection * view.camera.view;
    const RenderSceneSnapshot scene = CaptureRenderSceneSnapshot(root.get());

    RenderProductDesc desc;
    desc.width = width;
    desc.height = height;
    desc.device = RenderDevice::Cuda;
    desc.mode = RenderProductMode::Minimal;
    desc.outputs = {RenderOutputType::Rgb,
                    RenderOutputType::LinearDepth,
                    RenderOutputType::WorldNormal,
                    RenderOutputType::InstanceId,
                    RenderOutputType::SemanticId};
    RenderProduct product(desc);
    const std::shared_ptr<RenderFrame> frame = product.Capture(scene, view);
    ASSERT_NE(frame, nullptr);
    EXPECT_EQ(product.GetDevice(), RenderDevice::Cuda);

    std::vector<std::uint8_t> rgb(static_cast<std::size_t>(width) * height * 3u);
    std::vector<float> depth(static_cast<std::size_t>(width) * height);
    std::vector<float> normal(static_cast<std::size_t>(width) * height * 3u);
    std::vector<std::uint32_t> instance(static_cast<std::size_t>(width) * height);
    std::vector<std::uint32_t> semantic(static_cast<std::size_t>(width) * height);
    EXPECT_TRUE(frame->Get(RenderOutputType::Rgb)->CopyToHost(rgb.data(), rgb.size()));
    EXPECT_TRUE(frame->Get(RenderOutputType::LinearDepth)
                        ->CopyToHost(depth.data(), depth.size() * sizeof(float)));
    EXPECT_TRUE(frame->Get(RenderOutputType::WorldNormal)
                        ->CopyToHost(normal.data(), normal.size() * sizeof(float)));
    EXPECT_TRUE(frame->Get(RenderOutputType::InstanceId)
                        ->CopyToHost(instance.data(), instance.size() * sizeof(std::uint32_t)));
    EXPECT_TRUE(frame->Get(RenderOutputType::SemanticId)
                        ->CopyToHost(semantic.data(), semantic.size() * sizeof(std::uint32_t)));

    const std::size_t center = static_cast<std::size_t>(height / 2) * width + width / 2;
    EXPECT_NEAR(depth[center], 2.5f, 1.0e-3f);
    EXPECT_NEAR(normal[center * 3u], 0.0f, 1.0e-4f);
    EXPECT_NEAR(normal[center * 3u + 1u], -1.0f, 1.0e-4f);
    EXPECT_NEAR(normal[center * 3u + 2u], 0.0f, 1.0e-4f);
    EXPECT_NE(instance[center], 0u);
    EXPECT_NE(semantic[center], 0u);
    EXPECT_EQ(frame->GetInstancePaths().at(instance[center]), "box");
    EXPECT_EQ(frame->GetSemanticLabels().at(semantic[center]), "crate");

    EXPECT_TRUE(std::isinf(depth.front()));
    EXPECT_EQ(instance.front(), 0u);
    EXPECT_EQ(semantic.front(), 0u);
    EXPECT_FLOAT_EQ(normal[0], 0.0f);
    EXPECT_FLOAT_EQ(normal[1], 0.0f);
    EXPECT_FLOAT_EQ(normal[2], 0.0f);
}

TEST(LuisaRendererGpu, CompositesGaussianBackgroundBehindProxyAovs) {
#if !GOBOT_TEST_HAS_GSPLAT_INFERENCE
    GTEST_SKIP() << "The Luisa renderer was built without gsplat inference support.";
#endif
    const char* run_gpu_test = std::getenv("GOBOT_RUN_LUISA_GPU_TEST");
    if (run_gpu_test == nullptr || std::string{run_gpu_test} != "1") {
        GTEST_SKIP() << "Set GOBOT_RUN_LUISA_GPU_TEST=1 on a CUDA runner.";
    }
    ASSERT_EQ(setenv("GOBOT_LUISA_RENDERER_LIBRARY", GOBOT_TEST_LUISA_MODULE_PATH, 1), 0);

    HeadlessRenderContext context;
    ASSERT_TRUE(context.Initialize()) << context.GetLastError();
    const std::filesystem::path ply_path = "/tmp/gobot-gsplat-gpu-test.ply";
    {
        std::ofstream stream(ply_path);
        stream << "ply\nformat ascii 1.0\nelement vertex 2\n"
                  "property float x\nproperty float y\nproperty float z\n"
                  "property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n"
                  "property float opacity\n"
                  "property float scale_0\nproperty float scale_1\nproperty float scale_2\n"
                  "property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n"
                  "end_header\n"
                  "0 0 0 1.75 -1.75 -1.75 10 -0.693147 -0.693147 -0.693147 1 0 0 0\n"
                  "10 -2.9 0 -1.75 1.75 -1.75 10 -0.693147 -0.693147 -0.693147 1 0 0 0\n";
    }
    Ref<GaussianSplatResource> resource = MakeRef<GaussianSplatResource>();
    std::string load_error;
    ASSERT_TRUE(resource->LoadPly(ply_path.string(), &load_error)) << load_error;

    auto delete_node = [](Node3D* node) {
        if (node != nullptr) Object::Delete(node);
    };
    std::unique_ptr<Node3D, decltype(delete_node)> root(Object::New<Node3D>(), delete_node);
    auto* environment = Object::New<Environment3D>();
    environment->SetClearColor({0.0f, 0.0f, 0.0f, 1.0f});
    environment->SetSkyColor({0.0f, 0.0f, 0.0f, 1.0f});
    environment->SetGroundColor({0.0f, 0.0f, 0.0f, 1.0f});
    root->AddChild(environment);
    auto* gaussian = Object::New<GaussianSplat3D>();
    gaussian->SetName("environment");
    gaussian->SetSplat(resource);
    root->AddChild(gaussian);
    auto* proxy = Object::New<MeshInstance3D>();
    proxy->SetName("proxy");
    proxy->SetMesh(MakeRef<BoxMesh>());
    proxy->SetVisibleInRgb(false);
    proxy->SetCastShadow(false);
    auto proxy_material = MakeRef<PBRMaterial3D>();
    proxy_material->SetAlphaMode(AlphaMode::Blend);
    proxy_material->SetAlbedo({0.2f, 0.8f, 0.2f, 0.2f});
    proxy_material->SetEmissive({0.0f, 10.0f, 0.0f, 1.0f});
    proxy->SetMaterial(proxy_material);
    root->AddChild(proxy);

    constexpr int width = 64;
    constexpr int height = 64;
    Camera3D camera;
    camera.SetPerspective(55.0, 0.05, 20.0);
    camera.SetViewMatrix({0.0, -3.0, 0.0},
                         {0.0, 0.0, 0.0},
                         {0.0, 0.0, 1.0});
    RenderViewSnapshot view = CaptureRenderViewSnapshot(camera);
    view.camera.projection = Matrix4f::Perspective(55.0, 1.0, 0.05, 20.0);
    view.camera.view_projection = view.camera.projection * view.camera.view;

    RenderProductDesc desc;
    desc.width = width;
    desc.height = height;
    desc.device = RenderDevice::Cuda;
    desc.mode = RenderProductMode::Minimal;
    desc.outputs = {RenderOutputType::Rgb,
                    RenderOutputType::LinearDepth,
                    RenderOutputType::InstanceId};
    RenderProduct product(desc);
    const std::shared_ptr<RenderFrame> frame = product.Capture(
            CaptureRenderSceneSnapshot(root.get()), view);
    ASSERT_NE(frame, nullptr);

    std::vector<std::uint8_t> rgb(static_cast<std::size_t>(width) * height * 3u);
    std::vector<float> depth(static_cast<std::size_t>(width) * height);
    std::vector<std::uint32_t> instance(static_cast<std::size_t>(width) * height);
    ASSERT_TRUE(frame->Get(RenderOutputType::Rgb)->CopyToHost(rgb.data(), rgb.size()));
    ASSERT_TRUE(frame->Get(RenderOutputType::LinearDepth)
                        ->CopyToHost(depth.data(), depth.size() * sizeof(float)));
    ASSERT_TRUE(frame->Get(RenderOutputType::InstanceId)
                        ->CopyToHost(instance.data(), instance.size() * sizeof(std::uint32_t)));
    const std::size_t center = static_cast<std::size_t>(height / 2) * width + width / 2;
    EXPECT_GT(rgb[center * 3u], rgb[center * 3u + 1u] + 40u);
    EXPECT_GT(rgb[center * 3u], rgb[center * 3u + 2u] + 40u);
    EXPECT_NEAR(depth[center], 2.5f, 1.0e-3f);
    EXPECT_NE(instance[center], 0u);

    RenderProductDesc cpu_desc = desc;
    cpu_desc.device = RenderDevice::Cpu;
    RenderProduct cpu_product(cpu_desc);
    const std::shared_ptr<RenderFrame> cpu_frame = cpu_product.Capture(
            CaptureRenderSceneSnapshot(root.get()), view);
    ASSERT_NE(cpu_frame, nullptr);
    EXPECT_EQ(cpu_product.GetDevice(), RenderDevice::Cpu);
    std::vector<std::uint8_t> cpu_rgb(static_cast<std::size_t>(width) * height * 3u);
    ASSERT_TRUE(cpu_frame->Get(RenderOutputType::Rgb)->CopyToHost(
            cpu_rgb.data(), cpu_rgb.size()));
    EXPECT_EQ(cpu_rgb[center * 3u], rgb[center * 3u]);
    EXPECT_EQ(cpu_rgb[center * 3u + 1u], rgb[center * 3u + 1u]);
    EXPECT_EQ(cpu_rgb[center * 3u + 2u], rgb[center * 3u + 2u]);

    RenderServer* render_server = RenderServer::GetInstance();
    ASSERT_NE(render_server, nullptr);
    const RID viewport = render_server->ViewportCreate();
    render_server->ViewportSetSize(viewport, width, height);
    SceneRendererSettings settings;
    settings.mode = SceneRendererMode::Raster;
    settings.raster.anti_aliasing = RasterAntiAliasingMode::Disabled;
    render_server->SetSceneRendererSettings(settings);
    camera.SetAspect(1.0);
    render_server->RenderSceneToViewport(viewport, root.get(), &camera);
    const std::vector<std::uint8_t> viewport_rgb =
            render_server->ReadViewportRgbPixels(viewport, true);
    ASSERT_EQ(viewport_rgb.size(), rgb.size());
    EXPECT_GT(viewport_rgb[center * 3u], viewport_rgb[center * 3u + 1u] + 40u);
    EXPECT_GT(viewport_rgb[center * 3u], viewport_rgb[center * 3u + 2u] + 40u);
    const SceneRendererStats stats = render_server->GetSceneRendererStats();
    EXPECT_EQ(stats.active_mode, SceneRendererMode::Raster) << stats.status;
    EXPECT_EQ(stats.gaussian_count, 2u) << stats.status;
    EXPECT_GT(stats.gaussian_splat_ms, 0.0) << stats.status;

    settings.mode = SceneRendererMode::RealtimeRayTracing;
    settings.denoise = false;
    render_server->SetSceneRendererSettings(settings);
    render_server->RenderSceneToViewport(viewport, root.get(), &camera);
    const std::vector<std::uint8_t> ray_traced_rgb =
            render_server->ReadViewportRgbPixels(viewport, true);
    ASSERT_EQ(ray_traced_rgb.size(), rgb.size());
    EXPECT_LT(ray_traced_rgb[center * 3u], 10u);
    EXPECT_LT(ray_traced_rgb[center * 3u + 1u], 10u);
    EXPECT_LT(ray_traced_rgb[center * 3u + 2u], 10u);
    render_server->Free(viewport);
    std::filesystem::remove(ply_path);
}

TEST(LuisaRendererGpu, RendersProgressiveFramesAndResetsAccumulation) {
    const char* run_gpu_test = std::getenv("GOBOT_RUN_LUISA_GPU_TEST");
    if (run_gpu_test == nullptr || std::string{run_gpu_test} != "1") {
        GTEST_SKIP() << "Set GOBOT_RUN_LUISA_GPU_TEST=1 on a CUDA/OpenGL runner.";
    }

    ASSERT_EQ(setenv("GOBOT_LUISA_RENDERER_LIBRARY", GOBOT_TEST_LUISA_MODULE_PATH, 1), 0);

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

    constexpr int resized_width = 96;
    constexpr int resized_height = 72;
    render_server->ViewportSetSize(viewport, resized_width, resized_height);
    camera.SetAspect(static_cast<RealType>(resized_width) /
                     static_cast<RealType>(resized_height));
    render_server->RenderSceneToViewport(viewport, tree->GetRoot(), &camera);
    const SceneRendererStats resized_stats = render_server->GetSceneRendererStats();
    ASSERT_EQ(resized_stats.active_mode, SceneRendererMode::ProgressivePathTracing)
            << resized_stats.status;
    EXPECT_EQ(resized_stats.accumulated_samples, 1u);
    const std::vector<std::uint8_t> resized_pixels =
            render_server->ReadViewportRgbPixels(viewport, true);
    ASSERT_EQ(resized_pixels.size(),
              static_cast<std::size_t>(resized_width * resized_height * 3));

    render_server->RenderSceneToViewport(viewport, tree->GetRoot(), &camera);
    const SceneRendererStats consecutive_stats = render_server->GetSceneRendererStats();
    ASSERT_EQ(consecutive_stats.active_mode, SceneRendererMode::ProgressivePathTracing)
            << consecutive_stats.status;
    EXPECT_EQ(consecutive_stats.accumulated_samples, 2u);

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
