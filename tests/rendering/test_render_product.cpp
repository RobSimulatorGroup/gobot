#include <gtest/gtest.h>

#include <gobot/rendering/render_product.hpp>
#include <gobot/rendering/render_server.hpp>
#include <gobot/scene/camera_3d.hpp>
#include <gobot/scene/node_3d.hpp>

#include <memory>

namespace gobot {

TEST(RenderProduct, exposes_requested_cpu_buffers_and_metadata) {
    auto render_server = std::make_unique<RenderServer>();
    auto* root = Object::New<Node3D>();
    root->SetName("World");
    Camera3D camera;
    camera.SetViewMatrix({2.0, -2.0, 1.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 1.0});

    RenderProductDesc desc;
    desc.width = 8;
    desc.height = 6;
    desc.outputs = {
            RenderOutputType::Rgb,
            RenderOutputType::LinearDepth,
            RenderOutputType::WorldNormal,
            RenderOutputType::InstanceId,
            RenderOutputType::SemanticId};
    desc.device = RenderDevice::Cpu;
    RenderProduct product(desc);

    const auto frame = product.Capture(CaptureRenderSceneSnapshot(root),
                                       CaptureRenderViewSnapshot(camera));
    ASSERT_NE(frame, nullptr);
    EXPECT_EQ(frame->GetOutputs().size(), 5);
    EXPECT_EQ(frame->Get(RenderOutputType::Rgb)->GetShape(),
              (std::vector<std::size_t>{6, 8, 3}));
    EXPECT_EQ(frame->Get(RenderOutputType::LinearDepth)->GetShape(),
              (std::vector<std::size_t>{6, 8}));
    EXPECT_EQ(frame->Get(RenderOutputType::WorldNormal)->GetDataType(), RenderDataType::Float32);
    EXPECT_EQ(frame->Get(RenderOutputType::InstanceId)->GetDataType(), RenderDataType::UInt32);
    EXPECT_EQ(frame->Get(RenderOutputType::Rgb)->GetByteSize(), 6u * 8u * 3u);

    Object::Delete(root);
}

TEST(RenderProduct, retained_buffers_hold_slots_and_pool_exhaustion_is_explicit) {
    auto render_server = std::make_unique<RenderServer>();
    auto* root = Object::New<Node3D>();
    Camera3D camera;

    RenderProductDesc desc;
    desc.width = 4;
    desc.height = 3;
    desc.outputs = {RenderOutputType::LinearDepth};
    desc.device = RenderDevice::Cpu;
    desc.frame_slots = 2;
    RenderProduct product(desc);
    const RenderSceneSnapshot scene = CaptureRenderSceneSnapshot(root);
    const RenderViewSnapshot view = CaptureRenderViewSnapshot(camera);

    auto first_frame = product.Capture(scene, view);
    auto retained_first = first_frame->Get(RenderOutputType::LinearDepth);
    first_frame.reset();
    auto second_frame = product.Capture(scene, view);
    EXPECT_THROW(product.Capture(scene, view), std::runtime_error);

    second_frame.reset();
    EXPECT_NO_THROW(product.Capture(scene, view));
    retained_first.reset();

    Object::Delete(root);
}

TEST(RenderProduct, explicit_cuda_never_silently_falls_back) {
    auto render_server = std::make_unique<RenderServer>();
    auto* root = Object::New<Node3D>();
    Camera3D camera;

    RenderProductDesc desc;
    desc.width = 4;
    desc.height = 3;
    desc.device = RenderDevice::Cuda;
    RenderProduct product(desc);
    EXPECT_THROW(product.Capture(CaptureRenderSceneSnapshot(root),
                                 CaptureRenderViewSnapshot(camera)),
                 std::runtime_error);

    Object::Delete(root);
}

} // namespace gobot
