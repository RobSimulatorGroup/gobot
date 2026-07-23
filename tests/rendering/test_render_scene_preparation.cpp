#include <gtest/gtest.h>

#include <gobot/rendering/render_scene_preparation.hpp>

namespace gobot {
namespace {

RenderViewSnapshot TestView() {
    RenderViewSnapshot view;
    view.camera.view = Matrix4::LookAt(
            Vector3{0.0, -5.0, 0.0}, Vector3::Zero(), Vector3::UnitZ());
    view.camera.projection = Matrix4::Perspective(60.0, 1.0, 0.1, 20.0);
    view.camera.view_projection = view.camera.projection * view.camera.view;
    view.camera.world_position = {0.0, -5.0, 0.0};
    view.camera.z_near = 0.1;
    view.camera.z_far = 20.0;
    return view;
}

VisualMeshRenderItem Item(std::uint32_t id,
                          const Vector3& center,
                          AlphaMode alpha_mode) {
    VisualMeshRenderItem item;
    item.instance_id = id;
    item.surface_index = id;
    item.world_bounds = AABB::FromMinMax(
            center - Vector3::Constant(0.25), center + Vector3::Constant(0.25));
    item.material.alpha_mode = alpha_mode;
    return item;
}

} // namespace

TEST(RenderScenePreparation, transforms_axis_aligned_bounds_conservatively) {
    const AABB local = AABB::FromMinMax({-1.0, -2.0, -3.0}, {1.0, 2.0, 3.0});
    Matrix4 transform = Matrix4::Identity();
    transform(0, 0) = 2.0;
    transform(1, 1) = 3.0;
    transform(2, 2) = 4.0;
    transform(0, 3) = 5.0;
    transform(1, 3) = -1.0;
    transform(2, 3) = 2.0;

    const AABB world = local.Transformed(transform);
    ASSERT_TRUE(world.IsValid());
    EXPECT_TRUE(world.GetMin().isApprox(Vector3{3.0, -7.0, -10.0}, CMP_EPSILON));
    EXPECT_TRUE(world.GetMax().isApprox(Vector3{7.0, 5.0, 14.0}, CMP_EPSILON));
}

TEST(RenderScenePreparation, culls_bounds_and_builds_deterministic_queues) {
    RenderSceneSnapshot scene;
    scene.visual_meshes = {
            Item(1, {0.0, -2.0, 0.0}, AlphaMode::Opaque),
            Item(2, {0.0, 2.0, 0.0}, AlphaMode::Opaque),
            Item(3, {0.0, -1.0, 0.0}, AlphaMode::Blend),
            Item(4, {0.0, 3.0, 0.0}, AlphaMode::Blend),
            Item(5, {100.0, 0.0, 0.0}, AlphaMode::Mask)};

    const RenderDrawLists lists = BuildRenderDrawLists(scene, TestView(), true);
    ASSERT_EQ(lists.visible_count, 4u);
    ASSERT_EQ(lists.culled_count, 1u);
    ASSERT_EQ(lists.opaque.size(), 2u);
    EXPECT_EQ(lists.opaque[0].item->instance_id, 1u);
    EXPECT_EQ(lists.opaque[1].item->instance_id, 2u);
    ASSERT_EQ(lists.transparent.size(), 2u);
    EXPECT_EQ(lists.transparent[0].item->instance_id, 4u);
    EXPECT_EQ(lists.transparent[1].item->instance_id, 3u);
    ASSERT_EQ(lists.shadow_casters.size(), 3u);
    EXPECT_NE(std::find_if(lists.shadow_casters.begin(),
                           lists.shadow_casters.end(),
                           [](const PreparedRenderItem& prepared) {
                               return prepared.item->instance_id == 5u;
                           }),
              lists.shadow_casters.end());
}

TEST(RenderScenePreparation, invalid_bounds_are_kept_visible) {
    RenderSceneSnapshot scene;
    VisualMeshRenderItem item;
    item.instance_id = 9;
    scene.visual_meshes.push_back(item);

    const RenderDrawLists lists = BuildRenderDrawLists(scene, TestView(), true);
    EXPECT_EQ(lists.visible_count, 1u);
    EXPECT_EQ(lists.culled_count, 0u);
    ASSERT_EQ(lists.opaque.size(), 1u);
    EXPECT_EQ(lists.opaque.front().item->instance_id, 9u);
}

} // namespace gobot
