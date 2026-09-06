#include <gtest/gtest.h>

#include "gobot/rendering/scene_debug_data.hpp"
#include "gobot/scene/camera_3d.hpp"
#include "gobot/scene/sensor_3d.hpp"
#include "gobot/scene/terrain_3d.hpp"
#include "gobot/simulation/physics_debug_preview.hpp"

namespace gobot {

TEST(PhysicsDebugPreview, StaticFramesAndCameraMovesReusePreview) {
    auto* root = Node::New<Node3D>();
    auto* sensor = Node::New<RayCastSensor3D>();
    sensor->SetPosition({0, 0, 1});
    root->AddChild(sensor);
    auto* camera = Node::New<Camera3D>();
    root->AddChild(camera);
    auto* terrain = Node::New<Terrain3D>();
    terrain->AddBox({0, 0, -0.05}, {2, 2, 0.1});
    root->AddChild(terrain);
    PhysicsDebugPreview preview;
    const auto* initial = preview.Update(root);
    ASSERT_NE(initial, nullptr);
    ASSERT_EQ(initial->loose_sensors.size(), 1);
    ASSERT_EQ(initial->loose_sensors[0].hits.size(), 1);
    EXPECT_TRUE(initial->loose_sensors[0].hits[0].hit);
    for (int i = 0; i < 100; ++i) {
        camera->SetPosition({RealType(i), 2, 3});
        EXPECT_EQ(preview.Update(root), initial);
    }
    EXPECT_EQ(preview.GetBuildCount(), 1);
    sensor->SetPosition({0, 0, 2});
    ASSERT_NE(preview.Update(root), nullptr);
    EXPECT_EQ(preview.GetBuildCount(), 2);
    terrain->SetPosition({0, 0, 0.5});
    ASSERT_NE(preview.Update(root), nullptr);
    EXPECT_EQ(preview.GetBuildCount(), 3);
    terrain->SetBoxes({});
    ASSERT_NE(preview.Update(root), nullptr);
    EXPECT_FALSE(preview.Update(root)->loose_sensors[0].hits[0].hit);
    EXPECT_EQ(preview.GetBuildCount(), 4);
    sensor->SetVisualizeDebug(false);
    EXPECT_EQ(preview.Update(root), nullptr);
    Node::Delete(root);
    EXPECT_EQ(preview.Update(nullptr), nullptr);
}

TEST(PhysicsDebugPreview, SceneWithoutVisibleRaySensorsNeverBuildsAWorld) {
    auto* root = Node::New<Node3D>();
    PhysicsDebugPreview preview;
    for (int i = 0; i < 20; ++i) {
        EXPECT_EQ(preview.Update(root), nullptr);
    }
    EXPECT_EQ(preview.GetBuildCount(), 0);
    Node::Delete(root);
}
} // namespace gobot
