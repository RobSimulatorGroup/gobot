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

#include "gobot/rendering/scene_debug_data.hpp"
#include "gobot/simulation/simulation_contact_state.hpp"

TEST(PhysicsContactPresentation, ForceToggleIsIndependentAndPausedStateIsReusable) {
    gobot::PhysicsContactState contact;
    contact.robot_name = "robot";
    contact.link_name = "foot";
    gobot::SceneDebugData data;
    data.contacts.push_back(contact);
    data.settings.debug_draw_contact_forces = true;
    auto visibility = gobot::GetContactDebugVisibility(contact, data);
    EXPECT_FALSE(visibility.points_and_normals);
    EXPECT_TRUE(visibility.forces);
    data.settings.debug_draw_contacts = true;
    data.settings.debug_draw_contact_forces = false;
    visibility = gobot::GetContactDebugVisibility(contact, data);
    EXPECT_TRUE(visibility.points_and_normals);
    EXPECT_FALSE(visibility.forces);
    EXPECT_EQ(data.contacts.size(), 1); // toggles do not mutate or step the physics frame
}

TEST(PhysicsContactPresentation, RigidDisplayTransformPreservesForceUnitsAndSnapshot) {
    gobot::SimulationSnapshot snapshot;
    snapshot.environments = {0};
    gobot::SimulationContact contact;
    contact.body = {"robot", "foot", "sole"};
    contact.position = {1, 2, 3}; contact.normal = {0, 0, 1}; contact.force = {3, 4, 100};
    snapshot.contacts.push_back(contact);
    gobot::Affine3 transform = gobot::Affine3::Identity();
    transform.translation() = gobot::Vector3(-2, -3, 0);
    transform.linear() = Eigen::AngleAxis<gobot::RealType>(gobot::RealType(1.57079632679), gobot::Vector3::UnitZ()).toRotationMatrix();
    auto frame = gobot::CaptureContactStateFrame(snapshot, {{0, transform}});
    ASSERT_EQ(frame->state.contacts.size(), 1);
    EXPECT_TRUE(frame->state.contacts[0].force.isApprox(gobot::Vector3(-4, 3, 100), 1e-5));
    EXPECT_TRUE(frame->state.contacts[0].position.isApprox(gobot::Vector3(-4, -2, 3), 1e-5));
    EXPECT_EQ(snapshot.contacts[0].position, contact.position);
    snapshot.contacts.clear();
    EXPECT_TRUE(gobot::CaptureContactStateFrame(snapshot)->state.contacts.empty());
    EXPECT_EQ(frame->state.contacts.size(), 1);
    transform.linear() *= 2;
    EXPECT_THROW(gobot::CaptureContactStateFrame(snapshot, {{0, transform}}), std::invalid_argument);
}
