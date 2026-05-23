#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <fstream>

#include <gobot/core/io/resource_format_mjcf.hpp>
#include <gobot/core/io/resource_format_urdf.hpp>
#include <gobot/scene/collision_shape_3d.hpp>
#include <gobot/scene/joint_3d.hpp>
#include <gobot/scene/link_3d.hpp>
#include <gobot/scene/mesh_instance_3d.hpp>
#include <gobot/scene/resources/capsule_shape_3d.hpp>
#include <gobot/scene/resources/cylinder_shape_3d.hpp>
#include <gobot/scene/resources/material.hpp>
#include <gobot/scene/resources/packed_scene.hpp>
#include <gobot/scene/robot_3d.hpp>

namespace {

gobot::Node* FindNodeByName(gobot::Node* node, const std::string& name) {
    if (node == nullptr) {
        return nullptr;
    }
    if (node->GetName() == name) {
        return node;
    }

    for (std::size_t i = 0; i < node->GetChildCount(); ++i) {
        if (auto* found = FindNodeByName(node->GetChild(static_cast<int>(i)), name)) {
            return found;
        }
    }
    return nullptr;
}

gobot::Affine3 ComputeTransformFromRoot(const gobot::Node3D* node) {
    if (node == nullptr) {
        return gobot::Affine3::Identity();
    }

    gobot::Affine3 transform = node->GetTransform();
    const gobot::Node* parent = node->GetParent();
    while (parent != nullptr) {
        if (auto* parent_3d = gobot::Object::PointerCastTo<gobot::Node3D>(parent)) {
            transform = parent_3d->GetTransform() * transform;
        }
        parent = parent->GetParent();
    }
    return transform;
}

} // namespace

TEST(TestResourceFormatMJCF, recognizes_mjcf_xml_for_packed_scene) {
    const std::filesystem::path fixture_path =
            std::filesystem::current_path() / "tests/fixtures/mjcf/simple_robot.xml";

    gobot::Ref<gobot::ResourceFormatLoaderMJCF> loader = gobot::MakeRef<gobot::ResourceFormatLoaderMJCF>();

    std::vector<std::string> extensions;
    loader->GetRecognizedExtensionsForType("PackedScene", &extensions);

    EXPECT_NE(std::find(extensions.begin(), extensions.end(), "mjcf"), extensions.end());
    EXPECT_NE(std::find(extensions.begin(), extensions.end(), "xml"), extensions.end());
    EXPECT_TRUE(loader->RecognizePath(fixture_path.string(), "PackedScene"));
    EXPECT_TRUE(loader->HandlesType("PackedScene"));

    gobot::Ref<gobot::ResourceFormatLoaderURDF> urdf_loader = gobot::MakeRef<gobot::ResourceFormatLoaderURDF>();
    EXPECT_FALSE(urdf_loader->RecognizePath(fixture_path.string(), "PackedScene"));
}

TEST(TestResourceFormatMJCF, imports_bodies_joints_and_collision_geoms_as_scene_tree) {
    if (!gobot::ResourceFormatLoaderMJCF::IsMuJoCoAvailable()) {
        GTEST_SKIP() << "MuJoCo support is not enabled.";
    }

    const std::filesystem::path fixture_path =
            std::filesystem::current_path() / "tests/fixtures/mjcf/simple_robot.xml";
    const std::string path = fixture_path.string();

    gobot::Ref<gobot::ResourceFormatLoaderMJCF> loader = gobot::MakeRef<gobot::ResourceFormatLoaderMJCF>();
    gobot::Ref<gobot::Resource> resource = loader->Load(path);
    ASSERT_TRUE(resource.IsValid());

    gobot::Ref<gobot::PackedScene> packed_scene = gobot::dynamic_pointer_cast<gobot::PackedScene>(resource);
    ASSERT_TRUE(packed_scene.IsValid());

    gobot::Node* root_node = packed_scene->Instantiate();
    ASSERT_NE(root_node, nullptr);

    auto* robot = gobot::Object::PointerCastTo<gobot::Robot3D>(root_node);
    ASSERT_NE(robot, nullptr);
    EXPECT_EQ(robot->GetName(), "mjcf_test_bot");
    EXPECT_EQ(robot->GetSourcePath(), path);

    auto* cart = gobot::Object::PointerCastTo<gobot::Link3D>(FindNodeByName(robot, "cart"));
    ASSERT_NE(cart, nullptr);
    EXPECT_TRUE(cart->GetPosition().isApprox(gobot::Vector3(0.0, 0.0, 0.5), CMP_EPSILON));

    auto* hinge = gobot::Object::PointerCastTo<gobot::Joint3D>(FindNodeByName(robot, "hinge"));
    ASSERT_NE(hinge, nullptr);
    EXPECT_EQ(hinge->GetJointType(), gobot::JointType::Revolute);
    EXPECT_EQ(hinge->GetParentLink(), "cart");
    EXPECT_EQ(hinge->GetChildLink(), "pole");
    EXPECT_TRUE(hinge->GetAxis().isApprox(gobot::Vector3::UnitY(), CMP_EPSILON));
    EXPECT_NEAR(hinge->GetLowerLimit(), -1.57, 1.0e-6);
    EXPECT_NEAR(hinge->GetUpperLimit(), 1.57, 1.0e-6);
    EXPECT_NEAR(hinge->GetEffortLimit(), 5.0, 1.0e-6);

    auto* pole = gobot::Object::PointerCastTo<gobot::Link3D>(FindNodeByName(robot, "pole"));
    ASSERT_NE(pole, nullptr);
    EXPECT_EQ(pole->GetParent(), hinge);

    auto* pole_collision =
            gobot::Object::PointerCastTo<gobot::CollisionShape3D>(FindNodeByName(robot, "pole_collision"));
    ASSERT_NE(pole_collision, nullptr);
    gobot::Ref<gobot::CylinderShape3D> cylinder =
            gobot::dynamic_pointer_cast<gobot::CylinderShape3D>(pole_collision->GetShape());
    ASSERT_TRUE(cylinder.IsValid());
    EXPECT_NEAR(cylinder->GetRadius(), 0.02, 1.0e-6);
    EXPECT_NEAR(cylinder->GetHeight(), 1.0, 1.0e-6);
    EXPECT_FALSE(pole_collision->IsVisible());

    EXPECT_EQ(FindNodeByName(robot, "hinge_motor"), nullptr);

    gobot::Object::Delete(root_node);
}

TEST(TestResourceFormatMJCF, imports_mesh_geoms_as_visuals) {
    if (!gobot::ResourceFormatLoaderMJCF::IsMuJoCoAvailable()) {
        GTEST_SKIP() << "MuJoCo support is not enabled.";
    }

    const std::filesystem::path fixture_path =
            std::filesystem::temp_directory_path() / "gobot_mjcf_mesh_visual.xml";
    const std::filesystem::path mesh_path =
            std::filesystem::current_path() / "tests/fixtures/mesh/tetrahedron.obj";
    {
        std::ofstream file(fixture_path);
        file << R"(<mujoco model="mesh_visual_bot">
  <asset>
    <mesh name="tetrahedron" file=")" << mesh_path.string() << R"("/>
  </asset>
  <worldbody>
    <body name="base">
      <geom name="tetrahedron_visual" type="mesh" mesh="tetrahedron" rgba="0.25 0.5 0.75 1" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>
)";
    }

    gobot::Ref<gobot::ResourceFormatLoaderMJCF> loader = gobot::MakeRef<gobot::ResourceFormatLoaderMJCF>();
    gobot::Ref<gobot::Resource> resource = loader->Load(fixture_path.string());
    ASSERT_TRUE(resource.IsValid());

    gobot::Ref<gobot::PackedScene> packed_scene = gobot::dynamic_pointer_cast<gobot::PackedScene>(resource);
    ASSERT_TRUE(packed_scene.IsValid());

    gobot::Node* root_node = packed_scene->Instantiate();
    ASSERT_NE(root_node, nullptr);

    auto* visual = gobot::Object::PointerCastTo<gobot::MeshInstance3D>(
            FindNodeByName(root_node, "tetrahedron_visual"));
    ASSERT_NE(visual, nullptr);
    ASSERT_TRUE(visual->GetMesh().IsValid());
    EXPECT_EQ(visual->GetMesh()->GetPath(), mesh_path.lexically_normal().string());
    EXPECT_FALSE(visual->GetMeshMaterial().IsValid());
    const gobot::Vector3 position = visual->GetPosition();
    EXPECT_NEAR(position.x(), 0.0, 1.0e-9);
    EXPECT_NEAR(position.y(), 0.0, 1.0e-9);
    EXPECT_NEAR(position.z(), 0.0, 1.0e-9);
    EXPECT_TRUE(visual->GetRotationMatrix().isApprox(gobot::Matrix3::Identity(), 1.0e-9));
    EXPECT_TRUE(visual->GetScale().isApprox(gobot::Vector3::Ones(), 1.0e-9));
    EXPECT_FLOAT_EQ(visual->GetSurfaceColor().red(), 0.25f);
    EXPECT_FLOAT_EQ(visual->GetSurfaceColor().green(), 0.5f);
    EXPECT_FLOAT_EQ(visual->GetSurfaceColor().blue(), 0.75f);

    gobot::Object::Delete(root_node);
}

TEST(TestResourceFormatMJCF, imports_stand_keyframe_and_joint_anchor_transforms) {
    if (!gobot::ResourceFormatLoaderMJCF::IsMuJoCoAvailable()) {
        GTEST_SKIP() << "MuJoCo support is not enabled.";
    }

    const std::filesystem::path fixture_path =
            std::filesystem::temp_directory_path() / "gobot_mjcf_stand_joint_anchor.xml";
    {
        std::ofstream file(fixture_path);
        file << R"(<mujoco model="anchored_bot">
  <compiler angle="radian"/>
  <worldbody>
    <body name="trunk" pos="0 0 0.445">
      <freejoint name="floating_base_joint"/>
      <geom name="trunk_collision" type="box" size="0.05 0.04 0.03"/>
      <body name="hip" pos="0.2 0.1 0">
        <joint name="hip_joint" type="hinge" axis="1 0 0" range="-1 1" limited="true"/>
        <geom name="hip_collision" type="sphere" size="0.01"/>
      </body>
    </body>
  </worldbody>
  <keyframe>
    <key name="stand" qpos="0 0 0.27 1 0 0 0 0.5"/>
  </keyframe>
</mujoco>
)";
    }

    gobot::Ref<gobot::ResourceFormatLoaderMJCF> loader = gobot::MakeRef<gobot::ResourceFormatLoaderMJCF>();
    gobot::Ref<gobot::PackedScene> packed_scene =
            gobot::dynamic_pointer_cast<gobot::PackedScene>(loader->Load(fixture_path.string()));
    ASSERT_TRUE(packed_scene.IsValid());

    gobot::Node* root_node = packed_scene->Instantiate();
    ASSERT_NE(root_node, nullptr);

    auto* floating_joint =
            gobot::Object::PointerCastTo<gobot::Joint3D>(FindNodeByName(root_node, "floating_base_joint"));
    ASSERT_NE(floating_joint, nullptr);
    EXPECT_EQ(floating_joint->GetJointType(), gobot::JointType::Floating);
    EXPECT_TRUE(ComputeTransformFromRoot(floating_joint).translation().isApprox(
            gobot::Vector3(0.0, 0.0, 0.27), 1.0e-9));

    auto* trunk = gobot::Object::PointerCastTo<gobot::Link3D>(FindNodeByName(root_node, "trunk"));
    ASSERT_NE(trunk, nullptr);
    EXPECT_EQ(trunk->GetParent(), floating_joint);
    EXPECT_TRUE(trunk->GetPosition().isApprox(gobot::Vector3::Zero(), 1.0e-9));
    EXPECT_TRUE(ComputeTransformFromRoot(trunk).translation().isApprox(
            gobot::Vector3(0.0, 0.0, 0.27), 1.0e-9));

    auto* hip_joint = gobot::Object::PointerCastTo<gobot::Joint3D>(FindNodeByName(root_node, "hip_joint"));
    ASSERT_NE(hip_joint, nullptr);
    EXPECT_EQ(hip_joint->GetParent(), trunk);
    EXPECT_TRUE(hip_joint->GetPosition().isApprox(gobot::Vector3(0.2, 0.1, 0.0), 1.0e-9));
    EXPECT_TRUE(ComputeTransformFromRoot(hip_joint).translation().isApprox(
            gobot::Vector3(0.2, 0.1, 0.27), 1.0e-9));
    EXPECT_NEAR(hip_joint->GetJointPosition(), 0.5, 1.0e-9);

    auto* hip = gobot::Object::PointerCastTo<gobot::Link3D>(FindNodeByName(root_node, "hip"));
    ASSERT_NE(hip, nullptr);
    EXPECT_EQ(hip->GetParent(), hip_joint);
    EXPECT_TRUE(hip->GetPosition().isApprox(gobot::Vector3::Zero(), 1.0e-9));

    auto* hip_collision =
            gobot::Object::PointerCastTo<gobot::CollisionShape3D>(FindNodeByName(root_node, "hip_collision"));
    ASSERT_NE(hip_collision, nullptr);
    EXPECT_FALSE(hip_collision->IsVisible());

    gobot::Object::Delete(root_node);
}

TEST(TestResourceFormatMJCF, imports_named_material_rgba_for_mesh_visuals) {
    if (!gobot::ResourceFormatLoaderMJCF::IsMuJoCoAvailable()) {
        GTEST_SKIP() << "MuJoCo support is not enabled.";
    }

    const std::filesystem::path fixture_path =
            std::filesystem::temp_directory_path() / "gobot_mjcf_mesh_material.xml";
    const std::filesystem::path mesh_path =
            std::filesystem::current_path() / "tests/fixtures/mesh/tetrahedron.obj";
    {
        std::ofstream file(fixture_path);
        file << R"(<mujoco model="mesh_material_bot">
  <asset>
    <material name="dark" rgba="0.2 0.3 0.4 1" metallic="0.1" roughness="0.7" specular="0.6"/>
    <mesh name="tetrahedron" file=")" << mesh_path.string() << R"("/>
  </asset>
  <worldbody>
    <body name="base">
      <geom name="tetrahedron_visual" type="mesh" mesh="tetrahedron" material="dark" rgba="0.9 0.9 0.9 1" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>
)";
    }

    gobot::Ref<gobot::ResourceFormatLoaderMJCF> loader = gobot::MakeRef<gobot::ResourceFormatLoaderMJCF>();
    gobot::Ref<gobot::PackedScene> packed_scene =
            gobot::dynamic_pointer_cast<gobot::PackedScene>(loader->Load(fixture_path.string()));
    ASSERT_TRUE(packed_scene.IsValid());

    gobot::Node* root_node = packed_scene->Instantiate();
    ASSERT_NE(root_node, nullptr);

    auto* visual = gobot::Object::PointerCastTo<gobot::MeshInstance3D>(
            FindNodeByName(root_node, "tetrahedron_visual"));
    ASSERT_NE(visual, nullptr);
    EXPECT_FALSE(visual->GetMeshMaterial().IsValid());
    EXPECT_FLOAT_EQ(visual->GetSurfaceColor().red(), 0.2f);
    EXPECT_FLOAT_EQ(visual->GetSurfaceColor().green(), 0.3f);
    EXPECT_FLOAT_EQ(visual->GetSurfaceColor().blue(), 0.4f);

    gobot::Ref<gobot::PBRMaterial3D> material =
            gobot::dynamic_pointer_cast<gobot::PBRMaterial3D>(visual->GetMaterial());
    ASSERT_TRUE(material.IsValid());
    EXPECT_FLOAT_EQ(material->GetAlbedo().red(), 0.2f);
    EXPECT_FLOAT_EQ(material->GetAlbedo().green(), 0.3f);
    EXPECT_FLOAT_EQ(material->GetAlbedo().blue(), 0.4f);
    EXPECT_FLOAT_EQ(material->GetMetallic(), 0.1f);
    EXPECT_FLOAT_EQ(material->GetRoughness(), 0.7f);
    EXPECT_FLOAT_EQ(material->GetSpecular(), 0.6f);

    gobot::Object::Delete(root_node);
}

TEST(TestResourceFormatMJCF, imports_capsule_contact_and_position_actuator_semantics) {
    if (!gobot::ResourceFormatLoaderMJCF::IsMuJoCoAvailable()) {
        GTEST_SKIP() << "MuJoCo support is not enabled.";
    }

    const std::filesystem::path fixture_path =
            std::filesystem::temp_directory_path() / "gobot_mjcf_core_semantics.xml";
    {
        std::ofstream file(fixture_path);
        file << R"(<mujoco model="semantic_bot">
  <compiler angle="radian"/>
  <worldbody>
    <body name="base" pos="0 0 0.2">
      <body name="leg" pos="0 0 0">
        <joint name="knee" type="hinge" axis="0 1 0" limited="true" range="-2 -0.5"/>
        <geom name="leg_capsule" type="capsule" fromto="0 0 0 0 0 -0.4" size="0.03"
              friction="0.8 0.02 0.003" contype="2" conaffinity="4" condim="4"
              solref="0.01 0.9" solimp="0.7 0.8 0.02 0.5 2" margin="0.001" gap="0.0002"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="knee_position" joint="knee" kp="40" ctrlrange="-2 -0.5" forcerange="-12 12"/>
  </actuator>
  <keyframe>
    <key name="stand" qpos="-0.888"/>
  </keyframe>
</mujoco>
)";
    }

    gobot::Ref<gobot::ResourceFormatLoaderMJCF> loader = gobot::MakeRef<gobot::ResourceFormatLoaderMJCF>();
    gobot::Ref<gobot::PackedScene> packed_scene =
            gobot::dynamic_pointer_cast<gobot::PackedScene>(loader->Load(fixture_path.string()));
    ASSERT_TRUE(packed_scene.IsValid());

    gobot::Node* root_node = packed_scene->Instantiate();
    ASSERT_NE(root_node, nullptr);

    auto* knee = gobot::Object::PointerCastTo<gobot::Joint3D>(FindNodeByName(root_node, "knee"));
    ASSERT_NE(knee, nullptr);
    EXPECT_NEAR(knee->GetJointPosition(), -0.888, 1.0e-6);
    EXPECT_NEAR(knee->GetInitialPosition(), -0.888, 1.0e-6);
    EXPECT_EQ(knee->GetDriveMode(), gobot::JointDriveMode::Position);
    EXPECT_NEAR(knee->GetDriveStiffness(), 40.0, 1.0e-9);
    EXPECT_NEAR(knee->GetControlLowerLimit(), -2.0, 1.0e-9);
    EXPECT_NEAR(knee->GetControlUpperLimit(), -0.5, 1.0e-9);
    EXPECT_NEAR(knee->GetForceLowerLimit(), -12.0, 1.0e-9);
    EXPECT_NEAR(knee->GetForceUpperLimit(), 12.0, 1.0e-9);

    auto* collision =
            gobot::Object::PointerCastTo<gobot::CollisionShape3D>(FindNodeByName(root_node, "leg_capsule"));
    ASSERT_NE(collision, nullptr);
    gobot::Ref<gobot::CapsuleShape3D> capsule =
            gobot::dynamic_pointer_cast<gobot::CapsuleShape3D>(collision->GetShape());
    ASSERT_TRUE(capsule.IsValid());
    EXPECT_NEAR(capsule->GetRadius(), 0.03, 1.0e-6);
    EXPECT_NEAR(capsule->GetHeight(), 0.4, 1.0e-6);
    EXPECT_TRUE(collision->GetFriction().isApprox(gobot::Vector3(0.8, 0.02, 0.003), 1.0e-9));
    EXPECT_EQ(collision->GetContactType(), 2);
    EXPECT_EQ(collision->GetContactAffinity(), 4);
    EXPECT_EQ(collision->GetContactDimension(), 4);
    EXPECT_TRUE(collision->GetSolref().isApprox(gobot::Vector2(0.01, 0.9), 1.0e-9));
    ASSERT_EQ(collision->GetSolimp().size(), 5);
    EXPECT_NEAR(collision->GetSolimp()[0], 0.7, 1.0e-6);
    EXPECT_NEAR(collision->GetSolimp()[1], 0.8, 1.0e-6);
    EXPECT_NEAR(collision->GetSolimp()[2], 0.02, 1.0e-9);
    EXPECT_NEAR(collision->GetSolimp()[3], 0.5, 1.0e-9);
    EXPECT_NEAR(collision->GetSolimp()[4], 2.0, 1.0e-9);
    EXPECT_NEAR(collision->GetMargin(), 0.001, 1.0e-9);
    EXPECT_NEAR(collision->GetGap(), 0.0002, 1.0e-9);

    gobot::Object::Delete(root_node);
}
