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
#include <gobot/scene/resources/cylinder_shape_3d.hpp>
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
