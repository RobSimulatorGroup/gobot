#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>

#include <gobot/core/io/resource_format_mjcf.hpp>
#include <gobot/core/io/resource_format_urdf.hpp>
#include <gobot/scene/collision_shape_3d.hpp>
#include <gobot/scene/joint_3d.hpp>
#include <gobot/scene/link_3d.hpp>
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
