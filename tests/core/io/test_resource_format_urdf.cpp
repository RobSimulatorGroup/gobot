#include <gtest/gtest.h>

#include <algorithm>
#include <fstream>

#include <gobot/core/io/resource_format_urdf.hpp>
#include <gobot/scene/joint_3d.hpp>
#include <gobot/scene/link_3d.hpp>
#include <gobot/scene/resources/packed_scene.hpp>
#include <gobot/scene/robot_3d.hpp>

TEST(TestResourceFormatURDF, recognizes_urdf_extension_for_packed_scene) {
    gobot::Ref<gobot::ResourceFormatLoaderURDF> loader = gobot::MakeRef<gobot::ResourceFormatLoaderURDF>();

    std::vector<std::string> extensions;
    loader->GetRecognizedExtensionsForType("PackedScene", &extensions);

    EXPECT_NE(std::find(extensions.begin(), extensions.end(), "urdf"), extensions.end());
    EXPECT_TRUE(loader->HandlesType("PackedScene"));
}

TEST(TestResourceFormatURDF, imports_robot_links_joints_and_inertial_metadata) {
    const std::string path = "/tmp/gobot_test_robot.urdf";
    {
        std::ofstream file(path);
        file << R"(
<robot name="test_bot">
  <link name="base_link">
    <inertial>
      <origin xyz="0.1 0.2 0.3" rpy="0 0 0"/>
      <mass value="12.5"/>
      <inertia ixx="1.0" ixy="0.01" ixz="0.02" iyy="2.0" iyz="0.03" izz="3.0"/>
    </inertial>
  </link>
  <link name="upper_arm_link"/>
  <joint name="shoulder_pan_joint" type="revolute">
    <parent link="base_link"/>
    <child link="upper_arm_link"/>
    <origin xyz="0 0 1" rpy="0 0 1.57079632679"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="80" velocity="2"/>
  </joint>
</robot>
)";
    }

    gobot::Ref<gobot::ResourceFormatLoaderURDF> loader = gobot::MakeRef<gobot::ResourceFormatLoaderURDF>();
    gobot::Ref<gobot::Resource> resource = loader->Load(path);
    ASSERT_TRUE(resource.IsValid());

    gobot::Ref<gobot::PackedScene> packed_scene = gobot::dynamic_pointer_cast<gobot::PackedScene>(resource);
    ASSERT_TRUE(packed_scene.IsValid());

    gobot::Node* root_node = packed_scene->Instantiate();
    ASSERT_NE(root_node, nullptr);

    auto* robot = gobot::Object::PointerCastTo<gobot::Robot3D>(root_node);
    ASSERT_NE(robot, nullptr);
    EXPECT_EQ(robot->GetName(), "test_bot");
    EXPECT_EQ(robot->GetSourcePath(), path);

    ASSERT_EQ(robot->GetChildCount(), 1);
    auto* base = gobot::Object::PointerCastTo<gobot::Link3D>(robot->GetChild(0));
    ASSERT_NE(base, nullptr);
    EXPECT_EQ(base->GetName(), "base_link");
    EXPECT_FLOAT_EQ(base->GetMass(), 12.5);
    EXPECT_TRUE(base->GetCenterOfMass().isApprox(gobot::Vector3(0.1, 0.2, 0.3), CMP_EPSILON));
    EXPECT_TRUE(base->GetInertiaDiagonal().isApprox(gobot::Vector3(1.0, 2.0, 3.0), CMP_EPSILON));
    EXPECT_TRUE(base->GetInertiaOffDiagonal().isApprox(gobot::Vector3(0.01, 0.02, 0.03), CMP_EPSILON));

    ASSERT_EQ(base->GetChildCount(), 1);
    auto* joint = gobot::Object::PointerCastTo<gobot::Joint3D>(base->GetChild(0));
    ASSERT_NE(joint, nullptr);
    EXPECT_EQ(joint->GetName(), "shoulder_pan_joint");
    EXPECT_EQ(joint->GetJointType(), gobot::JointType::Revolute);
    EXPECT_EQ(joint->GetParentLink(), "base_link");
    EXPECT_EQ(joint->GetChildLink(), "upper_arm_link");
    EXPECT_TRUE(joint->GetPosition().isApprox(gobot::Vector3(0.0, 0.0, 1.0), CMP_EPSILON));
    EXPECT_TRUE(joint->GetAxis().isApprox(gobot::Vector3::UnitZ(), CMP_EPSILON));
    EXPECT_FLOAT_EQ(joint->GetLowerLimit(), -1.57);
    EXPECT_FLOAT_EQ(joint->GetUpperLimit(), 1.57);
    EXPECT_FLOAT_EQ(joint->GetEffortLimit(), 80.0);
    EXPECT_FLOAT_EQ(joint->GetVelocityLimit(), 2.0);

    ASSERT_EQ(joint->GetChildCount(), 1);
    auto* child_link = gobot::Object::PointerCastTo<gobot::Link3D>(joint->GetChild(0));
    ASSERT_NE(child_link, nullptr);
    EXPECT_EQ(child_link->GetName(), "upper_arm_link");

    gobot::Object::Delete(root_node);
}
