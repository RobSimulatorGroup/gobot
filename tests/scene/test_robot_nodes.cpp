#include <gtest/gtest.h>

#include <gobot/scene/joint_3d.hpp>
#include <gobot/scene/link_3d.hpp>
#include <gobot/scene/robot_3d.hpp>

TEST(TestRobotNodes, stores_robot_link_and_joint_metadata) {
    auto* robot = gobot::Object::New<gobot::Robot3D>();
    robot->SetName("robot");
    robot->SetSourcePath("res://robots/arm.urdf");

    auto* base = gobot::Object::New<gobot::Link3D>();
    base->SetName("base_link");
    base->SetMass(12.5);
    base->SetCenterOfMass({0.1, 0.2, 0.3});
    base->SetInertiaDiagonal({1.0, 2.0, 3.0});
    base->SetInertiaOffDiagonal({0.01, 0.02, 0.03});

    auto* shoulder = gobot::Object::New<gobot::Joint3D>();
    shoulder->SetName("shoulder_pan_joint");
    shoulder->SetJointType(gobot::JointType::Revolute);
    shoulder->SetParentLink("base_link");
    shoulder->SetChildLink("upper_arm_link");
    shoulder->SetAxis({0.0, 0.0, 2.0});
    shoulder->SetLowerLimit(-1.57);
    shoulder->SetUpperLimit(1.57);
    shoulder->SetEffortLimit(80.0);
    shoulder->SetVelocityLimit(2.0);

    robot->AddChild(base);
    robot->AddChild(shoulder);

    EXPECT_EQ(robot->GetSourcePath(), "res://robots/arm.urdf");
    EXPECT_FLOAT_EQ(base->GetMass(), 12.5);
    EXPECT_TRUE(base->GetCenterOfMass().isApprox(gobot::Vector3(0.1, 0.2, 0.3), CMP_EPSILON));
    EXPECT_TRUE(base->GetInertiaDiagonal().isApprox(gobot::Vector3(1.0, 2.0, 3.0), CMP_EPSILON));
    EXPECT_TRUE(base->GetInertiaOffDiagonal().isApprox(gobot::Vector3(0.01, 0.02, 0.03), CMP_EPSILON));
    EXPECT_EQ(shoulder->GetJointType(), gobot::JointType::Revolute);
    EXPECT_EQ(shoulder->GetParentLink(), "base_link");
    EXPECT_EQ(shoulder->GetChildLink(), "upper_arm_link");
    EXPECT_TRUE(shoulder->GetAxis().isApprox(gobot::Vector3::UnitZ(), CMP_EPSILON));
    EXPECT_FLOAT_EQ(shoulder->GetLowerLimit(), -1.57);
    EXPECT_FLOAT_EQ(shoulder->GetUpperLimit(), 1.57);
    EXPECT_FLOAT_EQ(shoulder->GetEffortLimit(), 80.0);
    EXPECT_FLOAT_EQ(shoulder->GetVelocityLimit(), 2.0);

    gobot::Object::Delete(robot);
}

TEST(TestRobotNodes, reflected_properties_are_available) {
    EXPECT_TRUE(gobot::Type::get<gobot::Robot3D>().get_property("source_path").is_valid());

    EXPECT_TRUE(gobot::Type::get<gobot::Link3D>().get_property("mass").is_valid());
    EXPECT_TRUE(gobot::Type::get<gobot::Link3D>().get_property("center_of_mass").is_valid());
    EXPECT_TRUE(gobot::Type::get<gobot::Link3D>().get_property("inertia_diagonal").is_valid());
    EXPECT_TRUE(gobot::Type::get<gobot::Link3D>().get_property("inertia_off_diagonal").is_valid());

    EXPECT_TRUE(gobot::Type::get<gobot::Joint3D>().get_property("joint_type").is_valid());
    EXPECT_TRUE(gobot::Type::get<gobot::Joint3D>().get_property("parent_link").is_valid());
    EXPECT_TRUE(gobot::Type::get<gobot::Joint3D>().get_property("child_link").is_valid());
    EXPECT_TRUE(gobot::Type::get<gobot::Joint3D>().get_property("axis").is_valid());
    EXPECT_TRUE(gobot::Type::get<gobot::Joint3D>().get_property("lower_limit").is_valid());
    EXPECT_TRUE(gobot::Type::get<gobot::Joint3D>().get_property("upper_limit").is_valid());
    EXPECT_TRUE(gobot::Type::get<gobot::Joint3D>().get_property("effort_limit").is_valid());
    EXPECT_TRUE(gobot::Type::get<gobot::Joint3D>().get_property("velocity_limit").is_valid());
}
