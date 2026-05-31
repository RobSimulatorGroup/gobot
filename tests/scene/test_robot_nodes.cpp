#include <gtest/gtest.h>

#include <cmath>

#include <gobot/scene/joint_3d.hpp>
#include <gobot/scene/link_3d.hpp>
#include <gobot/scene/robot_3d.hpp>
#include <gobot/scene/sensor_3d.hpp>

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

    EXPECT_TRUE(gobot::Type::get<gobot::Sensor3D>().get_property("enabled").is_valid());
    EXPECT_TRUE(gobot::Type::get<gobot::Sensor3D>().get_property("sensor_period").is_valid());
    EXPECT_TRUE(gobot::Type::get<gobot::Sensor3D>().get_property("noise_stddev").is_valid());
    EXPECT_TRUE(gobot::Type::get<gobot::Sensor3D>().get_property("visualize_debug").is_valid());
    EXPECT_TRUE(gobot::Type::get<gobot::AngularMomentumSensor3D>().is_valid());
    EXPECT_TRUE(gobot::Type::get<gobot::ContactSensor3D>().get_property("radius").is_valid());
    EXPECT_TRUE(gobot::Type::get<gobot::ContactSensor3D>().get_property("min_threshold").is_valid());
    EXPECT_TRUE(gobot::Type::get<gobot::ContactSensor3D>().get_property("max_threshold").is_valid());
    EXPECT_TRUE(gobot::Type::get<gobot::TerrainHeightSensor3D>().get_property("sample_offsets").is_valid());
}

TEST(TestRobotNodes, stores_sensor_metadata) {
    auto* imu = gobot::Object::New<gobot::IMUSensor3D>();
    imu->SetName("imu");
    imu->SetEnabled(false);
    imu->SetSensorPeriod(0.01);
    imu->SetNoiseStddev(0.02);
    imu->SetVisualizeDebug(true);

    EXPECT_FALSE(imu->IsEnabled());
    EXPECT_NEAR(imu->GetSensorPeriod(), 0.01, 1.0e-6);
    EXPECT_NEAR(imu->GetNoiseStddev(), 0.02, 1.0e-6);
    EXPECT_TRUE(imu->ShouldVisualizeDebug());
    gobot::Object::Delete(imu);

    auto* angular_momentum = gobot::Object::New<gobot::AngularMomentumSensor3D>();
    angular_momentum->SetName("root_angmom");
    angular_momentum->SetSensorPeriod(0.02);
    EXPECT_EQ(angular_momentum->GetName(), "root_angmom");
    EXPECT_NEAR(angular_momentum->GetSensorPeriod(), 0.02, 1.0e-6);
    gobot::Object::Delete(angular_momentum);

    auto* contact = gobot::Object::New<gobot::ContactSensor3D>();
    contact->SetName("foot_contact");
    contact->SetRadius(0.05);
    contact->SetMinThreshold(0.1);
    contact->SetMaxThreshold(100.0);

    EXPECT_NEAR(contact->GetRadius(), 0.05, 1.0e-6);
    EXPECT_NEAR(contact->GetMinThreshold(), 0.1, 1.0e-6);
    EXPECT_NEAR(contact->GetMaxThreshold(), 100.0, 1.0e-6);
    gobot::Object::Delete(contact);

    auto* terrain_height = gobot::Object::New<gobot::TerrainHeightSensor3D>();
    terrain_height->SetName("terrain_scan");
    terrain_height->SetSampleOffsets({{0.1, 0.0, 0.0}, {0.2, 0.1, -0.1}});

    ASSERT_EQ(terrain_height->GetSampleOffsets().size(), 2);
    EXPECT_TRUE(terrain_height->GetSampleOffsets()[0].isApprox(gobot::Vector3(0.1, 0.0, 0.0), CMP_EPSILON));
    EXPECT_TRUE(terrain_height->GetSampleOffsets()[1].isApprox(gobot::Vector3(0.2, 0.1, -0.1), CMP_EPSILON));
    gobot::Object::Delete(terrain_height);
}

TEST(TestRobotNodes, motion_mode_applies_joint_position_delta_from_entered_pose) {
    auto* robot = gobot::Object::New<gobot::Robot3D>();
    robot->SetName("robot");

    auto* joint = gobot::Object::New<gobot::Joint3D>();
    joint->SetName("joint");
    joint->SetJointType(gobot::JointType::Revolute);
    joint->SetAxis(gobot::Vector3::UnitZ());
    joint->SetLowerLimit(-2.0);
    joint->SetUpperLimit(2.0);
    joint->SetJointPosition(0.9);

    auto* link = gobot::Object::New<gobot::Link3D>();
    link->SetName("link");
    link->SetPosition({1.0, 0.0, 0.0});

    robot->AddChild(joint);
    joint->AddChild(link);

    robot->SetMode(gobot::RobotMode::Motion);
    EXPECT_TRUE(link->GetPosition().isApprox(gobot::Vector3(1.0, 0.0, 0.0), CMP_EPSILON));

    joint->SetJointPosition(1.1);
    const gobot::Vector3 expected{
            std::cos(0.2),
            std::sin(0.2),
            0.0};
    EXPECT_TRUE(link->GetPosition().isApprox(expected, CMP_EPSILON));

    robot->SetMode(gobot::RobotMode::Assembly);
    EXPECT_TRUE(link->GetPosition().isApprox(gobot::Vector3(1.0, 0.0, 0.0), CMP_EPSILON));

    gobot::Object::Delete(robot);
}
