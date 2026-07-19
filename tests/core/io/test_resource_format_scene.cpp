/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * This file is created by Qiqi Wu, 23-1-15
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <gobot/core/ref_counted.hpp>
#include <gobot/core/io/resource_format_scene.hpp>
#include <gobot/core/io/python_script.hpp>
#include <gobot/core/config/project_setting.hpp>
#include <gobot/scene/collision_shape_3d.hpp>
#include <gobot/scene/resources/cylinder_shape_3d.hpp>
#include <gobot/scene/mesh_instance_3d.hpp>
#include <gobot/scene/node.hpp>
#include <gobot/scene/node_3d.hpp>
#include <gobot/scene/resources/array_mesh.hpp>
#include <gobot/scene/resources/packed_scene.hpp>
#include <gobot/scene/resources/primitive_mesh.hpp>
#include <gobot/scene/link_3d.hpp>
#include <gobot/scene/robot_3d.hpp>
#include <gobot/scene/sensor_3d.hpp>
#include <gobot/rendering/render_server.hpp>
#include <gobot/core/types.hpp>
#include <gobot/log.hpp>

#include <cstdlib>
#include <fstream>

namespace {

gobot::Json LoadJsonFile(const std::string& path) {
    std::ifstream input(path);
    gobot::Json json;
    input >> json;
    return json;
}

gobot::Json CoreSceneJson(gobot::Json json) {
    json.erase("__SUB_RESOURCES__");
    json.erase("__EXT_RESOURCES__");
    return json;
}

} // namespace

class TestResourceFormatScene : public testing::Test {
protected:
    static void SetUpTestSuite() {
        static gobot::Ref<gobot::ResourceFormatLoaderScene> resource_loader_scene;
        static gobot::Ref<gobot::ResourceFormatSaverScene> resource_saver_scene;
        static gobot::Ref<gobot::ResourceFormatLoaderPythonScript> resource_loader_python_script;

        resource_saver_scene = gobot::MakeRef<gobot::ResourceFormatSaverScene>();
        gobot::ResourceSaver::AddResourceFormatSaver(resource_saver_scene, true);

        resource_loader_scene = gobot::MakeRef<gobot::ResourceFormatLoaderScene>();
        gobot::ResourceLoader::AddResourceFormatLoader(resource_loader_scene, true);

        resource_loader_python_script = gobot::MakeRef<gobot::ResourceFormatLoaderPythonScript>();
        gobot::ResourceLoader::AddResourceFormatLoader(resource_loader_python_script, true);
    }

    static void TearDownTestSuite() {
    }

    void SetUp() override {
        setenv("HOME", "/tmp/gobot-test-home", 1);
        std::filesystem::create_directories("/tmp/test_project");
        auto* project_setting = gobot::ProjectSettings::GetInstance();
        project_setting->SetProjectPath("/tmp/test_project");

        render_server = std::make_unique<gobot::RenderServer>();
    }

    void TearDown() override {
    }

    gobot::ProjectSettings project_settings;
    std::unique_ptr<gobot::RenderServer> render_server;
};

TEST_F(TestResourceFormatScene, test_save_load) {
    gobot::Ref<gobot::CylinderShape3D> cy = gobot::MakeRef<gobot::CylinderShape3D>();
    cy->SetRadius(1.1);

    USING_ENUM_BITWISE_OPERATORS;
    gobot::ResourceSaver::Save(cy, "res://cyl.jres",
                               gobot::ResourceSaverFlags::ReplaceSubResourcePaths |
                               gobot::ResourceSaverFlags::ChangePath);

    gobot::Ref<gobot::Resource> cylinder = gobot::ResourceLoader::Load("res://cyl.jres");
    ASSERT_TRUE(cylinder->get_type().get_name() == "CylinderShape3D");
    cy = gobot::dynamic_pointer_cast<gobot::CylinderShape3D>(cylinder);
    ASSERT_TRUE(cy->GetRadius() ==  1.1f);
}

TEST_F(TestResourceFormatScene, test_subresource) {
    gobot::Ref<gobot::BoxMesh> box_mesh = gobot::MakeRef<gobot::BoxMesh>();
    box_mesh->SetWidth(1.1);
    auto material_3d = gobot::MakeRef<gobot::PBRMaterial3D>();
    material_3d->SetAlbedo(gobot::Color(0.5f, 0.5f, 0.1f));
    box_mesh->SetMaterial(material_3d);


    USING_ENUM_BITWISE_OPERATORS;
    gobot::ResourceSaver::Save(box_mesh, "res://box_mesh.jres",
                               gobot::ResourceSaverFlags::ReplaceSubResourcePaths |
                               gobot::ResourceSaverFlags::ChangePath);

    gobot::Ref<gobot::Resource> box = gobot::ResourceLoader::Load("res://box_mesh.jres");
    ASSERT_TRUE(box->get_type().get_name() == "BoxMesh");
    auto box_load = gobot::dynamic_pointer_cast<gobot::BoxMesh>(box);
    ASSERT_TRUE(box_load->GetWidth() ==  1.1f);
    auto material3d = gobot::dynamic_pointer_cast<gobot::PBRMaterial3D>(box_load->GetMaterial());
    ASSERT_TRUE(material3d.IsValid());
    ASSERT_TRUE(material3d->GetAlbedo().blue() == 0.1f);
}

TEST_F(TestResourceFormatScene, test_extresource) {
    gobot::Ref<gobot::BoxMesh> box_mesh = gobot::MakeRef<gobot::BoxMesh>();
    box_mesh->SetWidth(1.1);
    auto material_3d = gobot::MakeRef<gobot::PBRMaterial3D>();
    material_3d->SetAlbedo(gobot::Color(0.5f, 0.5f, 0.1f));
    material_3d->SetPath("res://material_ext.jres", true);
    USING_ENUM_BITWISE_OPERATORS;
    ASSERT_TRUE(gobot::ResourceSaver::Save(material_3d, "res://material_ext.jres",
                                           gobot::ResourceSaverFlags::ReplaceSubResourcePaths));


    box_mesh->SetMaterial(material_3d);

    ASSERT_TRUE(gobot::ResourceSaver::Save(box_mesh, "res://box_mesh_ext.jres",
                                           gobot::ResourceSaverFlags::ReplaceSubResourcePaths));
    material_3d->SetPath("", true);

    gobot::Ref<gobot::Resource> box = gobot::ResourceLoader::Load("res://box_mesh_ext.jres");
    ASSERT_TRUE(box->get_type().get_name() == "BoxMesh");
    auto box_load = gobot::dynamic_pointer_cast<gobot::BoxMesh>(box);
    ASSERT_TRUE(box_load->GetWidth() ==  1.1f);
    auto material3d = gobot::dynamic_pointer_cast<gobot::PBRMaterial3D>(box_load->GetMaterial());
    ASSERT_TRUE(material3d.IsValid());
    ASSERT_TRUE(material3d->GetAlbedo().blue() == 0.1f);
}

TEST_F(TestResourceFormatScene, packed_scene_round_trips_nodes_transforms_and_mesh_resource) {
    auto* root = gobot::Object::New<gobot::Node3D>();
    root->SetName("RobotRoot");
    root->SetPosition({1.0f, 2.0f, 3.0f});

    auto* box = gobot::Object::New<gobot::MeshInstance3D>();
    box->SetName("Box");
    box->SetPosition({4.0f, 5.0f, 6.0f});

    gobot::Ref<gobot::BoxMesh> box_mesh = gobot::MakeRef<gobot::BoxMesh>();
    box_mesh->SetSize({2.0f, 3.0f, 4.0f});
    box->SetMesh(box_mesh);
    root->AddChild(box);

    gobot::Ref<gobot::PackedScene> packed_scene = gobot::MakeRef<gobot::PackedScene>();
    ASSERT_TRUE(packed_scene->Pack(root));

    USING_ENUM_BITWISE_OPERATORS;
    ASSERT_TRUE(gobot::ResourceSaver::Save(packed_scene, "res://robot_scene.jscn",
                                           gobot::ResourceSaverFlags::ReplaceSubResourcePaths |
                                           gobot::ResourceSaverFlags::ChangePath));

    gobot::Ref<gobot::Resource> loaded_resource = gobot::ResourceLoader::Load(
            "res://robot_scene.jscn", "PackedScene", gobot::ResourceFormatLoader::CacheMode::Ignore);
    ASSERT_TRUE(loaded_resource.IsValid());

    gobot::Ref<gobot::PackedScene> loaded_scene = gobot::dynamic_pointer_cast<gobot::PackedScene>(loaded_resource);
    ASSERT_TRUE(loaded_scene.IsValid());
    ASSERT_EQ(loaded_scene->GetState()->GetNodeCount(), 2);

    gobot::Node* instance = loaded_scene->Instantiate();
    ASSERT_NE(instance, nullptr);
    EXPECT_EQ(instance->GetName(), "RobotRoot");

    auto* instanced_root = gobot::Object::PointerCastTo<gobot::Node3D>(instance);
    ASSERT_NE(instanced_root, nullptr);
    EXPECT_TRUE(instanced_root->GetPosition().isApprox(gobot::Vector3(1.0f, 2.0f, 3.0f), CMP_EPSILON));
    ASSERT_EQ(instanced_root->GetChildCount(), 1);

    auto* instanced_box = gobot::Object::PointerCastTo<gobot::MeshInstance3D>(instanced_root->GetChild(0));
    ASSERT_NE(instanced_box, nullptr);
    EXPECT_EQ(instanced_box->GetName(), "Box");
    EXPECT_TRUE(instanced_box->GetPosition().isApprox(gobot::Vector3(4.0f, 5.0f, 6.0f), CMP_EPSILON));

    gobot::Ref<gobot::BoxMesh> instanced_mesh = gobot::dynamic_pointer_cast<gobot::BoxMesh>(instanced_box->GetMesh());
    ASSERT_TRUE(instanced_mesh.IsValid());
    EXPECT_TRUE(instanced_mesh->GetSize().isApprox(gobot::Vector3(2.0f, 3.0f, 4.0f), CMP_EPSILON));

    gobot::Object::Delete(root);
    gobot::Object::Delete(instance);
}

TEST_F(TestResourceFormatScene, packed_scene_round_trips_collision_shape_resource) {
    auto* root = gobot::Object::New<gobot::Node3D>();
    root->SetName("RobotRoot");

    auto* collision_shape = gobot::Object::New<gobot::CollisionShape3D>();
    collision_shape->SetName("Collision");
    collision_shape->SetPosition({1.0f, 2.0f, 3.0f});
    collision_shape->SetPriority(3);

    gobot::Ref<gobot::CylinderShape3D> cylinder_shape = gobot::MakeRef<gobot::CylinderShape3D>();
    cylinder_shape->SetRadius(0.25f);
    cylinder_shape->SetHeight(2.0f);
    collision_shape->SetShape(cylinder_shape);
    root->AddChild(collision_shape);

    gobot::Ref<gobot::PackedScene> packed_scene = gobot::MakeRef<gobot::PackedScene>();
    ASSERT_TRUE(packed_scene->Pack(root));

    USING_ENUM_BITWISE_OPERATORS;
    ASSERT_TRUE(gobot::ResourceSaver::Save(packed_scene, "res://collision_scene.jscn",
                                           gobot::ResourceSaverFlags::ReplaceSubResourcePaths |
                                           gobot::ResourceSaverFlags::ChangePath));

    gobot::Ref<gobot::Resource> loaded_resource = gobot::ResourceLoader::Load(
            "res://collision_scene.jscn", "PackedScene", gobot::ResourceFormatLoader::CacheMode::Ignore);
    ASSERT_TRUE(loaded_resource.IsValid());

    gobot::Ref<gobot::PackedScene> loaded_scene = gobot::dynamic_pointer_cast<gobot::PackedScene>(loaded_resource);
    ASSERT_TRUE(loaded_scene.IsValid());

    gobot::Node* instance = loaded_scene->Instantiate();
    ASSERT_NE(instance, nullptr);

    auto* instanced_root = gobot::Object::PointerCastTo<gobot::Node3D>(instance);
    ASSERT_NE(instanced_root, nullptr);
    ASSERT_EQ(instanced_root->GetChildCount(), 1);

    auto* instanced_collision = gobot::Object::PointerCastTo<gobot::CollisionShape3D>(instanced_root->GetChild(0));
    ASSERT_NE(instanced_collision, nullptr);
    EXPECT_EQ(instanced_collision->GetName(), "Collision");
    EXPECT_TRUE(instanced_collision->GetPosition().isApprox(gobot::Vector3(1.0f, 2.0f, 3.0f), CMP_EPSILON));
    EXPECT_EQ(instanced_collision->GetPriority(), 3);

    gobot::Ref<gobot::CylinderShape3D> instanced_shape =
            gobot::dynamic_pointer_cast<gobot::CylinderShape3D>(instanced_collision->GetShape());
    ASSERT_TRUE(instanced_shape.IsValid());
    EXPECT_FLOAT_EQ(instanced_shape->GetRadius(), 0.25f);
    EXPECT_FLOAT_EQ(instanced_shape->GetHeight(), 2.0f);

    gobot::Object::Delete(root);
    gobot::Object::Delete(instance);
}

TEST_F(TestResourceFormatScene, packed_scene_round_trips_link_inertial_orientation) {
    auto* robot = gobot::Object::New<gobot::Robot3D>();
    robot->SetName("InertialBot");

    auto* base_link = gobot::Object::New<gobot::Link3D>();
    base_link->SetName("base");
    base_link->SetHasInertial(true);
    base_link->SetMass(2.0);
    base_link->SetCenterOfMass({0.1, 0.2, 0.3});
    base_link->SetInertiaOrientation(gobot::Quaternion(0.5, 0.5, -0.5, 0.5));
    base_link->SetInertiaDiagonal({1.0, 2.0, 3.0});
    robot->AddChild(base_link);

    gobot::Ref<gobot::PackedScene> packed_scene = gobot::MakeRef<gobot::PackedScene>();
    ASSERT_TRUE(packed_scene->Pack(robot));

    USING_ENUM_BITWISE_OPERATORS;
    ASSERT_TRUE(gobot::ResourceSaver::Save(packed_scene, "res://inertial_scene.jscn",
                                           gobot::ResourceSaverFlags::ReplaceSubResourcePaths |
                                           gobot::ResourceSaverFlags::ChangePath));

    gobot::Ref<gobot::Resource> loaded_resource = gobot::ResourceLoader::Load(
            "res://inertial_scene.jscn", "PackedScene", gobot::ResourceFormatLoader::CacheMode::Ignore);
    ASSERT_TRUE(loaded_resource.IsValid());
    gobot::Ref<gobot::PackedScene> loaded_scene = gobot::dynamic_pointer_cast<gobot::PackedScene>(loaded_resource);
    ASSERT_TRUE(loaded_scene.IsValid());

    gobot::Node* instance = loaded_scene->Instantiate();
    ASSERT_NE(instance, nullptr);
    auto* loaded_robot = gobot::Object::PointerCastTo<gobot::Robot3D>(instance);
    ASSERT_NE(loaded_robot, nullptr);
    ASSERT_EQ(loaded_robot->GetChildCount(), 1);

    auto* loaded_link = gobot::Object::PointerCastTo<gobot::Link3D>(loaded_robot->GetChild(0));
    ASSERT_NE(loaded_link, nullptr);
    EXPECT_TRUE(loaded_link->HasInertial());
    EXPECT_NEAR(loaded_link->GetMass(), 2.0, 1.0e-9);
    EXPECT_TRUE(loaded_link->GetCenterOfMass().isApprox(gobot::Vector3(0.1, 0.2, 0.3), CMP_EPSILON));
    EXPECT_TRUE(loaded_link->GetInertiaOrientation().isApprox(gobot::Quaternion(0.5, 0.5, -0.5, 0.5), CMP_EPSILON));
    EXPECT_TRUE(loaded_link->GetInertiaDiagonal().isApprox(gobot::Vector3(1.0, 2.0, 3.0), CMP_EPSILON));

    gobot::Object::Delete(robot);
    gobot::Object::Delete(instance);
}

TEST_F(TestResourceFormatScene, packed_scene_round_trips_sensor_nodes_under_link) {
    auto* robot = gobot::Object::New<gobot::Robot3D>();
    robot->SetName("SensorBot");

    auto* base_link = gobot::Object::New<gobot::Link3D>();
    base_link->SetName("base");
    robot->AddChild(base_link);

    auto* imu = gobot::Object::New<gobot::IMUSensor3D>();
    imu->SetName("imu");
    imu->SetPosition({0.1, 0.2, 0.3});
    imu->SetSensorPeriod(0.005);
    imu->SetNoiseStddev(0.01);
    imu->SetVisualizeDebug(true);
    base_link->AddChild(imu);

    auto* angular_momentum = gobot::Object::New<gobot::AngularMomentumSensor3D>();
    angular_momentum->SetName("root_angmom");
    angular_momentum->SetSensorPeriod(0.02);
    angular_momentum->SetPosition({0.0, 0.1, 0.0});
    base_link->AddChild(angular_momentum);

    auto* contact = gobot::Object::New<gobot::ContactSensor3D>();
    contact->SetName("foot_contact");
    contact->SetEnabled(false);
    contact->SetRadius(0.04);
    contact->SetMinThreshold(0.2);
    contact->SetMaxThreshold(50.0);
    contact->SetPosition({0.0, 0.0, -0.2});
    base_link->AddChild(contact);

    gobot::Ref<gobot::PackedScene> packed_scene = gobot::MakeRef<gobot::PackedScene>();
    ASSERT_TRUE(packed_scene->Pack(robot));

    USING_ENUM_BITWISE_OPERATORS;
    ASSERT_TRUE(gobot::ResourceSaver::Save(packed_scene, "res://sensor_scene.jscn",
                                           gobot::ResourceSaverFlags::ReplaceSubResourcePaths |
                                           gobot::ResourceSaverFlags::ChangePath));

    gobot::Ref<gobot::Resource> loaded_resource = gobot::ResourceLoader::Load(
            "res://sensor_scene.jscn", "PackedScene", gobot::ResourceFormatLoader::CacheMode::Ignore);
    ASSERT_TRUE(loaded_resource.IsValid());
    gobot::Ref<gobot::PackedScene> loaded_scene = gobot::dynamic_pointer_cast<gobot::PackedScene>(loaded_resource);
    ASSERT_TRUE(loaded_scene.IsValid());

    gobot::Node* instance = loaded_scene->Instantiate();
    ASSERT_NE(instance, nullptr);
    auto* loaded_robot = gobot::Object::PointerCastTo<gobot::Robot3D>(instance);
    ASSERT_NE(loaded_robot, nullptr);
    ASSERT_EQ(loaded_robot->GetChildCount(), 1);

    auto* loaded_link = gobot::Object::PointerCastTo<gobot::Link3D>(loaded_robot->GetChild(0));
    ASSERT_NE(loaded_link, nullptr);
    ASSERT_EQ(loaded_link->GetChildCount(), 3);

    auto* loaded_imu = gobot::Object::PointerCastTo<gobot::IMUSensor3D>(loaded_link->GetChild(0));
    ASSERT_NE(loaded_imu, nullptr);
    EXPECT_EQ(loaded_imu->GetName(), "imu");
    EXPECT_TRUE(loaded_imu->GetPosition().isApprox(gobot::Vector3(0.1, 0.2, 0.3), CMP_EPSILON));
    EXPECT_NEAR(loaded_imu->GetSensorPeriod(), 0.005, 1.0e-6);
    EXPECT_NEAR(loaded_imu->GetNoiseStddev(), 0.01, 1.0e-6);
    EXPECT_TRUE(loaded_imu->ShouldVisualizeDebug());

    auto* loaded_angular_momentum =
            gobot::Object::PointerCastTo<gobot::AngularMomentumSensor3D>(loaded_link->GetChild(1));
    ASSERT_NE(loaded_angular_momentum, nullptr);
    EXPECT_EQ(loaded_angular_momentum->GetName(), "root_angmom");
    EXPECT_NEAR(loaded_angular_momentum->GetSensorPeriod(), 0.02, 1.0e-6);
    EXPECT_TRUE(loaded_angular_momentum->GetPosition().isApprox(gobot::Vector3(0.0, 0.1, 0.0), CMP_EPSILON));

    auto* loaded_contact = gobot::Object::PointerCastTo<gobot::ContactSensor3D>(loaded_link->GetChild(2));
    ASSERT_NE(loaded_contact, nullptr);
    EXPECT_EQ(loaded_contact->GetName(), "foot_contact");
    EXPECT_FALSE(loaded_contact->IsEnabled());
    EXPECT_NEAR(loaded_contact->GetRadius(), 0.04, 1.0e-6);
    EXPECT_NEAR(loaded_contact->GetMinThreshold(), 0.2, 1.0e-6);
    EXPECT_NEAR(loaded_contact->GetMaxThreshold(), 50.0, 1.0e-6);
    EXPECT_TRUE(loaded_contact->GetPosition().isApprox(gobot::Vector3(0.0, 0.0, -0.2), CMP_EPSILON));

    gobot::Object::Delete(robot);
    gobot::Object::Delete(instance);
}

TEST_F(TestResourceFormatScene, packed_scene_save_load_instantiate_repack_keeps_core_json_semantics) {
    auto* robot = gobot::Object::New<gobot::Robot3D>();
    robot->SetName("RoundTripBot");

    auto* base_link = gobot::Object::New<gobot::Link3D>();
    base_link->SetName("base");
    base_link->SetPosition({0.1, 0.2, 0.3});
    robot->AddChild(base_link);

    auto* imu = gobot::Object::New<gobot::IMUSensor3D>();
    imu->SetName("imu");
    imu->SetSensorPeriod(0.01);
    imu->SetNoiseStddev(0.02);
    base_link->AddChild(imu);

    auto* angular_momentum = gobot::Object::New<gobot::AngularMomentumSensor3D>();
    angular_momentum->SetName("root_angmom");
    base_link->AddChild(angular_momentum);

    auto* contact = gobot::Object::New<gobot::ContactSensor3D>();
    contact->SetName("foot_contact");
    contact->SetPosition({0.0, 0.0, -0.25});
    contact->SetRadius(0.05);
    contact->SetMinThreshold(0.1);
    contact->SetMaxThreshold(20.0);
    base_link->AddChild(contact);

    gobot::Ref<gobot::PackedScene> first_pack = gobot::MakeRef<gobot::PackedScene>();
    ASSERT_TRUE(first_pack->Pack(robot));

    USING_ENUM_BITWISE_OPERATORS;
    ASSERT_TRUE(gobot::ResourceSaver::Save(first_pack, "res://round_trip_first.jscn",
                                           gobot::ResourceSaverFlags::ReplaceSubResourcePaths |
                                           gobot::ResourceSaverFlags::ChangePath));

    gobot::Ref<gobot::PackedScene> loaded_scene = gobot::dynamic_pointer_cast<gobot::PackedScene>(
            gobot::ResourceLoader::Load("res://round_trip_first.jscn",
                                        "PackedScene",
                                        gobot::ResourceFormatLoader::CacheMode::Ignore));
    ASSERT_TRUE(loaded_scene.IsValid());
    gobot::Node* instance = loaded_scene->Instantiate();
    ASSERT_NE(instance, nullptr);

    gobot::Ref<gobot::PackedScene> second_pack = gobot::MakeRef<gobot::PackedScene>();
    ASSERT_TRUE(second_pack->Pack(instance));
    ASSERT_TRUE(gobot::ResourceSaver::Save(second_pack, "res://round_trip_second.jscn",
                                           gobot::ResourceSaverFlags::ReplaceSubResourcePaths |
                                           gobot::ResourceSaverFlags::ChangePath));

    EXPECT_EQ(CoreSceneJson(LoadJsonFile("/tmp/test_project/round_trip_first.jscn")),
              CoreSceneJson(LoadJsonFile("/tmp/test_project/round_trip_second.jscn")));

    gobot::Object::Delete(robot);
    gobot::Object::Delete(instance);
}

TEST_F(TestResourceFormatScene, packed_scene_round_trips_node_python_script_resource) {
    auto* root = gobot::Object::New<gobot::Node3D>();
    root->SetName("ScriptedRobot");

    gobot::Ref<gobot::PythonScript> script = gobot::MakeRef<gobot::PythonScript>();
    script->SetSourceCode("def reset(env):\n    return None\n");
    root->SetScript(script);

    gobot::Ref<gobot::PackedScene> packed_scene = gobot::MakeRef<gobot::PackedScene>();
    ASSERT_TRUE(packed_scene->Pack(root));

    USING_ENUM_BITWISE_OPERATORS;
    ASSERT_TRUE(gobot::ResourceSaver::Save(packed_scene, "res://scripted_robot.jscn",
                                           gobot::ResourceSaverFlags::ReplaceSubResourcePaths |
                                           gobot::ResourceSaverFlags::ChangePath));

    gobot::Ref<gobot::Resource> loaded_resource = gobot::ResourceLoader::Load(
            "res://scripted_robot.jscn", "PackedScene", gobot::ResourceFormatLoader::CacheMode::Ignore);
    ASSERT_TRUE(loaded_resource.IsValid());

    gobot::Ref<gobot::PackedScene> loaded_scene = gobot::dynamic_pointer_cast<gobot::PackedScene>(loaded_resource);
    ASSERT_TRUE(loaded_scene.IsValid());

    gobot::Node* instance = loaded_scene->Instantiate();
    ASSERT_NE(instance, nullptr);

    gobot::Ref<gobot::PythonScript> loaded_script =
            gobot::dynamic_pointer_cast<gobot::PythonScript>(instance->GetScript());
    ASSERT_TRUE(loaded_script.IsValid());
    EXPECT_EQ(loaded_script->GetSourceCode(), "def reset(env):\n    return None\n");

    gobot::Object::Delete(root);
    gobot::Object::Delete(instance);
}

TEST_F(TestResourceFormatScene, packed_scene_saves_external_node_python_script_reference) {
    std::filesystem::create_directories("/tmp/test_project/scripts");
    {
        std::ofstream output("/tmp/test_project/scripts/controller.py", std::ios::out | std::ios::trunc);
        output << "import gobot\n\nclass Script(gobot.NodeScript):\n    pass\n";
    }

    gobot::Ref<gobot::Resource> script_resource =
            gobot::ResourceLoader::Load("res://scripts/controller.py",
                                        "PythonScript",
                                        gobot::ResourceFormatLoader::CacheMode::Replace);
    gobot::Ref<gobot::PythonScript> script =
            gobot::dynamic_pointer_cast<gobot::PythonScript>(script_resource);
    ASSERT_TRUE(script.IsValid());
    EXPECT_EQ(script->GetPath(), "res://scripts/controller.py");

    auto* root = gobot::Object::New<gobot::Node3D>();
    root->SetName("ScriptedRobot");
    root->SetScript(script);

    gobot::Ref<gobot::PackedScene> packed_scene = gobot::MakeRef<gobot::PackedScene>();
    ASSERT_TRUE(packed_scene->Pack(root));

    USING_ENUM_BITWISE_OPERATORS;
    ASSERT_TRUE(gobot::ResourceSaver::Save(packed_scene, "res://scripted_robot_external.jscn",
                                           gobot::ResourceSaverFlags::ReplaceSubResourcePaths |
                                           gobot::ResourceSaverFlags::ChangePath));

    std::ifstream saved_file("/tmp/test_project/scripted_robot_external.jscn");
    ASSERT_TRUE(saved_file.is_open());
    gobot::Json saved_json;
    saved_file >> saved_json;
    ASSERT_TRUE(saved_json.contains("__EXT_RESOURCES__"));
    ASSERT_EQ(saved_json["__EXT_RESOURCES__"].size(), 1);
    EXPECT_EQ(saved_json["__EXT_RESOURCES__"][0]["__PATH__"], "res://scripts/controller.py");

    gobot::Ref<gobot::Resource> loaded_resource = gobot::ResourceLoader::Load(
            "res://scripted_robot_external.jscn", "PackedScene", gobot::ResourceFormatLoader::CacheMode::Ignore);
    gobot::Ref<gobot::PackedScene> loaded_scene = gobot::dynamic_pointer_cast<gobot::PackedScene>(loaded_resource);
    ASSERT_TRUE(loaded_scene.IsValid());

    gobot::Node* instance = loaded_scene->Instantiate();
    ASSERT_NE(instance, nullptr);
    gobot::Ref<gobot::PythonScript> loaded_script =
            gobot::dynamic_pointer_cast<gobot::PythonScript>(instance->GetScript());
    ASSERT_TRUE(loaded_script.IsValid());
    EXPECT_EQ(loaded_script->GetPath(), "res://scripts/controller.py");

    gobot::Object::Delete(root);
    gobot::Object::Delete(instance);
}

TEST_F(TestResourceFormatScene, packed_scene_saves_child_scene_as_external_instance) {
    auto* child_root = gobot::Object::New<gobot::Node3D>();
    child_root->SetName("RobotAsset");

    auto* child_mesh = gobot::Object::New<gobot::MeshInstance3D>();
    child_mesh->SetName("AssetMesh");
    child_mesh->SetPosition({1.0f, 2.0f, 3.0f});
    child_root->AddChild(child_mesh);

    gobot::Ref<gobot::PackedScene> child_scene = gobot::MakeRef<gobot::PackedScene>();
    ASSERT_TRUE(child_scene->Pack(child_root));

    USING_ENUM_BITWISE_OPERATORS;
    ASSERT_TRUE(gobot::ResourceSaver::Save(child_scene, "res://robot_asset.jscn",
                                           gobot::ResourceSaverFlags::ReplaceSubResourcePaths |
                                           gobot::ResourceSaverFlags::ChangePath));
    child_scene = gobot::dynamic_pointer_cast<gobot::PackedScene>(
            gobot::ResourceLoader::Load("res://robot_asset.jscn", "PackedScene"));
    ASSERT_TRUE(child_scene.IsValid());
    EXPECT_EQ(child_scene->GetPath(), "res://robot_asset.jscn");

    auto* parent_root = gobot::Object::New<gobot::Node3D>();
    parent_root->SetName("World");

    gobot::Node* child_instance_node = child_scene->Instantiate();
    ASSERT_NE(child_instance_node, nullptr);
    auto* child_instance_root = gobot::Object::PointerCastTo<gobot::Node3D>(child_instance_node);
    ASSERT_NE(child_instance_root, nullptr);
    child_instance_root->SetName("robot");
    child_instance_root->SetPosition({4.0f, 5.0f, 6.0f});
    child_instance_root->SetSceneInstance(child_scene);
    parent_root->AddChild(child_instance_root);

    gobot::Ref<gobot::PackedScene> parent_scene = gobot::MakeRef<gobot::PackedScene>();
    ASSERT_TRUE(parent_scene->Pack(parent_root));
    ASSERT_EQ(parent_scene->GetState()->GetNodeCount(), 2);
    ASSERT_TRUE(parent_scene->GetState()->GetNodeInstance(1).IsValid());

    ASSERT_TRUE(gobot::ResourceSaver::Save(parent_scene, "res://world_with_instance.jscn",
                                           gobot::ResourceSaverFlags::ReplaceSubResourcePaths |
                                           gobot::ResourceSaverFlags::ChangePath));

    std::ifstream input("/tmp/test_project/world_with_instance.jscn");
    ASSERT_TRUE(input.is_open());
    gobot::Json saved_json;
    input >> saved_json;
    ASSERT_EQ(saved_json["__EXT_RESOURCES__"].size(), 1);
    EXPECT_EQ(saved_json["__EXT_RESOURCES__"][0]["__PATH__"], "res://robot_asset.jscn");
    ASSERT_EQ(saved_json["__NODES__"].size(), 2);
    EXPECT_TRUE(saved_json["__NODES__"][1].contains("instance"));
    EXPECT_EQ(saved_json["__NODES__"][1]["name"], "robot");

    gobot::Ref<gobot::Resource> loaded_resource = gobot::ResourceLoader::Load(
            "res://world_with_instance.jscn", "PackedScene", gobot::ResourceFormatLoader::CacheMode::Ignore);
    ASSERT_TRUE(loaded_resource.IsValid());

    gobot::Ref<gobot::PackedScene> loaded_scene = gobot::dynamic_pointer_cast<gobot::PackedScene>(loaded_resource);
    ASSERT_TRUE(loaded_scene.IsValid());
    ASSERT_EQ(loaded_scene->GetState()->GetNodeCount(), 2);
    ASSERT_TRUE(loaded_scene->GetState()->GetNodeInstance(1).IsValid());

    gobot::Node* loaded_instance = loaded_scene->Instantiate();
    ASSERT_NE(loaded_instance, nullptr);
    auto* loaded_root = gobot::Object::PointerCastTo<gobot::Node3D>(loaded_instance);
    ASSERT_NE(loaded_root, nullptr);
    ASSERT_EQ(loaded_root->GetChildCount(), 1);

    auto* loaded_robot = gobot::Object::PointerCastTo<gobot::Node3D>(loaded_root->GetChild(0));
    ASSERT_NE(loaded_robot, nullptr);
    EXPECT_EQ(loaded_robot->GetName(), "robot");
    EXPECT_TRUE(loaded_robot->GetPosition().isApprox(gobot::Vector3(4.0f, 5.0f, 6.0f), CMP_EPSILON));
    ASSERT_EQ(loaded_robot->GetChildCount(), 1);
    EXPECT_EQ(loaded_robot->GetChild(0)->GetName(), "AssetMesh");
    EXPECT_TRUE(loaded_robot->GetSceneInstance().IsValid());

    gobot::Object::Delete(child_root);
    gobot::Object::Delete(parent_root);
    gobot::Object::Delete(loaded_instance);
}

TEST_F(TestResourceFormatScene, packed_scene_subresources_do_not_serialize_resource_path) {
    auto* root = gobot::Object::New<gobot::Node3D>();
    root->SetName("World");

    auto* visual = gobot::Object::New<gobot::MeshInstance3D>();
    visual->SetName("Box");
    visual->SetMesh(gobot::MakeRef<gobot::BoxMesh>());
    root->AddChild(visual);

    gobot::Ref<gobot::PackedScene> scene = gobot::MakeRef<gobot::PackedScene>();
    ASSERT_TRUE(scene->Pack(root));

    USING_ENUM_BITWISE_OPERATORS;
    ASSERT_TRUE(gobot::ResourceSaver::Save(scene, "res://subresource_path_scene.jscn",
                                           gobot::ResourceSaverFlags::ReplaceSubResourcePaths |
                                           gobot::ResourceSaverFlags::ChangePath));

    std::ifstream input("/tmp/test_project/subresource_path_scene.jscn");
    ASSERT_TRUE(input.is_open());
    gobot::Json saved_json;
    input >> saved_json;

    ASSERT_TRUE(saved_json.contains("__SUB_RESOURCES__"));
    ASSERT_FALSE(saved_json["__SUB_RESOURCES__"].empty());
    for (const auto& sub_resource : saved_json["__SUB_RESOURCES__"]) {
        EXPECT_FALSE(sub_resource.contains("resource_path")) << sub_resource.dump(4);
    }

    gobot::Object::Delete(root);
}

TEST_F(TestResourceFormatScene, packed_scene_instance_node_is_serialized_as_black_box_boundary) {
    auto* child_root = gobot::Object::New<gobot::Node3D>();
    child_root->SetName("RobotAsset");

    auto* child_mesh = gobot::Object::New<gobot::MeshInstance3D>();
    child_mesh->SetName("InternalAssetMesh");
    child_root->AddChild(child_mesh);

    gobot::Ref<gobot::PackedScene> child_scene = gobot::MakeRef<gobot::PackedScene>();
    ASSERT_TRUE(child_scene->Pack(child_root));

    USING_ENUM_BITWISE_OPERATORS;
    ASSERT_TRUE(gobot::ResourceSaver::Save(child_scene, "res://black_box_robot_asset.jscn",
                                           gobot::ResourceSaverFlags::ReplaceSubResourcePaths |
                                           gobot::ResourceSaverFlags::ChangePath));

    child_scene = gobot::dynamic_pointer_cast<gobot::PackedScene>(
            gobot::ResourceLoader::Load("res://black_box_robot_asset.jscn", "PackedScene",
                                        gobot::ResourceFormatLoader::CacheMode::Ignore));
    ASSERT_TRUE(child_scene.IsValid());
    ASSERT_EQ(child_scene->GetState()->GetNodeCount(), 2);

    auto* world_root = gobot::Object::New<gobot::Node3D>();
    world_root->SetName("World");

    gobot::Node* instance_node = child_scene->Instantiate();
    ASSERT_NE(instance_node, nullptr);
    auto* instance_root = gobot::Object::PointerCastTo<gobot::Node3D>(instance_node);
    ASSERT_NE(instance_root, nullptr);
    instance_root->SetName("RobotInWorld");
    instance_root->SetPosition({7.0f, 8.0f, 9.0f});
    instance_root->SetSceneInstance(child_scene);
    world_root->AddChild(instance_root);

    gobot::Ref<gobot::PackedScene> world_scene = gobot::MakeRef<gobot::PackedScene>();
    ASSERT_TRUE(world_scene->Pack(world_root));

    ASSERT_EQ(world_scene->GetState()->GetNodeCount(), 2);
    const gobot::SceneState::NodeData* instance_data = world_scene->GetState()->GetNodeData(1);
    ASSERT_NE(instance_data, nullptr);
    EXPECT_EQ(instance_data->name, "RobotInWorld");
    EXPECT_TRUE(instance_data->instance.IsValid());

    for (std::size_t i = 0; i < world_scene->GetState()->GetNodeCount(); ++i) {
        const gobot::SceneState::NodeData* node_data = world_scene->GetState()->GetNodeData(i);
        ASSERT_NE(node_data, nullptr);
        EXPECT_NE(node_data->name, "InternalAssetMesh");
    }

    bool stored_position = false;
    for (const auto& property : instance_data->properties) {
        if (property.name == "position") {
            gobot::Vector3 position = property.value.convert<gobot::Vector3>();
            EXPECT_TRUE(position.isApprox(gobot::Vector3(7.0f, 8.0f, 9.0f), CMP_EPSILON));
            stored_position = true;
        }
    }
    EXPECT_TRUE(stored_position);

    gobot::Object::Delete(child_root);
    gobot::Object::Delete(world_root);
}

TEST_F(TestResourceFormatScene, packed_scene_rejects_instance_without_resource_path) {
    auto* child_root = gobot::Object::New<gobot::Node3D>();
    child_root->SetName("UnsavedAsset");
    gobot::Ref<gobot::PackedScene> child_scene = gobot::MakeRef<gobot::PackedScene>();
    ASSERT_TRUE(child_scene->Pack(child_root));
    ASSERT_TRUE(child_scene->GetPath().empty());

    auto* parent_root = gobot::Object::New<gobot::Node3D>();
    parent_root->SetName("World");
    auto* child_instance = gobot::Object::New<gobot::Node3D>();
    child_instance->SetName("UnsavedAssetInstance");
    child_instance->SetSceneInstance(child_scene);
    parent_root->AddChild(child_instance);

    gobot::Ref<gobot::PackedScene> parent_scene = gobot::MakeRef<gobot::PackedScene>();
    ASSERT_TRUE(parent_scene->Pack(parent_root));

    USING_ENUM_BITWISE_OPERATORS;
    EXPECT_FALSE(gobot::ResourceSaver::Save(parent_scene, "res://invalid_world_with_instance.jscn",
                                            gobot::ResourceSaverFlags::ReplaceSubResourcePaths |
                                            gobot::ResourceSaverFlags::ChangePath));

    gobot::Object::Delete(child_root);
    gobot::Object::Delete(parent_root);
}

TEST_F(TestResourceFormatScene, malformed_external_resource_path_fails_without_throwing) {
    const std::string malformed_path = "/tmp/test_project/malformed_instance.jscn";
    std::ofstream output(malformed_path);
    ASSERT_TRUE(output.is_open());
    output << R"json({
        "__EXT_RESOURCES__": [
            {
                "__ID__": "MissingPath",
                "__PATH__": "",
                "__TYPE__": "PackedScene"
            }
        ],
        "__META_TYPE__": "SCENE",
        "__NODES__": [
            {
                "instance": "ExtResource(MissingPath)",
                "name": "Broken",
                "parent": -1,
                "properties": {},
                "type": "Node3D"
            }
        ],
        "__TYPE__": "PackedScene",
        "__VERSION__": 2
    })json";
    output.close();

    EXPECT_NO_THROW({
        gobot::Ref<gobot::Resource> loaded = gobot::ResourceLoader::Load(
                "res://malformed_instance.jscn", "PackedScene", gobot::ResourceFormatLoader::CacheMode::Ignore);
        EXPECT_FALSE(loaded.IsValid());
    });
}

TEST_F(TestResourceFormatScene, packed_scene_round_trips_mesh_with_null_material_and_surface_color) {
    gobot::Ref<gobot::BoxMesh> mesh = gobot::MakeRef<gobot::BoxMesh>();
    mesh->SetSize({1.0f, 2.0f, 3.0f});
    mesh->SetPath("res://external_visual_mesh.jres", true);

    USING_ENUM_BITWISE_OPERATORS;
    ASSERT_TRUE(gobot::ResourceSaver::Save(mesh, "res://external_visual_mesh.jres",
                                           gobot::ResourceSaverFlags::ReplaceSubResourcePaths));

    auto* root = gobot::Object::New<gobot::Node3D>();
    root->SetName("Robot");

    auto* visual = gobot::Object::New<gobot::MeshInstance3D>();
    visual->SetName("visual");
    visual->SetMesh(mesh);
    visual->SetMaterial({});
    visual->SetSurfaceColor({0.9f, 0.2f, 0.3f, 1.0f});
    root->AddChild(visual);

    gobot::Ref<gobot::PackedScene> packed_scene = gobot::MakeRef<gobot::PackedScene>();
    ASSERT_TRUE(packed_scene->Pack(root));
    ASSERT_TRUE(gobot::ResourceSaver::Save(packed_scene, "res://visual_null_material.jscn",
                                           gobot::ResourceSaverFlags::ReplaceSubResourcePaths |
                                           gobot::ResourceSaverFlags::ChangePath));

    gobot::Ref<gobot::Resource> loaded_resource = gobot::ResourceLoader::Load(
            "res://visual_null_material.jscn", "PackedScene", gobot::ResourceFormatLoader::CacheMode::Ignore);
    ASSERT_TRUE(loaded_resource.IsValid());

    gobot::Ref<gobot::PackedScene> loaded_scene = gobot::dynamic_pointer_cast<gobot::PackedScene>(loaded_resource);
    ASSERT_TRUE(loaded_scene.IsValid());

    gobot::Node* instance = loaded_scene->Instantiate();
    ASSERT_NE(instance, nullptr);
    ASSERT_EQ(instance->GetChildCount(), 1);

    auto* loaded_visual = gobot::Object::PointerCastTo<gobot::MeshInstance3D>(instance->GetChild(0));
    ASSERT_NE(loaded_visual, nullptr);
    EXPECT_TRUE(loaded_visual->GetMesh().IsValid());
    EXPECT_FALSE(loaded_visual->GetMaterial().IsValid());
    EXPECT_FLOAT_EQ(loaded_visual->GetSurfaceColor().red(), 0.9f);
    EXPECT_FLOAT_EQ(loaded_visual->GetSurfaceColor().green(), 0.2f);
    EXPECT_FLOAT_EQ(loaded_visual->GetSurfaceColor().blue(), 0.3f);

    gobot::Object::Delete(root);
    gobot::Object::Delete(instance);
}

TEST_F(TestResourceFormatScene, external_mesh_material_round_trips_as_shared_mesh_material) {
    gobot::Ref<gobot::ArrayMesh> mesh = gobot::MakeRef<gobot::ArrayMesh>();

    gobot::Ref<gobot::PBRMaterial3D> material = gobot::MakeRef<gobot::PBRMaterial3D>();
    material->SetAlbedo({0.15f, 0.35f, 0.85f, 1.0f});
    material->SetRoughness(0.25f);
    mesh->SetMaterial(material);
    mesh->SetPath("res://external_array_mesh.jres", true);

    USING_ENUM_BITWISE_OPERATORS;
    ASSERT_TRUE(gobot::ResourceSaver::Save(mesh, "res://external_array_mesh.jres",
                                           gobot::ResourceSaverFlags::ReplaceSubResourcePaths));

    auto* root = gobot::Object::New<gobot::Node3D>();
    root->SetName("Scene");

    auto* visual = gobot::Object::New<gobot::MeshInstance3D>();
    visual->SetName("visual");
    visual->SetMesh(mesh);
    root->AddChild(visual);

    gobot::Ref<gobot::PackedScene> packed_scene = gobot::MakeRef<gobot::PackedScene>();
    ASSERT_TRUE(packed_scene->Pack(root));
    ASSERT_TRUE(gobot::ResourceSaver::Save(packed_scene, "res://external_mesh_material_override.jscn",
                                           gobot::ResourceSaverFlags::ReplaceSubResourcePaths |
                                           gobot::ResourceSaverFlags::ChangePath));

    gobot::Ref<gobot::Resource> loaded_resource = gobot::ResourceLoader::Load(
            "res://external_mesh_material_override.jscn", "PackedScene", gobot::ResourceFormatLoader::CacheMode::Ignore);
    ASSERT_TRUE(loaded_resource.IsValid());

    gobot::Ref<gobot::PackedScene> loaded_scene = gobot::dynamic_pointer_cast<gobot::PackedScene>(loaded_resource);
    ASSERT_TRUE(loaded_scene.IsValid());

    gobot::Node* instance = loaded_scene->Instantiate();
    ASSERT_NE(instance, nullptr);

    auto* loaded_visual = gobot::Object::PointerCastTo<gobot::MeshInstance3D>(instance->GetChild(0));
    ASSERT_NE(loaded_visual, nullptr);
    EXPECT_TRUE(loaded_visual->GetMesh().IsValid());
    EXPECT_FALSE(loaded_visual->GetMaterial().IsValid());

    gobot::Ref<gobot::ArrayMesh> loaded_mesh =
            gobot::dynamic_pointer_cast<gobot::ArrayMesh>(loaded_visual->GetMesh());
    ASSERT_TRUE(loaded_mesh.IsValid());
    gobot::Ref<gobot::PBRMaterial3D> loaded_material =
            gobot::dynamic_pointer_cast<gobot::PBRMaterial3D>(loaded_mesh->GetMaterial());
    ASSERT_TRUE(loaded_material.IsValid());
    EXPECT_FLOAT_EQ(loaded_material->GetAlbedo().red(), 0.15f);
    EXPECT_FLOAT_EQ(loaded_material->GetAlbedo().green(), 0.35f);
    EXPECT_FLOAT_EQ(loaded_material->GetAlbedo().blue(), 0.85f);
    EXPECT_FLOAT_EQ(loaded_material->GetRoughness(), 0.25f);

    gobot::Object::Delete(root);
    gobot::Object::Delete(instance);
}

TEST_F(TestResourceFormatScene, imported_mesh_material_is_saved_as_mesh_material) {
    gobot::Ref<gobot::ArrayMesh> mesh = gobot::MakeRef<gobot::ArrayMesh>();
    mesh->SetPath("res://assets/H2/meshes/torso_link.stl", true);

    gobot::Ref<gobot::PBRMaterial3D> material = gobot::MakeRef<gobot::PBRMaterial3D>();
    material->SetAlbedo({0.7f, 0.16f, 0.16f, 1.0f});
    mesh->SetMaterial(material);

    auto* root = gobot::Object::New<gobot::Node3D>();
    root->SetName("Scene");

    auto* visual = gobot::Object::New<gobot::MeshInstance3D>();
    visual->SetName("torso_link_visual");
    visual->SetMesh(mesh);
    root->AddChild(visual);

    gobot::Ref<gobot::PackedScene> packed_scene = gobot::MakeRef<gobot::PackedScene>();
    ASSERT_TRUE(packed_scene->Pack(root));
    ASSERT_EQ(packed_scene->GetState()->GetNodeCount(), 2);

    const gobot::SceneState::NodeData* visual_data = packed_scene->GetState()->GetNodeData(1);
    ASSERT_NE(visual_data, nullptr);

    gobot::Ref<gobot::PBRMaterial3D> saved_mesh_material;
    for (const auto& property : visual_data->properties) {
        if (property.name == "mesh_material") {
            saved_mesh_material = gobot::dynamic_pointer_cast<gobot::PBRMaterial3D>(
                    property.value.convert<gobot::Ref<gobot::Resource>>());
            break;
        }
    }

    ASSERT_TRUE(saved_mesh_material.IsValid());
    EXPECT_FLOAT_EQ(saved_mesh_material->GetAlbedo().red(), 0.7f);
    EXPECT_FLOAT_EQ(saved_mesh_material->GetAlbedo().green(), 0.16f);
    EXPECT_FLOAT_EQ(saved_mesh_material->GetAlbedo().blue(), 0.16f);

    gobot::Object::Delete(root);
}

TEST_F(TestResourceFormatScene, imported_mesh_material_and_instance_override_are_distinct) {
    gobot::Ref<gobot::ArrayMesh> mesh = gobot::MakeRef<gobot::ArrayMesh>();
    mesh->SetPath("res://assets/H2/meshes/torso_link.stl", true);

    gobot::Ref<gobot::PBRMaterial3D> mesh_material = gobot::MakeRef<gobot::PBRMaterial3D>();
    mesh_material->SetAlbedo({0.7f, 0.16f, 0.16f, 1.0f});
    mesh->SetMaterial(mesh_material);

    gobot::Ref<gobot::PBRMaterial3D> override_material = gobot::MakeRef<gobot::PBRMaterial3D>();
    override_material->SetAlbedo({0.1f, 0.8f, 0.2f, 1.0f});

    auto* root = gobot::Object::New<gobot::Node3D>();
    root->SetName("Scene");

    auto* visual = gobot::Object::New<gobot::MeshInstance3D>();
    visual->SetName("torso_link_visual");
    visual->SetMesh(mesh);
    visual->SetMaterial(override_material);
    root->AddChild(visual);

    gobot::Ref<gobot::PackedScene> packed_scene = gobot::MakeRef<gobot::PackedScene>();
    ASSERT_TRUE(packed_scene->Pack(root));
    ASSERT_EQ(packed_scene->GetState()->GetNodeCount(), 2);

    const gobot::SceneState::NodeData* visual_data = packed_scene->GetState()->GetNodeData(1);
    ASSERT_NE(visual_data, nullptr);

    gobot::Ref<gobot::PBRMaterial3D> saved_mesh_material;
    gobot::Ref<gobot::PBRMaterial3D> saved_override_material;
    for (const auto& property : visual_data->properties) {
        if (property.name == "mesh_material") {
            saved_mesh_material = gobot::dynamic_pointer_cast<gobot::PBRMaterial3D>(
                    property.value.convert<gobot::Ref<gobot::Resource>>());
        } else if (property.name == "material") {
            saved_override_material = gobot::dynamic_pointer_cast<gobot::PBRMaterial3D>(
                    property.value.convert<gobot::Ref<gobot::Resource>>());
        }
    }

    ASSERT_TRUE(saved_mesh_material.IsValid());
    ASSERT_TRUE(saved_override_material.IsValid());
    EXPECT_FLOAT_EQ(saved_mesh_material->GetAlbedo().red(), 0.7f);
    EXPECT_FLOAT_EQ(saved_mesh_material->GetAlbedo().green(), 0.16f);
    EXPECT_FLOAT_EQ(saved_mesh_material->GetAlbedo().blue(), 0.16f);
    EXPECT_FLOAT_EQ(saved_override_material->GetAlbedo().red(), 0.1f);
    EXPECT_FLOAT_EQ(saved_override_material->GetAlbedo().green(), 0.8f);
    EXPECT_FLOAT_EQ(saved_override_material->GetAlbedo().blue(), 0.2f);

    gobot::Node* instance = packed_scene->Instantiate();
    ASSERT_NE(instance, nullptr);

    auto* loaded_visual = gobot::Object::PointerCastTo<gobot::MeshInstance3D>(instance->GetChild(0));
    ASSERT_NE(loaded_visual, nullptr);
    auto loaded_mesh_material = gobot::dynamic_pointer_cast<gobot::PBRMaterial3D>(loaded_visual->GetMeshMaterial());
    auto loaded_override_material = gobot::dynamic_pointer_cast<gobot::PBRMaterial3D>(loaded_visual->GetMaterial());
    ASSERT_TRUE(loaded_mesh_material.IsValid());
    ASSERT_TRUE(loaded_override_material.IsValid());
    EXPECT_FLOAT_EQ(loaded_mesh_material->GetAlbedo().red(), 0.7f);
    EXPECT_FLOAT_EQ(loaded_override_material->GetAlbedo().green(), 0.8f);

    gobot::Object::Delete(root);
    gobot::Object::Delete(instance);
}

TEST_F(TestResourceFormatScene, built_in_mesh_material_and_instance_override_round_trip) {
    gobot::Ref<gobot::BoxMesh> mesh = gobot::MakeRef<gobot::BoxMesh>();
    mesh->SetSize({1.0f, 2.0f, 3.0f});

    gobot::Ref<gobot::PBRMaterial3D> mesh_material = gobot::MakeRef<gobot::PBRMaterial3D>();
    mesh_material->SetAlbedo({0.2f, 0.3f, 0.9f, 1.0f});
    mesh->SetMaterial(mesh_material);

    gobot::Ref<gobot::PBRMaterial3D> override_material = gobot::MakeRef<gobot::PBRMaterial3D>();
    override_material->SetAlbedo({0.9f, 0.5f, 0.1f, 1.0f});
    override_material->SetRoughness(0.4f);

    auto* root = gobot::Object::New<gobot::Node3D>();
    root->SetName("Scene");

    auto* visual = gobot::Object::New<gobot::MeshInstance3D>();
    visual->SetName("box_visual");
    visual->SetMesh(mesh);
    visual->SetMaterial(override_material);
    root->AddChild(visual);

    gobot::Ref<gobot::PackedScene> packed_scene = gobot::MakeRef<gobot::PackedScene>();
    ASSERT_TRUE(packed_scene->Pack(root));

    USING_ENUM_BITWISE_OPERATORS;
    ASSERT_TRUE(gobot::ResourceSaver::Save(packed_scene, "res://built_in_mesh_material_override.jscn",
                                           gobot::ResourceSaverFlags::ReplaceSubResourcePaths |
                                           gobot::ResourceSaverFlags::ChangePath));

    gobot::Ref<gobot::Resource> loaded_resource = gobot::ResourceLoader::Load(
            "res://built_in_mesh_material_override.jscn", "PackedScene", gobot::ResourceFormatLoader::CacheMode::Ignore);
    ASSERT_TRUE(loaded_resource.IsValid());

    gobot::Ref<gobot::PackedScene> loaded_scene = gobot::dynamic_pointer_cast<gobot::PackedScene>(loaded_resource);
    ASSERT_TRUE(loaded_scene.IsValid());

    gobot::Node* instance = loaded_scene->Instantiate();
    ASSERT_NE(instance, nullptr);
    ASSERT_EQ(instance->GetChildCount(), 1);

    auto* loaded_visual = gobot::Object::PointerCastTo<gobot::MeshInstance3D>(instance->GetChild(0));
    ASSERT_NE(loaded_visual, nullptr);

    gobot::Ref<gobot::BoxMesh> loaded_mesh = gobot::dynamic_pointer_cast<gobot::BoxMesh>(loaded_visual->GetMesh());
    ASSERT_TRUE(loaded_mesh.IsValid());

    auto loaded_mesh_material = gobot::dynamic_pointer_cast<gobot::PBRMaterial3D>(loaded_mesh->GetMaterial());
    auto loaded_override_material = gobot::dynamic_pointer_cast<gobot::PBRMaterial3D>(loaded_visual->GetMaterial());
    ASSERT_TRUE(loaded_mesh_material.IsValid());
    ASSERT_TRUE(loaded_override_material.IsValid());
    EXPECT_FLOAT_EQ(loaded_mesh_material->GetAlbedo().red(), 0.2f);
    EXPECT_FLOAT_EQ(loaded_mesh_material->GetAlbedo().green(), 0.3f);
    EXPECT_FLOAT_EQ(loaded_mesh_material->GetAlbedo().blue(), 0.9f);
    EXPECT_FLOAT_EQ(loaded_override_material->GetAlbedo().red(), 0.9f);
    EXPECT_FLOAT_EQ(loaded_override_material->GetAlbedo().green(), 0.5f);
    EXPECT_FLOAT_EQ(loaded_override_material->GetAlbedo().blue(), 0.1f);
    EXPECT_FLOAT_EQ(loaded_override_material->GetRoughness(), 0.4f);

    gobot::Object::Delete(root);
    gobot::Object::Delete(instance);
}

TEST_F(TestResourceFormatScene, array_mesh_round_trips_multiple_surfaces_and_nested_pbr_assets) {
    auto image = gobot::MakeRef<gobot::Image>(
            2, 1, false, gobot::ImageFormat::RGBA8,
            std::vector<std::uint8_t>{255, 0, 0, 255, 0, 255, 0, 255});
    auto texture = gobot::MakeRef<gobot::Texture2D>(image);
    texture->SetWrapU(gobot::TextureWrap::ClampToEdge);

    auto first_material = gobot::MakeRef<gobot::PBRMaterial3D>();
    first_material->SetAlbedoTexture(texture);
    first_material->SetMetallic(0.35f);
    first_material->SetRoughness(0.65f);

    auto second_material = gobot::MakeRef<gobot::PBRMaterial3D>();
    second_material->SetAlbedo({0.1f, 0.3f, 0.9f, 0.75f});
    second_material->SetAlphaMode(gobot::AlphaMode::Blend);
    second_material->SetDoubleSided(true);

    gobot::MeshSurfaceData first_surface;
    first_surface.vertices = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}};
    first_surface.indices = {0, 1, 2};
    first_surface.uv0 = {{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}};
    first_surface.material = first_material;

    gobot::MeshSurfaceData second_surface;
    second_surface.vertices = {{0.0, 0.0, 1.0}, {1.0, 0.0, 1.0}, {0.0, 1.0, 1.0}};
    second_surface.indices = {0, 1, 2};
    second_surface.material = second_material;

    auto mesh = gobot::MakeRef<gobot::ArrayMesh>();
    mesh->SetSurfaces({first_surface, second_surface});
    const std::uint64_t revision = mesh->GetRevision();
    mesh->SetSurfaces({first_surface, second_surface});
    EXPECT_GT(mesh->GetRevision(), revision);

    USING_ENUM_BITWISE_OPERATORS;
    ASSERT_TRUE(gobot::ResourceSaver::Save(
            mesh, "res://multi_surface_mesh.jres",
            gobot::ResourceSaverFlags::ReplaceSubResourcePaths |
                    gobot::ResourceSaverFlags::ChangePath));

    auto loaded_resource = gobot::ResourceLoader::Load(
            "res://multi_surface_mesh.jres", "ArrayMesh",
            gobot::ResourceFormatLoader::CacheMode::Ignore);
    auto loaded_mesh = gobot::dynamic_pointer_cast<gobot::ArrayMesh>(loaded_resource);
    ASSERT_TRUE(loaded_mesh.IsValid());
    const gobot::MeshSurfaceList surfaces = loaded_mesh->GetSurfaces();
    ASSERT_EQ(surfaces.size(), 2);
    ASSERT_EQ(surfaces[0].uv0.size(), 3);
    ASSERT_EQ(surfaces[0].tangents.size(), 3);
    EXPECT_TRUE(surfaces[1].uv0.empty());
    EXPECT_TRUE(surfaces[1].tangents.empty());

    auto loaded_first = gobot::dynamic_pointer_cast<gobot::PBRMaterial3D>(surfaces[0].material);
    auto loaded_second = gobot::dynamic_pointer_cast<gobot::PBRMaterial3D>(surfaces[1].material);
    ASSERT_TRUE(loaded_first.IsValid());
    ASSERT_TRUE(loaded_second.IsValid());
    EXPECT_FLOAT_EQ(loaded_first->GetMetallic(), 0.35f);
    EXPECT_FLOAT_EQ(loaded_first->GetRoughness(), 0.65f);
    EXPECT_EQ(loaded_second->GetAlphaMode(), gobot::AlphaMode::Blend);
    EXPECT_TRUE(loaded_second->IsDoubleSided());

    const auto loaded_texture = loaded_first->GetAlbedoTexture();
    ASSERT_TRUE(loaded_texture.IsValid());
    EXPECT_EQ(loaded_texture->GetWrapU(), gobot::TextureWrap::ClampToEdge);
    ASSERT_TRUE(loaded_texture->GetImage().IsValid());
    EXPECT_EQ(loaded_texture->GetImage()->GetSize(), gobot::Vector2i(2, 1));
    EXPECT_EQ(loaded_texture->GetImage()->GetDataRef(), image->GetDataRef());
}
