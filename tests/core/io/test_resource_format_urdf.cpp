#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <filesystem>

#include <gobot/core/io/resource_format_mesh.hpp>
#include <gobot/core/io/resource_format_urdf.hpp>
#include <gobot/core/io/resource_loader.hpp>
#include <gobot/core/config/project_setting.hpp>
#include <gobot/rendering/render_server.hpp>
#include <gobot/scene/collision_shape_3d.hpp>
#include <gobot/scene/joint_3d.hpp>
#include <gobot/scene/link_3d.hpp>
#include <gobot/scene/mesh_instance_3d.hpp>
#include <gobot/scene/resources/array_mesh.hpp>
#include <gobot/scene/resources/box_shape_3d.hpp>
#include <gobot/scene/resources/packed_scene.hpp>
#include <gobot/scene/robot_3d.hpp>
#include <gobot/scene/scene_tree.hpp>
#include <gobot/scene/window.hpp>

namespace {

class ScopedMeshLoader {
public:
    ScopedMeshLoader() {
        loader_ = gobot::MakeRef<gobot::ResourceFormatLoaderMesh>();
        gobot::ResourceLoader::AddResourceFormatLoader(loader_, true);
    }

    ~ScopedMeshLoader() {
        gobot::ResourceLoader::RemoveResourceFormatLoader(loader_);
    }

private:
    gobot::Ref<gobot::ResourceFormatLoaderMesh> loader_;
};

class ScopedProjectSettings {
public:
    ScopedProjectSettings() {
        settings_ = gobot::Object::New<gobot::ProjectSettings>();
    }

    ~ScopedProjectSettings() {
        gobot::Object::Delete(settings_);
    }

private:
    gobot::ProjectSettings* settings_{nullptr};
};

int CountArrayMeshVisuals(const gobot::Node* node) {
    int count = 0;
    const auto* mesh_instance = gobot::Object::PointerCastTo<gobot::MeshInstance3D>(node);
    if (mesh_instance != nullptr &&
        gobot::dynamic_pointer_cast<gobot::ArrayMesh>(mesh_instance->GetMesh()).IsValid()) {
        ++count;
    }

    for (std::size_t i = 0; i < node->GetChildCount(); ++i) {
        count += CountArrayMeshVisuals(node->GetChild(static_cast<int>(i)));
    }

    return count;
}

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

void ExpectNode3DGlobalPosition(gobot::Node* root, const std::string& name, const gobot::Vector3& expected) {
    auto* node = gobot::Object::PointerCastTo<gobot::Node3D>(FindNodeByName(root, name));
    ASSERT_NE(node, nullptr) << name;
    EXPECT_TRUE(node->GetGlobalPosition().isApprox(expected, 1e-5)) << name
                                                                    << " actual="
                                                                    << node->GetGlobalPosition().transpose()
                                                                    << " expected="
                                                                    << expected.transpose();
}

const gobot::SceneState::PropertyData* FindNodeProperty(const gobot::SceneState::NodeData& node_data,
                                                        const std::string& property_name) {
    auto it = std::find_if(node_data.properties.begin(),
                           node_data.properties.end(),
                           [&property_name](const gobot::SceneState::PropertyData& property) {
                               return property.name == property_name;
                           });
    return it == node_data.properties.end() ? nullptr : &(*it);
}

} // namespace

TEST(TestResourceFormatURDF, recognizes_urdf_extension_for_packed_scene) {
    gobot::Ref<gobot::ResourceFormatLoaderURDF> loader = gobot::MakeRef<gobot::ResourceFormatLoaderURDF>();

    std::vector<std::string> extensions;
    loader->GetRecognizedExtensionsForType("PackedScene", &extensions);

    EXPECT_NE(std::find(extensions.begin(), extensions.end(), "urdf"), extensions.end());
    EXPECT_NE(std::find(extensions.begin(), extensions.end(), "xml"), extensions.end());
    EXPECT_TRUE(loader->HandlesType("PackedScene"));
}

TEST(TestResourceFormatURDF, imports_robot_links_joints_and_inertial_metadata) {
    const std::filesystem::path fixture_path =
            std::filesystem::current_path() / "tests/fixtures/urdf/simple_robot.urdf";
    const std::string path = fixture_path.string();

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
    ASSERT_EQ(base->GetChildCount(), 3);
    auto* visual = gobot::Object::PointerCastTo<gobot::MeshInstance3D>(base->GetChild(0));
    ASSERT_NE(visual, nullptr);
    ASSERT_TRUE(visual->GetMesh().IsValid());
    EXPECT_EQ(visual->GetMesh()->GetPath(),
              (fixture_path.parent_path() / "meshes/base_visual.dae").lexically_normal().string());
    EXPECT_TRUE(visual->GetScale().isApprox(gobot::Vector3(0.1, 0.2, 0.3), CMP_EPSILON));
    EXPECT_GT(visual->GetSurfaceColor().alpha(), 0.0f);

    auto* collision = gobot::Object::PointerCastTo<gobot::CollisionShape3D>(base->GetChild(1));
    ASSERT_NE(collision, nullptr);
    gobot::Ref<gobot::BoxShape3D> box_shape =
            gobot::dynamic_pointer_cast<gobot::BoxShape3D>(collision->GetShape());
    ASSERT_TRUE(box_shape.IsValid());
    EXPECT_TRUE(box_shape->GetSize().isApprox(gobot::Vector3(0.4, 0.2, 0.1), CMP_EPSILON));

    auto* joint = gobot::Object::PointerCastTo<gobot::Joint3D>(base->GetChild(2));
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
    EXPECT_FLOAT_EQ(joint->GetJointPosition(), 0.0);

    ASSERT_EQ(joint->GetChildCount(), 1);
    auto* child_link = gobot::Object::PointerCastTo<gobot::Link3D>(joint->GetChild(0));
    ASSERT_NE(child_link, nullptr);
    EXPECT_EQ(child_link->GetName(), "upper_arm_link");

    gobot::Object::Delete(root_node);
}

TEST(TestResourceFormatURDF, stores_limited_joint_position_at_limit_midpoint) {
    const std::filesystem::path fixture_path =
            std::filesystem::temp_directory_path() / "gobot_midpoint_joint.urdf";
    {
        std::ofstream file(fixture_path);
        file << R"(<?xml version="1.0"?>
<robot name="midpoint_bot">
  <link name="base_link"/>
  <link name="upper_arm_link"/>
  <joint name="offset_joint" type="revolute">
    <parent link="base_link"/>
    <child link="upper_arm_link"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="1.0" upper="3.0" effort="80" velocity="2"/>
  </joint>
</robot>
)";
    }

    gobot::Ref<gobot::ResourceFormatLoaderURDF> loader = gobot::MakeRef<gobot::ResourceFormatLoaderURDF>();
    gobot::Ref<gobot::Resource> resource = loader->Load(fixture_path.string());
    gobot::Ref<gobot::PackedScene> packed_scene = gobot::dynamic_pointer_cast<gobot::PackedScene>(resource);
    ASSERT_TRUE(packed_scene.IsValid());
    ASSERT_TRUE(packed_scene->GetState().IsValid());

    const gobot::SceneState::NodeData* joint_node = nullptr;
    for (std::size_t i = 0; i < packed_scene->GetState()->GetNodeCount(); ++i) {
        const gobot::SceneState::NodeData* node_data = packed_scene->GetState()->GetNodeData(i);
        if (node_data != nullptr && node_data->type == "Joint3D" && node_data->name == "offset_joint") {
            joint_node = node_data;
            break;
        }
    }
    ASSERT_NE(joint_node, nullptr);

    const gobot::SceneState::PropertyData* joint_position = FindNodeProperty(*joint_node, "joint_position");
    ASSERT_NE(joint_position, nullptr);

    bool success = false;
    const gobot::RealType value = joint_position->value.convert<gobot::RealType>(&success);
    ASSERT_TRUE(success);
    EXPECT_FLOAT_EQ(value, 2.0);

    gobot::Node* root_node = packed_scene->Instantiate();
    ASSERT_NE(root_node, nullptr);
    auto* joint = gobot::Object::PointerCastTo<gobot::Joint3D>(FindNodeByName(root_node, "offset_joint"));
    ASSERT_NE(joint, nullptr);
    EXPECT_FLOAT_EQ(joint->GetJointPosition(), 2.0);
    gobot::Object::Delete(root_node);
}

TEST(TestResourceFormatURDF, marks_empty_root_link_as_virtual_root) {
    const std::filesystem::path fixture_path =
            std::filesystem::temp_directory_path() / "gobot_virtual_root.urdf";
    {
        std::ofstream file(fixture_path);
        file << R"(<?xml version="1.0"?>
<robot name="virtual_root_bot">
  <link name="world"/>
  <link name="base_link"/>
  <joint name="world_to_base" type="fixed">
    <parent link="world"/>
    <child link="base_link"/>
  </joint>
</robot>)";
    }

    gobot::Ref<gobot::ResourceFormatLoaderURDF> loader = gobot::MakeRef<gobot::ResourceFormatLoaderURDF>();
    gobot::Ref<gobot::Resource> resource = loader->Load(fixture_path.string());
    ASSERT_TRUE(resource.IsValid());

    auto packed_scene = gobot::dynamic_pointer_cast<gobot::PackedScene>(resource);
    ASSERT_TRUE(packed_scene.IsValid());

    gobot::Node* root_node = packed_scene->Instantiate();
    ASSERT_NE(root_node, nullptr);

    const auto* world_data = packed_scene->GetState()->GetNodeData(1);
    ASSERT_NE(world_data, nullptr);
    ASSERT_EQ(world_data->name, "world");
    const auto* world_role_property = FindNodeProperty(*world_data, "role");
    ASSERT_NE(world_role_property, nullptr);
    EXPECT_EQ(world_role_property->value.convert<gobot::LinkRole>(), gobot::LinkRole::VirtualRoot);

    auto* world_link = gobot::Object::PointerCastTo<gobot::Link3D>(FindNodeByName(root_node, "world"));
    ASSERT_NE(world_link, nullptr);
    EXPECT_EQ(world_link->GetRole(), gobot::LinkRole::VirtualRoot);

    auto* base_link = gobot::Object::PointerCastTo<gobot::Link3D>(FindNodeByName(root_node, "base_link"));
    ASSERT_NE(base_link, nullptr);
    EXPECT_EQ(base_link->GetRole(), gobot::LinkRole::Physical);

    gobot::Object::Delete(root_node);
}

TEST(TestResourceFormatURDF, keeps_root_link_with_physical_content_physical) {
    const std::filesystem::path fixture_path =
            std::filesystem::temp_directory_path() / "gobot_physical_root.urdf";
    {
        std::ofstream file(fixture_path);
        file << R"(<?xml version="1.0"?>
<robot name="physical_root_bot">
  <link name="base_link">
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>
</robot>)";
    }

    gobot::Ref<gobot::ResourceFormatLoaderURDF> loader = gobot::MakeRef<gobot::ResourceFormatLoaderURDF>();
    gobot::Ref<gobot::Resource> resource = loader->Load(fixture_path.string());
    ASSERT_TRUE(resource.IsValid());

    auto packed_scene = gobot::dynamic_pointer_cast<gobot::PackedScene>(resource);
    ASSERT_TRUE(packed_scene.IsValid());

    gobot::Node* root_node = packed_scene->Instantiate();
    ASSERT_NE(root_node, nullptr);

    auto* base_link = gobot::Object::PointerCastTo<gobot::Link3D>(FindNodeByName(root_node, "base_link"));
    ASSERT_NE(base_link, nullptr);
    EXPECT_EQ(base_link->GetRole(), gobot::LinkRole::Physical);

    gobot::Object::Delete(root_node);
}

TEST(TestResourceFormatURDF, keeps_root_link_with_zero_mass_inertial_physical) {
    const std::filesystem::path fixture_path =
            std::filesystem::temp_directory_path() / "gobot_zero_mass_root.urdf";
    {
        std::ofstream file(fixture_path);
        file << R"(<?xml version="1.0"?>
<robot name="zero_mass_root_bot">
  <link name="base_link">
    <inertial>
      <mass value="0.0"/>
      <inertia ixx="0" iyy="0" izz="0" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>
</robot>)";
    }

    gobot::Ref<gobot::ResourceFormatLoaderURDF> loader = gobot::MakeRef<gobot::ResourceFormatLoaderURDF>();
    gobot::Ref<gobot::Resource> resource = loader->Load(fixture_path.string());
    ASSERT_TRUE(resource.IsValid());

    auto packed_scene = gobot::dynamic_pointer_cast<gobot::PackedScene>(resource);
    ASSERT_TRUE(packed_scene.IsValid());

    gobot::Node* root_node = packed_scene->Instantiate();
    ASSERT_NE(root_node, nullptr);

    auto* base_link = gobot::Object::PointerCastTo<gobot::Link3D>(FindNodeByName(root_node, "base_link"));
    ASSERT_NE(base_link, nullptr);
    EXPECT_EQ(base_link->GetRole(), gobot::LinkRole::Physical);

    gobot::Object::Delete(root_node);
}

TEST(TestResourceFormatURDF, keeps_root_link_with_empty_inertial_tag_physical) {
    const std::filesystem::path fixture_path =
            std::filesystem::temp_directory_path() / "gobot_empty_inertial_root.urdf";
    {
        std::ofstream file(fixture_path);
        file << R"(<?xml version="1.0"?>
<robot name="empty_inertial_root_bot">
  <link name="base_link">
    <inertial></inertial>
  </link>
</robot>)";
    }

    gobot::Ref<gobot::ResourceFormatLoaderURDF> loader = gobot::MakeRef<gobot::ResourceFormatLoaderURDF>();
    gobot::Ref<gobot::Resource> resource = loader->Load(fixture_path.string());
    ASSERT_TRUE(resource.IsValid());

    auto packed_scene = gobot::dynamic_pointer_cast<gobot::PackedScene>(resource);
    ASSERT_TRUE(packed_scene.IsValid());

    gobot::Node* root_node = packed_scene->Instantiate();
    ASSERT_NE(root_node, nullptr);

    auto* base_link = gobot::Object::PointerCastTo<gobot::Link3D>(FindNodeByName(root_node, "base_link"));
    ASSERT_NE(base_link, nullptr);
    EXPECT_EQ(base_link->GetRole(), gobot::LinkRole::Physical);

    gobot::Object::Delete(root_node);
}

TEST(TestResourceFormatURDF, copies_external_package_meshes_into_project_assets) {
    const std::filesystem::path temp_root =
            std::filesystem::temp_directory_path() / "gobot_urdf_import_project_assets_test";
    const std::filesystem::path project_path = temp_root / "project";
    const std::filesystem::path package_path = temp_root / "robot_pkg";
    const std::filesystem::path mesh_path = package_path / "meshes" / "base_visual.stl";
    const std::filesystem::path urdf_path = package_path / "robot.urdf";

    std::filesystem::remove_all(temp_root);
    std::filesystem::create_directories(project_path);
    std::filesystem::create_directories(mesh_path.parent_path());

    {
        std::ofstream package_file(package_path / "package.xml");
        package_file << R"(<package><name>robot_pkg</name></package>)";
    }
    {
        std::ofstream mesh_file(mesh_path);
        mesh_file << "solid base_visual\nendsolid base_visual\n";
    }
    {
        std::ofstream urdf_file(urdf_path);
        urdf_file << R"(<?xml version="1.0"?>
<robot name="asset_bot">
  <link name="base_link">
    <visual>
      <geometry>
        <mesh filename="package://robot_pkg/meshes/base_visual.stl"/>
      </geometry>
    </visual>
  </link>
</robot>
)";
    }

    ScopedProjectSettings project_settings;
    ASSERT_TRUE(gobot::ProjectSettings::GetInstance()->SetProjectPath(project_path.string()));

    gobot::Ref<gobot::ResourceFormatLoaderURDF> loader = gobot::MakeRef<gobot::ResourceFormatLoaderURDF>();
    gobot::Ref<gobot::PackedScene> packed_scene =
            gobot::dynamic_pointer_cast<gobot::PackedScene>(loader->Load(urdf_path.string()));
    ASSERT_TRUE(packed_scene.IsValid());

    gobot::Node* root_node = packed_scene->Instantiate();
    ASSERT_NE(root_node, nullptr);
    auto* visual = gobot::Object::PointerCastTo<gobot::MeshInstance3D>(
            FindNodeByName(root_node, "base_link_visual"));
    ASSERT_NE(visual, nullptr);
    ASSERT_TRUE(visual->GetMesh().IsValid());
    EXPECT_EQ(visual->GetMesh()->GetPath(), "res://assets/robot_pkg/meshes/base_visual.stl");
    EXPECT_TRUE(std::filesystem::exists(project_path / "assets" / "robot_pkg" / "meshes" / "base_visual.stl"));

    gobot::Object::Delete(root_node);
    std::filesystem::remove_all(temp_root);
}

TEST(TestResourceFormatURDF, imports_external_robot_urdf_when_requested) {
    const char* test_urdf_path = std::getenv("GOBOT_TEST_URDF");
    if (test_urdf_path == nullptr || std::string(test_urdf_path).empty()) {
        GTEST_SKIP() << "Set GOBOT_TEST_URDF to run this test with a real robot URDF.";
    }

    const std::string path = test_urdf_path;
    if (!std::filesystem::exists(path)) {
        GTEST_SKIP() << "GOBOT_TEST_URDF does not point to an existing file: " << path;
    }

    gobot::Ref<gobot::ResourceFormatLoaderURDF> loader = gobot::MakeRef<gobot::ResourceFormatLoaderURDF>();
    gobot::Ref<gobot::Resource> resource = loader->Load(path);
    ASSERT_TRUE(resource.IsValid());

    gobot::Ref<gobot::PackedScene> packed_scene = gobot::dynamic_pointer_cast<gobot::PackedScene>(resource);
    ASSERT_TRUE(packed_scene.IsValid());
    ASSERT_TRUE(packed_scene->GetState().IsValid());

    EXPECT_GT(packed_scene->GetState()->GetNodeCount(), 1);

    gobot::Node* root_node = packed_scene->Instantiate();
    ASSERT_NE(root_node, nullptr);

    auto* robot = gobot::Object::PointerCastTo<gobot::Robot3D>(root_node);
    ASSERT_NE(robot, nullptr);
    EXPECT_FALSE(robot->GetName().empty());
    ASSERT_GT(robot->GetChildCount(), 0);

    auto* base = gobot::Object::PointerCastTo<gobot::Link3D>(robot->GetChild(0));
    ASSERT_NE(base, nullptr);
    EXPECT_FALSE(base->GetName().empty());
    ASSERT_GT(base->GetChildCount(), 0);

    auto* visual = gobot::Object::PointerCastTo<gobot::MeshInstance3D>(base->GetChild(0));
    if (visual != nullptr && visual->GetMesh().IsValid()) {
        const std::string visual_mesh_path = visual->GetMesh()->GetPath();
        EXPECT_NE(visual_mesh_path.find("package://"), 0);
        EXPECT_TRUE(std::filesystem::path(visual_mesh_path).is_absolute() ||
                    visual_mesh_path.starts_with("res://"));
    }

    gobot::Object::Delete(root_node);
}

TEST(TestResourceFormatURDF, imports_external_robot_visual_meshes_when_rendering_is_available) {
#ifndef GOBOT_HAS_ASSIMP
    GTEST_SKIP() << "Assimp support is not enabled.";
#else
    const char* test_urdf_path = std::getenv("GOBOT_TEST_URDF");
    if (test_urdf_path == nullptr || std::string(test_urdf_path).empty()) {
        GTEST_SKIP() << "Set GOBOT_TEST_URDF to run this test with a real robot URDF.";
    }

    const std::string path = test_urdf_path;
    if (!std::filesystem::exists(path)) {
        GTEST_SKIP() << "GOBOT_TEST_URDF does not point to an existing file: " << path;
    }

    ScopedProjectSettings project_settings;
    auto render_server = std::make_unique<gobot::RenderServer>();
    ScopedMeshLoader mesh_loader;

    gobot::Ref<gobot::ResourceFormatLoaderURDF> loader = gobot::MakeRef<gobot::ResourceFormatLoaderURDF>();
    gobot::Ref<gobot::Resource> resource = loader->Load(path);
    gobot::Ref<gobot::PackedScene> packed_scene = gobot::dynamic_pointer_cast<gobot::PackedScene>(resource);
    ASSERT_TRUE(packed_scene.IsValid());

    gobot::Node* root_node = packed_scene->Instantiate();
    ASSERT_NE(root_node, nullptr);
    EXPECT_GT(CountArrayMeshVisuals(root_node), 0);

    gobot::Object::Delete(root_node);
#endif
}

TEST(TestResourceFormatURDF, external_robot_link_global_transforms_follow_urdf_joint_chain) {
    const char* test_urdf_path = std::getenv("GOBOT_TEST_URDF");
    if (test_urdf_path == nullptr || std::string(test_urdf_path).empty()) {
        GTEST_SKIP() << "Set GOBOT_TEST_URDF to run this test with a real robot URDF.";
    }

    const std::string path = test_urdf_path;
    if (!std::filesystem::exists(path)) {
        GTEST_SKIP() << "GOBOT_TEST_URDF does not point to an existing file: " << path;
    }

    ScopedProjectSettings project_settings;
    auto render_server = std::make_unique<gobot::RenderServer>();
    auto* tree = gobot::Object::New<gobot::SceneTree>(false);
    tree->Initialize();

    gobot::Ref<gobot::ResourceFormatLoaderURDF> loader = gobot::MakeRef<gobot::ResourceFormatLoaderURDF>();
    gobot::Ref<gobot::Resource> resource = loader->Load(path);
    gobot::Ref<gobot::PackedScene> packed_scene = gobot::dynamic_pointer_cast<gobot::PackedScene>(resource);
    ASSERT_TRUE(packed_scene.IsValid());

    gobot::Node* root_node = packed_scene->Instantiate();
    ASSERT_NE(root_node, nullptr);
    tree->GetRoot()->AddChild(root_node);

    ExpectNode3DGlobalPosition(root_node, "base", {0.0, 0.0, 0.0});
    ExpectNode3DGlobalPosition(root_node, "FL_hip", {0.1934, 0.0465, 0.0});
    ExpectNode3DGlobalPosition(root_node, "FL_thigh", {0.1934, 0.142, 0.0});
    ExpectNode3DGlobalPosition(root_node, "FL_calf", {0.1934, 0.142, -0.213});
    ExpectNode3DGlobalPosition(root_node, "FL_foot", {0.1934, 0.142, -0.426});
    ExpectNode3DGlobalPosition(root_node, "RR_hip", {-0.1934, -0.0465, 0.0});
    ExpectNode3DGlobalPosition(root_node, "RR_thigh", {-0.1934, -0.142, 0.0});
    ExpectNode3DGlobalPosition(root_node, "RR_calf", {-0.1934, -0.142, -0.213});
    ExpectNode3DGlobalPosition(root_node, "RR_foot", {-0.1934, -0.142, -0.426});

    tree->GetRoot()->RemoveChild(root_node);
    gobot::Object::Delete(root_node);
    tree->Finalize();
    gobot::Object::Delete(tree);
}
