#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
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

    ASSERT_EQ(joint->GetChildCount(), 1);
    auto* child_link = gobot::Object::PointerCastTo<gobot::Link3D>(joint->GetChild(0));
    ASSERT_NE(child_link, nullptr);
    EXPECT_EQ(child_link->GetName(), "upper_arm_link");

    gobot::Object::Delete(root_node);
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
