#include <gtest/gtest.h>

#include <filesystem>
#include <thread>

#include "gobot/core/config/project_setting.hpp"
#include "gobot/core/io/resource_loader.hpp"
#include "gobot/core/io/resource_format_scene.hpp"
#include "gobot/core/io/resource_saver.hpp"
#include "gobot/main/engine_context.hpp"
#include "gobot/scene/node_3d.hpp"
#include "gobot/scene/resources/array_mesh.hpp"
#include "gobot/scene/resources/packed_scene.hpp"
#include "gobot/simulation/simulation_server.hpp"

namespace gobot {
namespace {
class EngineContextIsolation : public testing::Test {
protected:
    void SetUp() override {
        loader = MakeRef<ResourceFormatLoaderScene>();
        saver = MakeRef<ResourceFormatSaverScene>();
        ResourceLoader::AddResourceFormatLoader(loader, true);
        ResourceSaver::AddResourceFormatSaver(saver, true);
        directory = std::filesystem::temp_directory_path() /
            ("gobot-context-" + Resource::GenerateResourceUniqueId());
        std::filesystem::create_directories(directory / "a");
        std::filesystem::create_directories(directory / "b");
        ASSERT_TRUE(first.SetProjectPath((directory / "a").string()));
        ASSERT_TRUE(second.SetProjectPath((directory / "b").string()));
    }
    void TearDown() override {
        ResourceLoader::RemoveResourceFormatLoader(loader);
        ResourceSaver::RemoveResourceFormatSaver(saver);
        std::filesystem::remove_all(directory);
    }
    Ref<ResourceFormatLoaderScene> loader;
    Ref<ResourceFormatSaverScene> saver;
    std::filesystem::path directory;
    ProjectSettings first{false};
    ProjectSettings second{false};
};
} // namespace

TEST_F(EngineContextIsolation, NestedAndThreadLocalResolutionDoesNotChangeDefault) {
    auto* original = ProjectSettings::GetInstance();
    {
        ProjectSettings::Scope outer(&first);
        EXPECT_EQ(ProjectSettings::GetInstance(), &first);
        {
            ProjectSettings::Scope inner(&second);
            EXPECT_EQ(ProjectSettings::GetInstance(), &second);
        }
        EXPECT_EQ(ProjectSettings::GetInstance(), &first);
        std::thread other([&] {
            ProjectSettings::Scope scope(&second);
            EXPECT_EQ(ProjectSettings::GetInstance(), &second);
        });
        other.join();
        EXPECT_EQ(ProjectSettings::GetInstance(), &first);
    }
    EXPECT_EQ(ProjectSettings::GetInstance(), original);
}

TEST_F(EngineContextIsolation, SameAuthoredResourcePathHasIndependentCacheIdentity) {
    auto a = MakeRef<ArrayMesh>();
    auto b = MakeRef<ArrayMesh>();
    {
        ProjectSettings::Scope scope(&first);
        a->SetPath("res://shared.jres");
        EXPECT_EQ(ResourceCache::GetRef("res://shared.jres").Get(), a.Get());
    }
    {
        ProjectSettings::Scope scope(&second);
        EXPECT_FALSE(ResourceCache::Has("res://shared.jres"));
        b->SetPath("res://shared.jres");
        EXPECT_EQ(ResourceCache::GetRef("res://shared.jres").Get(), b.Get());
        a.Reset();
        EXPECT_EQ(ResourceCache::GetRef("res://shared.jres").Get(), b.Get());
        EXPECT_EQ(b->GetPath(), "res://shared.jres");
    }
    ProjectSettings::Scope scope(&first);
    EXPECT_FALSE(ResourceCache::Has("res://shared.jres"));
}

TEST_F(EngineContextIsolation, RepeatedLoadsUseOwningProject) {
    auto write_scene = [](ProjectSettings* settings, const char* name) {
        ProjectSettings::Scope scope(settings);
        auto* root = Node::New<Node3D>();
        root->SetName(name);
        auto scene = MakeRef<PackedScene>();
        EXPECT_TRUE(scene->Pack(root));
        EXPECT_TRUE(ResourceSaver::Save(scene, "res://world.jscn"));
        Node::Delete(root);
    };
    write_scene(&first, "First");
    write_scene(&second, "Second");
    SimulationServer first_sim(PhysicsBackendType::Null, false);
    SimulationServer second_sim(PhysicsBackendType::Null, false);
    EngineContext a(&first, &first_sim);
    EngineContext b(&second, &second_sim);
    for (int i = 0; i < 5; ++i) {
        ASSERT_TRUE(a.LoadScene("res://world.jscn")) << a.GetLastError();
        ASSERT_TRUE(b.LoadScene("res://world.jscn")) << b.GetLastError();
        EXPECT_EQ(a.GetSceneRoot()->GetName(), "First");
        EXPECT_EQ(b.GetSceneRoot()->GetName(), "Second");
        EXPECT_NE(a.GetProjectPath(), b.GetProjectPath());
    }
}

TEST_F(EngineContextIsolation, SceneBindingLifecycleStaysWithinItsContext) {
    SimulationServer first_sim(PhysicsBackendType::Null, false);
    SimulationServer second_sim(PhysicsBackendType::Null, false);
    std::vector<std::pair<ObjectID, std::uint64_t>> first_events;
    std::vector<ObjectID> second_events;
    {
        EngineContext a(&first, &first_sim);
        EngineContext b(&second, &second_sim);
        a.SetSceneBindingCallback([&](Node* root, std::uint64_t epoch) {
            first_events.emplace_back(root ? root->GetInstanceId() : ObjectID{}, epoch);
        });
        b.SetSceneBindingCallback([&](Node* root, std::uint64_t) {
            second_events.push_back(root ? root->GetInstanceId() : ObjectID{});
        });
        auto* first_root = Node::New<Node3D>();
        auto* second_root = Node::New<Node3D>();
        a.SetSceneRoot(first_root, true);
        b.SetSceneRoot(second_root, true);
        ASSERT_EQ(first_events.size(), 1);
        ASSERT_EQ(second_events.size(), 1);
        EXPECT_EQ(first_events[0].first, first_root->GetInstanceId());
        EXPECT_EQ(second_events[0], second_root->GetInstanceId());
        a.ClearScene();
        ASSERT_EQ(first_events.size(), 2);
        EXPECT_EQ(first_events[1].first, ObjectID{});
        EXPECT_GT(first_events[1].second, first_events[0].second);
        EXPECT_EQ(second_events.size(), 1);
        EXPECT_EQ(b.GetSceneRoot(), second_root);
    }
    ASSERT_EQ(first_events.size(), 3);
    ASSERT_EQ(second_events.size(), 2);
    EXPECT_EQ(first_events.back().first, ObjectID{});
    EXPECT_EQ(second_events.back(), ObjectID{});
}

} // namespace gobot
