#include <gtest/gtest.h>
#include <array>
#include <limits>
#include "gobot/scene/deformable_body_3d.hpp"
#include "gobot/scene/scene_tree.hpp"
#include "gobot/scene/window.hpp"
#include "gobot/simulation/simulation_scene_sync.hpp"

namespace gobot {
TEST(SimulationSceneSync, LocalVerticesIgnoreDisplayTransformsAndWorldVerticesAreConverted) {
    SceneTree tree(false);
    tree.Initialize();
    auto* body = Object::New<DeformableBody3D>();
    auto mesh = MakeRef<TetrahedralMesh>();
    mesh->SetVertices({{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}});
    mesh->SetTetrahedra({0, 1, 2, 3});
    body->SetMesh(mesh);
    tree.GetRoot()->AddChild(body);
    body->SetPosition({10, 20, 30});
    const std::array<DeformableBody3D*, 1> bodies{body};
    const std::array<std::size_t, 1> counts{4};
    std::vector<RealType> vertices{0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2};
    SimulationSceneSync::ApplyDeformableVertices(tree.GetRoot(), bodies, vertices, 4, counts,
                                                SimulationVertexSpace::Local);
    EXPECT_TRUE(body->GetRuntimeVertices()[1].isApprox(Vector3(2, 0, 0)));
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 3; ++j) vertices[i * 3 + j] += RealType(10 * (j + 1));
    SimulationSceneSync::ApplyDeformableVertices(tree.GetRoot(), bodies, vertices, 4, counts,
                                                SimulationVertexSpace::World);
    EXPECT_TRUE(body->GetRuntimeVertices()[1].isApprox(Vector3(2, 0, 0)));
    EXPECT_TRUE(mesh->GetVertices()[1].isApprox(Vector3(1, 0, 0)));
}

TEST(SimulationSceneSync, InvalidLaterBodyDoesNotApplyEarlierVertices) {
    SceneTree tree(false);
    tree.Initialize();
    auto mesh = MakeRef<TetrahedralMesh>();
    mesh->SetVertices({{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}});
    mesh->SetTetrahedra({0, 1, 2, 3});
    std::array<DeformableBody3D*, 2> bodies{};
    for (auto& body : bodies) {
        body = Object::New<DeformableBody3D>();
        body->SetMesh(mesh);
        tree.GetRoot()->AddChild(body);
        body->SetRuntimeVertices(mesh->GetVertices());
    }
    const std::array<std::size_t, 2> counts{4, 4};
    std::vector<RealType> vertices(24, 2);
    vertices.back() = std::numeric_limits<RealType>::quiet_NaN();
    EXPECT_THROW(SimulationSceneSync::ApplyDeformableVertices(tree.GetRoot(), bodies, vertices, 4,
            counts, SimulationVertexSpace::Local), std::invalid_argument);
    EXPECT_TRUE(bodies[0]->GetRuntimeVertices()[1].isApprox(Vector3(1, 0, 0)));
    vertices.back() = 2;
    auto duplicate = bodies;
    duplicate[1] = bodies[0];
    EXPECT_THROW(SimulationSceneSync::ApplyDeformableVertices(tree.GetRoot(), duplicate, vertices, 4,
            counts, SimulationVertexSpace::Local), std::invalid_argument);
    EXPECT_TRUE(bodies[0]->GetRuntimeVertices()[1].isApprox(Vector3(1, 0, 0)));
}
} // namespace gobot
