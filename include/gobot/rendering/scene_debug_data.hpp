#pragma once

#include "gobot/physics/physics_types.hpp"

namespace gobot {
class Node;

struct DeformableDebugGeometry {
    Color surface_color;
    std::vector<float> triangles;
    std::vector<float> lines;
};

struct SceneDebugData {
    std::vector<float> collision_lines;
    std::vector<DeformableDebugGeometry> deformables;
    std::vector<PhysicsSensorState> sensors;
    std::vector<PhysicsContactState> contacts;
    PhysicsWorldSettings settings;
};

GOBOT_EXPORT SceneDebugData CaptureSceneDebugData(const Node* root,
                                                  const PhysicsSceneState* state,
                                                  const PhysicsWorldSettings& settings,
                                                  bool show_collision_shapes);
} // namespace gobot
