#pragma once

#include "gobot/physics/backends/null_physics_world.hpp"

namespace gobot {
class Node;

class GOBOT_EXPORT PhysicsDebugPreview {
public:
    const PhysicsSceneState* Update(const Node* scene_root);
    void Clear();
    std::uint64_t GetBuildCount() const { return build_count_; }

private:
    Ref<NullPhysicsWorld> world_;
    std::uint64_t fingerprint_{0};
    std::uint64_t build_count_{0};
    bool initialized_{false};
};
} // namespace gobot
