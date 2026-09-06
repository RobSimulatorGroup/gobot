#include "gobot/simulation/physics_debug_preview.hpp"
#include "gobot/physics/physics_scene_compiler.hpp"

namespace gobot {
const PhysicsSceneState* PhysicsDebugPreview::Update(const Node* scene_root) {
    if (scene_root == nullptr) {
        Clear();
        return nullptr;
    }
    const auto fingerprint = PhysicsSceneCompiler::GetSensorPreviewFingerprint(scene_root);
    if (!initialized_ || fingerprint_ != fingerprint) {
        auto snapshot = PhysicsSceneCompiler::CaptureSensorPreview(scene_root);
        world_.Reset();
        if (!snapshot.loose_sensors.empty()) {
            world_ = MakeRef<NullPhysicsWorld>();
            if (!world_->Build(std::move(snapshot))) {
                world_.Reset();
            }
            ++build_count_;
        }
        fingerprint_ = fingerprint;
        initialized_ = true;
    }
    return world_.IsValid() ? &world_->GetSceneState() : nullptr;
}

void PhysicsDebugPreview::Clear() {
    world_.Reset();
    initialized_ = false;
}
} // namespace gobot
