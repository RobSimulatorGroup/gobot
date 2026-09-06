/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>
#include <vector>

#include "gobot/core/object_id.hpp"
#include "gobot/physics/physics_types.hpp"

namespace gobot {

class Joint3D;
class Link3D;
class Node;
class Robot3D;

struct PhysicsRobotSceneBinding {
    ObjectID robot_id{};
    std::vector<ObjectID> link_ids;
    std::vector<ObjectID> joint_ids;
};

struct PhysicsSceneBindings {
    ObjectID scene_root_id{};
    std::vector<PhysicsRobotSceneBinding> robots;
    std::vector<ObjectID> deformable_ids;
};

enum class PhysicsSceneCompileSeverity {
    Warning,
    Error
};

struct PhysicsSceneCompileDiagnostic {
    PhysicsSceneCompileSeverity severity{PhysicsSceneCompileSeverity::Error};
    std::string path;
    std::string message;
};

struct CompiledPhysicsScene {
    PhysicsSceneSnapshot snapshot;
    PhysicsSceneBindings bindings;
    std::vector<PhysicsSceneCompileDiagnostic> diagnostics;
};

class GOBOT_EXPORT PhysicsSceneCompiler {
public:
    static std::uint64_t GetSensorPreviewFingerprint(const Node* scene_root);
    static PhysicsSceneSnapshot CaptureSensorPreview(const Node* scene_root);

    static bool Compile(const Node* scene_root,
                        CompiledPhysicsScene* compiled_scene,
                        std::string* error = nullptr);
};

} // namespace gobot
