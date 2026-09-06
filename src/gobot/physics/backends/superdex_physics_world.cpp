/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/physics/backends/superdex_physics_world.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <deque>
#include <limits>
#include <numeric>
#include <optional>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include <superdex_physics.h>
#include <mochi_physics/mochi_physics_experimental.h>
#if __has_include(<mochi_physics/utils/step_profiling.h>)
#include <mochi_physics/utils/step_profiling.h>
#define GOBOT_SUPERDEX_HAS_STEP_PROFILING 1
#else
#define GOBOT_SUPERDEX_HAS_STEP_PROFILING 0
#endif

#include "gobot/core/registration.hpp"
#include "gobot/log.hpp"
#include "gobot/physics/backends/superdex_conversions.hpp"
#include "gobot/physics/joint_controller.hpp"
#include "gobot/physics/physics_server.hpp"

namespace gobot {
namespace {

constexpr std::size_t kInvalidIndex = std::numeric_limits<std::size_t>::max();
constexpr int kPrimitiveSegments = 16;
constexpr int kCapsuleHemisphereRings = 4;

mochi::Real3 ToMochi(const Vector3& value) {
    const Vector3 converted = superdex::ToMochiVector(value);
    return {static_cast<mochi::real>(converted.x()),
            static_cast<mochi::real>(converted.y()),
            static_cast<mochi::real>(converted.z())};
}

Vector3 FromMochi(const mochi::Real3& value) {
    return superdex::FromMochiVector(
            {static_cast<RealType>(value[0]),
             static_cast<RealType>(value[1]),
             static_cast<RealType>(value[2])});
}

mochi::TransformRT ToMochi(const Affine3& value) {
    const Affine3 converted = superdex::ToMochiFrame(value);
    Quaternion rotation(converted.linear());
    rotation.normalize();
    return mochi::TransformRT(
            mochi::Quaternion(static_cast<mochi::real>(rotation.x()),
                              static_cast<mochi::real>(rotation.y()),
                              static_cast<mochi::real>(rotation.z()),
                              static_cast<mochi::real>(rotation.w())),
            ToMochi(value.translation()));
}

mochi::Real6 ToMochiInertia(const PhysicsLinkSnapshot& link) {
    const Matrix3 tensor = superdex::ToMochiInertiaTensor(link);
    return {
            static_cast<mochi::real>(tensor(0, 0)),
            static_cast<mochi::real>(tensor(0, 1)),
            static_cast<mochi::real>(tensor(0, 2)),
            static_cast<mochi::real>(tensor(1, 1)),
            static_cast<mochi::real>(tensor(1, 2)),
            static_cast<mochi::real>(tensor(2, 2)),
    };
}

Affine3 FromMochi(const mochi::TransformRT& value) {
    const mochi::Real4 rotation_data = value.GetRotation().ToReal4();
    Quaternion rotation(static_cast<RealType>(rotation_data[3]),
                        static_cast<RealType>(rotation_data[0]),
                        static_cast<RealType>(rotation_data[1]),
                        static_cast<RealType>(rotation_data[2]));
    Affine3 converted = Affine3::Identity();
    converted.linear() = rotation.normalized().toRotationMatrix();
    const mochi::Real3 translation = value.GetTranslation();
    converted.translation() = Vector3{static_cast<RealType>(translation[0]),
                                      static_cast<RealType>(translation[1]),
                                      static_cast<RealType>(translation[2])};
    return superdex::FromMochiFrame(converted);
}

mochi::ContactParams ToMochiContact(const PhysicsMaterialSnapshot& material,
                                    RealType contact_offset = 0.0,
                                    RealType rest_offset = 0.0) {
    mochi::ContactParams params;
    params.coulombFrictionCoefficient = static_cast<mochi::real>(
            std::max<RealType>(0.0, material.sliding_friction));
    params.normalViscousDampingCoefficient = static_cast<mochi::real>(
            std::max<RealType>(0.0, material.contact_damping));
    if (material.contact_compliance > 0.0) {
        params.penaltyCoefficient = static_cast<mochi::real>(
                1.0 / material.contact_compliance);
    }
    if (contact_offset != 0.0 || rest_offset != 0.0) {
        params.penaltyThresholdDefault = static_cast<mochi::real>(
                std::max<RealType>(0.0, contact_offset - rest_offset));
        params.penaltyThresholdExtraPadding = static_cast<mochi::real>(
                std::max<RealType>(0.0, contact_offset));
    }
    // Mochi's 5 mm default smoothing distance is larger than its 1 mm
    // default contact threshold. Scale it to Gobot's authored threshold so
    // configured contact stiffness is effective before visible penetration.
    const RealType penalty_threshold = static_cast<RealType>(
            params.penaltyThresholdDefault);
    params.penaltySmoothingHalfDistance = static_cast<mochi::real>(
            std::clamp(penalty_threshold * 0.25, 1.0e-5, 2.5e-4));
    return params;
}

mochi::ArticulatedJointType ToMochiJointType(JointType type) {
    switch (superdex::ToJointKind(type)) {
        case superdex::JointKind::Hard:
            return mochi::ArticulatedJointType::Hard;
        case superdex::JointKind::Revolute:
            return mochi::ArticulatedJointType::Revolute;
        case superdex::JointKind::Prismatic:
            return mochi::ArticulatedJointType::Prismatic;
        case superdex::JointKind::Free:
            return mochi::ArticulatedJointType::Free;
        case superdex::JointKind::Unsupported:
            return mochi::ArticulatedJointType::Invalid;
    }
    return mochi::ArticulatedJointType::Invalid;
}

struct TriangleMeshBuilder {
    std::vector<Vector3> vertices;
    std::vector<int> indices;

    int AddVertex(const Affine3& transform, const Vector3& vertex) {
        const Vector3 converted = superdex::ToMochiVector(transform * vertex);
        vertices.push_back(converted);
        return static_cast<int>(vertices.size() - 1);
    }

    void AddTriangle(int a, int b, int c) {
        indices.push_back(a);
        indices.push_back(b);
        indices.push_back(c);
    }

    void AppendBox(const Affine3& transform, const Vector3& size) {
        const Vector3 half = size * 0.5;
        const std::array<Vector3, 8> corners = {
                Vector3{-half.x(), -half.y(), -half.z()},
                Vector3{half.x(), -half.y(), -half.z()},
                Vector3{half.x(), half.y(), -half.z()},
                Vector3{-half.x(), half.y(), -half.z()},
                Vector3{-half.x(), -half.y(), half.z()},
                Vector3{half.x(), -half.y(), half.z()},
                Vector3{half.x(), half.y(), half.z()},
                Vector3{-half.x(), half.y(), half.z()},
        };
        const int base = static_cast<int>(vertices.size());
        for (const Vector3& corner : corners) {
            AddVertex(transform, corner);
        }
        constexpr std::array<int, 36> triangles = {
                0, 2, 1, 0, 3, 2,
                4, 5, 6, 4, 6, 7,
                0, 1, 5, 0, 5, 4,
                3, 7, 6, 3, 6, 2,
                0, 4, 7, 0, 7, 3,
                1, 2, 6, 1, 6, 5,
        };
        for (std::size_t index = 0; index < triangles.size(); index += 3) {
            AddTriangle(base + triangles[index],
                        base + triangles[index + 1],
                        base + triangles[index + 2]);
        }
    }

    void AppendCylinder(const Affine3& transform, RealType radius, RealType height) {
        const int base = static_cast<int>(vertices.size());
        const RealType half_height = height * 0.5;
        for (int ring = 0; ring < 2; ++ring) {
            const RealType z = ring == 0 ? -half_height : half_height;
            for (int segment = 0; segment < kPrimitiveSegments; ++segment) {
                const RealType angle = 2.0 * Math_PI * static_cast<RealType>(segment) /
                                       static_cast<RealType>(kPrimitiveSegments);
                AddVertex(transform, {radius * std::cos(angle), radius * std::sin(angle), z});
            }
        }
        const int bottom_center = AddVertex(transform, {0.0, 0.0, -half_height});
        const int top_center = AddVertex(transform, {0.0, 0.0, half_height});
        for (int segment = 0; segment < kPrimitiveSegments; ++segment) {
            const int next = (segment + 1) % kPrimitiveSegments;
            const int bottom = base + segment;
            const int bottom_next = base + next;
            const int top = base + kPrimitiveSegments + segment;
            const int top_next = base + kPrimitiveSegments + next;
            AddTriangle(bottom, bottom_next, top_next);
            AddTriangle(bottom, top_next, top);
            AddTriangle(bottom_center, bottom_next, bottom);
            AddTriangle(top_center, top, top_next);
        }
    }

    void AppendSphere(const Affine3& transform, RealType radius) {
        const int bottom_pole = AddVertex(transform, {0.0, 0.0, -radius});
        std::vector<std::vector<int>> rings;
        constexpr int latitude_segments = 2 * kCapsuleHemisphereRings;
        for (int latitude = 1; latitude < latitude_segments; ++latitude) {
            const RealType angle = -Math_HALF_PI +
                                   Math_PI * static_cast<RealType>(latitude) /
                                           static_cast<RealType>(latitude_segments);
            std::vector<int> indices;
            indices.reserve(kPrimitiveSegments);
            for (int segment = 0; segment < kPrimitiveSegments; ++segment) {
                const RealType azimuth = 2.0 * Math_PI * static_cast<RealType>(segment) /
                                         static_cast<RealType>(kPrimitiveSegments);
                indices.push_back(AddVertex(
                        transform,
                        {radius * std::cos(angle) * std::cos(azimuth),
                         radius * std::cos(angle) * std::sin(azimuth),
                         radius * std::sin(angle)}));
            }
            rings.push_back(std::move(indices));
        }
        const int top_pole = AddVertex(transform, {0.0, 0.0, radius});
        for (int segment = 0; segment < kPrimitiveSegments; ++segment) {
            const int next = (segment + 1) % kPrimitiveSegments;
            AddTriangle(bottom_pole, rings.front()[next], rings.front()[segment]);
            for (std::size_t ring = 0; ring + 1 < rings.size(); ++ring) {
                AddTriangle(rings[ring][segment], rings[ring][next], rings[ring + 1][next]);
                AddTriangle(rings[ring][segment], rings[ring + 1][next], rings[ring + 1][segment]);
            }
            AddTriangle(top_pole, rings.back()[segment], rings.back()[next]);
        }
    }

    void AppendCapsule(const Affine3& transform, RealType radius, RealType height) {
        if (height <= CMP_EPSILON) {
            AppendSphere(transform, radius);
            return;
        }
        const RealType half_height = height * 0.5;
        std::vector<std::vector<int>> rings;
        const int bottom_pole = AddVertex(transform, {0.0, 0.0, -half_height - radius});
        for (int ring = 1; ring <= kCapsuleHemisphereRings; ++ring) {
            const RealType angle = -Math_HALF_PI +
                                   Math_HALF_PI * static_cast<RealType>(ring) /
                                           static_cast<RealType>(kCapsuleHemisphereRings);
            std::vector<int> indices;
            indices.reserve(kPrimitiveSegments);
            for (int segment = 0; segment < kPrimitiveSegments; ++segment) {
                const RealType azimuth = 2.0 * Math_PI * static_cast<RealType>(segment) /
                                         static_cast<RealType>(kPrimitiveSegments);
                indices.push_back(AddVertex(
                        transform,
                        {radius * std::cos(angle) * std::cos(azimuth),
                         radius * std::cos(angle) * std::sin(azimuth),
                         -half_height + radius * std::sin(angle)}));
            }
            rings.push_back(std::move(indices));
        }
        {
            std::vector<int> equator;
            equator.reserve(kPrimitiveSegments);
            for (int segment = 0; segment < kPrimitiveSegments; ++segment) {
                const RealType azimuth = 2.0 * Math_PI * static_cast<RealType>(segment) /
                                         static_cast<RealType>(kPrimitiveSegments);
                equator.push_back(AddVertex(
                        transform,
                        {radius * std::cos(azimuth), radius * std::sin(azimuth), half_height}));
            }
            rings.push_back(std::move(equator));
        }
        for (int ring = 1; ring < kCapsuleHemisphereRings; ++ring) {
            const RealType angle = Math_HALF_PI * static_cast<RealType>(ring) /
                                   static_cast<RealType>(kCapsuleHemisphereRings);
            std::vector<int> indices;
            indices.reserve(kPrimitiveSegments);
            for (int segment = 0; segment < kPrimitiveSegments; ++segment) {
                const RealType azimuth = 2.0 * Math_PI * static_cast<RealType>(segment) /
                                         static_cast<RealType>(kPrimitiveSegments);
                indices.push_back(AddVertex(
                        transform,
                        {radius * std::cos(angle) * std::cos(azimuth),
                         radius * std::cos(angle) * std::sin(azimuth),
                         half_height + radius * std::sin(angle)}));
            }
            rings.push_back(std::move(indices));
        }
        const int top_pole = AddVertex(transform, {0.0, 0.0, half_height + radius});
        for (int segment = 0; segment < kPrimitiveSegments; ++segment) {
            const int next = (segment + 1) % kPrimitiveSegments;
            AddTriangle(bottom_pole, rings.front()[next], rings.front()[segment]);
            for (std::size_t ring = 0; ring + 1 < rings.size(); ++ring) {
                AddTriangle(rings[ring][segment], rings[ring][next], rings[ring + 1][next]);
                AddTriangle(rings[ring][segment], rings[ring + 1][next], rings[ring + 1][segment]);
            }
            AddTriangle(top_pole, rings.back()[segment], rings.back()[next]);
        }
    }

    bool AppendShape(const PhysicsShapeSnapshot& shape, const Affine3& local_transform) {
        if (shape.disabled) {
            return true;
        }
        switch (shape.type) {
            case PhysicsShapeType::Box:
                AppendBox(local_transform, shape.box_size);
                return true;
            case PhysicsShapeType::Cylinder:
                AppendCylinder(local_transform, shape.radius, shape.height);
                return true;
            case PhysicsShapeType::Capsule:
                AppendCapsule(local_transform, shape.radius, shape.height);
                return true;
            case PhysicsShapeType::Sphere: {
                AppendSphere(local_transform, shape.radius);
                return true;
            }
            case PhysicsShapeType::Mesh: {
                if (shape.vertices.empty() || shape.indices.size() < 3) {
                    return false;
                }
                const int base = static_cast<int>(vertices.size());
                for (const Vector3& vertex : shape.vertices) {
                    AddVertex(local_transform, vertex);
                }
                for (std::uint32_t index : shape.indices) {
                    if (index >= shape.vertices.size()) {
                        return false;
                    }
                    indices.push_back(base + static_cast<int>(index));
                }
                return indices.size() % 3 == 0;
            }
            case PhysicsShapeType::Unknown:
                return false;
        }
        return false;
    }

    bool IsValid() const {
        return vertices.size() >= 3 && indices.size() >= 3 && indices.size() % 3 == 0;
    }
};

std::vector<mochi::real> FlattenVertices(const std::vector<Vector3>& vertices) {
    std::vector<mochi::real> result;
    result.reserve(vertices.size() * 3);
    for (const Vector3& vertex : vertices) {
        result.push_back(static_cast<mochi::real>(vertex.x()));
        result.push_back(static_cast<mochi::real>(vertex.y()));
        result.push_back(static_cast<mochi::real>(vertex.z()));
    }
    return result;
}

PhysicsSolverConvergenceStatus FromMochi(mochi::ConvergenceStatus status) {
    switch (status) {
        case mochi::ConvergenceStatus::None:
            return PhysicsSolverConvergenceStatus::Unknown;
        case mochi::ConvergenceStatus::Converged:
            return PhysicsSolverConvergenceStatus::Converged;
        case mochi::ConvergenceStatus::Stopped:
            return PhysicsSolverConvergenceStatus::Stopped;
        case mochi::ConvergenceStatus::Diverged:
            return PhysicsSolverConvergenceStatus::Diverged;
        case mochi::ConvergenceStatus::Count:
            break;
    }
    return PhysicsSolverConvergenceStatus::Unknown;
}

} // namespace

struct SuperDexPhysicsWorld::Impl {
    struct ActorMetadata {
        std::string robot_name;
        std::string link_name;
        std::string shape_name;
        std::string shape_path;
        std::uint64_t shape_stable_id{0};
        std::size_t deformable_index{kInvalidIndex};
    };

    struct LayerInfo {
        std::string name;
        std::uint32_t layer{1};
        std::uint32_t mask{1};
    };

    struct LinkBinding {
        mochi::Actor* actor{nullptr};
        int mochi_link_index{-1};
    };

    struct JointBinding {
        int mochi_joint_index{-1};
        int dof_offset{-1};
        int dof_size{0};
        RealType position_offset{0.0};
        JointController controller;
    };

    struct RobotBinding {
        mochi::Actor* actor{nullptr};
        bool articulated{false};
        std::vector<LinkBinding> links;
        std::vector<JointBinding> joints;
    };

    struct DeformableBinding {
        mochi::Actor* actor{nullptr};
        std::size_t snapshot_index{kInvalidIndex};
        std::vector<Vector3> previous_vertices;
    };

    mochi::Context* context{nullptr};
    mochi::Scene* scene{nullptr};
    std::vector<mochi::ShapeHandle> shapes;
    std::vector<RobotBinding> robots;
    std::vector<DeformableBinding> deformables;
    std::deque<ActorMetadata> metadata;
    std::unordered_map<std::uint64_t, ActorMetadata*> metadata_by_actor;
    std::vector<mochi::Actor*> query_actors;
    std::vector<LayerInfo> layers;
    mochi::DynamicArray<std::uint8_t> initial_state;
    PhysicsSolverDiagnostics diagnostics;

    ~Impl() {
        Release();
    }

    void Release() {
        query_actors.clear();
        metadata_by_actor.clear();
        metadata.clear();
        deformables.clear();
        robots.clear();
        layers.clear();
        initial_state.clear();
        if (context != nullptr && scene != nullptr) {
            context->DestroyScene(scene);
        }
        scene = nullptr;
        shapes.clear();
        if (context != nullptr) {
            mochi::DestroyContext(context);
        }
        context = nullptr;
        diagnostics = {};
    }

    mochi::ShapeHandle CreateTriangleShape(const TriangleMeshBuilder& mesh, mochi::Error& error) {
        if (!mesh.IsValid()) {
            return {};
        }
        const std::vector<mochi::real> coordinates = FlattenVertices(mesh.vertices);
        mochi::ShapeHandle shape = context->CreateTriMeshShape(coordinates, mesh.indices, error);
        if (error.IsOK() && shape.IsValid()) {
            shapes.push_back(shape);
        }
        return shape;
    }

    mochi::ShapeHandle CreateLinkShape(const PhysicsLinkSnapshot& link, mochi::Error& error) {
        std::vector<const PhysicsShapeSnapshot*> enabled_shapes;
        for (const PhysicsShapeSnapshot& shape : link.collision_shapes) {
            if (!shape.disabled) {
                enabled_shapes.push_back(&shape);
            }
        }
        if (enabled_shapes.empty()) {
            return {};
        }
        if (enabled_shapes.size() == 1 && enabled_shapes.front()->type == PhysicsShapeType::Sphere) {
            const PhysicsShapeSnapshot& sphere = *enabled_shapes.front();
            const Affine3 local = link.global_transform.inverse() * sphere.global_transform;
            const RealType scale = local.linear().colwise().norm().maxCoeff();
            Affine3 sphere_transform = Affine3::Identity();
            sphere_transform.translation() = local.translation();
            TriangleMeshBuilder mesh;
            mesh.AppendSphere(sphere_transform, sphere.radius * scale);
            return CreateTriangleShape(mesh, error);
        }

        TriangleMeshBuilder mesh;
        for (const PhysicsShapeSnapshot* shape : enabled_shapes) {
            if (!mesh.AppendShape(*shape, link.global_transform.inverse() * shape->global_transform)) {
                return {};
            }
        }
        return CreateTriangleShape(mesh, error);
    }

    ActorMetadata* AddMetadata(mochi::Actor* actor,
                               std::string robot_name,
                               std::string link_name,
                               const PhysicsShapeSnapshot* shape,
                               std::size_t deformable_index = kInvalidIndex) {
        metadata.push_back({});
        ActorMetadata& item = metadata.back();
        item.robot_name = std::move(robot_name);
        item.link_name = std::move(link_name);
        item.deformable_index = deformable_index;
        if (shape != nullptr) {
            item.shape_name = shape->name;
            item.shape_path = shape->scene_path;
            item.shape_stable_id = shape->stable_id;
        }
        actor->SetUserData(&item);
        metadata_by_actor[actor->GetHandle().value] = &item;
        return &item;
    }

    void AddLayer(std::uint32_t layer, std::uint32_t mask) {
        const std::string name = superdex::MakeContactLayerName(layer, mask);
        const auto found = std::find_if(layers.begin(), layers.end(), [&](const LayerInfo& value) {
            return value.name == name;
        });
        if (found == layers.end()) {
            layers.push_back({name, layer, mask});
        }
    }

    bool RegisterQueries(mochi::Actor* actor,
                         bool node_contact_forces,
                         std::string* error_message) {
        mochi::Error error;
        if (actor->IsQuerySupported(mochi::QueryType::ContactPoints)) {
            actor->RegisterQuery(mochi::QueryType::ContactPoints, error);
        }
        if (error.IsOK() && node_contact_forces &&
            actor->IsQuerySupported(mochi::QueryType::NodeContactForces)) {
            actor->RegisterQuery(mochi::QueryType::NodeContactForces, error);
        }
        if (!error.IsOK()) {
            *error_message = error.ToString();
            return false;
        }
        query_actors.push_back(actor);
        return true;
    }
};

namespace {

const PhysicsShapeSnapshot* FirstEnabledShape(const PhysicsLinkSnapshot& link) {
    for (const PhysicsShapeSnapshot& shape : link.collision_shapes) {
        if (!shape.disabled) {
            return &shape;
        }
    }
    return nullptr;
}

bool UsesAnalyticSphereCollider(const PhysicsLinkSnapshot& link) {
    const PhysicsShapeSnapshot* enabled_shape = nullptr;
    for (const PhysicsShapeSnapshot& shape : link.collision_shapes) {
        if (shape.disabled) {
            continue;
        }
        if (enabled_shape != nullptr) {
            return false;
        }
        enabled_shape = &shape;
    }
    return enabled_shape != nullptr && enabled_shape->type == PhysicsShapeType::Sphere;
}

mochi::ColliderType ColliderTypeForLink(const PhysicsLinkSnapshot& link,
                                        bool exact_mesh = true) {
    if (UsesAnalyticSphereCollider(link)) {
        // Mochi dynamic actors need a surface mesh for boundary sampling, but
        // the explicit collider type keeps contact evaluation analytic.
        return mochi::ColliderType::Sphere;
    }
    return exact_mesh ? mochi::ColliderType::Mesh : mochi::ColliderType::Auto;
}

std::pair<std::uint32_t, std::uint32_t> LinkLayerAndMask(const PhysicsLinkSnapshot& link) {
    if (const PhysicsShapeSnapshot* shape = FirstEnabledShape(link)) {
        return {shape->collision_layer, shape->collision_mask};
    }
    return {1U, 1U};
}

bool IsFiniteSnapshot(const PhysicsSceneSnapshot& snapshot, std::string* error) {
    for (const PhysicsRobotSnapshot& robot : snapshot.robots) {
        for (const PhysicsJointSnapshot& joint : robot.joints) {
            if (static_cast<JointType>(joint.joint_type) == JointType::Planar) {
                *error = "SuperDex does not support Planar joints in its first experimental release: " +
                         joint.scene_path;
                return false;
            }
        }
    }
    if (!snapshot.couplings.empty()) {
        *error = "SuperDex is a unified solver and rejects legacy PhysicsCoupling3D nodes.";
        return false;
    }
    for (const PhysicsDeformableSnapshot& deformable : snapshot.deformables) {
        if (deformable.stable_id == 0 || deformable.vertices.empty()) {
            *error = "SuperDex deformables require a stable ID and non-empty authored vertices: " +
                     deformable.scene_path;
            return false;
        }
        if (deformable.model != 0 && deformable.model != 1) {
            *error = "SuperDex supports only tetrahedral (model 0) and thin-shell "
                     "(model 1) deformables: " +
                     deformable.scene_path;
            return false;
        }
        if (!deformable.global_transform.matrix().allFinite() ||
            !std::all_of(deformable.vertices.begin(),
                         deformable.vertices.end(),
                         [](const Vector3& vertex) { return vertex.allFinite(); })) {
            *error = "SuperDex deformable transforms and vertices must be finite: " +
                     deformable.scene_path;
            return false;
        }
        if (!(deformable.density > 0.0) || !(deformable.young_modulus > 0.0) ||
            !(deformable.poisson_ratio > -1.0 && deformable.poisson_ratio < 0.5) ||
            !(deformable.damping >= 0.0) ||
            !std::isfinite(deformable.density) ||
            !std::isfinite(deformable.young_modulus) ||
            !std::isfinite(deformable.poisson_ratio) ||
            !std::isfinite(deformable.damping)) {
            *error = "SuperDex deformable material parameters are invalid: " + deformable.scene_path;
            return false;
        }
        const auto indices_are_valid = [&deformable](const auto& indices) {
            return std::all_of(indices.begin(), indices.end(), [&deformable](std::uint32_t index) {
                return index < deformable.vertices.size();
            });
        };
        if (deformable.model == 0 &&
            (deformable.tetrahedra.empty() || deformable.tetrahedra.size() % 4 != 0 ||
             !indices_are_valid(deformable.tetrahedra))) {
            *error = "SuperDex volumetric deformable connectivity is invalid: " +
                     deformable.scene_path;
            return false;
        }
        if (deformable.model == 1 &&
            (deformable.surface_triangles.empty() ||
             deformable.surface_triangles.size() % 3 != 0 ||
             !indices_are_valid(deformable.surface_triangles) ||
             !(deformable.thickness > 0.0) ||
             !(deformable.bending_stiffness >= 0.0) ||
             !std::isfinite(deformable.thickness) ||
             !std::isfinite(deformable.bending_stiffness))) {
            *error = "SuperDex thin-shell geometry or material parameters are invalid: " +
                     deformable.scene_path;
            return false;
        }
    }
    return true;
}

std::vector<std::size_t> SortRobotLinks(const PhysicsRobotSnapshot& robot, std::string* error) {
    std::unordered_map<std::string, std::size_t> links_by_name;
    for (std::size_t index = 0; index < robot.links.size(); ++index) {
        links_by_name[robot.links[index].name] = index;
    }
    std::vector<int> indegree(robot.links.size(), 0);
    std::vector<std::vector<std::size_t>> children(robot.links.size());
    for (const PhysicsJointSnapshot& joint : robot.joints) {
        if (joint.child_link.empty()) {
            continue;
        }
        const auto child = links_by_name.find(joint.child_link);
        if (child == links_by_name.end()) {
            *error = "SuperDex joint references an unknown child link: " + joint.scene_path;
            return {};
        }
        if (!joint.parent_link.empty()) {
            const auto parent = links_by_name.find(joint.parent_link);
            if (parent == links_by_name.end()) {
                *error = "SuperDex joint references an unknown parent link: " + joint.scene_path;
                return {};
            }
            ++indegree[child->second];
            children[parent->second].push_back(child->second);
        }
    }
    std::deque<std::size_t> queue;
    for (std::size_t index = 0; index < indegree.size(); ++index) {
        if (indegree[index] == 0) {
            queue.push_back(index);
        }
    }
    std::vector<std::size_t> result;
    while (!queue.empty()) {
        const std::size_t link = queue.front();
        queue.pop_front();
        result.push_back(link);
        for (const std::size_t child : children[link]) {
            if (--indegree[child] == 0) {
                queue.push_back(child);
            }
        }
    }
    if (result.size() != robot.links.size()) {
        *error = "SuperDex requires each Robot3D articulation to be an acyclic link tree: " +
                 robot.scene_path;
        return {};
    }
    return result;
}

} // namespace

SuperDexPhysicsWorld::SuperDexPhysicsWorld()
    : impl_(std::make_unique<Impl>()) {
}

SuperDexPhysicsWorld::~SuperDexPhysicsWorld() = default;

PhysicsBackendType SuperDexPhysicsWorld::GetBackendType() const {
    return PhysicsBackendType::SuperDex;
}

bool SuperDexPhysicsWorld::IsAvailable() const {
    return true;
}

const std::string& SuperDexPhysicsWorld::GetLastError() const {
    return last_error_;
}

PhysicsBackendCapabilities SuperDexPhysicsWorld::GetCapabilities() const {
    PhysicsBackendCapabilities capabilities;
    capabilities.deterministic_fixed_step = true;
    capabilities.runtime_checkpoint = true;
    capabilities.exact_contact_wrench = true;
    capabilities.solver_substeps = true;
    capabilities.environment_batch = false;
    capabilities.masked_reset = false;
    capabilities.device_native = false;
    capabilities.graph_capture = false;
    capabilities.sensor_batch = false;
    return capabilities;
}

PhysicsSolverDiagnostics SuperDexPhysicsWorld::GetSolverDiagnostics() const {
    return impl_->diagnostics;
}

bool SuperDexPhysicsWorld::Build(PhysicsSceneSnapshot scene_snapshot) {
    std::string validation_error;
    if (!IsFiniteSnapshot(scene_snapshot, &validation_error)) {
        SetLastError(std::move(validation_error));
        return false;
    }
    const SuperDexSolverSettings& solver_settings = settings_.superdex_solver;
#if !GOBOT_SUPERDEX_HAS_STEP_PROFILING
    if (solver_settings.record_solver_timings) {
        SetLastError("SuperDex stage profiling requires rebuilding/installing the SDK with step_profiling.h support.");
        return false;
    }
#endif
    if (solver_settings.execution_mode == SuperDexExecutionMode::Cuda) {
#if !GOBOT_SUPERDEX_HAS_CUDA
        SetLastError("SuperDex CUDA execution was requested, but this build contains only the CPU solver.");
        return false;
#endif
    }
    if (solver_settings.newton_iterations <= 0 ||
        solver_settings.line_search_iterations <= 0 ||
        solver_settings.linear_iterations < -1 ||
        solver_settings.substeps <= 0) {
        SetLastError("SuperDex solver iteration counts and substeps are invalid.");
        return false;
    }

    impl_->Release();
    if (!PhysicsWorld::Build(std::move(scene_snapshot))) {
        return false;
    }
    const auto validate_snapshot_storage = [this](std::string_view stage) {
        for (const PhysicsRobotSnapshot& robot : scene_snapshot_.robots) {
            if (robot.standalone_rigid_body && robot.links.size() != 1) {
                SetLastError("SuperDex snapshot storage changed unexpectedly after " +
                             std::string(stage) + ": '" + robot.name + "' has " +
                             std::to_string(robot.links.size()) + " link(s).");
                return false;
            }
        }
        return true;
    };
    if (!validate_snapshot_storage("the backend-neutral build")) {
        return false;
    }

    impl_->context = mochi::CreateContext(0);
    if (impl_->context == nullptr) {
        SetLastError("SuperDex failed to create a Mochi CPU context.");
        return false;
    }
    if (!validate_snapshot_storage("Mochi context creation")) {
        impl_->Release();
        return false;
    }
    impl_->scene = impl_->context->CreateScene("Gobot SuperDex World");
    if (impl_->scene == nullptr) {
        SetLastError("SuperDex failed to create a Mochi scene.");
        impl_->Release();
        return false;
    }
    if (!validate_snapshot_storage("Mochi scene creation")) {
        impl_->Release();
        return false;
    }
    impl_->scene->SetGravity(ToMochi(settings_.gravity));
#if GOBOT_SUPERDEX_HAS_STEP_PROFILING
    mochi::SetStepProfilingEnabled(*impl_->scene, solver_settings.record_solver_timings);
#endif
    mochi::SolverParams mochi_solver = impl_->scene->GetSolverParams();
    mochi_solver.nonLinearSolver.maxIter = solver_settings.newton_iterations;
    mochi_solver.nonLinearSolver.lineSearchMaxIter = solver_settings.line_search_iterations;
    // Convergence is reported through PhysicsSolverDiagnostics. Keep normal
    // headless runs machine-readable instead of emitting a warning per step.
    mochi_solver.nonLinearSolver.verbosity = mochi::VerbosityLevel::Error;
    mochi_solver.linearSolver.maxIter = solver_settings.linear_iterations;
    mochi_solver.linearSolver.verbosity = mochi::VerbosityLevel::Error;
    if (solver_settings.linear_solver == SuperDexLinearSolver::CG) {
        mochi_solver.linearSolver.solverType = mochi::LinearSolverType::CG;
    } else if (solver_settings.linear_solver == SuperDexLinearSolver::GMRES) {
        mochi_solver.linearSolver.solverType = mochi::LinearSolverType::GMRES;
    }
    mochi::Error solver_error;
    impl_->scene->SetSolverParams(mochi_solver, solver_error);
    if (!solver_error.IsOK()) {
        SetLastError("SuperDex rejected solver settings: " + solver_error.ToString());
        impl_->Release();
        return false;
    }
    if (!validate_snapshot_storage("Mochi solver configuration")) {
        impl_->Release();
        return false;
    }

    impl_->robots.resize(scene_snapshot_.robots.size());
    for (std::size_t robot_index = 0; robot_index < scene_snapshot_.robots.size(); ++robot_index) {
        const PhysicsRobotSnapshot& robot = scene_snapshot_.robots[robot_index];
        Impl::RobotBinding& binding = impl_->robots[robot_index];
        binding.links.resize(robot.links.size());
        binding.joints.resize(robot.joints.size());

        if (robot.standalone_rigid_body) {
            if (robot.links.size() != 1) {
                SetLastError("A standalone SuperDex RigidBody3D must compile to exactly one link; '" +
                             robot.name + "' has " + std::to_string(robot.links.size()) + ".");
                impl_->Release();
                return false;
            }
            const PhysicsLinkSnapshot& link = robot.links.front();
            mochi::Error error;
            const mochi::ShapeHandle shape = impl_->CreateLinkShape(link, error);
            if (!error.IsOK() || !shape.IsValid()) {
                SetLastError("SuperDex failed to create rigid shape '" + link.scene_path + "': " +
                             error.ToString());
                impl_->Release();
                return false;
            }
            const auto [layer, mask] = LinkLayerAndMask(link);
            const PhysicsShapeSnapshot* first_shape = FirstEnabledShape(link);
            mochi::RigidActorParams params;
            params.name = robot.name;
            params.layer = superdex::MakeContactLayerName(layer, mask);
            params.shape = shape;
            params.colliderType = ColliderTypeForLink(link);
            params.worldFromLocal = ToMochi(link.global_transform);
            params.isStatic = link.mass <= 0.0;
            if (!params.isStatic) {
                params.mass = static_cast<mochi::real>(link.mass);
                params.centerOfMass = ToMochi(link.center_of_mass);
                params.momentOfInertia = ToMochiInertia(link);
            }
            params.contact = first_shape != nullptr
                    ? ToMochiContact(first_shape->material,
                                     first_shape->contact_offset,
                                     first_shape->rest_offset)
                    : mochi::ContactParams{};
            mochi::Actor* actor = impl_->scene->CreateRigidActor(params, error);
            if (!error.IsOK() || actor == nullptr) {
                SetLastError("SuperDex failed to create rigid actor '" + robot.scene_path + "': " +
                             error.ToString());
                impl_->Release();
                return false;
            }
            binding.actor = actor;
            binding.links[0] = {actor, 0};
            impl_->AddMetadata(actor, robot.name, link.name, first_shape);
            impl_->AddLayer(layer, mask);
            if (!impl_->RegisterQueries(actor, false, &validation_error)) {
                SetLastError("SuperDex failed to register rigid contact queries: " + validation_error);
                impl_->Release();
                return false;
            }
            continue;
        }

        const std::vector<std::size_t> sorted_links = SortRobotLinks(robot, &validation_error);
        if (sorted_links.size() != robot.links.size()) {
            SetLastError(std::move(validation_error));
            impl_->Release();
            return false;
        }
        std::unordered_map<std::string, std::size_t> original_link_by_name;
        std::vector<int> mochi_index_by_original(robot.links.size(), -1);
        for (std::size_t index = 0; index < robot.links.size(); ++index) {
            original_link_by_name[robot.links[index].name] = index;
        }
        for (std::size_t index = 0; index < sorted_links.size(); ++index) {
            mochi_index_by_original[sorted_links[index]] = static_cast<int>(index);
        }
        std::vector<int> incoming_joint(robot.links.size(), -1);
        for (std::size_t joint_index = 0; joint_index < robot.joints.size(); ++joint_index) {
            const auto child = original_link_by_name.find(robot.joints[joint_index].child_link);
            if (child != original_link_by_name.end()) {
                if (incoming_joint[child->second] >= 0) {
                    SetLastError("SuperDex requires one incoming joint per articulated link: " +
                                 robot.joints[joint_index].child_link);
                    impl_->Release();
                    return false;
                }
                incoming_joint[child->second] = static_cast<int>(joint_index);
            }
        }

        mochi::ArticulatedActorParams params;
        params.name = robot.name;
        params.worldFromRoot = mochi::TransformRT::Identity();
        params.links.resize(sorted_links.size());
        params.joints.resize(sorted_links.size());
        for (std::size_t mochi_index = 0; mochi_index < sorted_links.size(); ++mochi_index) {
            const std::size_t original_index = sorted_links[mochi_index];
            const PhysicsLinkSnapshot& link = robot.links[original_index];
            mochi::ArticulatedLinkParams& link_params = params.links[mochi_index];
            mochi::ArticulatedJointParams& joint_params = params.joints[mochi_index];
            link_params.name = link.name;
            joint_params.name = "root_" + link.name;

            const int authored_joint_index = incoming_joint[original_index];
            const PhysicsJointSnapshot* joint = authored_joint_index >= 0
                    ? &robot.joints[static_cast<std::size_t>(authored_joint_index)]
                    : nullptr;
            if (joint != nullptr) {
                joint_params.name = joint->name;
                joint_params.type = ToMochiJointType(static_cast<JointType>(joint->joint_type));
                if (joint_params.type == mochi::ArticulatedJointType::Invalid) {
                    SetLastError("SuperDex encountered an unsupported joint: " + joint->scene_path);
                    impl_->Release();
                    return false;
                }
                joint_params.axis = ToMochi(joint->axis.normalized());
                joint_params.friction.viscous = static_cast<mochi::real>(
                        std::max<RealType>(0.0, joint->damping));
                joint_params.friction.coulomb = static_cast<mochi::real>(
                        std::max<RealType>(0.0, joint->friction_loss));
                if (joint->armature > 0.0) {
                    joint_params.inertia = static_cast<mochi::real>(joint->armature);
                }
                if (static_cast<JointType>(joint->joint_type) != JointType::Continuous &&
                    joint->upper_limit > joint->lower_limit &&
                    (joint_params.type == mochi::ArticulatedJointType::Revolute ||
                     joint_params.type == mochi::ArticulatedJointType::Prismatic)) {
                    const mochi::Real3 axis = joint_params.axis;
                    const RealType lower_limit =
                            joint->lower_limit - joint->joint_position;
                    const RealType upper_limit =
                            joint->upper_limit - joint->joint_position;
                    joint_params.minLimit = mochi::Real3{
                            axis[0] * static_cast<mochi::real>(lower_limit),
                            axis[1] * static_cast<mochi::real>(lower_limit),
                            axis[2] * static_cast<mochi::real>(lower_limit)};
                    joint_params.maxLimit = mochi::Real3{
                            axis[0] * static_cast<mochi::real>(upper_limit),
                            axis[1] * static_cast<mochi::real>(upper_limit),
                            axis[2] * static_cast<mochi::real>(upper_limit)};
                }
                joint_params.parentLinkFromJoint = joint->parent_link.empty()
                        ? ToMochi(joint->global_transform)
                        : ToMochi(robot.links[original_link_by_name.at(joint->parent_link)]
                                          .global_transform.inverse() *
                                  joint->global_transform);
                link_params.parentJointFromLink = ToMochi(
                        joint->global_transform.inverse() * link.global_transform);
                if (!joint->parent_link.empty()) {
                    link_params.parentLink = mochi_index_by_original[
                            original_link_by_name.at(joint->parent_link)];
                }
            } else {
                joint_params.type = mochi::ArticulatedJointType::Hard;
                joint_params.parentLinkFromJoint = mochi::TransformRT::Identity();
                link_params.parentJointFromLink = ToMochi(link.global_transform);
                link_params.parentLink = -1;
            }

            mochi::Error shape_error;
            link_params.shape = impl_->CreateLinkShape(link, shape_error);
            if (!shape_error.IsOK()) {
                SetLastError("SuperDex failed to create articulated link shape '" + link.scene_path +
                             "': " + shape_error.ToString());
                impl_->Release();
                return false;
            }
            link_params.colliderType = ColliderTypeForLink(
                    link, robot.joints.empty());
            const auto [layer, mask] = LinkLayerAndMask(link);
            const PhysicsShapeSnapshot* first_shape = FirstEnabledShape(link);
            link_params.layer = superdex::MakeContactLayerName(layer, mask);
            link_params.contact = first_shape != nullptr
                    ? ToMochiContact(first_shape->material,
                                     first_shape->contact_offset,
                                     first_shape->rest_offset)
                    : mochi::ContactParams{};
            if (link.mass > 0.0) {
                link_params.mass = static_cast<mochi::real>(link.mass);
                link_params.centerOfMass = ToMochi(link.center_of_mass);
                link_params.momentOfInertia = ToMochiInertia(link);
            }
            impl_->AddLayer(layer, mask);
        }

        mochi::Error actor_error;
        mochi::Actor* actor = impl_->scene->CreateArticulatedActor(params, actor_error);
        if (!actor_error.IsOK() || actor == nullptr) {
            SetLastError("SuperDex failed to create articulated actor '" + robot.scene_path + "': " +
                         actor_error.ToString());
            impl_->Release();
            return false;
        }
        binding.actor = actor;
        binding.articulated = true;
        const mochi::Span<mochi::ActorHandle const> nested_handles =
                actor->GetNestedLinkActors(actor_error);
        const mochi::ArticulatedShapeInfo shape_info = actor->GetArticulatedShapeInfo(actor_error);
        if (!actor_error.IsOK() || nested_handles.size() != sorted_links.size() ||
            shape_info.dofInfo.size() != sorted_links.size()) {
            SetLastError("SuperDex articulated actor returned inconsistent link topology: " +
                         actor_error.ToString());
            impl_->Release();
            return false;
        }
        for (std::size_t mochi_index = 0; mochi_index < sorted_links.size(); ++mochi_index) {
            const std::size_t original_index = sorted_links[mochi_index];
            mochi::Actor* link_actor = impl_->scene->GetActor(nested_handles[mochi_index]);
            if (link_actor == nullptr) {
                SetLastError("SuperDex failed to resolve a nested articulated link actor.");
                impl_->Release();
                return false;
            }
            binding.links[original_index] = {link_actor, static_cast<int>(mochi_index)};
            const PhysicsLinkSnapshot& link = robot.links[original_index];
            impl_->AddMetadata(link_actor, robot.name, link.name, FirstEnabledShape(link));
            if (!impl_->RegisterQueries(link_actor, false, &validation_error)) {
                SetLastError("SuperDex failed to register articulated contact queries: " +
                             validation_error);
                impl_->Release();
                return false;
            }
            const int authored_joint_index = incoming_joint[original_index];
            if (authored_joint_index >= 0) {
                Impl::JointBinding& joint_binding =
                        binding.joints[static_cast<std::size_t>(authored_joint_index)];
                const mochi::ArticulatedDofInfo& dof_info = shape_info.dofInfo[mochi_index];
                joint_binding.mochi_joint_index = static_cast<int>(mochi_index);
                joint_binding.dof_offset = dof_info.offset;
                joint_binding.dof_size = dof_info.GetSize();
                JointControllerGains gains = settings_.default_joint_gains;
                const PhysicsJointSnapshot& joint =
                        robot.joints[static_cast<std::size_t>(authored_joint_index)];
                joint_binding.position_offset = joint.joint_position;
                if (joint.drive_stiffness > 0.0) {
                    gains.position_stiffness = joint.drive_stiffness;
                }
                if (joint.drive_damping > 0.0) {
                    gains.velocity_damping = joint.drive_damping;
                }
                joint_binding.controller.SetGains(gains);
                joint_binding.controller.SetActuatorModel(joint.actuator_model);
            }
        }

        std::vector<mochi::real> pose(static_cast<std::size_t>(actor->GetNumDofs()),
                                      mochi::real{0});
        std::vector<mochi::real> velocity(pose.size(), mochi::real{0});
        // Link and joint frames were compiled from the SceneTree's current
        // authored pose. Mochi therefore starts at relative zero; Gobot's
        // absolute joint coordinate is restored through position_offset.
        actor->SetArticulatedPoseFromJoints(pose, actor_error);
        actor->SetArticulatedJointVelocities(velocity, actor_error);
        if (!actor_error.IsOK()) {
            SetLastError("SuperDex failed to initialize articulated state: " + actor_error.ToString());
            impl_->Release();
            return false;
        }
    }

    for (const PhysicsTerrainSnapshot& terrain : scene_snapshot_.terrains) {
        const std::string layer_name = superdex::MakeContactLayerName(
                terrain.collision_layer, terrain.collision_mask);
        impl_->AddLayer(terrain.collision_layer, terrain.collision_mask);
        const auto create_static_mesh_actor = [&](TriangleMeshBuilder mesh,
                                                  std::string_view name) -> bool {
            mochi::Error error;
            const mochi::ShapeHandle shape = impl_->CreateTriangleShape(mesh, error);
            if (!error.IsOK() || !shape.IsValid()) {
                validation_error = "SuperDex failed to create terrain mesh: " + error.ToString();
                return false;
            }
            mochi::RigidActorParams params;
            params.name = std::string(name);
            params.layer = layer_name;
            params.shape = shape;
            params.colliderType = mochi::ColliderType::Mesh;
            params.isStatic = true;
            params.contact = ToMochiContact(
                    terrain.material, terrain.contact_offset, terrain.rest_offset);
            mochi::Actor* actor = impl_->scene->CreateRigidActor(params, error);
            if (!error.IsOK() || actor == nullptr) {
                validation_error = "SuperDex failed to create terrain actor: " + error.ToString();
                return false;
            }
            impl_->AddMetadata(actor, {}, terrain.name, nullptr);
            return impl_->RegisterQueries(actor, false, &validation_error);
        };
        for (std::size_t index = 0; index < terrain.boxes.size(); ++index) {
            TriangleMeshBuilder mesh;
            mesh.AppendBox(terrain.boxes[index].global_transform, terrain.boxes[index].size);
            if (!create_static_mesh_actor(std::move(mesh), terrain.name + "_box_" + std::to_string(index))) {
                SetLastError(validation_error);
                impl_->Release();
                return false;
            }
        }
        for (std::size_t field_index = 0; field_index < terrain.heightfields.size(); ++field_index) {
            const PhysicsTerrainHeightFieldSnapshot& field = terrain.heightfields[field_index];
            TriangleMeshBuilder mesh;
            if (field.rows < 2 || field.cols < 2 ||
                field.heights.size() < static_cast<std::size_t>(field.rows * field.cols)) {
                SetLastError("SuperDex terrain heightfield has invalid dimensions.");
                impl_->Release();
                return false;
            }
            for (int row = 0; row < field.rows; ++row) {
                for (int col = 0; col < field.cols; ++col) {
                    const RealType x = -field.size.x() * 0.5 +
                                       field.size.x() * static_cast<RealType>(col) /
                                               static_cast<RealType>(field.cols - 1);
                    const RealType y = field.size.y() * 0.5 -
                                       field.size.y() * static_cast<RealType>(row) /
                                               static_cast<RealType>(field.rows - 1);
                    const RealType z = field.z_offset +
                                       field.heights[static_cast<std::size_t>(row * field.cols + col)];
                    mesh.AddVertex(field.global_transform, {x, y, z});
                }
            }
            for (int row = 0; row + 1 < field.rows; ++row) {
                for (int col = 0; col + 1 < field.cols; ++col) {
                    const int a = row * field.cols + col;
                    const int b = a + 1;
                    const int c = a + field.cols;
                    const int d = c + 1;
                    mesh.AddTriangle(a, b, d);
                    mesh.AddTriangle(a, d, c);
                }
            }
            if (!create_static_mesh_actor(
                        std::move(mesh), terrain.name + "_heightfield_" + std::to_string(field_index))) {
                SetLastError(validation_error);
                impl_->Release();
                return false;
            }
        }
        for (std::size_t patch_index = 0; patch_index < terrain.mesh_patches.size(); ++patch_index) {
            const PhysicsTerrainMeshPatchSnapshot& patch = terrain.mesh_patches[patch_index];
            TriangleMeshBuilder mesh;
            for (const Vector3& vertex : patch.vertices) {
                mesh.AddVertex(patch.global_transform, vertex);
            }
            for (std::uint32_t index : patch.indices) {
                if (index >= patch.vertices.size()) {
                    SetLastError("SuperDex terrain mesh contains an out-of-range index.");
                    impl_->Release();
                    return false;
                }
                mesh.indices.push_back(static_cast<int>(index));
            }
            if (!create_static_mesh_actor(
                        std::move(mesh), terrain.name + "_mesh_" + std::to_string(patch_index))) {
                SetLastError(validation_error);
                impl_->Release();
                return false;
            }
        }
    }

    for (std::size_t shape_index = 0;
         shape_index < scene_snapshot_.loose_collision_shapes.size(); ++shape_index) {
        const PhysicsShapeSnapshot& shape_snapshot = scene_snapshot_.loose_collision_shapes[shape_index];
        if (shape_snapshot.disabled) {
            continue;
        }
        PhysicsLinkSnapshot temporary_link;
        temporary_link.name = shape_snapshot.name;
        temporary_link.global_transform = Affine3::Identity();
        temporary_link.collision_shapes.push_back(shape_snapshot);
        mochi::Error error;
        const mochi::ShapeHandle shape = impl_->CreateLinkShape(temporary_link, error);
        if (!error.IsOK() || !shape.IsValid()) {
            SetLastError("SuperDex failed to create loose collision shape: " + error.ToString());
            impl_->Release();
            return false;
        }
        mochi::RigidActorParams params;
        params.name = shape_snapshot.name;
        params.layer = superdex::MakeContactLayerName(
                shape_snapshot.collision_layer, shape_snapshot.collision_mask);
        params.shape = shape;
        params.colliderType = ColliderTypeForLink(temporary_link);
        params.isStatic = true;
        params.contact = ToMochiContact(shape_snapshot.material,
                                        shape_snapshot.contact_offset,
                                        shape_snapshot.rest_offset);
        mochi::Actor* actor = impl_->scene->CreateRigidActor(params, error);
        if (!error.IsOK() || actor == nullptr) {
            SetLastError("SuperDex failed to create loose collision actor: " + error.ToString());
            impl_->Release();
            return false;
        }
        impl_->AddMetadata(actor, {}, {}, &shape_snapshot);
        impl_->AddLayer(shape_snapshot.collision_layer, shape_snapshot.collision_mask);
        if (!impl_->RegisterQueries(actor, false, &validation_error)) {
            SetLastError("SuperDex failed to register loose-shape contact queries: " + validation_error);
            impl_->Release();
            return false;
        }
    }

    impl_->deformables.resize(scene_snapshot_.deformables.size());
    for (std::size_t index = 0; index < scene_snapshot_.deformables.size(); ++index) {
        const PhysicsDeformableSnapshot& deformable = scene_snapshot_.deformables[index];
        std::vector<mochi::real> coordinates;
        coordinates.reserve(deformable.vertices.size() * 3);
        for (const Vector3& vertex : deformable.vertices) {
            const mochi::Real3 converted = ToMochi(vertex);
            coordinates.insert(coordinates.end(), converted.begin(), converted.end());
        }
        std::vector<int> connectivity;
        mochi::Error error;
        mochi::ShapeHandle shape;
        if (deformable.model == 0) {
            if (deformable.tetrahedra.empty() || deformable.tetrahedra.size() % 4 != 0) {
                SetLastError("SuperDex volumetric deformable requires tetrahedral connectivity: " +
                             deformable.scene_path);
                impl_->Release();
                return false;
            }
            connectivity.assign(deformable.tetrahedra.begin(), deformable.tetrahedra.end());
            shape = impl_->context->CreateTetMeshShape(coordinates, connectivity, error);
        } else {
            if (deformable.surface_triangles.empty() ||
                deformable.surface_triangles.size() % 3 != 0) {
                SetLastError("SuperDex thin shell requires triangle connectivity: " +
                             deformable.scene_path);
                impl_->Release();
                return false;
            }
            connectivity.assign(
                    deformable.surface_triangles.begin(), deformable.surface_triangles.end());
            shape = impl_->context->CreateTriMeshShape(coordinates, connectivity, error);
        }
        if (!error.IsOK() || !shape.IsValid()) {
            SetLastError("SuperDex failed to create deformable shape '" + deformable.scene_path +
                         "': " + error.ToString());
            impl_->Release();
            return false;
        }
        impl_->shapes.push_back(shape);
        const std::string layer_name = superdex::MakeContactLayerName(
                deformable.collision_layer, deformable.collision_mask);
        mochi::Actor* actor = nullptr;
        if (deformable.model == 0) {
            mochi::SoftActorParams params;
            params.name = deformable.name;
            params.layer = layer_name;
            params.worldFromLocal = ToMochi(deformable.global_transform);
            params.shape = shape;
            params.material.type = mochi::SoftMaterialType::NeoHookean;
            params.material.neoHookean.youngsModulus =
                    static_cast<mochi::real>(deformable.young_modulus);
            params.material.neoHookean.poissonRatio =
                    static_cast<mochi::real>(deformable.poisson_ratio);
            params.material.density = static_cast<mochi::real>(deformable.density);
            params.material.massDampingCoefficient =
                    static_cast<mochi::real>(std::max<RealType>(0.0, deformable.damping));
            params.contact = ToMochiContact(deformable.material);
            params.hasGravity = !deformable.kinematic;
            params.hasInertia = !deformable.kinematic;
            mochi::experimental::ExperimentalSoftActorParams experimental_params;
            // Gobot exposes vertices relative to the authored Node3D transform.
            // Mochi's default recentring moves rigid-like displacement into the
            // actor root, which would make those local vertices appear frozen.
            experimental_params.useRecentering = false;
            // A regular soft actor samples other colliders but is not itself a
            // collider. The SDF makes tetrahedral bodies participate on both
            // sides of deformable-deformable contact (for example bag fill).
            experimental_params.colliderType = mochi::ColliderType::Sdf;
            actor = mochi::experimental::CreateSoftActor(
                    impl_->scene, params, experimental_params, error);
        } else {
            mochi::experimental::ShellActorParams params;
            params.name = deformable.name;
            params.layer = layer_name;
            params.worldFromLocal = ToMochi(deformable.global_transform);
            params.shape = shape;
            params.material = mochi::experimental::ShellMaterialParamsFrom3dIsotropic(
                    static_cast<mochi::real>(deformable.young_modulus),
                    static_cast<mochi::real>(deformable.poisson_ratio),
                    static_cast<mochi::real>(deformable.density),
                    static_cast<mochi::real>(deformable.thickness),
                    error);
            const mochi::real authored_bending = static_cast<mochi::real>(
                    std::max<RealType>(0.0, deformable.bending_stiffness));
            const mochi::real generated_bending =
                    std::abs(params.material.bendingAlpha) +
                    std::abs(params.material.bendingBeta);
            if (generated_bending > mochi::real{0}) {
                const mochi::real scale = authored_bending / generated_bending;
                params.material.bendingAlpha *= scale;
                params.material.bendingBeta *= scale;
            }
            params.material.massDampingCoefficient =
                    static_cast<mochi::real>(std::max<RealType>(0.0, deformable.damping));
            params.contact = ToMochiContact(deformable.material);
            const RealType contact_radius = superdex::ComputeShellContactRadius(
                    deformable.vertices,
                    deformable.surface_triangles,
                    deformable.thickness);
            params.pointCloudCollider.radius = static_cast<mochi::real>(contact_radius);
            if (superdex::RequiresShellContactQuadrature(
                        deformable.vertices,
                        deformable.surface_triangles,
                        contact_radius)) {
                // Coarse preview meshes can leave gaps wider than the point-cloud
                // radius between nodal colliders. Q6 adds contact samples without
                // adding shell degrees of freedom.
                params.pointCloudCollider.colliderTriangleElementType =
                        mochi::ActorBoundaryElementType::P1Q6;
            }
            params.pointCloudCollider.selfContact = deformable.self_collision_enabled;
            params.hasGravity = !deformable.kinematic;
            actor = mochi::experimental::CreateShellActor(impl_->scene, params, error);
        }
        if (!error.IsOK() || actor == nullptr) {
            SetLastError("SuperDex failed to create deformable actor '" + deformable.scene_path +
                         "': " + error.ToString());
            impl_->Release();
            return false;
        }
        if (deformable.kinematic) {
            std::vector<int> node_indices(deformable.vertices.size());
            std::iota(node_indices.begin(), node_indices.end(), 0);
            std::vector<mochi::real> world_positions;
            world_positions.reserve(deformable.vertices.size() * 3);
            for (const Vector3& vertex : deformable.vertices) {
                const mochi::Real3 point = ToMochi(deformable.global_transform * vertex);
                world_positions.insert(world_positions.end(), point.begin(), point.end());
            }
            actor->AddBoundaryConditionNodesWorldPermanent(
                    node_indices, world_positions, error);
        }
        if (!error.IsOK()) {
            SetLastError("SuperDex failed to constrain kinematic deformable: " + error.ToString());
            impl_->Release();
            return false;
        }
        impl_->deformables[index] = {actor, index, deformable.vertices};
        impl_->AddMetadata(actor, {}, deformable.name, nullptr, index);
        impl_->AddLayer(deformable.collision_layer, deformable.collision_mask);
        if (!actor->IsQuerySupported(mochi::QueryType::NodePositions)) {
            SetLastError("SuperDex deformable actor does not support node-position queries: " +
                         deformable.scene_path);
            impl_->Release();
            return false;
        }
        actor->RegisterQuery(mochi::QueryType::NodePositions, error);
        if (!error.IsOK()) {
            SetLastError("SuperDex failed to register deformable node-position queries: " +
                         error.ToString());
            impl_->Release();
            return false;
        }
        if (!impl_->RegisterQueries(
                    actor,
                    solver_settings.record_deformable_contact_forces,
                    &validation_error)) {
            SetLastError("SuperDex failed to register deformable contact queries: " +
                         validation_error);
            impl_->Release();
            return false;
        }
        if (deformable.self_collision_enabled) {
            impl_->scene->EnableActorContactSymmetric(
                    actor->GetHandle(),
                    actor->GetHandle(),
                    true,
                    mochi::IncludeNestedActors::Yes,
                    error);
            if (!error.IsOK()) {
                SetLastError("SuperDex failed to enable deformable self-contact: " + error.ToString());
                impl_->Release();
                return false;
            }
        }
    }

    for (std::size_t a = 0; a < impl_->layers.size(); ++a) {
        for (std::size_t b = a; b < impl_->layers.size(); ++b) {
            const Impl::LayerInfo& layer_a = impl_->layers[a];
            const Impl::LayerInfo& layer_b = impl_->layers[b];
            mochi::Error error;
            impl_->scene->EnableLayerContactSymmetric(
                    layer_a.name,
                    layer_b.name,
                    superdex::ShouldEnableContact(
                            layer_a.layer, layer_a.mask, layer_b.layer, layer_b.mask),
                    error);
            if (!error.IsOK()) {
                SetLastError("SuperDex failed to configure contact layers: " + error.ToString());
                impl_->Release();
                return false;
            }
        }
    }

    last_error_.clear();
    Step(0.0);
    if (!last_error_.empty()) {
        impl_->Release();
        return false;
    }
    mochi::Error state_error;
    impl_->scene->CaptureStateToBytes(impl_->initial_state, state_error);
    if (!state_error.IsOK()) {
        SetLastError("SuperDex failed to capture its initial state: " + state_error.ToString());
        impl_->Release();
        return false;
    }
    return true;
}

bool SuperDexPhysicsWorld::ApplyForces(RealType delta_time) {
    if (impl_->scene == nullptr) {
        SetLastError("SuperDex world has not been built.");
        return false;
    }

    std::vector<std::vector<mochi::real>> robot_forces(impl_->robots.size());
    for (std::size_t robot_index = 0; robot_index < impl_->robots.size(); ++robot_index) {
        Impl::RobotBinding& binding = impl_->robots[robot_index];
        if (binding.actor == nullptr) {
            continue;
        }
        binding.actor->ClearExternalForces();
        robot_forces[robot_index].assign(
                static_cast<std::size_t>(binding.actor->GetNumDofs()), mochi::real{0});

        if (!binding.articulated || robot_index >= scene_state_.robots.size() ||
            robot_index >= scene_snapshot_.robots.size()) {
            continue;
        }
        PhysicsRobotState& robot_state = scene_state_.robots[robot_index];
        const PhysicsRobotSnapshot& robot_snapshot = scene_snapshot_.robots[robot_index];
        for (std::size_t joint_index = 0;
             joint_index < binding.joints.size() &&
             joint_index < robot_state.joints.size() &&
             joint_index < robot_snapshot.joints.size();
             ++joint_index) {
            Impl::JointBinding& joint_binding = binding.joints[joint_index];
            if (joint_binding.dof_size != 1 || joint_binding.dof_offset < 0 ||
                static_cast<std::size_t>(joint_binding.dof_offset) >=
                        robot_forces[robot_index].size()) {
                continue;
            }
            PhysicsJointState& joint_state = robot_state.joints[joint_index];
            const RealType effort = joint_binding.controller.ComputeEffort(
                    MakeJointControllerState(joint_state),
                    MakeJointControllerCommand(joint_state),
                    MakeJointControllerLimits(robot_snapshot.joints[joint_index]),
                    delta_time);
            robot_forces[robot_index][static_cast<std::size_t>(joint_binding.dof_offset)] +=
                    static_cast<mochi::real>(effort);

            const JointControllerTelemetry& telemetry = joint_binding.controller.GetTelemetry();
            joint_state.commanded_target = telemetry.commanded_target;
            joint_state.applied_target = telemetry.applied_target;
            joint_state.tracking_error = telemetry.tracking_error;
            joint_state.applied_effort = telemetry.applied_effort;
            joint_state.command_delayed = telemetry.delayed;
            joint_state.command_deadband_applied = telemetry.deadband_applied;
            joint_state.command_rate_limited = telemetry.rate_limited;
            joint_state.effort_saturated = telemetry.effort_saturated;
        }
    }
    for (Impl::DeformableBinding& binding : impl_->deformables) {
        if (binding.actor != nullptr) {
            binding.actor->ClearExternalForces();
        }
    }

    for (const PhysicsExternalForce& external_force : external_forces_) {
        std::size_t robot_index = kInvalidIndex;
        std::size_t link_index = kInvalidIndex;
        for (std::size_t candidate_robot = 0;
             candidate_robot < scene_snapshot_.robots.size();
             ++candidate_robot) {
            if (scene_snapshot_.robots[candidate_robot].name != external_force.robot_name) {
                continue;
            }
            robot_index = candidate_robot;
            for (std::size_t candidate_link = 0;
                 candidate_link < scene_snapshot_.robots[candidate_robot].links.size();
                 ++candidate_link) {
                if (scene_snapshot_.robots[candidate_robot].links[candidate_link].name ==
                    external_force.link_name) {
                    link_index = candidate_link;
                    break;
                }
            }
            break;
        }
        if (robot_index >= impl_->robots.size() || link_index == kInvalidIndex ||
            link_index >= impl_->robots[robot_index].links.size() ||
            robot_index >= scene_state_.robots.size() ||
            link_index >= scene_state_.robots[robot_index].links.size()) {
            SetLastError("SuperDex external force references a missing robot link.");
            return false;
        }

        Impl::RobotBinding& robot_binding = impl_->robots[robot_index];
        mochi::Actor* link_actor = robot_binding.links[link_index].actor;
        if (link_actor == nullptr || link_actor->IsStatic()) {
            SetLastError("SuperDex cannot apply an external force to static or missing link '" +
                         external_force.robot_name + "::" + external_force.link_name + "'.");
            return false;
        }
        const PhysicsLinkSnapshot& link_snapshot =
                scene_snapshot_.robots[robot_index].links[link_index];
        const PhysicsLinkState& link_state = scene_state_.robots[robot_index].links[link_index];
        const Vector3 center_of_mass =
                link_state.global_transform * link_snapshot.center_of_mass;
        const Vector3 world_point = external_force.use_spring
                ? link_state.global_transform * external_force.local_point
                : external_force.point;

        Vector3 force = external_force.force;
        if (external_force.use_spring) {
            const Vector3 moment_arm = world_point - center_of_mass;
            const Vector3 point_velocity = link_state.linear_velocity +
                                           link_state.angular_velocity.cross(moment_arm);
            const RealType mass = link_snapshot.mass > CMP_EPSILON
                    ? link_snapshot.mass
                    : RealType{1};
            constexpr RealType kSpringStiffness = 100.0;
            constexpr RealType kSpringDamping = 10.0;
            force = (external_force.target_point - world_point) *
                            (kSpringStiffness * mass) -
                    point_velocity * (kSpringDamping * mass);
            if (external_force.force.squaredNorm() > CMP_EPSILON2) {
                const Vector3 axis = external_force.force.normalized();
                force = axis * force.dot(axis);
            }
        }
        const Vector3 torque = (world_point - center_of_mass).cross(force);
        const mochi::Real3 force_mochi = ToMochi(force);
        const mochi::Real3 torque_mochi = ToMochi(torque);
        const std::array<mochi::real, 6> wrench = {
                force_mochi[0], force_mochi[1], force_mochi[2],
                torque_mochi[0], torque_mochi[1], torque_mochi[2]};

        std::vector<mochi::real>& generalized = robot_forces[robot_index];
        if (!robot_binding.articulated) {
            if (generalized.size() < wrench.size()) {
                SetLastError("SuperDex rigid actor exposes an invalid force layout.");
                return false;
            }
            for (std::size_t index = 0; index < wrench.size(); ++index) {
                generalized[index] += wrench[index];
            }
            continue;
        }

        mochi::Error error;
        const mochi::Span<mochi::real const> jacobian =
                link_actor->GetArticulatedJacobian(error);
        if (!error.IsOK() || jacobian.size() != generalized.size() * wrench.size()) {
            SetLastError("SuperDex failed to map a link wrench through its articulated Jacobian: " +
                         error.ToString());
            return false;
        }
        for (std::size_t dof = 0; dof < generalized.size(); ++dof) {
            for (std::size_t row = 0; row < wrench.size(); ++row) {
                generalized[dof] += jacobian[row * generalized.size() + dof] * wrench[row];
            }
        }
    }

    for (std::size_t robot_index = 0; robot_index < impl_->robots.size(); ++robot_index) {
        mochi::Actor* actor = impl_->robots[robot_index].actor;
        std::vector<mochi::real>& forces = robot_forces[robot_index];
        if (actor == nullptr || forces.empty() ||
            std::none_of(forces.begin(), forces.end(), [](mochi::real value) {
                return value != mochi::real{0};
            })) {
            continue;
        }
        std::vector<int> dofs(forces.size());
        std::iota(dofs.begin(), dofs.end(), 0);
        mochi::Error error;
        actor->SetExternalForcesOnDofs(dofs, forces, error);
        if (!error.IsOK()) {
            SetLastError("SuperDex failed to apply rigid or articulated forces: " +
                         error.ToString());
            return false;
        }
    }

    for (const PhysicsDeformableExternalForces& external_forces :
         deformable_external_forces_) {
        const auto snapshot = std::find_if(
                scene_snapshot_.deformables.begin(),
                scene_snapshot_.deformables.end(),
                [&external_forces](const PhysicsDeformableSnapshot& candidate) {
                    return candidate.stable_id == external_forces.stable_id;
                });
        if (snapshot == scene_snapshot_.deformables.end()) {
            SetLastError("SuperDex deformable force references an unknown stable ID.");
            return false;
        }
        const std::size_t deformable_index = static_cast<std::size_t>(
                std::distance(scene_snapshot_.deformables.begin(), snapshot));
        if (deformable_index >= impl_->deformables.size() ||
            impl_->deformables[deformable_index].actor == nullptr ||
            external_forces.forces.size() != snapshot->vertices.size()) {
            SetLastError("SuperDex deformable force payload has incompatible dimensions.");
            return false;
        }
        mochi::Actor* actor = impl_->deformables[deformable_index].actor;
        const std::size_t force_dof_count = external_forces.forces.size() * 3;
        if (static_cast<std::size_t>(actor->GetNumDofs()) < force_dof_count) {
            SetLastError("SuperDex deformable actor exposes fewer DOFs than authored vertices.");
            return false;
        }
        std::vector<int> dofs(force_dof_count);
        std::iota(dofs.begin(), dofs.end(), 0);
        std::vector<mochi::real> forces;
        forces.reserve(force_dof_count);
        for (const Vector3& force : external_forces.forces) {
            const mochi::Real3 converted = ToMochi(force);
            forces.insert(forces.end(), converted.begin(), converted.end());
        }
        mochi::Error error;
        actor->SetExternalForcesOnDofs(dofs, forces, error);
        if (!error.IsOK()) {
            SetLastError("SuperDex failed to apply deformable nodal forces: " + error.ToString());
            return false;
        }
    }

    return true;
}

bool SuperDexPhysicsWorld::PushStateToSuperDex() {
    if (impl_->scene == nullptr) {
        SetLastError("SuperDex world has not been built.");
        return false;
    }

    for (std::size_t robot_index = 0; robot_index < impl_->robots.size(); ++robot_index) {
        if (robot_index >= scene_state_.robots.size() ||
            robot_index >= scene_snapshot_.robots.size()) {
            SetLastError("SuperDex robot state dimensions do not match the compiled scene.");
            return false;
        }
        Impl::RobotBinding& binding = impl_->robots[robot_index];
        PhysicsRobotState& robot_state = scene_state_.robots[robot_index];
        const PhysicsRobotSnapshot& robot_snapshot = scene_snapshot_.robots[robot_index];
        if (binding.actor == nullptr) {
            continue;
        }
        mochi::Error error;
        if (!binding.articulated) {
            if (robot_state.links.empty()) {
                SetLastError("SuperDex standalone rigid body is missing link state.");
                return false;
            }
            binding.actor->SetRootTransform(ToMochi(robot_state.links.front().global_transform), error);
            if (error.IsOK()) {
                binding.actor->SetVelocity(ToMochi(robot_state.links.front().linear_velocity),
                                           ToMochi(robot_state.links.front().angular_velocity),
                                           error);
            }
            if (!error.IsOK()) {
                SetLastError("SuperDex failed to write rigid state: " + error.ToString());
                return false;
            }
            continue;
        }

        std::vector<mochi::TransformRT> link_transforms(binding.links.size());
        for (std::size_t original_index = 0; original_index < binding.links.size(); ++original_index) {
            const Impl::LinkBinding& link_binding = binding.links[original_index];
            if (link_binding.mochi_link_index < 0 ||
                original_index >= robot_state.links.size()) {
                SetLastError("SuperDex articulated link mapping is incomplete.");
                return false;
            }
            link_transforms[static_cast<std::size_t>(link_binding.mochi_link_index)] =
                    ToMochi(robot_state.links[original_index].global_transform);
        }
        binding.actor->SetArticulatedPoseFromLinks(link_transforms, error);
        if (!error.IsOK()) {
            SetLastError("SuperDex failed to write articulated link transforms: " + error.ToString());
            return false;
        }

        std::vector<mochi::real> pose(
                static_cast<std::size_t>(binding.actor->GetNumDofs()), mochi::real{0});
        std::vector<mochi::real> velocity(pose.size(), mochi::real{0});
        binding.actor->GetArticulatedPose(pose, error);
        if (error.IsOK()) {
            binding.actor->GetArticulatedJointVelocities(velocity, error);
        }
        if (!error.IsOK()) {
            SetLastError("SuperDex failed to read articulated state before writing it: " +
                         error.ToString());
            return false;
        }
        for (std::size_t joint_index = 0;
             joint_index < binding.joints.size() &&
             joint_index < robot_state.joints.size() &&
             joint_index < robot_snapshot.joints.size();
             ++joint_index) {
            const Impl::JointBinding& joint_binding = binding.joints[joint_index];
            if (joint_binding.dof_offset < 0) {
                continue;
            }
            const std::size_t offset = static_cast<std::size_t>(joint_binding.dof_offset);
            if (joint_binding.dof_size == 1 && offset < pose.size()) {
                pose[offset] = static_cast<mochi::real>(
                        robot_state.joints[joint_index].position -
                        joint_binding.position_offset);
                velocity[offset] = static_cast<mochi::real>(robot_state.joints[joint_index].velocity);
            } else if (joint_binding.dof_size == 6 && offset + 5 < velocity.size()) {
                const std::string& child_link = robot_snapshot.joints[joint_index].child_link;
                const auto link = std::find_if(
                        robot_state.links.begin(),
                        robot_state.links.end(),
                        [&child_link](const PhysicsLinkState& candidate) {
                            return candidate.link_name == child_link;
                        });
                if (link != robot_state.links.end()) {
                    const mochi::Real3 linear = ToMochi(link->linear_velocity);
                    const mochi::Real3 angular = ToMochi(link->angular_velocity);
                    for (std::size_t axis = 0; axis < 3; ++axis) {
                        velocity[offset + axis] = linear[axis];
                        velocity[offset + 3 + axis] = angular[axis];
                    }
                }
            }
        }
        binding.actor->SetArticulatedPoseFromJoints(pose, error);
        if (error.IsOK()) {
            binding.actor->SetArticulatedJointVelocities(velocity, error);
        }
        if (!error.IsOK()) {
            SetLastError("SuperDex failed to write articulated joint state: " + error.ToString());
            return false;
        }
    }

    for (std::size_t index = 0; index < impl_->deformables.size(); ++index) {
        if (index >= scene_state_.deformables.size() ||
            impl_->deformables[index].actor == nullptr) {
            SetLastError("SuperDex deformable state dimensions do not match the compiled scene.");
            return false;
        }
        const PhysicsDeformableState& state = scene_state_.deformables[index];
        std::vector<mochi::real> positions;
        positions.reserve(state.local_vertices.size() * 3);
        for (const Vector3& vertex : state.local_vertices) {
            const mochi::Real3 converted = ToMochi(vertex);
            positions.insert(positions.end(), converted.begin(), converted.end());
        }
        std::vector<mochi::real> velocities;
        velocities.reserve(state.local_velocities.size() * 3);
        for (const Vector3& velocity : state.local_velocities) {
            const mochi::Real3 converted = ToMochi(velocity);
            velocities.insert(velocities.end(), converted.begin(), converted.end());
        }
        if (velocities.empty()) {
            velocities.assign(positions.size(), mochi::real{0});
        }
        mochi::Error error;
        impl_->deformables[index].actor->SetNodePositionsLocal(positions, error);
        if (error.IsOK()) {
            impl_->deformables[index].actor->SetNodeVelocitiesLocal(velocities, error);
        }
        if (!error.IsOK()) {
            SetLastError("SuperDex failed to write deformable state: " + error.ToString());
            return false;
        }
        impl_->deformables[index].previous_vertices = state.local_vertices;
    }

    return true;
}

bool SuperDexPhysicsWorld::SyncStateFromSuperDex(RealType delta_time) {
    if (impl_->scene == nullptr) {
        SetLastError("SuperDex world has not been built.");
        return false;
    }

    for (std::size_t robot_index = 0; robot_index < impl_->robots.size(); ++robot_index) {
        if (robot_index >= scene_state_.robots.size()) {
            SetLastError("SuperDex robot state dimensions do not match the compiled scene.");
            return false;
        }
        Impl::RobotBinding& binding = impl_->robots[robot_index];
        PhysicsRobotState& robot_state = scene_state_.robots[robot_index];
        mochi::Error error;
        for (std::size_t link_index = 0;
             link_index < binding.links.size() && link_index < robot_state.links.size();
             ++link_index) {
            mochi::Actor* actor = binding.links[link_index].actor;
            if (actor == nullptr) {
                continue;
            }
            robot_state.links[link_index].global_transform = FromMochi(actor->GetRootTransform());
            robot_state.links[link_index].linear_velocity = FromMochi(actor->GetLinearVelocity(error));
            if (error.IsOK()) {
                robot_state.links[link_index].angular_velocity =
                        FromMochi(actor->GetAngularVelocity(error));
            }
            if (!error.IsOK()) {
                SetLastError("SuperDex failed to read link state: " + error.ToString());
                return false;
            }
        }
        if (!binding.articulated) {
            continue;
        }
        std::vector<mochi::real> pose(
                static_cast<std::size_t>(binding.actor->GetNumDofs()), mochi::real{0});
        std::vector<mochi::real> velocity(pose.size(), mochi::real{0});
        binding.actor->GetArticulatedPose(pose, error);
        if (error.IsOK()) {
            binding.actor->GetArticulatedJointVelocities(velocity, error);
        }
        if (!error.IsOK()) {
            SetLastError("SuperDex failed to read articulated joint state: " + error.ToString());
            return false;
        }
        for (std::size_t joint_index = 0;
             joint_index < binding.joints.size() && joint_index < robot_state.joints.size();
             ++joint_index) {
            const Impl::JointBinding& joint_binding = binding.joints[joint_index];
            if (joint_binding.dof_size != 1 || joint_binding.dof_offset < 0 ||
                static_cast<std::size_t>(joint_binding.dof_offset) >= pose.size()) {
                continue;
            }
            const std::size_t offset = static_cast<std::size_t>(joint_binding.dof_offset);
            robot_state.joints[joint_index].position =
                    static_cast<RealType>(pose[offset]) + joint_binding.position_offset;
            robot_state.joints[joint_index].velocity = static_cast<RealType>(velocity[offset]);
            robot_state.joints[joint_index].effort =
                    robot_state.joints[joint_index].applied_effort;
        }
    }

    for (std::size_t index = 0; index < impl_->deformables.size(); ++index) {
        if (index >= scene_state_.deformables.size() ||
            index >= scene_snapshot_.deformables.size() ||
            impl_->deformables[index].actor == nullptr) {
            SetLastError("SuperDex deformable state dimensions do not match the compiled scene.");
            return false;
        }
        Impl::DeformableBinding& binding = impl_->deformables[index];
        PhysicsDeformableState& state = scene_state_.deformables[index];
        mochi::Error error;
        const mochi::Span<mochi::real const> positions =
                binding.actor->GetNodePositionsLocal(error);
        if (!error.IsOK() || positions.size() % 3 != 0) {
            SetLastError("SuperDex failed to read deformable node positions: " + error.ToString());
            return false;
        }
        const std::size_t vertex_count = positions.size() / 3;
        if (vertex_count != scene_snapshot_.deformables[index].vertices.size()) {
            SetLastError("SuperDex deformable node count changed unexpectedly.");
            return false;
        }
        std::vector<Vector3> vertices(vertex_count);
        for (std::size_t vertex = 0; vertex < vertex_count; ++vertex) {
            vertices[vertex] = FromMochi({positions[3 * vertex],
                                          positions[3 * vertex + 1],
                                          positions[3 * vertex + 2]});
            if (!vertices[vertex].allFinite()) {
                SetLastError("SuperDex produced a non-finite deformable vertex.");
                return false;
            }
        }
        if (delta_time > 0.0) {
            state.local_velocities.assign(vertex_count, Vector3::Zero());
        } else if (state.local_velocities.size() != vertex_count) {
            state.local_velocities.assign(vertex_count, Vector3::Zero());
        }
        if (delta_time > 0.0 && binding.previous_vertices.size() == vertex_count) {
            for (std::size_t vertex = 0; vertex < vertex_count; ++vertex) {
                state.local_velocities[vertex] =
                        (vertices[vertex] - binding.previous_vertices[vertex]) / delta_time;
            }
        }
        state.local_vertices = vertices;
        binding.previous_vertices = std::move(vertices);

        state.contact_forces_world.clear();
        if (settings_.superdex_solver.record_deformable_contact_forces &&
            binding.actor->IsQuerySupported(mochi::QueryType::NodeContactForces)) {
            state.contact_forces_world.assign(vertex_count, Vector3::Zero());
            const mochi::Span<mochi::NodeContactForce const> node_forces =
                    binding.actor->GetNodeContactForcesWorld(error);
            if (!error.IsOK()) {
                SetLastError("SuperDex failed to read deformable contact forces: " +
                             error.ToString());
                return false;
            }
            for (const mochi::NodeContactForce& node_force : node_forces) {
                if (node_force.index >= 0 &&
                    static_cast<std::size_t>(node_force.index) < vertex_count) {
                    state.contact_forces_world[static_cast<std::size_t>(node_force.index)] +=
                            FromMochi(node_force.force);
                }
            }
        }
    }

    scene_state_.contacts.clear();
    std::unordered_set<std::string> recorded_contacts;
    for (mochi::Actor* query_actor : impl_->query_actors) {
        if (query_actor == nullptr ||
            !query_actor->IsQuerySupported(mochi::QueryType::ContactPoints)) {
            continue;
        }
        mochi::Error error;
        const mochi::Span<mochi::ContactPoint const> contacts =
                query_actor->GetContactPointsWorld(error);
        if (!error.IsOK()) {
            SetLastError("SuperDex failed to read contact points: " + error.ToString());
            return false;
        }
        for (const mochi::ContactPoint& contact : contacts) {
            const std::string key = std::to_string(contact.actorA.value) + ":" +
                                    std::to_string(contact.actorB.value) + ":" +
                                    std::to_string(contact.sampleIndex) + ":" +
                                    std::to_string(contact.elementIndex);
            if (!recorded_contacts.insert(key).second) {
                continue;
            }
            const auto metadata_a = impl_->metadata_by_actor.find(contact.actorA.value);
            const auto metadata_b = impl_->metadata_by_actor.find(contact.actorB.value);
            const Impl::ActorMetadata* actor_a = metadata_a == impl_->metadata_by_actor.end()
                    ? nullptr
                    : metadata_a->second;
            const Impl::ActorMetadata* actor_b = metadata_b == impl_->metadata_by_actor.end()
                    ? nullptr
                    : metadata_b->second;
            const Vector3 position =
                    (FromMochi(contact.posA) + FromMochi(contact.posB)) * RealType{0.5};
            const Vector3 normal = FromMochi(contact.normal).normalized();
            const Vector3 force = FromMochi(contact.force);
            const auto add_contact = [&](const Impl::ActorMetadata* owner,
                                         const Impl::ActorMetadata* other,
                                         RealType sign) {
                if (owner == nullptr) {
                    return;
                }
                PhysicsContactState state;
                state.robot_name = owner->robot_name;
                state.link_name = owner->link_name;
                state.shape_name = owner->shape_name;
                state.shape_path = owner->shape_path;
                state.shape_stable_id = owner->shape_stable_id;
                if (other != nullptr) {
                    state.other_robot_name = other->robot_name;
                    state.other_link_name = other->link_name;
                    state.other_shape_name = other->shape_name;
                    state.other_shape_path = other->shape_path;
                    state.other_shape_stable_id = other->shape_stable_id;
                }
                state.position = position;
                state.normal = sign * normal;
                state.force = sign * force;
                state.normal_force = std::max<RealType>(
                        0.0, state.force.dot(state.normal));
                state.tangent_force =
                        state.force - state.normal * state.normal_force;
                state.normal_impulse = state.normal_force *
                                       (delta_time > 0.0
                                                ? delta_time
                                                : settings_.fixed_time_step);
                state.distance = static_cast<RealType>(contact.distance);
                scene_state_.contacts.push_back(std::move(state));
            };
            add_contact(actor_a, actor_b, RealType{1});
            add_contact(actor_b, actor_a, RealType{-1});
        }
    }

    UpdateSensorGlobalTransformsAndRaycastSensors(
            scene_state_, static_cast<RealType>(impl_->scene->GetTotalSimulationTime()));
    return true;
}

void SuperDexPhysicsWorld::UpdateDiagnostics() {
    if (impl_->scene == nullptr) {
        impl_->diagnostics = {};
        return;
    }
    const mochi::PerformanceStats performance = impl_->scene->GetPerformanceStats();
    const mochi::SolverStats solver = impl_->scene->GetSolverStats();
    impl_->diagnostics.total_step_time_seconds = performance.totalStepDurationSec;
    impl_->diagnostics.solve_time_seconds = performance.solveStepDurationSec;
    impl_->diagnostics.newton_iterations = solver.maxNonLinearIters;
    impl_->diagnostics.line_search_iterations = solver.maxLineSearchIters;
    impl_->diagnostics.residual_norm = static_cast<RealType>(solver.residualNorm);
    impl_->diagnostics.convergence = FromMochi(solver.convergenceStatus);
    impl_->diagnostics.execution_device =
            settings_.superdex_solver.execution_mode == SuperDexExecutionMode::Cuda
                    ? "cuda-linear-solver"
                    : "cpu";
    impl_->diagnostics.device_native = false;
    impl_->diagnostics.graph_capture = false;
    impl_->diagnostics.timings_available =
            GOBOT_SUPERDEX_HAS_STEP_PROFILING && settings_.superdex_solver.record_solver_timings;
    impl_->diagnostics.linear_iterations = 0;
    impl_->diagnostics.stage_timings.clear();
}

bool SuperDexPhysicsWorld::RestoreCompatibleState(const PhysicsSceneState& previous_state) {
    if (!PhysicsWorld::RestoreCompatibleState(previous_state)) {
        return false;
    }
    if (!PushStateToSuperDex() || !SyncStateFromSuperDex(0.0)) {
        return false;
    }
    last_error_.clear();
    return true;
}

void SuperDexPhysicsWorld::Reset() {
    PhysicsWorld::Reset();
    if (impl_->scene == nullptr || impl_->initial_state.empty()) {
        SetLastError("SuperDex world has not been built.");
        return;
    }
    mochi::Error error;
    impl_->scene->RestoreStateFromBytes(impl_->initial_state, error);
    if (!error.IsOK()) {
        SetLastError("SuperDex failed to restore its initial state: " + error.ToString());
        return;
    }
    for (Impl::RobotBinding& robot : impl_->robots) {
        for (Impl::JointBinding& joint : robot.joints) {
            joint.controller.Reset();
        }
    }
    for (std::size_t index = 0;
         index < impl_->deformables.size() && index < scene_state_.deformables.size();
         ++index) {
        impl_->deformables[index].previous_vertices =
                scene_state_.deformables[index].local_vertices;
    }
    impl_->scene->Step(0.0);
    if (!SyncStateFromSuperDex(0.0)) {
        return;
    }
    UpdateDiagnostics();
    last_error_.clear();
}

PhysicsStepResult SuperDexPhysicsWorld::Step(RealType delta_time) {
    if (impl_->scene == nullptr) {
        SetLastError("SuperDex world has not been built.");
        return {.error = last_error_};
    }
#if GOBOT_SUPERDEX_HAS_STEP_PROFILING
    if (mochi::GetStepProfile(*impl_->scene).enabled != settings_.superdex_solver.record_solver_timings) {
        mochi::SetStepProfilingEnabled(*impl_->scene, settings_.superdex_solver.record_solver_timings);
    }
#else
    if (settings_.superdex_solver.record_solver_timings) {
        SetLastError("SuperDex stage profiling requires rebuilding/installing the SDK with step_profiling.h support.");
        return {.error = last_error_};
    }
#endif
    if (delta_time == 0.0) {
        impl_->scene->Step(0.0);
        UpdateDiagnostics();
        if (SyncStateFromSuperDex(0.0)) {
            last_error_.clear();
        }
        return {.completed = last_error_.empty(), .diagnostics = GetSolverDiagnostics(), .error = last_error_};
    }
    const RealType resolved_delta_time = delta_time;
    if (!(resolved_delta_time > 0.0) || !std::isfinite(resolved_delta_time)) {
        SetLastError("SuperDex step duration must be positive and finite.");
        return {.error = last_error_};
    }
    const int substeps = settings_.superdex_solver.substeps;
    if (substeps <= 0) {
        SetLastError("SuperDex substep count must be positive.");
        return {.error = last_error_};
    }
    const RealType substep_time = resolved_delta_time / static_cast<RealType>(substeps);
    const auto step_start = std::chrono::steady_clock::now();
    const bool record_timings = settings_.superdex_solver.record_solver_timings;
    double force_time_seconds = 0.0;
    double sync_time_seconds = 0.0;
    std::uint64_t force_calls = 0;
    std::uint64_t sync_calls = 0;
#if GOBOT_SUPERDEX_HAS_STEP_PROFILING
    mochi::StepProfile step_profile;
#endif
    double solve_time_seconds = 0.0;
    int newton_iterations = 0;
    int line_search_iterations = 0;
    RealType residual_norm = 0.0;
    bool has_residual = false;
    PhysicsStepResult result;
    last_error_.clear();
    PhysicsSolverConvergenceStatus convergence =
            PhysicsSolverConvergenceStatus::Converged;
    const auto merge_convergence = [&convergence](PhysicsSolverConvergenceStatus status) {
        if (status == PhysicsSolverConvergenceStatus::Diverged ||
            convergence == PhysicsSolverConvergenceStatus::Diverged) {
            convergence = PhysicsSolverConvergenceStatus::Diverged;
        } else if (status == PhysicsSolverConvergenceStatus::Stopped ||
                   convergence == PhysicsSolverConvergenceStatus::Stopped) {
            convergence = PhysicsSolverConvergenceStatus::Stopped;
        } else if (status == PhysicsSolverConvergenceStatus::Unknown ||
                   convergence == PhysicsSolverConvergenceStatus::Unknown) {
            convergence = PhysicsSolverConvergenceStatus::Unknown;
        }
    };
    for (int substep = 0; substep < substeps; ++substep) {
        auto stage_start = record_timings ? std::chrono::steady_clock::now()
                                         : std::chrono::steady_clock::time_point{};
        const bool forces_applied = ApplyForces(substep_time);
        if (record_timings) {
            force_time_seconds += std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - stage_start).count();
            ++force_calls;
        }
        if (!forces_applied) {
            break;
        }
        impl_->scene->Step(static_cast<double>(substep_time));
#if GOBOT_SUPERDEX_HAS_STEP_PROFILING
        if (record_timings) {
            const auto profile = mochi::GetStepProfile(*impl_->scene);
            for (std::size_t i = 0; i < profile.stages.size(); ++i) {
                step_profile.stages[i].seconds += profile.stages[i].seconds;
                step_profile.stages[i].calls += profile.stages[i].calls;
            }
            step_profile.linearIterations += profile.linearIterations;
        }
#endif
        // Mochi advances its clock even when an island diverges and restores
        // that island's previous deformation. Report that time, then stop.
        result.advanced_time += substep_time;
        const mochi::PerformanceStats performance = impl_->scene->GetPerformanceStats();
        const mochi::SolverStats solver = impl_->scene->GetSolverStats();
        solve_time_seconds += performance.solveStepDurationSec;
        newton_iterations += solver.maxNonLinearIters;
        line_search_iterations += solver.maxLineSearchIters;
        const RealType substep_residual = static_cast<RealType>(solver.residualNorm);
        if (!has_residual || !std::isfinite(substep_residual)) {
            residual_norm = substep_residual;
        } else if (std::isfinite(residual_norm)) {
            residual_norm = std::max(residual_norm, substep_residual);
        }
        has_residual = true;
        merge_convergence(FromMochi(solver.convergenceStatus));
        // Controllers and external-force mappings for the next substep must
        // consume the state produced by this one. This also leaves nodal
        // velocities as the final substep velocity instead of a stale
        // full-step finite difference.
        if (record_timings) {
            stage_start = std::chrono::steady_clock::now();
        }
        const bool state_synced = SyncStateFromSuperDex(substep_time);
        if (record_timings) {
            sync_time_seconds += std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - stage_start).count();
            ++sync_calls;
        }
        if (!state_synced) {
            result.state_valid = false;
            break;
        }
        if (convergence == PhysicsSolverConvergenceStatus::Diverged) {
            SetLastError("SuperDex solver diverged; simulation paused and reset is required.");
            break;
        }
    }
    UpdateDiagnostics();
    impl_->diagnostics.total_step_time_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - step_start).count();
    impl_->diagnostics.solve_time_seconds = solve_time_seconds;
    impl_->diagnostics.newton_iterations = newton_iterations;
    impl_->diagnostics.line_search_iterations = line_search_iterations;
    impl_->diagnostics.residual_norm = residual_norm;
    impl_->diagnostics.convergence = convergence;
#if GOBOT_SUPERDEX_HAS_STEP_PROFILING
    if (record_timings) {
        constexpr std::array stage_names{
                "sdk_pre_step", "sdk_islands", "sdk_post_step", "island_prepare",
                "island_newton", "island_queries", "collision_detection",
                "contact_jacobians", "assembly", "linear_setup", "linear_solve", "line_search"};
        static_assert(stage_names.size() == static_cast<std::size_t>(mochi::StepProfileStage::Count));
        auto& timings = impl_->diagnostics.stage_timings;
        timings.reserve(stage_names.size() + 2);
        timings.push_back({"apply_forces", force_time_seconds, force_calls, false});
        timings.push_back({"state_sync", sync_time_seconds, sync_calls, false});
        for (std::size_t i = 0; i < stage_names.size(); ++i) {
            timings.push_back({stage_names[i], step_profile.stages[i].seconds,
                               step_profile.stages[i].calls, i >= 3});
        }
        impl_->diagnostics.linear_iterations = step_profile.linearIterations;
    }
#endif
    result.completed = last_error_.empty();
    result.diagnostics = impl_->diagnostics;
    result.error = last_error_;
    return result;
}

Ref<PhysicsRuntimeCheckpoint> SuperDexPhysicsWorld::CaptureCheckpoint() const {
    if (impl_->scene == nullptr) {
        return {};
    }
    Ref<PhysicsRuntimeCheckpoint> checkpoint = PhysicsWorld::CaptureCheckpoint();
    if (!checkpoint.IsValid()) {
        return {};
    }
    mochi::DynamicArray<std::uint8_t> state;
    mochi::Error error;
    impl_->scene->CaptureStateToBytes(state, error);
    if (!error.IsOK()) {
        return {};
    }
    checkpoint->backend_state_blobs_.resize(1);
    checkpoint->backend_state_blobs_.front().assign(state.begin(), state.end());
    for (const Impl::RobotBinding& robot : impl_->robots) {
        for (const Impl::JointBinding& joint : robot.joints) {
            checkpoint->controller_states_.push_back(
                    joint.controller.CaptureRuntimeState());
        }
    }
    return checkpoint;
}

bool SuperDexPhysicsWorld::RestoreCheckpoint(
        const Ref<PhysicsRuntimeCheckpoint>& checkpoint,
        const std::vector<std::size_t>& environment_indices) {
    std::vector<std::size_t> resolved_indices;
    std::string validation_error;
    if (!ValidateCheckpoint(
                checkpoint, environment_indices, &resolved_indices, &validation_error)) {
        SetLastError(std::move(validation_error));
        return false;
    }
    std::size_t controller_count = 0;
    for (const Impl::RobotBinding& robot : impl_->robots) {
        controller_count += robot.joints.size();
    }
    if (resolved_indices.size() != 1 || resolved_indices.front() != 0 ||
        checkpoint->backend_state_blobs_.size() != 1 ||
        checkpoint->backend_state_blobs_.front().empty() ||
        checkpoint->controller_states_.size() != controller_count) {
        SetLastError("SuperDex runtime checkpoint payload has incompatible dimensions.");
        return false;
    }
    mochi::Error error;
    impl_->scene->RestoreStateFromBytes(checkpoint->backend_state_blobs_.front(), error);
    if (!error.IsOK()) {
        SetLastError("SuperDex rejected runtime checkpoint bytes: " + error.ToString());
        return false;
    }
    scene_state_ = checkpoint->scene_states_.front();
    external_forces_ = checkpoint->external_forces_;
    deformable_external_forces_ = checkpoint->deformable_external_forces_;
    std::size_t controller_index = 0;
    for (Impl::RobotBinding& robot : impl_->robots) {
        for (Impl::JointBinding& joint : robot.joints) {
            joint.controller.RestoreRuntimeState(
                    checkpoint->controller_states_[controller_index++]);
        }
    }
    for (std::size_t index = 0;
         index < impl_->deformables.size() && index < scene_state_.deformables.size();
         ++index) {
        impl_->deformables[index].previous_vertices =
                scene_state_.deformables[index].local_vertices;
    }
    // Mochi captures solver state, while node/contact queries are derived caches.
    // Refresh those caches after restore without advancing simulation time.
    impl_->scene->Step(0.0);
    if (!SyncStateFromSuperDex(0.0)) {
        return false;
    }
    UpdateDiagnostics();
    last_error_.clear();
    return true;
}

bool SuperDexPhysicsWorld::ResetJointState(const std::string& robot_name,
                                           const std::string& joint_name,
                                           RealType position,
                                           RealType velocity) {
    if (!PhysicsWorld::ResetJointState(robot_name, joint_name, position, velocity)) {
        return false;
    }
    if (!PushStateToSuperDex() || !SyncStateFromSuperDex(0.0)) {
        return false;
    }
    last_error_.clear();
    return true;
}

bool SuperDexPhysicsWorld::ResetLinkState(const std::string& robot_name,
                                          const std::string& link_name,
                                          const Vector3& position,
                                          const Quaternion& orientation,
                                          const Vector3& linear_velocity,
                                          const Vector3& angular_velocity) {
    if (!PhysicsWorld::ResetLinkState(robot_name,
                                      link_name,
                                      position,
                                      orientation,
                                      linear_velocity,
                                      angular_velocity)) {
        return false;
    }
    if (!PushStateToSuperDex() || !SyncStateFromSuperDex(0.0)) {
        return false;
    }
    last_error_.clear();
    return true;
}

bool SuperDexPhysicsWorld::WriteEnvironmentLinkVelocity(
        std::size_t environment_index,
        const std::string& robot_name,
        const std::string& link_name,
        const Vector3& linear_velocity,
        const Vector3& angular_velocity) {
    if (environment_index != 0) {
        SetLastError("SuperDex supports only environment 0 in its first experimental release.");
        return false;
    }
    const PhysicsLinkState* link_state = FindMutableLinkState(robot_name, link_name);
    if (link_state == nullptr) {
        SetLastError("Cannot set velocity for a missing SuperDex robot link.");
        return false;
    }
    return ResetLinkState(
            robot_name,
            link_name,
            Vector3(link_state->global_transform.translation()),
            link_state->global_transform.GetQuaternion(),
            linear_velocity,
            angular_velocity);
}

PhysicsRaycastHit SuperDexPhysicsWorld::RaycastTerrain(
        const PhysicsRaycastQuery& query) const {
    return RaycastTerrainFallback(query);
}

namespace {

Ref<PhysicsWorld> CreateSuperDexPhysicsWorld() {
    return MakeRef<SuperDexPhysicsWorld>();
}

const bool s_superdex_backend_registered = PhysicsServer::RegisterBackend(
        {
                PhysicsBackendType::SuperDex,
                "SuperDex",
                true,
                true,
#if GOBOT_SUPERDEX_HAS_CUDA
                true,
#else
                false,
#endif
                true,
                "Experimental unified rigid, articulated, and deformable physics backend."
        },
        &CreateSuperDexPhysicsWorld);

} // namespace

} // namespace gobot

GOBOT_REGISTRATION {

    Class_<gobot::SuperDexPhysicsWorld>("SuperDexPhysicsWorld")
            .constructor()(CtorAsRawPtr);

    gobot::Type::register_wrapper_converter_for_base_classes<
            Ref<gobot::SuperDexPhysicsWorld>, Ref<gobot::PhysicsWorld>>();

};
