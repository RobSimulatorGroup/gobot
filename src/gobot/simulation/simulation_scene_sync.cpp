#include "gobot/simulation/simulation_scene_sync.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <vector>
#include "gobot/scene/deformable_body_3d.hpp"
#include "gobot/scene/node.hpp"

namespace gobot {
void SimulationSceneSync::ApplyDeformableVertices(
        Node* root, std::span<DeformableBody3D* const> bodies,
        std::span<const RealType> positions, std::size_t width,
        std::span<const std::size_t> counts, SimulationVertexSpace space) {
    if (space != SimulationVertexSpace::World && space != SimulationVertexSpace::Local)
        throw std::invalid_argument("Unknown deformable vertex coordinate space");
    if (!root || counts.size() != bodies.size() ||
        (bodies.empty() ? !positions.empty() :
         positions.size() / bodies.size() / 3 != width || positions.size() % bodies.size() != 0 ||
         (positions.size() / bodies.size()) % 3 != 0))
        throw std::invalid_argument("Invalid deformable vertex batch dimensions or scene root");
    std::unordered_set<DeformableBody3D*> unique;
    std::vector<std::vector<Vector3>> staged;
    staged.reserve(bodies.size());
    for (std::size_t i = 0; i < bodies.size(); ++i) {
        auto* body = bodies[i];
        if (!body || (body != root && !root->IsAncestorOf(body)))
            throw std::invalid_argument("Deformable vertex batch contains a body outside the active scene");
        if (!unique.insert(body).second)
            throw std::invalid_argument("Deformable vertex batch contains a duplicate body");
        const auto authored_count = body->GetModel() == DeformableBodyModel::ThinShell
                ? (body->GetSurfaceMesh().IsValid() ? body->GetSurfaceMesh()->GetVertexCount() : 0)
                : (body->GetMesh().IsValid() ? body->GetMesh()->GetVertexCount() : 0);
        if (!authored_count || counts[i] != authored_count || counts[i] > width)
            throw std::invalid_argument("Deformable vertex count does not match the authored mesh topology");
        Affine3 inverse = Affine3::Identity();
        if (space == SimulationVertexSpace::World) {
            const auto transform = body->GetGlobalTransform();
            const auto determinant = transform.linear().determinant();
            const auto scale = transform.linear().col(0).norm() * transform.linear().col(1).norm() *
                    transform.linear().col(2).norm();
            if (!transform.matrix().allFinite() || !std::isfinite(determinant) || !std::isfinite(scale) ||
                scale <= 0 || std::abs(determinant) <= std::numeric_limits<RealType>::epsilon() * RealType(128) * scale)
                throw std::invalid_argument("Deformable body has a non-invertible runtime transform");
            inverse = transform.inverse();
        }
        auto& local = staged.emplace_back();
        local.reserve(counts[i]);
        for (std::size_t vertex = 0; vertex < counts[i]; ++vertex) {
            const auto offset = (i * width + vertex) * 3;
            Vector3 value(positions[offset], positions[offset + 1], positions[offset + 2]);
            if (!value.allFinite()) throw std::invalid_argument("Deformable vertex batch contains a non-finite position");
            value = inverse * value;
            if (!value.allFinite()) throw std::invalid_argument("Deformable local vertex conversion is non-finite");
            local.push_back(value);
        }
    }
    for (std::size_t i = 0; i < bodies.size(); ++i) bodies[i]->SetRuntimeVertices(staged[i]);
}
} // namespace gobot
