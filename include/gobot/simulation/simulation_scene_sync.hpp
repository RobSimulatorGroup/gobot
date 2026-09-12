#pragma once

#include <span>
#include "gobot/core/math/math_defs.hpp"
#include "gobot_export.h"

namespace gobot {
class Node;
class DeformableBody3D;

enum class SimulationVertexSpace { World, Local };

// Scene-owner service. Validate and stage the complete batch before changing
// any runtime node, so a stale binding cannot leave half a displayed frame.
class GOBOT_EXPORT SimulationSceneSync {
public:
    static void ApplyDeformableVertices(
            Node* root, std::span<DeformableBody3D* const> bodies,
            std::span<const RealType> padded_positions, std::size_t width,
            std::span<const std::size_t> counts, SimulationVertexSpace space);
};
} // namespace gobot
