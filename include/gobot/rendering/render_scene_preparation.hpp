/*
 * Backend-neutral per-view render-list preparation.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "gobot/core/math/aabb.hpp"
#include "gobot/rendering/scene_render_items.hpp"

#include "gobot_export.h"

#include <array>
#include <cstddef>
#include <vector>

namespace gobot {

struct RenderFrustumPlane {
    Vector3 normal = Vector3::Zero();
    RealType distance = 0.0;
};

class GOBOT_EXPORT RenderFrustum {
public:
    [[nodiscard]] static RenderFrustum FromViewProjection(const Matrix4& view_projection);
    [[nodiscard]] bool Intersects(const AABB& bounds) const;
    [[nodiscard]] bool IsValid() const;

private:
    std::array<RenderFrustumPlane, 6> planes_{};
    bool valid_ = false;
};

struct PreparedRenderItem {
    const VisualMeshRenderItem* item = nullptr;
    RealType camera_depth = 0.0;
};

struct RenderDrawLists {
    std::vector<PreparedRenderItem> opaque;
    std::vector<PreparedRenderItem> alpha_masked;
    std::vector<PreparedRenderItem> transparent;
    std::vector<PreparedRenderItem> shadow_casters;
    std::size_t visible_count = 0;
    std::size_t culled_count = 0;
};

GOBOT_EXPORT RenderDrawLists BuildRenderDrawLists(const RenderSceneSnapshot& scene,
                                                  const RenderViewSnapshot& view,
                                                  bool frustum_culling);

} // namespace gobot
