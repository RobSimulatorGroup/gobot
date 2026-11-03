/* SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include "gobot/scene/resources/shape_3d.hpp"

namespace gobot {
// Infinite static half-space boundary with local outward normal +Z.
// Rendering extents belong to a separate mesh, not the collision surface.
class GOBOT_EXPORT PlaneShape3D : public Shape3D {
    GOBCLASS(PlaneShape3D, Shape3D);
public:
    PlaneShape3D() = default;
};
} // namespace gobot
