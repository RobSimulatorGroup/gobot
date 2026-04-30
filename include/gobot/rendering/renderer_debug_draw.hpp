/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
*/

#pragma once

#include "gobot/core/rid.hpp"

namespace gobot {

class Camera3D;
class Node;

class RendererDebugDraw {
public:
    virtual ~RendererDebugDraw() = default;

    virtual void RenderEditorDebug(const RID& render_target, const Camera3D* camera, const Node* scene_root) = 0;
};

}
