/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-3-19
*/

#pragma once

#include "gobot/rendering/render_types.hpp"
#include "gobot/rendering/render_pass.hpp"

namespace gobot {

class Texture;
class Camera3D;
class SceneTree;

class SceneRenderer {
public:
    SceneRenderer();

    ~SceneRenderer();

    void SetRenderTarget(const RenderRID& texture_rid);

    void OnRenderer(const SceneTree* scene_tree);

    void FinalPass();

    void DebugPass();

private:
    Camera3D* camera_ = nullptr;

    RenderRID view_texture_{};
    RenderRID view_frame_buffer_{};
};


}
