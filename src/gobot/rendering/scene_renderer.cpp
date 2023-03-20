/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-3-19
*/

#include "gobot/rendering/scene_renderer.hpp"
#include "gobot/scene/resources/texture.hpp"
#include "gobot/rendering/debug_draw/debug_draw.hpp"

namespace gobot {

SceneRenderer::SceneRenderer(uint32_t width, uint32_t height)
{
}

SceneRenderer::~SceneRenderer() {
}

void SceneRenderer::SetRenderTarget(const Texture* texture) {

}

void SceneRenderer::Resize(uint32_t width, uint32_t height) {

}

void SceneRenderer::OnRenderer(const SceneTree* scene_tree) {
    FinalPass();
}

void SceneRenderer::DebugPass() {
    RenderPass pass("DebugPass");

    DebugDrawEncoder dde;

    dde.Begin(0);
    dde.DrawWorldAxis(65.0);

    dde.Push();
    bx::Aabb aabb =
            {
                    {  5.0f, 1.0f, 1.0f },
                    { 10.0f, 5.0f, 5.0f },
            };
    dde.SetWireframe(true);
    dde.Draw(aabb);
    dde.Pop();

    {
        const bx::Vec3 normal = { 0.0f,  1.0f, 0.0f };
        const bx::Vec3 pos    = { 0.0f, -2.0f, 0.0f };

        bx::Plane plane(bx::init::None);
        bx::calcPlane(plane, normal, pos);

        dde.DrawGrid(Axis::Y, pos, 128, 1.0f);
    }

    dde.End();

//    bgfx::setViewFrameBuffer(0, m_fbh);
}


void SceneRenderer::FinalPass() {
    RenderPass pass("FinalPass");
    pass.Clear();
//    pass.SetViewTransform(view, proj);
}



}
