/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
*/

#pragma once

#include "glad/glad.h"
#include "gobot/rendering/renderer_debug_draw.hpp"
#include "gobot/rendering/scene_render_items.hpp"
#include "gobot/core/math/matrix.hpp"

namespace gobot::opengl {

class GLRendererDebugDraw : public RendererDebugDraw {
public:
    struct LineBuffer {
        GLuint vao = 0;
        GLuint vertex_buffer = 0;
        GLsizei vertex_count = 0;
    };

    GLRendererDebugDraw() = default;

    ~GLRendererDebugDraw() override;

    void RenderEditorDebug(const RID& render_target, const Camera3D* camera, const Node* scene_root) override;

private:
    GLuint program_ = 0;
    LineBuffer editor_grid_;
    LineBuffer world_axes_;
    LineBuffer collision_lines_;

    void EnsureProgram();

    void EnsureEditorGrid();

    void EnsureWorldAxes();

    void DrawEditorGrid();

    void DrawWorldAxes();

    void DrawCollisionDebug(const SceneRenderItems& render_items);
};

}
