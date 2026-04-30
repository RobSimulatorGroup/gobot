/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-23
*/

#include "gobot/rendering/render_server.hpp"
#include "gobot/scene/node.hpp"
#include "gobot/scene/scene_tree.hpp"
#include "gobot/scene/window.hpp"
#include "gobot/error_macros.hpp"
#include "gobot/rendering/rendering_server_globals.hpp"
#include "gobot/rendering/renderer_compositor.hpp"
#include "gobot/rendering/texture_storage.hpp"
#include "gobot/rendering/renderer_viewport.hpp"
#include "gobot/drivers/opengl/rasterizer_gl.hpp"


namespace gobot {

RenderServer* RenderServer::s_singleton = nullptr;

RenderServer::RenderServer(RendererType p_renderer_type)
    : renderer_type_(p_renderer_type)
{
    s_singleton =  this;

    RSG::viewport = new RendererViewport();
    if (renderer_type_ == RendererType::OpenGL46) {
        opengl::GLRasterizer::MakeCurrent();
    }

    RSG::rasterizer = Rasterizer::Create();

    RSG::texture_storage = RSG::rasterizer->GetTextureStorage();
    RSG::material_storage = RSG::rasterizer->GetMaterialStorage();
    RSG::mesh_storage = RSG::rasterizer->GetMeshStorage();
    RSG::scene = RSG::rasterizer->GetScene();
    RSG::debug_draw = RSG::rasterizer->GetDebugDraw();
    RSG::utilities = RSG::rasterizer->GetUtilities();

}

RendererType RenderServer::GetRendererType() {
    return renderer_type_;
}

RenderServer::~RenderServer() {
    s_singleton = nullptr;
    delete RSG::rasterizer;
    delete RSG::viewport;
    RSG::rasterizer = nullptr;
    RSG::viewport = nullptr;
    RSG::texture_storage = nullptr;
    RSG::material_storage = nullptr;
    RSG::mesh_storage = nullptr;
    RSG::scene = nullptr;
    RSG::debug_draw = nullptr;
    RSG::utilities = nullptr;
}

RenderServer* RenderServer::GetInstance() {
    ERR_FAIL_COND_V_MSG(s_singleton == nullptr, nullptr, "Must call this after initialize RenderingServer");
    return s_singleton;
}

bool RenderServer::HasInstance() {
    return s_singleton != nullptr;
}

void RenderServer::Draw() {
    SceneTree::GetInstance()->GetRoot()->GetWindow()->SwapBuffers();
}

void RenderServer::Free(const RID& p_rid) {
    if (p_rid.IsNull()) [[unlikely]] {
        return;
    }
    if (RSG::viewport->Free(p_rid)) {
        return;
    }
    if (RSG::mesh_storage->OwnsMesh(p_rid)) {
        RSG::mesh_storage->MeshFree(p_rid);
        return;
    }
    if (RSG::utilities->Free(p_rid)) {
        return;
    }
}

void* RenderServer::GetRenderTargetColorTextureNativeHandle(const RID& p_view_port) {
    return RSG::viewport->GetRenderTargetColorTextureNativeHandle(p_view_port);
}

RID RenderServer::MeshCreate() {
    auto rid = RSG::mesh_storage->MeshAllocate();
    RSG::mesh_storage->MeshInitialize(rid);
    return rid;
}

void RenderServer::MeshSetBox(const RID& mesh, const Vector3& size) {
    RSG::mesh_storage->MeshSetBox(mesh, size);
}

void RenderServer::MeshSetSurface(const RID& mesh,
                                  const std::vector<Vector3>& vertices,
                                  const std::vector<uint32_t>& indices) {
    RSG::mesh_storage->MeshSetSurface(mesh, vertices, indices);
}

void RenderServer::MeshSetCylinder(const RID& mesh, RealType radius, RealType height, int radial_segments) {
    RSG::mesh_storage->MeshSetCylinder(mesh, radius, height, radial_segments);
}

void RenderServer::MeshSetSphere(const RID& mesh, RealType radius, int radial_segments, int rings) {
    RSG::mesh_storage->MeshSetSphere(mesh, radius, radial_segments, rings);
}

void RenderServer::RenderSceneToViewport(const RID& viewport, const Node* scene_root, const Camera3D* camera) {
    ERR_FAIL_COND(RSG::scene == nullptr);
    const RID render_target = RSG::viewport->GetViewportRenderTarget(viewport);
    ERR_FAIL_COND(render_target.IsNull());

    RSG::scene->RenderScene(render_target, scene_root, camera);
}

void RenderServer::RenderEditorDebugToViewport(const RID& viewport, const Camera3D* camera, const Node* scene_root) {
    ERR_FAIL_COND(RSG::debug_draw == nullptr);
    const RID render_target = RSG::viewport->GetViewportRenderTarget(viewport);
    ERR_FAIL_COND(render_target.IsNull());

    RSG::debug_draw->RenderEditorDebug(render_target, camera, scene_root);
}



}
