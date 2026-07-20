/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * This file is created by Qiqi Wu, 23-2-23
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/rendering/render_server.hpp"
#include "gobot/core/profile.hpp"
#include "gobot/scene/node.hpp"
#include "gobot/scene/scene_tree.hpp"
#include "gobot/scene/window.hpp"
#include "gobot/error_macros.hpp"
#include "gobot/rendering/rendering_server_globals.hpp"
#include "gobot/rendering/renderer_compositor.hpp"
#include "gobot/rendering/scene_render_items.hpp"
#include "gobot/rendering/texture_storage.hpp"
#include "gobot/rendering/renderer_viewport.hpp"
#include "gobot/scene/camera_3d.hpp"


namespace gobot {

RenderServer* RenderServer::s_singleton = nullptr;

RenderServer::RenderServer(RendererType p_renderer_type)
    : renderer_type_(p_renderer_type)
{
    s_singleton =  this;

    RSG::viewport = new RendererViewport();
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
    GOBOT_PROFILE_ZONE("RenderServer::Draw");
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
    if (RSG::texture_storage->OwnsTexture(p_rid)) {
        RSG::texture_storage->TextureFree(p_rid);
        return;
    }
    if (RSG::utilities->Free(p_rid)) {
        return;
    }
}

void* RenderServer::GetRenderTargetColorTextureNativeHandle(const RID& p_view_port) {
    return RSG::viewport->GetRenderTargetColorTextureNativeHandle(p_view_port);
}

std::vector<std::uint8_t> RenderServer::ReadViewportRgbPixels(const RID& p_view_port, bool p_flip_y) {
    return RSG::viewport->ReadViewportRgbPixels(p_view_port, p_flip_y);
}

bool RenderServer::ReadViewportOutput(const RID& viewport,
                                      RenderOutputType output,
                                      void* destination,
                                      std::size_t destination_size,
                                      bool flip_y) {
    return RSG::viewport->ReadViewportOutput(
            viewport, output, destination, destination_size, flip_y);
}

RID RenderServer::MeshCreate() {
    auto rid = RSG::mesh_storage->MeshAllocate();
    RSG::mesh_storage->MeshInitialize(rid);
    return rid;
}

RID RenderServer::TextureCreate() {
    return RSG::texture_storage->TextureAllocate();
}

void RenderServer::Texture2DInitialize(const RID& texture, const Ref<Image>& image) {
    RSG::texture_storage->Texture2DInitialize(texture, image);
}

void RenderServer::TextureSetData(const RID& texture, const Ref<Image>& image) {
    RSG::texture_storage->TextureSetData(texture, image);
}

void RenderServer::MeshSetBox(const RID& mesh, const Vector3& size) {
    RSG::mesh_storage->MeshSetBox(mesh, size);
}

void RenderServer::MeshSetSurface(const RID& mesh,
                                  const std::vector<Vector3>& vertices,
                                  const std::vector<uint32_t>& indices,
                                  const std::vector<Vector3>& normals,
                                  const std::vector<Color>& colors) {
    RSG::mesh_storage->MeshSetSurface(mesh, vertices, indices, normals, colors);
}

void RenderServer::MeshSetCylinder(const RID& mesh, RealType radius, RealType height, int radial_segments) {
    RSG::mesh_storage->MeshSetCylinder(mesh, radius, height, radial_segments);
}

void RenderServer::MeshSetSphere(const RID& mesh, RealType radius, int radial_segments, int rings) {
    RSG::mesh_storage->MeshSetSphere(mesh, radius, radial_segments, rings);
}

void RenderServer::RenderSceneToViewport(const RID& viewport, const Node* scene_root, const Camera3D* camera) {
    GOBOT_PROFILE_ZONE("RenderServer::RenderSceneToViewport");
    ERR_FAIL_COND(RSG::scene == nullptr);
    ERR_FAIL_COND(scene_root == nullptr);
    ERR_FAIL_COND(camera == nullptr);
    RenderSceneSnapshot scene_snapshot;
    RenderViewSnapshot view_snapshot;
    {
        GOBOT_PROFILE_ZONE("RenderServer::CaptureRenderSnapshots");
        scene_snapshot = CaptureRenderSceneSnapshot(scene_root);
        view_snapshot = CaptureRenderViewSnapshot(*camera);
    }
    GOBOT_PROFILE_PLOT("visual_meshes", static_cast<double>(scene_snapshot.visual_meshes.size()));
    RenderSnapshotsToViewport(viewport, scene_snapshot, view_snapshot);
}

void RenderServer::RenderSnapshotsToViewport(const RID& viewport,
                                             const RenderSceneSnapshot& scene,
                                             const RenderViewSnapshot& view) {
    ERR_FAIL_COND(RSG::scene == nullptr);
    const RID render_target = RSG::viewport->GetViewportRenderTarget(viewport);
    ERR_FAIL_COND(render_target.IsNull());
    RSG::scene->RenderScene(render_target, scene, view);
}

void RenderServer::SetSceneRendererSettings(const SceneRendererSettings& settings) {
    ERR_FAIL_COND(RSG::scene == nullptr);
    RSG::scene->SetSettings(settings);
}

SceneRendererSettings RenderServer::GetSceneRendererSettings() const {
    return RSG::scene != nullptr ? RSG::scene->GetSettings() : SceneRendererSettings{};
}

SceneRendererCapabilities RenderServer::GetSceneRendererCapabilities() const {
    return RSG::scene != nullptr ? RSG::scene->GetCapabilities() : SceneRendererCapabilities{};
}

SceneRendererStats RenderServer::GetSceneRendererStats() const {
    return RSG::scene != nullptr ? RSG::scene->GetStats() : SceneRendererStats{};
}

bool RenderServer::CaptureCudaRenderProduct(const RenderSceneSnapshot& scene,
                                            const RenderViewSnapshot& view,
                                            int width,
                                            int height,
                                            std::uint32_t output_mask,
                                            std::uint32_t mode,
                                            RendererRenderProductFrame* frame,
                                            std::string* error) {
    if (RSG::scene == nullptr) {
        if (error != nullptr) {
            *error = "render scene backend is not initialized";
        }
        return false;
    }
    return RSG::scene->CaptureCudaRenderProduct(
            scene, view, width, height, output_mask, mode, frame, error);
}

void RenderServer::RenderEditorDebugToViewport(const RID& viewport,
                                               const Camera3D* camera,
                                               const Node* scene_root,
                                               const PhysicsWorld* physics_world,
                                               bool show_collision_shapes) {
    GOBOT_PROFILE_ZONE("RenderServer::RenderEditorDebugToViewport");
    ERR_FAIL_COND(RSG::debug_draw == nullptr);
    const RID render_target = RSG::viewport->GetViewportRenderTarget(viewport);
    ERR_FAIL_COND(render_target.IsNull());

    RSG::debug_draw->RenderEditorDebug(render_target, camera, scene_root, physics_world, show_collision_shapes);
}

void RenderServer::RenderDebugArrowsToViewport(const RID& viewport,
                                               const Camera3D* camera,
                                               const std::vector<DebugArrow>& arrows) {
    GOBOT_PROFILE_ZONE("RenderServer::RenderDebugArrowsToViewport");
    GOBOT_PROFILE_PLOT("debug_arrows", static_cast<double>(arrows.size()));
    ERR_FAIL_COND(RSG::debug_draw == nullptr);
    if (arrows.empty()) {
        return;
    }

    const RID render_target = RSG::viewport->GetViewportRenderTarget(viewport);
    ERR_FAIL_COND(render_target.IsNull());

    RSG::debug_draw->RenderDebugArrows(render_target, camera, arrows);
}



}
