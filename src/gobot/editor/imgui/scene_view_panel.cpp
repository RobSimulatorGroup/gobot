/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-3-10
*/


#include "gobot/editor/imgui/scene_view_panel.hpp"
#include "gobot/editor/node3d_editor.hpp"
#include "gobot/editor/editor.hpp"
#include "gobot/scene/camera3d.hpp"
#include "gobot/error_macros.hpp"
#include "gobot/editor/imgui/imgui_utilities.hpp"
#include "imgui_extension/fonts/MaterialDesign.inl"
#include "imgui_extension/icon_fonts/icons_material_design_icons.h"
#include "imgui_extension/gizmos/ImGuizmo.h"
#include "imgui.h"
#include "imgui_internal.h"

namespace gobot {

SceneViewPanel::SceneViewPanel()
{
    name_         = ICON_MDI_EYE " Viewer###scene_view";
    current_scene_ = nullptr;

    width_  = 1280;
    height_ = 800;

    scene_renderer_ = std::make_unique<SceneRenderer>(width_, height_);
}

void SceneViewPanel::OnImGui()
{
    ImGuiUtilities::ScopedStyle window_padding(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));

    auto flags = ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse;

    if(!ImGui::Begin(name_.toStdString().c_str(), &active_, flags) || current_scene_ ) {
        ImGui::End();
        return;
    }


    auto node3d_editor = Editor::GetInstance()->GetNode3dEditor();
    auto camera_3d = node3d_editor->GetCamera3D();


    ImVec2 offset = { 0.0f, 0.0f };

    {
        ToolBar();
        offset = ImGui::GetCursorPos(); // Usually ImVec2(0.0f, 50.0f);
    }

    if(!camera_3d) {
        ImGui::End();
        return;
    }

    ImGuizmo::SetDrawlist();
    auto scene_view_size     = ImGui::GetWindowContentRegionMax() - ImGui::GetWindowContentRegionMin() - offset * 0.5f; // - offset * 0.5f;
    auto scene_view_position = ImGui::GetWindowPos() + offset;

    scene_view_size.x -= static_cast<int>(scene_view_size.x) % 2 != 0 ? 1.0f : 0.0f;
    scene_view_size.y -= static_cast<int>(scene_view_size.y) % 2 != 0 ? 1.0f : 0.0f;

    real_t aspect = static_cast<real_t>(scene_view_size.x) / static_cast<real_t>(scene_view_size.y);

    if(!FloatEquals(aspect, camera_3d->GetAspect())) {
        camera_3d->SetAspect(aspect);
    }
//    node3d_editor->m_SceneViewPanelPosition = glm::vec2(scene_view_position.x, scene_view_position.y);


    scene_view_size.x -= static_cast<int>(scene_view_size.x) % 2 != 0 ? 1.0f : 0.0f;
    scene_view_size.y -= static_cast<int>(scene_view_size.y) % 2 != 0 ? 1.0f : 0.0f;

    Resize(static_cast<uint32_t>(scene_view_size.x), static_cast<uint32_t>(scene_view_size.y));

    ImGuiUtilities::Image(view_texture_.Get(), {scene_view_size.x, scene_view_size.y});

//    auto windowSize = ImGui::GetWindowSize();
//    ImVec2 minBound = scene_view_position;
//
//    ImVec2 maxBound   = { minBound.x + windowSize.x, minBound.y + windowSize.y };
//    bool updateCamera = ImGui::IsMouseHoveringRect(minBound, maxBound); // || Input::Get().GetMouseMode() == MouseMode::Captured;
//
//    app.SetSceneActive(ImGui::IsWindowFocused() && !ImGuizmo::IsUsing() && updateCamera);
//
//    ImGuizmo::SetRect(scene_view_position.x, scene_view_position.y, scene_view_size.x, scene_view_size.y);
//
//    m_Editor->SetSceneViewActive(updateCamera);
//
//    if(m_Editor->ShowGrid())
//    {
//        if(camera->IsOrthographic())
//        {
//            m_Editor->Draw2DGrid(ImGui::GetWindowDrawList(), { transform->GetWorldPosition().x, transform->GetWorldPosition().y }, scene_view_position, {scene_view_size.x, scene_view_size.y }, 1.0f, 1.5f);
//        }
//    }
//
//    {
//        ImGui::GetWindowDrawList()->PushClipRect(scene_view_position, {scene_view_size.x + scene_view_position.x, scene_view_size.y + scene_view_position.y - 2.0f });
//    }
//
//
//
//    if(updateCamera && app.GetSceneActive() && !ImGuizmo::IsUsing() && Input::Get().GetMouseClicked(InputCode::MouseKey::ButtonLeft))
//    {
//
//        float dpi     = Application::Get().GetWindowDPI();
//        auto clickPos = Input::Get().GetMousePosition() - glm::vec2(scene_view_position.x / dpi, scene_view_position.y / dpi);
//
//        Maths::Ray ray = m_Editor->GetScreenRay(int(clickPos.x), int(clickPos.y), camera, int(scene_view_size.x / dpi), int(scene_view_size.y / dpi));
//        m_Editor->SelectObject(ray);
//    }
//
//    const ImGuiPayload* payload = ImGui::GetDragDropPayload();
//
//    if(ImGui::BeginDragDropTarget())
//    {
//        auto data = ImGui::AcceptDragDropPayload("AssetFile", ImGuiDragDropFlags_AcceptNoDrawDefaultRect);
//        if(data)
//        {
//            std::string file = (char*)data->Data;
//            m_Editor->FileOpenCallback(file);
//        }
//        ImGui::EndDragDropTarget();
//    }
//
//    if(app.GetSceneManager()->GetCurrentScene())
//        DrawGizmos(scene_view_size.x, scene_view_size.y, offset.x, offset.y, app.GetSceneManager()->GetCurrentScene());

    ImGui::End();
}

void SceneViewPanel::Resize(uint32_t width, uint32_t height) {
    bool resize = false;
    ERR_FAIL_COND_MSG(width == 0 || height == 0, "Game View Dimensions 0");

    if(width_ != width || height_ != height) {
        resize   = true;
        width_  = width;
        height_ = height;
    }

    if(resize) {
        if(!view_texture_) {
            view_texture_ = MakeRef<Texture2D>(width_, height_, false, 1, TextureFormat::RGBA8, TextureFlags::RT);
        } else {
            view_texture_->Resize(width_, height_);
        }
        scene_renderer_->SetRenderTarget(view_texture_.Get());
        scene_renderer_->Resize(width, height);
    }
}

void SceneViewPanel::ToolBar()
{
    auto node3d_editor = Editor::GetInstance()->GetNode3dEditor();
    ImGui::Indent();
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.0f, 0.0f, 0.0f, 0.0f));
    bool selected = false;

    {
        selected = node3d_editor->GetImGuizmoOperation() == 4;
        if(selected)
            ImGui::PushStyleColor(ImGuiCol_Text, ImGuiUtilities::GetSelectedColor());
        ImGui::SameLine();
        if(ImGui::Button(ICON_MDI_CURSOR_DEFAULT))
            node3d_editor->SetImGuizmoOperation(4);

        if(selected)
            ImGui::PopStyleColor();
        ImGuiUtilities::Tooltip("Select");
    }

    ImGui::SameLine();
    ImGui::SeparatorEx(ImGuiSeparatorFlags_Vertical);
    ImGui::SameLine();

    {
        selected = node3d_editor->GetImGuizmoOperation() == ImGuizmo::TRANSLATE;
        if(selected)
            ImGui::PushStyleColor(ImGuiCol_Text, ImGuiUtilities::GetSelectedColor());
        ImGui::SameLine();
        if(ImGui::Button(ICON_MDI_ARROW_ALL))
            node3d_editor->SetImGuizmoOperation(ImGuizmo::TRANSLATE);

        if(selected)
            ImGui::PopStyleColor();
        ImGuiUtilities::Tooltip("Translate");
    }

    {
        selected = node3d_editor->GetImGuizmoOperation() == ImGuizmo::ROTATE;
        if(selected)
            ImGui::PushStyleColor(ImGuiCol_Text, ImGuiUtilities::GetSelectedColor());

        ImGui::SameLine();
        if(ImGui::Button(ICON_MDI_ROTATE_3D))
            node3d_editor->SetImGuizmoOperation(ImGuizmo::ROTATE);

        if(selected)
            ImGui::PopStyleColor();
        ImGuiUtilities::Tooltip("Rotate");
    }

    {
        selected = node3d_editor->GetImGuizmoOperation() == ImGuizmo::SCALE;
        if(selected)
            ImGui::PushStyleColor(ImGuiCol_Text, ImGuiUtilities::GetSelectedColor());

        ImGui::SameLine();
        if(ImGui::Button(ICON_MDI_ARROW_EXPAND_ALL))
            node3d_editor->SetImGuizmoOperation(ImGuizmo::SCALE);

        if(selected)
            ImGui::PopStyleColor();
        ImGuiUtilities::Tooltip("Scale");
    }

    ImGui::SameLine();
    ImGui::SeparatorEx(ImGuiSeparatorFlags_Vertical);
    ImGui::SameLine();

    {
        selected = node3d_editor->GetImGuizmoOperation() == ImGuizmo::UNIVERSAL;
        if(selected)
            ImGui::PushStyleColor(ImGuiCol_Text, ImGuiUtilities::GetSelectedColor());

        ImGui::SameLine();
        if(ImGui::Button(ICON_MDI_CROP_ROTATE))
            node3d_editor->SetImGuizmoOperation(ImGuizmo::UNIVERSAL);

        if(selected)
            ImGui::PopStyleColor();
        ImGuiUtilities::Tooltip("Universal");
    }

    ImGui::SameLine();
    ImGui::SeparatorEx(ImGuiSeparatorFlags_Vertical);
    ImGui::SameLine();

    {
        selected = node3d_editor->GetImGuizmoOperation() == ImGuizmo::BOUNDS;
        if(selected)
            ImGui::PushStyleColor(ImGuiCol_Text, ImGuiUtilities::GetSelectedColor());

        ImGui::SameLine();
        if(ImGui::Button(ICON_MDI_BORDER_NONE))
            node3d_editor->SetImGuizmoOperation(ImGuizmo::BOUNDS);

        if(selected)
            ImGui::PopStyleColor();
        ImGuiUtilities::Tooltip("Bounds");
    }

    ImGui::SameLine();
    ImGui::SeparatorEx(ImGuiSeparatorFlags_Vertical);
    ImGui::SameLine();

//    ImGui::SameLine();
//    {
//        selected = (node3d_editor->SnapGuizmo() == true);
//
//        if(selected)
//            ImGui::PushStyleColor(ImGuiCol_Text, ImGuiUtilities::GetSelectedColour());
//
//        if(ImGui::Button(ICON_MDI_MAGNET))
//            node3d_editor->SnapGuizmo() = !selected;
//
//        if(selected)
//            ImGui::PopStyleColor();
//        ImGuiUtilities::Tooltip("Snap");
//    }

    ImGui::SameLine();
    ImGui::SeparatorEx(ImGuiSeparatorFlags_Vertical);
    ImGui::SameLine();

//    if(ImGui::Button("Gizmos " ICON_MDI_CHEVRON_DOWN))
//        ImGui::OpenPopup("GizmosPopup");
//    if(ImGui::BeginPopup("GizmosPopup"))
//    {
//        {
//            ImGui::Checkbox("Grid", &m_Editor->ShowGrid());
//            ImGui::Checkbox("Selected Gizmos", &m_Editor->ShowGizmos());
//            ImGui::Checkbox("View Selected", &m_Editor->ShowViewSelected());
//
//            ImGui::Separator();
//            ImGui::Checkbox("Camera", &m_ShowComponentGizmoMap[typeid(Camera).hash_code()]);
//            ImGui::Checkbox("Light", &m_ShowComponentGizmoMap[typeid(Graphics::Light).hash_code()]);
//            ImGui::Checkbox("Audio", &m_ShowComponentGizmoMap[typeid(SoundComponent).hash_code()]);
//
//            ImGui::Separator();
//
//            uint32_t flags = m_Editor->GetSettings().m_DebugDrawFlags;
//
//            bool showAABB = flags & EditorDebugFlags::MeshBoundingBoxes;
//            if(ImGui::Checkbox("Mesh AABB", &showAABB))
//            {
//                if(showAABB)
//                    flags += EditorDebugFlags::MeshBoundingBoxes;
//                else
//                    flags -= EditorDebugFlags::MeshBoundingBoxes;
//            }
//
//            bool showSpriteBox = flags & EditorDebugFlags::SpriteBoxes;
//            if(ImGui::Checkbox("Sprite Box", &showSpriteBox))
//            {
//                if(showSpriteBox)
//                    flags += EditorDebugFlags::SpriteBoxes;
//                else
//                    flags -= EditorDebugFlags::SpriteBoxes;
//            }
//
//            bool showCameraFrustums = flags & EditorDebugFlags::CameraFrustum;
//            if(ImGui::Checkbox("Camera Frustums", &showCameraFrustums))
//            {
//                if(showCameraFrustums)
//                    flags += EditorDebugFlags::CameraFrustum;
//                else
//                    flags -= EditorDebugFlags::CameraFrustum;
//            }
//
//            m_Editor->GetSettings().m_DebugDrawFlags = flags;
//            ImGui::Separator();
//
//            auto physics2D = Application::Get().GetSystem<B2PhysicsEngine>();
//
//            if(physics2D)
//            {
//                uint32_t flags = physics2D->GetDebugDrawFlags();
//
//                bool show2DShapes = flags & b2Draw::e_shapeBit;
//                if(ImGui::Checkbox("Shapes (2D)", &show2DShapes))
//                {
//                    if(show2DShapes)
//                        flags += b2Draw::e_shapeBit;
//                    else
//                        flags -= b2Draw::e_shapeBit;
//                }
//
//                bool showCOG = flags & b2Draw::e_centerOfMassBit;
//                if(ImGui::Checkbox("Centre of Mass (2D)", &showCOG))
//                {
//                    if(showCOG)
//                        flags += b2Draw::e_centerOfMassBit;
//                    else
//                        flags -= b2Draw::e_centerOfMassBit;
//                }
//
//                bool showJoint = flags & b2Draw::e_jointBit;
//                if(ImGui::Checkbox("Joint Connection (2D)", &showJoint))
//                {
//                    if(showJoint)
//                        flags += b2Draw::e_jointBit;
//                    else
//                        flags -= b2Draw::e_jointBit;
//                }
//
//                bool showAABB = flags & b2Draw::e_aabbBit;
//                if(ImGui::Checkbox("AABB (2D)", &showAABB))
//                {
//                    if(showAABB)
//                        flags += b2Draw::e_aabbBit;
//                    else
//                        flags -= b2Draw::e_aabbBit;
//                }
//
//                bool showPairs = static_cast<bool>(flags & b2Draw::e_pairBit);
//                if(ImGui::Checkbox("Broadphase Pairs  (2D)", &showPairs))
//                {
//                    if(showPairs)
//                        flags += b2Draw::e_pairBit;
//                    else
//                        flags -= b2Draw::e_pairBit;
//                }
//
//                physics2D->SetDebugDrawFlags(flags);
//            }
//
//            auto physics3D = Application::Get().GetSystem<LumosPhysicsEngine>();
//
//            if(physics3D)
//            {
//                uint32_t flags = physics3D->GetDebugDrawFlags();
//
//                bool showCollisionShapes = flags & PhysicsDebugFlags::COLLISIONVOLUMES;
//                if(ImGui::Checkbox("Collision Volumes", &showCollisionShapes))
//                {
//                    if(showCollisionShapes)
//                        flags += PhysicsDebugFlags::COLLISIONVOLUMES;
//                    else
//                        flags -= PhysicsDebugFlags::COLLISIONVOLUMES;
//                }
//
//                bool showConstraints = static_cast<bool>(flags & PhysicsDebugFlags::CONSTRAINT);
//                if(ImGui::Checkbox("Constraints", &showConstraints))
//                {
//                    if(showConstraints)
//                        flags += PhysicsDebugFlags::CONSTRAINT;
//                    else
//                        flags -= PhysicsDebugFlags::CONSTRAINT;
//                }
//
//                bool showManifolds = static_cast<bool>(flags & PhysicsDebugFlags::MANIFOLD);
//                if(ImGui::Checkbox("Manifolds", &showManifolds))
//                {
//                    if(showManifolds)
//                        flags += PhysicsDebugFlags::MANIFOLD;
//                    else
//                        flags -= PhysicsDebugFlags::MANIFOLD;
//                }
//
//                bool showCollisionNormals = flags & PhysicsDebugFlags::COLLISIONNORMALS;
//                if(ImGui::Checkbox("Collision Normals", &showCollisionNormals))
//                {
//                    if(showCollisionNormals)
//                        flags += PhysicsDebugFlags::COLLISIONNORMALS;
//                    else
//                        flags -= PhysicsDebugFlags::COLLISIONNORMALS;
//                }
//
//                bool showAABB = flags & PhysicsDebugFlags::AABB;
//                if(ImGui::Checkbox("AABB", &showAABB))
//                {
//                    if(showAABB)
//                        flags += PhysicsDebugFlags::AABB;
//                    else
//                        flags -= PhysicsDebugFlags::AABB;
//                }
//
//                bool showLinearVelocity = flags & PhysicsDebugFlags::LINEARVELOCITY;
//                if(ImGui::Checkbox("Linear Velocity", &showLinearVelocity))
//                {
//                    if(showLinearVelocity)
//                        flags += PhysicsDebugFlags::LINEARVELOCITY;
//                    else
//                        flags -= PhysicsDebugFlags::LINEARVELOCITY;
//                }
//
//                bool LINEARFORCE = flags & PhysicsDebugFlags::LINEARFORCE;
//                if(ImGui::Checkbox("Linear Force", &LINEARFORCE))
//                {
//                    if(LINEARFORCE)
//                        flags += PhysicsDebugFlags::LINEARFORCE;
//                    else
//                        flags -= PhysicsDebugFlags::LINEARFORCE;
//                }
//
//                bool BROADPHASE = flags & PhysicsDebugFlags::BROADPHASE;
//                if(ImGui::Checkbox("Broadphase", &BROADPHASE))
//                {
//                    if(BROADPHASE)
//                        flags += PhysicsDebugFlags::BROADPHASE;
//                    else
//                        flags -= PhysicsDebugFlags::BROADPHASE;
//                }
//
//                bool showPairs = flags & PhysicsDebugFlags::BROADPHASE_PAIRS;
//                if(ImGui::Checkbox("Broadphase Pairs", &showPairs))
//                {
//                    if(showPairs)
//                        flags += PhysicsDebugFlags::BROADPHASE_PAIRS;
//                    else
//                        flags -= PhysicsDebugFlags::BROADPHASE_PAIRS;
//                }
//
//                physics3D->SetDebugDrawFlags(flags);
//            }
//
//            ImGui::EndPopup();
//        }
//    }

//    ImGui::SameLine();
//    ImGui::SeparatorEx(ImGuiSeparatorFlags_Vertical);
//    ImGui::SameLine();
//    // Editor Camera Settings
//
//    auto& camera = *m_Editor->GetCamera();
//    bool ortho   = camera.IsOrthographic();
//
//    selected = !ortho;
//    if(selected)
//        ImGui::PushStyleColor(ImGuiCol_Text, ImGuiUtilities::GetSelectedColour());
//    if(ImGui::Button(ICON_MDI_AXIS_ARROW " 3D"))
//    {
//        if(ortho)
//        {
//            camera.SetIsOrthographic(false);
//            m_Editor->GetEditorCameraController().SetCurrentMode(EditorCameraMode::ARCBALL);
//        }
//    }
//    if(selected)
//        ImGui::PopStyleColor();
//    ImGui::SameLine();
//
//    selected = ortho;
//    if(selected)
//        ImGui::PushStyleColor(ImGuiCol_Text, ImGuiUtilities::GetSelectedColour());
//    if(ImGui::Button(ICON_MDI_ANGLE_RIGHT "2D"))
//    {
//        if(!ortho)
//        {
//            camera.SetIsOrthographic(true);
//            m_Editor->GetEditorCameraTransform().SetLocalOrientation(glm::quat(glm::vec3(0.0f, 0.0f, 0.0f)));
//            m_Editor->GetEditorCameraController().SetCurrentMode(EditorCameraMode::TWODIM);
//        }
//    }
//    if(selected)
//        ImGui::PopStyleColor();

    ImGui::PopStyleColor();
    ImGui::Unindent();
}

}
