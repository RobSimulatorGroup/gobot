/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * This file is created by Qiqi Wu, 23-2-28
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "gobot/core/object.hpp"
#include "gobot/scene/camera_3d.hpp"
#include "gobot/scene/imgui_node.hpp"
#include "gobot/scene/resources/material.hpp"

struct ImVec2;

namespace gobot {

class Camera3D;
class SceneView3DPanel;

class GOBOT_EXPORT Node3DEditor : public ImGuiNode {
    GOBCLASS(Node3DEditor, ImGuiNode)
public:
    Node3DEditor();

    void ResetCamera();

    ~Node3DEditor() override;

    static Node3DEditor* GetInstance();

    void UpdateCamera(double interp_delta);

    void OnImGuizmo();

    FORCE_INLINE Camera3D* GetCamera3D() { return camera3d_; }

    FORCE_INLINE static uint32_t InvalidGuizmoOperation() { return UINT32_MAX; }

    FORCE_INLINE uint32_t GetImGuizmoOperation() const { return imguizmo_operation_; }

    FORCE_INLINE void SetImGuizmoOperation(uint32_t imGuizmo_operation) { imguizmo_operation_ = imGuizmo_operation; }

    void SetNeedUpdateCamera(bool update_camera);

    void SetBlockCameraInput(bool block_camera_input);

    bool& SnapGuizmo();

protected:
    void NotificationCallBack(NotificationType notification);

private:
    void ApplyCameraViewMatrix(const Matrix4& view_matrix);

    void DrawViewManipulator(const ImVec2& position, const ImVec2& size);

    void SetCameraOrbit(const Vector3& eye, const Vector3& at, const Vector3& up);

    static Node3DEditor* s_singleton;

    Camera3D* camera3d_;

    Vector2i mouse_position_last_{0, 0};
    Vector2i mouse_position_now_{0, 0};

    float mouse_speed_{0.0020f};
    float scroll_move_speed_{50.0f};
    float translation_speed_{0.02f};

    float horizontal_angle_{0.01f};
    float vertical_angle_{0.0};

    float distance_{0.0};

    bool mouse_down_{false};

    bool update_camera_{false};
    bool editing_{false};
    bool block_camera_input_{false};

    uint32_t imguizmo_operation_ = UINT32_MAX;

    bool snap_guizmo_{false};

    Ref<ShaderMaterial> shader_material_;

    SceneView3DPanel* scene_view3d_panel_;

};

}
