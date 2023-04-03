/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-28
*/

#pragma once

#include "gobot/core/object.hpp"
#include "gobot/scene/camera_3d.hpp"
#include "gobot/scene/imgui_node.hpp"

namespace gobot {

class Camera3D;

class GOBOT_EXPORT Node3DEditor : public ImGuiNode {
    GOBCLASS(Node3DEditor, ImGuiNode)
public:
    Node3DEditor();

    void ResetCamera();

    ~Node3DEditor() override;

    static Node3DEditor* GetInstance();

    void UpdateCamera(double interp_delta);

    void OnImGui();

    FORCE_INLINE Camera3D* GetCamera3D() { return camera3d_; }

    FORCE_INLINE static uint32_t InvalidGuizmoOperation() { return UINT32_MAX; }

    FORCE_INLINE uint32_t GetImGuizmoOperation() const { return imguizmo_operation_; }

    FORCE_INLINE void SetImGuizmoOperation(uint32_t imGuizmo_operation) { imguizmo_operation_ = imGuizmo_operation; }

    bool& SnapGuizmo();

protected:
    void NotificationCallBack(NotificationType notification);

private:
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

    uint32_t imguizmo_operation_ = UINT32_MAX;

    bool snap_guizmo_{false};

};

}
