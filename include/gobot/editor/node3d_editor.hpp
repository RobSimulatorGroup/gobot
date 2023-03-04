/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-28
*/

#pragma once

#include "gobot/core/object.hpp"
#include "gobot/scene/camera3d.hpp"

namespace gobot {

class Camera3D;

class Node3DEditor : public Node {
    GOBCLASS(Node3DEditor, Node)
public:
    Node3DEditor();

    void ResetCamera();

    ~Node3DEditor() override;

    static Node3DEditor* GetInstance();

    void UpdateCamera(double interp_delta);

protected:
    void NotificationCallBack(NotificationType notification);

private:
    static Node3DEditor* s_singleton;

    Camera3D* camera3d_;

    Vector2i mouse_position_last_{0, 0};
    Vector2i mouse_position_now_{0, 0};

    float mouse_speed_{0.0020f};
    float scroll_move_speed_{50.0f};

    float horizontal_angle_{0.01f};
    float vertical_angle_{0.0};

    Vector3 eye_;
    Vector3 at_;
    Vector3 up_;

    float distance_;

    bool mouse_down_{false};
};

}
