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
public:
    Node3DEditor();

    ~Node3DEditor();

    static Node3DEditor* GetInstance();

    void UpdateCameraByInput();

private:
    static Node3DEditor* s_singleton;

private:
    Camera3D* camera3d_;

};

}
