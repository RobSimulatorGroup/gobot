/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
*/

#include "gobot/scene/robot_3d.hpp"

#include <utility>

#include "gobot/core/registration.hpp"
#include "gobot/scene/joint_3d.hpp"

namespace gobot {

void Robot3D::SetSourcePath(std::string source_path) {
    source_path_ = std::move(source_path);
}

const std::string& Robot3D::GetSourcePath() const {
    return source_path_;
}

void Robot3D::SetMode(RobotMode mode) {
    if (mode_ == mode) {
        return;
    }

    if (mode == RobotMode::Motion) {
        SetJointMotionModeRecursive(this, true);
    } else {
        SetJointMotionModeRecursive(this, false);
    }
    mode_ = mode;
}

RobotMode Robot3D::GetMode() const {
    return mode_;
}

void Robot3D::SetJointMotionModeRecursive(Node* node, bool enabled) {
    if (!node) {
        return;
    }

    if (auto* joint = Object::PointerCastTo<Joint3D>(node)) {
        joint->SetMotionModeEnabled(enabled);
    }

    for (std::size_t i = 0; i < node->GetChildCount(); ++i) {
        SetJointMotionModeRecursive(node->GetChild(static_cast<int>(i)), enabled);
    }
}

} // namespace gobot

GOBOT_REGISTRATION {

    QuickEnumeration_<RobotMode>("RobotMode");

    Class_<Robot3D>("Robot3D")
            .constructor()(CtorAsRawPtr)
            .property("mode", &Robot3D::GetMode, &Robot3D::SetMode)
            .property("source_path", &Robot3D::GetSourcePath, &Robot3D::SetSourcePath);

};
