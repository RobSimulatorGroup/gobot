/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
*/

#include "gobot/scene/robot_3d.hpp"

#include <utility>

#include "gobot/core/registration.hpp"

namespace gobot {

void Robot3D::SetSourcePath(std::string source_path) {
    source_path_ = std::move(source_path);
}

const std::string& Robot3D::GetSourcePath() const {
    return source_path_;
}

} // namespace gobot

GOBOT_REGISTRATION {

    Class_<Robot3D>("Robot3D")
            .constructor()(CtorAsRawPtr)
            .property("source_path", &Robot3D::GetSourcePath, &Robot3D::SetSourcePath);

};
