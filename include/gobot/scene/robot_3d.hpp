/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
*/

#pragma once

#include <string>

#include "gobot/scene/node_3d.hpp"

namespace gobot {

class GOBOT_EXPORT Robot3D : public Node3D {
    GOBCLASS(Robot3D, Node3D)

public:
    Robot3D() = default;

    void SetSourcePath(std::string source_path);

    const std::string& GetSourcePath() const;

private:
    std::string source_path_;
};

}
