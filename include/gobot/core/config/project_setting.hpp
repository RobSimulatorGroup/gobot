/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 22-12-14
*/

#pragma once

#include "gobot/core/object.hpp"
#include "gobot/core/types.hpp"

namespace gobot {

class GOBOT_API ProjectSettings : public Object {
    GOBCLASS(ProjectSettings, Object)
public:
    static ProjectSettings* s_singleton;

    ProjectSettings();

    ~ProjectSettings();

    static ProjectSettings* GetSingleton();

    [[nodiscard]] String LocalizePath(const String &path) const;

    // For test
    void SetProjectPath(const String& project_path);

private:
    String project_path_;
};

}
