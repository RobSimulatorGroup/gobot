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

class GOBOT_EXPORT ProjectSettings : public Object {
    GOBCLASS(ProjectSettings, Object)
public:
    static ProjectSettings* s_singleton;

    ProjectSettings();

    ~ProjectSettings() override;

    static ProjectSettings* GetInstance();

    [[nodiscard]] std::string LocalizePath(std::string_view path) const;

    [[nodiscard]] std::string GlobalizePath(std::string_view path) const;

    bool SetProjectPath(const std::string& project_path);

    [[nodiscard]] FORCE_INLINE const std::string& GetProjectPath() const { return project_path_; }

private:
    std::string project_path_;
};

}
