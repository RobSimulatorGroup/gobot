/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * This file is created by Qiqi Wu, 22-12-14
 * SPDX-License-Identifier: Apache-2.0
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

    void ClearProjectPath();

    [[nodiscard]] FORCE_INLINE const std::string& GetProjectPath() const { return project_path_; }

private:
    std::string project_path_;
};

}
