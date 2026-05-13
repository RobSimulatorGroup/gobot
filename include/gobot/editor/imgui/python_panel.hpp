/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>

#include "gobot/scene/imgui_window.hpp"
#include "TextEditor.h"

namespace gobot {

class GOBOT_EXPORT PythonPanel : public ImGuiWindow {
    GOBCLASS(PythonPanel, ImGuiWindow)

public:
    PythonPanel();

    ~PythonPanel() override = default;

    void OnImGuiContent() override;

    bool OpenScript(const std::string& path);

    bool LoadScript(const std::string& path);

private:
    bool SavePythonScript();

    bool OpenInVSCode();

    void RunPythonScript();

    TextEditor editor_;
    std::string script_local_path_;
    std::string script_global_path_;
    bool script_dirty_{false};
};

} // namespace gobot
