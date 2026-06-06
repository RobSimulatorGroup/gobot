/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <filesystem>
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

    [[nodiscard]] bool IsScriptDirty() const { return script_dirty_; }

    [[nodiscard]] bool IsScriptExternallyModified() const { return script_externally_modified_; }

    [[nodiscard]] const std::string& GetScriptLocalPath() const { return script_local_path_; }

    [[nodiscard]] std::string GetEditorTextForTesting() const { return editor_.GetText(); }

    void SetEditorTextForTesting(const std::string& text);

    void CheckExternalScriptModificationForTesting();

private:
    bool ReadScriptFile(std::string* source_code) const;

    bool ReloadScriptFromDisk();

    void MarkScriptCleanAfterDiskSync();

    void CheckExternalScriptModification();

    void RecordScriptWriteTime();

    bool SavePythonScript();

    bool OpenInVSCode();

    void RunPythonScript();

    TextEditor editor_;
    std::string script_local_path_;
    std::string script_global_path_;
    std::filesystem::file_time_type script_last_write_time_{};
    bool script_dirty_{false};
    bool script_externally_modified_{false};
    bool script_write_time_valid_{false};
};

} // namespace gobot
