/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-3-19
*/

#pragma once

#include "gobot/scene/imgui_window.hpp"

namespace ImGui {
class FileBrowser;
}

namespace gobot {

class FileBrowserPanel : public ImGuiWindow {
    GOBCLASS(FileBrowserPanel, ImGuiWindow)
public:
public:
    FileBrowserPanel();

    ~FileBrowserPanel();

    void Open();

    void OnImGuiContent() override;

    void SetCurrentPath(const String& path);

    void SetOpenDirectory(bool value);

    FORCE_INLINE void SetCallback(const std::function<void(const String&)>& callback) { callback_ = callback; }

    bool IsOpen();

    void SetFileTypeFilters(const std::vector<const char*>& fileFilters);

    void ClearFileTypeFilters();

    String GetFilePath() const;

private:
    std::function<void(const String&)> callback_;
    ImGui::FileBrowser* file_browser_;
};

}