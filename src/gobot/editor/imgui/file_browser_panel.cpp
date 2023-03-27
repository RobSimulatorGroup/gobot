/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-3-19
*/

#include "imgui.h"
#include "gobot/editor/imgui/file_browser_panel.hpp"
#include "imgui_extension/file_browser/ImFileBrowser.h"
#include "imgui_extension/fonts/MaterialDesign.inl"
#include "imgui_extension/icon_fonts/icons_material_design_icons.h"

namespace gobot {

FileBrowserPanel::FileBrowserPanel()
{
    name_ = "FileBrowser";

    file_browser_ = new ImGui::FileBrowser(ImGuiFileBrowserFlags_CreateNewDir | ImGuiFileBrowserFlags_EnterNewFilename | ImGuiFileBrowserFlags_HideHiddenFiles);

    file_browser_->SetTitle("File Browser");
    file_browser_->SetLabels(ICON_MDI_FOLDER, ICON_MDI_FILE, ICON_MDI_FOLDER_OPEN);
    file_browser_->Refresh();
}

FileBrowserPanel::~FileBrowserPanel()
{
    delete file_browser_;
}

void FileBrowserPanel::OnImGui()
{
    file_browser_->Display();

    if(file_browser_->HasSelected()) {
        String temp_file_path = String::fromStdString(file_browser_->GetSelected().string());
        temp_file_path = temp_file_path.replace("\\", "/");
        callback_(temp_file_path);
        file_browser_->ClearSelected();
    }
}

bool FileBrowserPanel::IsOpen()
{
    return file_browser_->IsOpened();
}

void FileBrowserPanel::SetCurrentPath(const String& path)
{
    file_browser_->SetPwd(path.toStdString());
}

void FileBrowserPanel::Open()
{
    file_browser_->Open();
}

void FileBrowserPanel::SetOpenDirectory(bool value)
{
    auto flags = file_browser_->GetFlags();

    if(value) {
        flags |= ImGuiFileBrowserFlags_SelectDirectory;
    } else {
        flags &= ~(ImGuiFileBrowserFlags_SelectDirectory);
    }
    file_browser_->SetFlags(flags);
}

void FileBrowserPanel::SetFileTypeFilters(const std::vector<const char*>& fileFilters)
{
    file_browser_->SetFileFilters(fileFilters);
}

void FileBrowserPanel::ClearFileTypeFilters()
{
    file_browser_->ClearFilters();
}

String FileBrowserPanel::GetPath() const
{
    return file_browser_->GetPath().c_str();
}

}