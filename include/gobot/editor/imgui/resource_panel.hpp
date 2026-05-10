/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-3-20
*/

#pragma once

#include "gobot/scene/imgui_window.hpp"
#include "gobot_export.h"

class ImGuiTextFilter;
namespace ImGui {
class FileBrowser;
}

namespace gobot {

struct GOBOT_EXPORT DirectoryInformation
{
    DirectoryInformation* parent{nullptr};
    std::vector<DirectoryInformation*> children;

    std::string local_path;
    std::string global_path;
    std::string this_path;
    bool is_directory{};

public:
    DirectoryInformation(const std::string& _this_path, DirectoryInformation* _parent);
};

class GOBOT_EXPORT ResourcePanel : public ImGuiWindow {
    GOBCLASS(ResourcePanel, ImGuiWindow)
public:
    ResourcePanel();

    ~ResourcePanel() override;

    bool MoveFile(const std::string& file_path, const std::string& move_path);
    void OnImGuiContent() override;

    void ChangeDirectory(DirectoryInformation* directory);


    bool RenderFile(int dirIndex, bool folder, int shownIndex, bool gridView);

    void DrawResourceTree(DirectoryInformation* dir_info, bool root = false);

    void RenderBottom();

    void DrawNewFolderPopup();

    void DrawNewPythonScriptPopup();

    bool CreateFolderInCurrentDirectory(const std::string& folder_name);

    bool CreatePythonScriptInCurrentDirectory(const std::string& file_name);

    bool SetProjectPath(const std::string& project_path);

    bool SelectResource(const std::string& local_path);

    void LoadProjectHistory();

    void SaveProjectHistory() const;

    void AddProjectHistory(const std::string& project_path);

    void RemoveProjectHistory(const std::string& project_path);

    void DrawProjectSelector();

    void DrawDeleteProjectPopup();

    void DrawDeleteResourceFilePopup();

    void OpenProjectBrowser();

    void HandleProjectBrowser();

    DirectoryInformation* ProcessDirectory(const std::string& directory_path,
                                           DirectoryInformation* parent);

    void Refresh();

private:
    std::string project_path_;
    std::string move_path_;
    std::vector<std::string> project_history_;
    std::string project_history_file_;
    std::string pending_delete_project_path_;
    std::string pending_delete_resource_file_global_path_;
    std::string pending_delete_resource_file_local_path_;
    bool request_delete_project_popup_{false};
    bool request_delete_resource_file_popup_{false};


    bool is_dragging_;
    bool is_in_list_view_;
    bool show_hidden_files_;
    int grid_items_per_row_;
    float grid_size_ = 50.0f;
    char new_folder_name_[128]{"resources"};
    char new_python_script_name_[128]{"script.py"};

    ImGuiTextFilter* filter_{nullptr};
    ImGui::FileBrowser* project_browser_{nullptr};
    bool requested_initial_project_browser_{false};


    bool update_navigation_path_;
    DirectoryInformation* current_dir_{nullptr};
    DirectoryInformation* selected_resource_{nullptr};
    DirectoryInformation* base_project_dir_{nullptr};

    DirectoryInformation* next_directory_{nullptr};
    DirectoryInformation* previous_directory_{nullptr};
    std::vector<DirectoryInformation*> bread_crumb_data_;

    std::unordered_map<std::string, std::unique_ptr<DirectoryInformation>> directories_;
};

}
