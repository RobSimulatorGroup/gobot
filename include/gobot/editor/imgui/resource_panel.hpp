/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-3-20
*/

#pragma once
#include "gobot/scene/imgui_window.hpp"

class ImGuiTextFilter;

namespace gobot {

struct DirectoryInformation
{
    DirectoryInformation* parent{nullptr};
    std::vector<DirectoryInformation*> children;

    std::filesystem::path file_path;
    bool is_file;

public:
    DirectoryInformation(const std::filesystem::path& fname, bool _is_file) {
        file_path = fname;
        is_file = _is_file;
    }
};

class ResourcePanel : public ImGuiWindow {
    GOBCLASS(ResourcePanel, ImGuiWindow)
public:
    ResourcePanel();

    ~ResourcePanel() override;

    bool MoveFile(const String& file_path, const String& move_path);
    void OnImGuiContent() override;

    void ChangeDirectory(const DirectoryInformation* directory);


    bool RenderFile(int dirIndex, bool folder, int shownIndex, bool gridView);

    void RenderBottom();

    void DrawFolder(const DirectoryInformation* dir_info, bool default_open = false);

    String ProcessDirectory(const std::filesystem::path& directory_path,
                            const DirectoryInformation* parent);

    void Refresh();

private:

    String project_path_;
    String move_path_;
    String last_nav_path_;


    bool is_dragging_;
    bool is_in_list_view_;
    bool show_hidden_files_;
    int grid_items_per_row_;
    float grid_size_ = 50.0f;

    ImGuiTextFilter* filter_{nullptr};


    bool update_navigation_path_;
    DirectoryInformation* current_dir_{nullptr};
    DirectoryInformation* base_project_dir_{nullptr};

    DirectoryInformation* next_directory_{nullptr};
    DirectoryInformation* previous_directory_{nullptr};
    std::vector<DirectoryInformation*> bread_crumb_data_;

    std::unordered_map<String, std::unique_ptr<DirectoryInformation>> directories_;
};

}
