/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-3-20
*/

#include "gobot/editor/imgui/resource_panel.hpp"
#include "gobot/core/config/project_setting.hpp"
#include "gobot/core/string_utils.hpp"
#include "gobot/editor/editor.hpp"
#include "gobot/editor/imgui/imgui_utilities.hpp"
#include "imgui_extension/icon_fonts/icons_material_design_icons.h"
#include "imgui_extension/file_browser/ImFileBrowser.h"
#include "imgui.h"
#include "imgui_internal.h"

#include <algorithm>
#include <cstring>
#include <fstream>

namespace gobot {
namespace {

std::string DefaultProjectHistoryFile() {
    const char* home = std::getenv("HOME");
    if (home == nullptr || std::string(home).empty()) {
        return ".gobot/projects.json";
    }
    return (std::filesystem::path(home) / ".gobot" / "projects.json").string();
}

std::string ProjectDisplayName(const std::string& project_path) {
    std::filesystem::path path(project_path);
    std::string name = path.filename().string();
    return name.empty() ? project_path : name;
}

std::string ToLower(std::string value) {
    std::ranges::transform(value, value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

bool IsNativeSceneFile(const DirectoryInformation* dir_info) {
    if (dir_info == nullptr || dir_info->is_directory) {
        return false;
    }

    return ToLower(std::filesystem::path(dir_info->global_path).extension().string()) == ".jscn";
}

} // namespace

DirectoryInformation::DirectoryInformation(const std::string& _this_path, DirectoryInformation* _parent)
    : parent(_parent)
{
    this_path = _this_path;
    if (parent) {
        local_path = PathJoin(parent->local_path, this_path);
        global_path = ProjectSettings::GetInstance()->GlobalizePath(local_path);
    } else {
        local_path = this_path;
        global_path = ProjectSettings::GetInstance()->GlobalizePath(local_path);
    }

    is_directory = std::filesystem::is_directory(global_path);
}


ResourcePanel::ResourcePanel()
{
    SetName("ResourcePanel");
    SetImGuiWindow(ICON_MDI_FOLDER_STAR " Resources", "resources");

    show_hidden_files_ = false;
    is_dragging_ = false;
    is_in_list_view_ = true;
    update_navigation_path_ = true;

    filter_ = new ImGuiTextFilter();
    project_browser_ = new ImGui::FileBrowser(ImGuiFileBrowserFlags_SelectDirectory |
                                              ImGuiFileBrowserFlags_CreateNewDir |
                                              ImGuiFileBrowserFlags_CloseOnEsc);
    project_browser_->SetTitle("Open Gobot Project");
    project_browser_->SetOkText("Open Project");
    project_history_file_ = DefaultProjectHistoryFile();
    LoadProjectHistory();

    project_path_ = ProjectSettings::GetInstance()->GetProjectPath();

    if (!project_path_.empty()) {
        Refresh();
    }
}

void ResourcePanel::ChangeDirectory(DirectoryInformation* directory)
{
    if(!directory)
        return;

    previous_directory_ = current_dir_;
    current_dir_ = directory;
    update_navigation_path_ = true;
}

ResourcePanel::~ResourcePanel() {
    delete filter_;
    delete project_browser_;
}

bool ResourcePanel::MoveFile(const std::string& file_path, const std::string& move_path)
{
    auto moved_path = PathJoin(file_path, move_path);
    return std::filesystem::exists(moved_path);
}

void ResourcePanel::OnImGuiContent() {
    HandleProjectBrowser();

    if (request_delete_project_popup_) {
        ImGui::OpenPopup("Delete Project");
        request_delete_project_popup_ = false;
    }
    if (request_delete_resource_file_popup_) {
        ImGui::OpenPopup("Delete Resource File");
        request_delete_resource_file_popup_ = false;
    }

    if (project_path_.empty() || base_project_dir_ == nullptr || current_dir_ == nullptr) {
        DrawProjectSelector();
        if (ImGui::BeginPopupModal("Delete Project", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
            DrawDeleteProjectPopup();
        }
        return;
    }

    if(ImGui::BeginDragDropTarget()) {
        auto data = ImGui::AcceptDragDropPayload("selectable", ImGuiDragDropFlags_AcceptNoDrawDefaultRect);
        if(data) {
            std::string file = (char*)data->Data;
            if(MoveFile(file, move_path_)) {
                LOG_INFO("Moved File: " + file + " to " + move_path_);
            }
            is_dragging_ = false;
        }
        ImGui::EndDragDropTarget();
    }

    ImGui::BeginChild("##directory_structure", ImVec2(0, ImGui::GetWindowHeight() - ImGui::GetFrameHeightWithSpacing() * 2.6f));
    {
        ImGui::BeginChild("##directory_breadcrumbs", ImVec2(ImGui::GetColumnWidth(), ImGui::GetFrameHeightWithSpacing()));

        if (ImGui::Button(ICON_MDI_FOLDER_OPEN)) {
            OpenProjectBrowser();
        }
        ImGui::SameLine();
        if (ImGui::Button(ICON_MDI_FOLDER_PLUS)) {
            ImGui::OpenPopup("NewResourceFolderPopup");
        }
        ImGui::SameLine();

        if(update_navigation_path_) {
            bread_crumb_data_.clear();
            auto current = current_dir_;
            while(current) {
                if(current->parent != nullptr) {
                    bread_crumb_data_.push_back(current);
                    current = current->parent;
                } else {
                    bread_crumb_data_.push_back(base_project_dir_);
                    current = nullptr;
                }
            }

            std::reverse(bread_crumb_data_.begin(), bread_crumb_data_.end());
            update_navigation_path_ = false;
        }

        ImGui::TextUnformatted(ICON_MDI_MAGNIFY);
        ImGui::SameLine();

        filter_->Draw("##Filter", ImGui::GetContentRegionAvail().x - ImGui::GetStyle().IndentSpacing);

        ImGui::EndChild();

        {
            ImGui::BeginChild("##Scrolling");
            DrawResourceTree(base_project_dir_, true);

            if (ImGui::BeginPopupContextWindow("##ResourceWindowContext",
                                               ImGuiPopupFlags_MouseButtonRight |
                                               ImGuiPopupFlags_NoOpenOverItems)) {
                if(ImGui::Selectable("Import New")) {
                    // TODO(wqq)
                }

                if(ImGui::Selectable("Refresh")) {
                    Refresh();
                }

                if(ImGui::Selectable("New folder")) {
                    ImGui::OpenPopup("NewResourceFolderPopup");
                }

                ImGui::EndPopup();
            }

            if (ImGui::BeginPopup("NewResourceFolderPopup")) {
                DrawNewFolderPopup();
            }
            ImGui::EndChild();
        }

        ImGui::EndChild();

        ImGui::SeparatorEx(ImGuiSeparatorFlags_Horizontal);
        RenderBottom();
    }

    if(ImGui::BeginDragDropTarget()) {
        auto data = ImGui::AcceptDragDropPayload("selectable", ImGuiDragDropFlags_AcceptNoDrawDefaultRect);
        if(data) {
            std::string a = (char*)data->Data;
            if(MoveFile(a, move_path_)) {
                LOG_INFO("Moved File: " + a + " to " + move_path_);
            }
            is_dragging_ = false;
        }
        ImGui::EndDragDropTarget();
    }

    if (ImGui::BeginPopupModal("Delete Project", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        DrawDeleteProjectPopup();
    }
    if (ImGui::BeginPopupModal("Delete Resource File", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        DrawDeleteResourceFilePopup();
    }
}

bool ResourcePanel::RenderFile(int dirIndex, bool folder, int shownIndex, bool gridView)
{
    bool double_clicked = false;

    if(gridView) {
        ImGui::BeginGroup();

        if(ImGui::Button(folder ? ICON_MDI_FOLDER : ICON_MDI_FILE)) {
        }

        if(ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
            double_clicked = true;
        }

        auto newFname = SimplifyPath(current_dir_->children[dirIndex]->this_path.c_str());

        ImGui::TextUnformatted(newFname.c_str());
        ImGui::EndGroup();

        if((shownIndex + 1) % grid_items_per_row_ != 0)
            ImGui::SameLine();
    } else {
        ImGui::TextUnformatted(folder ? ICON_MDI_FOLDER : ICON_MDI_FILE);
        ImGui::SameLine();
        if(ImGui::Selectable(current_dir_->children[dirIndex]->this_path.c_str(),
                             false, ImGuiSelectableFlags_AllowDoubleClick)) {
            if(ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
                double_clicked = true;
            }
        }
    }

    ImGuiUtilities::Tooltip(current_dir_->children[dirIndex]->global_path.c_str());

    if(double_clicked) {
        if(folder) {
            ChangeDirectory(current_dir_->children[dirIndex]);
        } else {
//            m_Editor->FileOpenCallback(m_BasePath + "/" + m_CurrentDir->Children[dirIndex]->FilePath.string());
        }
    }

    if(ImGui::BeginDragDropSource(ImGuiDragDropFlags_SourceAllowNullID))
    {
        ImGui::TextUnformatted(ICON_MDI_FILE);

        ImGui::SameLine();
        move_path_ = project_path_ + "/" + current_dir_->children[dirIndex]->global_path;
        ImGui::TextUnformatted(move_path_.c_str());
        size_t size = sizeof(const char*) + strlen(move_path_.c_str());
        ImGui::SetDragDropPayload("AssetFile", move_path_.c_str(), size);
        is_dragging_ = true;
        ImGui::EndDragDropSource();
    }

    return double_clicked;
}

void ResourcePanel::DrawResourceTree(DirectoryInformation* dir_info, bool root)
{
    if (dir_info == nullptr) {
        return;
    }

    if (filter_->IsActive() && !filter_->PassFilter(dir_info->global_path.c_str())) {
        return;
    }

    ImGuiTreeNodeFlags node_flags = ImGuiTreeNodeFlags_OpenOnArrow |
                                    ImGuiTreeNodeFlags_OpenOnDoubleClick |
                                    ImGuiTreeNodeFlags_SpanAvailWidth;
    if (dir_info == current_dir_) {
        node_flags |= ImGuiTreeNodeFlags_Selected;
    }

    if (!dir_info->is_directory || dir_info->children.empty()) {
        node_flags |= ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
    }
    if (root) {
        node_flags |= ImGuiTreeNodeFlags_DefaultOpen;
    }

    const char* icon = dir_info->is_directory ? ICON_MDI_FOLDER : ICON_MDI_FILE;
    const std::string label = std::string(icon) + " " + dir_info->this_path + "##" + dir_info->global_path;
    const bool open = ImGui::TreeNodeEx(label.c_str(), node_flags);
    if (ImGui::IsItemClicked()) {
        if (dir_info->is_directory) {
            ChangeDirectory(dir_info);
        }
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("%s", dir_info->local_path.c_str());
    }

    if (IsNativeSceneFile(dir_info) && ImGui::BeginDragDropSource(ImGuiDragDropFlags_SourceAllowNullID)) {
        ImGui::TextUnformatted(ICON_MDI_FILE);
        ImGui::SameLine();
        ImGui::TextUnformatted(dir_info->local_path.c_str());
        ImGui::SetDragDropPayload("GobotSceneResource",
                                  dir_info->local_path.c_str(),
                                  dir_info->local_path.size() + 1);
        ImGui::EndDragDropSource();
    }

    if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left) && ImGui::IsItemHovered()) {
        if (dir_info->is_directory) {
            ChangeDirectory(dir_info);
        } else if (IsNativeSceneFile(dir_info)) {
            Editor::GetInstance()->OpenSceneFromPath(dir_info->local_path);
        }
    }

    const std::string popup_id = "##ResourceFileContext_" + dir_info->global_path;
    if (!dir_info->is_directory && ImGui::BeginPopupContextItem(popup_id.c_str())) {
        const char* delete_label = IsNativeSceneFile(dir_info) ?
                ICON_MDI_DELETE " Delete Scene File" :
                ICON_MDI_DELETE " Delete File";
        if (ImGui::MenuItem(delete_label)) {
            pending_delete_resource_file_global_path_ = dir_info->global_path;
            pending_delete_resource_file_local_path_ = dir_info->local_path;
            request_delete_resource_file_popup_ = true;
        }
        ImGui::EndPopup();
    }

    if (open && dir_info->is_directory && !dir_info->children.empty()) {
        for (DirectoryInformation* child : dir_info->children) {
            if (filter_->IsActive() && !filter_->PassFilter(child->global_path.c_str())) {
                continue;
            }
            DrawResourceTree(child);
        }
        ImGui::TreePop();
    }
}

void ResourcePanel::RenderBottom()
{
    ImGui::BeginChild("##nav", ImVec2(ImGui::GetColumnWidth(), ImGui::GetFontSize() * 1.8f), true, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
    {
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.1f, 0.2f, 0.7f, 0.0f));

        for(auto& directory : bread_crumb_data_) {
            const std::string& directory_name = directory->this_path;
            if(ImGui::SmallButton(directory_name.c_str()))
                ChangeDirectory(directory);

            ImGui::SameLine();
        }

        ImGui::PopStyleColor();

        ImGui::SameLine();
    }

    if(!is_in_list_view_) {
        ImGui::SliderFloat("##GridSize", &grid_size_, 40.0f, 400.0f);
    }
    ImGui::EndChild();
}

bool ResourcePanel::SetProjectPath(const std::string& project_path)
{
    if (project_path.empty()) {
        return false;
    }

    std::error_code error;
    std::filesystem::create_directories(project_path, error);
    if (error) {
        LOG_ERROR("Failed to create project directory '{}': {}", project_path, error.message());
        return false;
    }

    if (!ProjectSettings::GetInstance()->SetProjectPath(project_path)) {
        return false;
    }

    Refresh();
    AddProjectHistory(project_path_);
    LOG_INFO("Opened project: {}", project_path_);
    return true;
}

void ResourcePanel::LoadProjectHistory()
{
    project_history_.clear();
    std::ifstream input(project_history_file_);
    if (!input.is_open()) {
        return;
    }

    Json json;
    try {
        input >> json;
    } catch (const std::exception& error) {
        LOG_ERROR("Failed to read project history '{}': {}", project_history_file_, error.what());
        return;
    }

    if (!json.is_object() || !json.contains("projects") || !json["projects"].is_array()) {
        return;
    }

    for (const auto& project_json : json["projects"]) {
        if (!project_json.is_string()) {
            continue;
        }
        const std::string path = project_json.get<std::string>();
        if (!path.empty() && std::filesystem::exists(path)) {
            project_history_.push_back(path);
        }
    }
}

void ResourcePanel::SaveProjectHistory() const
{
    const std::filesystem::path history_path(project_history_file_);
    std::error_code error;
    std::filesystem::create_directories(history_path.parent_path(), error);
    if (error) {
        LOG_ERROR("Failed to create project history directory '{}': {}",
                  history_path.parent_path().string(), error.message());
        return;
    }

    Json json;
    json["projects"] = project_history_;
    std::ofstream output(project_history_file_);
    if (!output.is_open()) {
        LOG_ERROR("Failed to write project history '{}'.", project_history_file_);
        return;
    }
    output << json.dump(4);
}

void ResourcePanel::AddProjectHistory(const std::string& project_path)
{
    if (project_path.empty()) {
        return;
    }

    project_history_.erase(std::remove(project_history_.begin(), project_history_.end(), project_path),
                           project_history_.end());
    project_history_.insert(project_history_.begin(), project_path);
    constexpr std::size_t kMaxProjectHistory = 20;
    if (project_history_.size() > kMaxProjectHistory) {
        project_history_.resize(kMaxProjectHistory);
    }
    SaveProjectHistory();
}

void ResourcePanel::RemoveProjectHistory(const std::string& project_path)
{
    project_history_.erase(std::remove(project_history_.begin(), project_history_.end(), project_path),
                           project_history_.end());
    SaveProjectHistory();
}

void ResourcePanel::DrawProjectSelector()
{
    ImGui::BeginChild("##project_selector", ImVec2(0, 0), true);
    if (ImGui::Button(ICON_MDI_PLUS " Create")) {
        OpenProjectBrowser();
    }
    ImGui::SameLine();
    if (ImGui::Button(ICON_MDI_FOLDER_OPEN " Import")) {
        OpenProjectBrowser();
    }
    ImGui::SameLine();
    if (ImGui::Button(ICON_MDI_MAGNIFY " Scan")) {
        OpenProjectBrowser();
    }

    ImGui::SameLine();
    filter_->Draw("##ProjectFilter", ImGui::GetContentRegionAvail().x);

    ImGui::Separator();

    if (project_history_.empty()) {
        ImGui::TextDisabled("No projects yet. Create or import a project directory.");
    } else {
        std::string remove_project_path;
        for (const std::string& project_path : project_history_) {
            if (filter_->IsActive() && !filter_->PassFilter(project_path.c_str())) {
                continue;
            }

            ImGui::PushID(project_path.c_str());
            const float row_height = ImGui::GetTextLineHeightWithSpacing() * 2.2f;
            const bool selected = ImGui::Selectable("##ProjectRow", false,
                                                    ImGuiSelectableFlags_AllowDoubleClick,
                                                    ImVec2(0, row_height));
            const ImVec2 row_min = ImGui::GetItemRectMin();
            const ImVec2 text_pos = {row_min.x + ImGui::GetStyle().FramePadding.x,
                                     row_min.y + ImGui::GetStyle().FramePadding.y};
            ImDrawList* draw_list = ImGui::GetWindowDrawList();
            draw_list->AddText(text_pos,
                               ImGui::GetColorU32(ImGuiCol_Text),
                               (std::string(ICON_MDI_FOLDER_STAR " ") + ProjectDisplayName(project_path)).c_str());
            draw_list->AddText({text_pos.x + ImGui::GetFontSize() * 1.6f,
                                text_pos.y + ImGui::GetTextLineHeightWithSpacing()},
                               ImGui::GetColorU32(ImGuiCol_TextDisabled),
                               project_path.c_str());
            if (selected) {
                if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
                    SetProjectPath(project_path);
                }
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("%s", project_path.c_str());
            }
            if (ImGui::BeginPopupContextItem("##ProjectContext")) {
                if (ImGui::MenuItem(ICON_MDI_CLOSE " Remove From List")) {
                    remove_project_path = project_path;
                }
                if (ImGui::MenuItem(ICON_MDI_DELETE " Delete Project Folder")) {
                    pending_delete_project_path_ = project_path;
                    request_delete_project_popup_ = true;
                }
                ImGui::EndPopup();
            }
            ImGui::PopID();
        }
        if (!remove_project_path.empty()) {
            RemoveProjectHistory(remove_project_path);
        }
    }
    ImGui::EndChild();
}

void ResourcePanel::DrawDeleteProjectPopup()
{
    ImGui::TextUnformatted("Delete project folder?");
    ImGui::Spacing();
    ImGui::TextWrapped("%s", pending_delete_project_path_.c_str());
    ImGui::Spacing();
    ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.35f, 1.0f),
                       "This removes the folder and all files inside it.");
    ImGui::Separator();

    if (ImGui::Button("Delete")) {
        std::error_code error;
        std::filesystem::remove_all(pending_delete_project_path_, error);
        if (error) {
            LOG_ERROR("Failed to delete project '{}': {}", pending_delete_project_path_, error.message());
        } else {
            LOG_INFO("Deleted project folder: {}", pending_delete_project_path_);
            const bool deleting_current_project = pending_delete_project_path_ == project_path_;
            RemoveProjectHistory(pending_delete_project_path_);
            if (deleting_current_project) {
                ProjectSettings::GetInstance()->ClearProjectPath();
                project_path_.clear();
                Refresh();
            }
            pending_delete_project_path_.clear();
            ImGui::CloseCurrentPopup();
        }
    }
    ImGui::SameLine();
    if (ImGui::Button("Cancel")) {
        pending_delete_project_path_.clear();
        ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
}

void ResourcePanel::DrawDeleteResourceFilePopup()
{
    ImGui::TextUnformatted("Delete resource file?");
    ImGui::Spacing();
    ImGui::TextWrapped("%s", pending_delete_resource_file_local_path_.c_str());
    ImGui::Separator();

    if (ImGui::Button("Delete")) {
        std::error_code error;
        std::filesystem::remove(pending_delete_resource_file_global_path_, error);
        if (error) {
            LOG_ERROR("Failed to delete resource file '{}': {}",
                      pending_delete_resource_file_local_path_, error.message());
        } else {
            LOG_INFO("Deleted resource file: {}", pending_delete_resource_file_local_path_);
            pending_delete_resource_file_global_path_.clear();
            pending_delete_resource_file_local_path_.clear();
            Refresh();
            ImGui::CloseCurrentPopup();
        }
    }
    ImGui::SameLine();
    if (ImGui::Button("Cancel")) {
        pending_delete_resource_file_global_path_.clear();
        pending_delete_resource_file_local_path_.clear();
        ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
}

void ResourcePanel::OpenProjectBrowser()
{
    if (project_browser_ == nullptr) {
        return;
    }

    project_browser_->SetFlags(ImGuiFileBrowserFlags_SelectDirectory |
                               ImGuiFileBrowserFlags_CreateNewDir |
                               ImGuiFileBrowserFlags_CloseOnEsc);
    project_browser_->SetTitle("Open Gobot Project");
    project_browser_->SetOkText("Open Project");
    project_browser_->Open();
}

void ResourcePanel::HandleProjectBrowser()
{
    if (project_browser_ == nullptr) {
        return;
    }

    project_browser_->Display();
    if (!project_browser_->HasSelected()) {
        return;
    }

    const std::filesystem::path selected_path = project_browser_->GetSelected();
    SetProjectPath(selected_path.string());
    project_browser_->ClearSelected();
}

void ResourcePanel::DrawNewFolderPopup()
{
    ImGui::InputText("Name", new_folder_name_, sizeof(new_folder_name_));
    if (ImGui::Button("Create")) {
        if (CreateFolderInCurrentDirectory(new_folder_name_)) {
            ImGui::CloseCurrentPopup();
        }
    }
    ImGui::SameLine();
    if (ImGui::Button("Cancel")) {
        ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
}

bool ResourcePanel::CreateFolderInCurrentDirectory(const std::string& folder_name)
{
    if (current_dir_ == nullptr || !current_dir_->is_directory) {
        LOG_ERROR("Cannot create resource folder: no current directory.");
        return false;
    }

    if (folder_name.empty() || folder_name.find('/') != std::string::npos || folder_name.find('\\') != std::string::npos) {
        LOG_ERROR("Invalid resource folder name: '{}'.", folder_name);
        return false;
    }

    const std::string local_path = PathJoin(current_dir_->local_path, folder_name);
    const std::string global_path = ProjectSettings::GetInstance()->GlobalizePath(local_path);
    std::error_code error;
    if (!std::filesystem::exists(global_path) &&
            !std::filesystem::create_directories(global_path, error)) {
        LOG_ERROR("Failed to create resource folder '{}': {}", local_path, error.message());
        return false;
    }

    Refresh();
    if (auto it = directories_.find(global_path); it != directories_.end()) {
        ChangeDirectory(it->second.get());
    }

    std::strncpy(new_folder_name_, "resources", sizeof(new_folder_name_) - 1);
    new_folder_name_[sizeof(new_folder_name_) - 1] = '\0';
    return true;
}

DirectoryInformation* ResourcePanel::ProcessDirectory(const std::string& directory_path,
                                                      DirectoryInformation* parent)
{
    auto it = directories_.find(directory_path);
    if(it != directories_.end())
        return it->second.get();

    auto directory_info = std::make_unique<DirectoryInformation>(directory_path, parent);

    if(directory_info->is_directory) {
        for(const auto& file_info : std::filesystem::directory_iterator{directory_info->global_path}) {
            auto subdir = ProcessDirectory(file_info.path().filename(), directory_info.get());
            directory_info->children.push_back(subdir);
        }
    }

    auto res = directory_info.get();
    directories_[directory_info->global_path] = std::move(directory_info);
    return res;
}

void ResourcePanel::Refresh()
{
    project_path_ = ProjectSettings::GetInstance()->GetProjectPath();
    if (project_path_.empty()) {
        update_navigation_path_ = true;
        directories_.clear();
        previous_directory_ = nullptr;
        current_dir_ = nullptr;
        base_project_dir_ = nullptr;
        bread_crumb_data_.clear();
        return;
    }

    auto current_path = current_dir_ != nullptr ? current_dir_->global_path : std::string();

    update_navigation_path_ = true;

    directories_.clear();

    base_project_dir_ = ProcessDirectory("res://", nullptr);
    previous_directory_ = nullptr;
    current_dir_ = nullptr;

    if(!current_path.empty() && directories_.contains(current_path))
        current_dir_ = directories_[current_path].get();
    else
        ChangeDirectory(base_project_dir_);

}

}
