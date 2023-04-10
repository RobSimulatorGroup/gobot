/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-3-20
*/

#include "gobot/editor/imgui/resource_panel.hpp"
#include "gobot/core/config/project_setting.hpp"
#include "gobot/core/string_utils.hpp"
#include "gobot/editor/imgui/imgui_utilities.hpp"
#include "imgui_extension/icon_fonts/icons_material_design_icons.h"
#include "imgui.h"
#include "imgui_internal.h"
#include <QDir>

namespace gobot {

ResourcePanel::ResourcePanel() {
    SetName(ICON_MDI_FOLDER_STAR " Resources###resources");

    show_hidden_files_ = false;
    is_dragging_ = false;
    is_in_list_view_ = true;
    update_navigation_path_ = true;

    filter_ = new ImGuiTextFilter();

    project_path_ = ProjectSettings::GetInstance()->GetProjectPath();

    auto base_directory_handle = ProcessDirectory(std::filesystem::path(project_path_.toStdString()), nullptr);
    base_project_dir_ = directories_[base_directory_handle].get();

    ChangeDirectory(base_project_dir_);
}

void ResourcePanel::ChangeDirectory(const DirectoryInformation* directory)
{
    if(!directory)
        return;

    previous_directory_ = current_dir_;
    current_dir_ = const_cast<DirectoryInformation*>(directory);
    update_navigation_path_ = true;
}

ResourcePanel::~ResourcePanel() {
    delete filter_;
}

bool ResourcePanel::MoveFile(const String& file_path, const String& move_path)
{
    auto moved_path = PathJoin(file_path, move_path);
    QDir dir;
    return dir.exists(project_path_);
}

void ResourcePanel::OnImGuiContent() {
    auto window_size = ImGui::GetWindowSize();
    bool vertical   = window_size.y > window_size.x;
    {
        if(!vertical) {
            ImGui::BeginColumns("ResourcePanelColumns", 2, ImGuiOldColumnFlags_NoResize);
            ImGui::SetColumnWidth(0, ImGui::GetWindowContentRegionMax().x / 3.0f);
            ImGui::BeginChild("##folders_common");
        } else
            ImGui::BeginChild("##folders_common", ImVec2(0, ImGui::GetWindowHeight() / 3.0f));

        {
            ImGui::BeginChild("##folders");
            {
                DrawFolder(base_project_dir_, true);
            }
            ImGui::EndChild();
        }

        ImGui::EndChild();
    }


    if(ImGui::BeginDragDropTarget()) {
        auto data = ImGui::AcceptDragDropPayload("selectable", ImGuiDragDropFlags_AcceptNoDrawDefaultRect);
        if(data) {
            String file = (char*)data->Data;
            if(MoveFile(file, move_path_)) {
                LOG_INFO("Moved File: " + file + " to " + move_path_);
            }
            is_dragging_ = false;
        }
        ImGui::EndDragDropTarget();
    }
    float offset = 0.0f;
    if(!vertical) {
        ImGui::NextColumn();
    } else {
        offset = ImGui::GetWindowHeight() / 3.0f + 6.0f;
        ImGui::SeparatorEx(ImGuiSeparatorFlags_Horizontal);
    }

    ImGui::BeginChild("##directory_structure", ImVec2(0, ImGui::GetWindowHeight() - ImGui::GetFrameHeightWithSpacing() * 2.6f - offset));
    {
        ImGui::BeginChild("##directory_breadcrumbs", ImVec2(ImGui::GetColumnWidth(), ImGui::GetFrameHeightWithSpacing()));

        if(ImGui::Button(ICON_MDI_ARROW_LEFT)) {
            if(current_dir_ != base_project_dir_) {
                previous_directory_    = current_dir_;
                current_dir_           = current_dir_->parent;
                update_navigation_path_ = true;
            }
        }
        ImGui::SameLine();
        if(ImGui::Button(ICON_MDI_ARROW_RIGHT)) {
            previous_directory_ = current_dir_;
            // m_CurrentDir = m_LastNavPath;
            update_navigation_path_ = true;
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

        if(is_in_list_view_) {
            if(ImGui::Button(ICON_MDI_VIEW_GRID)) {
                is_in_list_view_ = !is_in_list_view_;
            }
            ImGui::SameLine();
        } else {
            if(ImGui::Button(ICON_MDI_VIEW_LIST)) {
                is_in_list_view_ = !is_in_list_view_;
            }
            ImGui::SameLine();
        }

        ImGui::TextUnformatted(ICON_MDI_MAGNIFY);
        ImGui::SameLine();

        filter_->Draw("##Filter", ImGui::GetContentRegionAvail().x - ImGui::GetStyle().IndentSpacing);

        ImGui::EndChild();

        {
            ImGui::BeginChild("##Scrolling");

            int shownIndex = 0;

            float xAvail = ImGui::GetContentRegionAvail().x;

            grid_items_per_row_ = (int)floor(xAvail / (grid_size_ + ImGui::GetStyle().ItemSpacing.x));
            grid_items_per_row_ = std::max(1, grid_items_per_row_);

            if(is_in_list_view_) {
                for(int i = 0; i < current_dir_->children.size(); i++) {
                    if(current_dir_->children.size() > 0) {
                        if(!show_hidden_files_ &&
                            IsHiddenFile(current_dir_->children[i]->file_path.filename().c_str())) {
                            continue;
                        }

                        if(filter_->IsActive()) {
                            if(!filter_->PassFilter(current_dir_->children[i]->file_path.filename().string().c_str())) {
                                continue;
                            }
                        }

                        bool doubleClicked = RenderFile(i, !current_dir_->children[i]->is_file, shownIndex, !is_in_list_view_);

                        if(doubleClicked)
                            break;
                        shownIndex++;
                    }
                }
            } else {
                for(int i = 0; i < current_dir_->children.size(); i++) {
                    if(!show_hidden_files_ &&
                        IsHiddenFile(current_dir_->children[i]->file_path.filename().c_str())) {
                        continue;
                    }

                    if(filter_->IsActive()) {
                        if(!filter_->PassFilter(current_dir_->children[i]->file_path.filename().string().c_str())) {
                            continue;
                        }
                    }

                    bool doubleClicked = RenderFile(i, !current_dir_->children[i]->is_file, shownIndex, !is_in_list_view_);

                    if(doubleClicked)
                        break;
                    shownIndex++;
                }
            }

            if(ImGui::BeginPopupContextWindow()) {
                if(ImGui::Selectable("Import New")) {
                    // TODO(wqq)
                }

                if(ImGui::Selectable("Refresh")) {
                    Refresh();
                }

                if(ImGui::Selectable("New folder")) {
                    // TODO(wqq)
//                    std::filesystem::create_directory(std::filesystem::path(
//                            m_BasePath + "/" + current_dir_->file_path.string() + "/NewFolder"));
                    Refresh();
                }

                ImGui::EndPopup();
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
            String a = (char*)data->Data;
            if(MoveFile(a, move_path_)) {
                LOG_INFO("Moved File: " + a + " to " + move_path_);
            }
            is_dragging_ = false;
        }
        ImGui::EndDragDropTarget();
    }
}

bool ResourcePanel::RenderFile(int dirIndex, bool folder, int shownIndex, bool gridView)
{
    bool doubleClicked = false;

    if(gridView) {
        ImGui::BeginGroup();

        if(ImGui::Button(folder ? ICON_MDI_FOLDER : ICON_MDI_FILE)) {
        }

        if(ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
            doubleClicked = true;
        }

        auto newFname = SimplifyPath(current_dir_->children[dirIndex]->file_path.filename().c_str());

        ImGui::TextUnformatted(newFname.toStdString().c_str());
        ImGui::EndGroup();

        if((shownIndex + 1) % grid_items_per_row_ != 0)
            ImGui::SameLine();
    } else {
        ImGui::TextUnformatted(folder ? ICON_MDI_FOLDER : ICON_MDI_FILE);
        ImGui::SameLine();
        if(ImGui::Selectable(current_dir_->children[dirIndex]->file_path.filename().string().c_str(),
                             false, ImGuiSelectableFlags_AllowDoubleClick)) {
            if(ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left))
            {
                doubleClicked = true;
            }
        }
    }

    ImGuiUtilities::Tooltip(current_dir_->children[dirIndex]->file_path.filename().string().c_str());

    if(doubleClicked) {
        if(folder) {
            previous_directory_    = current_dir_;
            current_dir_           = current_dir_->children[dirIndex];
            update_navigation_path_ = true;
        } else {
//            m_Editor->FileOpenCallback(m_BasePath + "/" + m_CurrentDir->Children[dirIndex]->FilePath.string());
        }
    }

    if(ImGui::BeginDragDropSource(ImGuiDragDropFlags_SourceAllowNullID))
    {
        ImGui::TextUnformatted(ICON_MDI_FILE);

        ImGui::SameLine();
        move_path_ = project_path_ + "/" + current_dir_->children[dirIndex]->file_path.string().c_str();
        ImGui::TextUnformatted(move_path_.toStdString().c_str());
        size_t size = sizeof(const char*) + strlen(move_path_.toStdString().c_str());
        ImGui::SetDragDropPayload("AssetFile", move_path_.toStdString().c_str(), size);
        is_dragging_ = true;
        ImGui::EndDragDropSource();
    }

    return doubleClicked;
}

void ResourcePanel::RenderBottom()
{
    ImGui::BeginChild("##nav", ImVec2(ImGui::GetColumnWidth(), ImGui::GetFontSize() * 1.8f), true, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
    {
        int secIdx = 0, newPwdLastSecIdx = -1;

        auto& AssetsDir = current_dir_->file_path;

        size_t PhysicalPathCount = 0;

        int dirIndex = 0;
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.1f, 0.2f, 0.7f, 0.0f));

        for(auto& directory : bread_crumb_data_) {
            const std::string& directoryName = directory->file_path.filename().string();
            if(ImGui::SmallButton(directoryName.c_str()))
                ChangeDirectory(directory);

            ImGui::SameLine();
        }

        ImGui::PopStyleColor();

        if(newPwdLastSecIdx >= 0) {
            int i = 0;
            std::filesystem::path newPwd;
            for(auto& sec : AssetsDir) {
                if(i++ > newPwdLastSecIdx)
                    break;
                newPwd /= sec;
            }
#ifdef _WIN32
            if(newPwdLastSecIdx == 0)
                    newPwd /= "\\";
#endif

            previous_directory_    = current_dir_;
            current_dir_           = directories_[newPwd.c_str()].get();
            update_navigation_path_ = true;
        }

        ImGui::SameLine();
    }

    if(!is_in_list_view_) {
        ImGui::SliderFloat("##GridSize", &grid_size_, 40.0f, 400.0f);
    }
    ImGui::EndChild();
}

String ResourcePanel::ProcessDirectory(const std::filesystem::path& directory_path,
                                       const DirectoryInformation* parent)
{
    auto* directory = directories_[String::fromStdString(directory_path.string())].get();
    if(directory)
        return directory->file_path.string().c_str();

    auto directory_info = std::make_unique<DirectoryInformation>(directory_path, !std::filesystem::is_directory(directory_path));
    directory_info->parent = const_cast<DirectoryInformation*>(parent);

    if(directory_path.c_str() == project_path_.toStdString().c_str())
        directory_info->file_path = project_path_.toStdString();
    else
        directory_info->file_path = std::filesystem::relative(directory_path, project_path_.toStdString());

    if(std::filesystem::is_directory(directory_path)) {
        for(auto entry : std::filesystem::directory_iterator(directory_path)) {
            if(!show_hidden_files_ && IsHiddenFile(entry.path().c_str())) {
                continue;
            }
            auto subdirHandle = ProcessDirectory(entry.path(), directory_info.get());
            directory_info->children.push_back(directories_[subdirHandle].get());
        }
    }

    String res = directory_info->file_path.string().c_str();
    directories_[directory_info->file_path.c_str()] = std::move(directory_info);
    return res;
}

void ResourcePanel::DrawFolder(const DirectoryInformation* dir_info, bool default_open)
{
    ImGuiTreeNodeFlags nodeFlags = ((dir_info == current_dir_) ? ImGuiTreeNodeFlags_Selected : 0);
    nodeFlags |= ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_OpenOnDoubleClick;

    if(dir_info->parent == nullptr)
        nodeFlags |= ImGuiTreeNodeFlags_Framed;

    const ImColor TreeLineColor = ImColor(128, 128, 128, 128);
    const float SmallOffsetX    = 6.0f * ImGui::GetWindowDpiScale();
    ImDrawList* drawList        = ImGui::GetWindowDrawList();

    if(!dir_info->is_file) {
        bool containsFolder = false;

        for(auto& file : dir_info->children) {
            if(!file->is_file) {
                containsFolder = true;
                break;
            }
        }
        if(!containsFolder)
            nodeFlags |= ImGuiTreeNodeFlags_Leaf;

        if(default_open)
            nodeFlags |= ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_Leaf;

        nodeFlags |= ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_AllowItemOverlap | ImGuiTreeNodeFlags_SpanAvailWidth;

        bool isOpen = ImGui::TreeNodeEx((void*)(intptr_t)dir_info, nodeFlags, "");

        const char* folderIcon = ((isOpen && containsFolder) || current_dir_ == dir_info) ? ICON_MDI_FOLDER_OPEN : ICON_MDI_FOLDER;
        ImGui::SameLine();
        ImGui::PushStyleColor(ImGuiCol_Text, ImGuiUtilities::GetIconColor());
        ImGui::Text("%s ", folderIcon);
        ImGui::PopStyleColor();
        ImGui::SameLine();
        ImGui::TextUnformatted((const char*)dir_info->file_path.filename().string().c_str());

        ImVec2 verticalLineStart = ImGui::GetCursorScreenPos();

        if(ImGui::IsItemClicked()) {
            previous_directory_    = current_dir_;
            current_dir_           = const_cast<DirectoryInformation*>(dir_info);
            update_navigation_path_ = true;
        }

        if(isOpen && containsFolder) {
            verticalLineStart.x += SmallOffsetX; // to nicely line up with the arrow symbol
            ImVec2 verticalLineEnd = verticalLineStart;

            for(int i = 0; i < dir_info->children.size(); i++) {
                if(!dir_info->children[i]->is_file) {
                    auto currentPos = ImGui::GetCursorScreenPos();

                    ImGui::Indent(10.0f);

                    bool containsFolderTemp = false;
                    for(auto& file : dir_info->children[i]->children) {
                        if(!file->is_file) {
                            containsFolderTemp = true;
                            break;
                        }
                    }
                    float HorizontalTreeLineSize = 16.0f * ImGui::GetWindowDpiScale(); // chosen arbitrarily

                    if(containsFolderTemp)
                        HorizontalTreeLineSize *= 0.5f;
                    DrawFolder(dir_info->children[i]);

                    const ImRect childRect = ImRect(currentPos, currentPos + ImVec2(0.0f, ImGui::GetFontSize()));

                    const float midpoint = (childRect.Min.y + childRect.Max.y) * 0.5f;
                    drawList->AddLine(ImVec2(verticalLineStart.x, midpoint), ImVec2(verticalLineStart.x + HorizontalTreeLineSize, midpoint), TreeLineColor);
                    verticalLineEnd.y = midpoint;

                    ImGui::Unindent(10.0f);
                }
            }

            drawList->AddLine(verticalLineStart, verticalLineEnd, TreeLineColor);

            ImGui::TreePop();
        }

        if(isOpen && !containsFolder)
            ImGui::TreePop();
    }

    if(is_dragging_ && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenBlockedByActiveItem)) {
        move_path_ = dir_info->file_path.string().c_str();
    }
}

void ResourcePanel::Refresh()
{
    project_path_ = ProjectSettings::GetInstance()->GetProjectPath();

    auto base_directory_handle = ProcessDirectory(std::filesystem::path(project_path_.toStdString()), nullptr);
    base_project_dir_ = directories_[base_directory_handle].get();

    auto current_path = current_dir_->file_path;

    update_navigation_path_ = true;

    directories_.clear();
    previous_directory_ = nullptr;
    current_dir_ = nullptr;

    if(directories_.find(current_path.string().c_str()) != directories_.end())
        current_dir_ = directories_[current_path.string().c_str()].get();
    else
        ChangeDirectory(base_project_dir_);

}

}
