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
#include <QFileInfo>

namespace gobot {

DirectoryInformation::DirectoryInformation(const String& _this_path, DirectoryInformation* _parent)
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

    is_file = QFileInfo(global_path).isFile();
}


ResourcePanel::ResourcePanel()
{
    SetName(ICON_MDI_FOLDER_STAR " Resources###resources");

    show_hidden_files_ = false;
    is_dragging_ = false;
    is_in_list_view_ = true;
    update_navigation_path_ = true;

    filter_ = new ImGuiTextFilter();

    project_path_ = ProjectSettings::GetInstance()->GetProjectPath();

    base_project_dir_ = ProcessDirectory("res://", nullptr);

    ChangeDirectory(base_project_dir_);
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
}

bool ResourcePanel::MoveFile(const String& file_path, const String& move_path)
{
    auto moved_path = PathJoin(file_path, move_path);
    QDir dir;
    return dir.exists(moved_path);
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

            int shown_index = 0;

            float x_avail = ImGui::GetContentRegionAvail().x;

            grid_items_per_row_ = (int)floor(x_avail / (grid_size_ + ImGui::GetStyle().ItemSpacing.x));
            grid_items_per_row_ = std::max(1, grid_items_per_row_);

            if(is_in_list_view_) {
                for(int i = 0; i < current_dir_->children.size(); i++) {
                    if(current_dir_->children.size() > 0) {
                        if(filter_->IsActive()) {
                            if(!filter_->PassFilter(current_dir_->children[i]->global_path.toStdString().c_str())) {
                                continue;
                            }
                        }

                        bool double_clicked = RenderFile(i, !current_dir_->children[i]->is_file, shown_index, !is_in_list_view_);

                        if(double_clicked)
                            break;
                        shown_index++;
                    }
                }
            } else {
                for(int i = 0; i < current_dir_->children.size(); i++) {
                    if(filter_->IsActive()) {
                        if(!filter_->PassFilter(current_dir_->children[i]->global_path.toStdString().c_str())) {
                            continue;
                        }
                    }

                    bool doubleClicked = RenderFile(i, !current_dir_->children[i]->is_file, shown_index, !is_in_list_view_);

                    if(doubleClicked)
                        break;
                    shown_index++;
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
    bool double_clicked = false;

    if(gridView) {
        ImGui::BeginGroup();

        if(ImGui::Button(folder ? ICON_MDI_FOLDER : ICON_MDI_FILE)) {
        }

        if(ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
            double_clicked = true;
        }

        auto newFname = SimplifyPath(current_dir_->children[dirIndex]->this_path.toStdString().c_str());

        ImGui::TextUnformatted(newFname.toStdString().c_str());
        ImGui::EndGroup();

        if((shownIndex + 1) % grid_items_per_row_ != 0)
            ImGui::SameLine();
    } else {
        ImGui::TextUnformatted(folder ? ICON_MDI_FOLDER : ICON_MDI_FILE);
        ImGui::SameLine();
        if(ImGui::Selectable(current_dir_->children[dirIndex]->this_path.toStdString().c_str(),
                             false, ImGuiSelectableFlags_AllowDoubleClick)) {
            if(ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left))
            {
                double_clicked = true;
            }
        }
    }

    ImGuiUtilities::Tooltip(current_dir_->children[dirIndex]->global_path.toStdString().c_str());

    if(double_clicked) {
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
        move_path_ = project_path_ + "/" + current_dir_->children[dirIndex]->global_path;
        ImGui::TextUnformatted(move_path_.toStdString().c_str());
        size_t size = sizeof(const char*) + strlen(move_path_.toStdString().c_str());
        ImGui::SetDragDropPayload("AssetFile", move_path_.toStdString().c_str(), size);
        is_dragging_ = true;
        ImGui::EndDragDropSource();
    }

    return double_clicked;
}

void ResourcePanel::RenderBottom()
{
    ImGui::BeginChild("##nav", ImVec2(ImGui::GetColumnWidth(), ImGui::GetFontSize() * 1.8f), true, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
    {
        int secIdx = 0, newPwdLastSecIdx = -1;

        auto& AssetsDir = current_dir_->global_path;

        size_t PhysicalPathCount = 0;

        int dirIndex = 0;
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.1f, 0.2f, 0.7f, 0.0f));

        for(auto& directory : bread_crumb_data_) {
            const std::string& directoryName = directory->this_path.toStdString();
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
//                newPwd /= sec;
            }
//#ifdef _WIN32
//            if(newPwdLastSecIdx == 0)
//                    newPwd /= "\\";
//#endif

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

DirectoryInformation* ResourcePanel::ProcessDirectory(const String& directory_path,
                                                      DirectoryInformation* parent)
{
    auto it = directories_.find(directory_path);
    if(it != directories_.end())
        return it->second.get();

    auto directory_info = std::make_unique<DirectoryInformation>(directory_path, parent);

    QDir dir(directory_info->global_path);

    if(!directory_info->is_file) {
        QStringList file_list;
        if (show_hidden_files_) {
            file_list = dir.entryList(QDir::AllEntries | QDir::Hidden | QDir::NoDotAndDotDot);
        } else  {
            file_list = dir.entryList(QDir::AllEntries | QDir::NoDotAndDotDot);
        }

        for(auto file_info : file_list) {
            auto subdir = ProcessDirectory(file_info, directory_info.get());
            directory_info->children.push_back(subdir);
        }
    }

    auto res = directory_info.get();
    directories_[directory_info->global_path] = std::move(directory_info);
    return res;
}

void ResourcePanel::DrawFolder(DirectoryInformation* dir_info, bool default_open)
{
    ImGuiTreeNodeFlags node_flags = ((dir_info == current_dir_) ? ImGuiTreeNodeFlags_Selected : 0);
    node_flags |= ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_OpenOnDoubleClick | ImGuiTreeNodeFlags_DefaultOpen;

    if(dir_info->parent == nullptr)
        node_flags |= ImGuiTreeNodeFlags_Framed;

    const ImColor tree_line_color = ImColor(128, 128, 128, 128);
    const float small_offset_x    = 6.0f * ImGui::GetWindowDpiScale();
    ImDrawList* draw_list        = ImGui::GetWindowDrawList();

    if(!dir_info->is_file) {
        bool contains_folder = false;

        for(auto& file : dir_info->children) {
            if(!file->is_file) {
                contains_folder = true;
                break;
            }
        }
        if(!contains_folder)
            node_flags |= ImGuiTreeNodeFlags_Leaf;

        if(default_open)
            node_flags |= ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_Leaf;

        node_flags |= ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_AllowItemOverlap | ImGuiTreeNodeFlags_SpanAvailWidth;

        bool is_open = ImGui::TreeNodeEx((void*)(intptr_t)dir_info, node_flags, "");

        const char* folder_icon = ((is_open && contains_folder) || current_dir_ == dir_info) ? ICON_MDI_FOLDER_OPEN : ICON_MDI_FOLDER;
        ImGui::SameLine();
        ImGui::PushStyleColor(ImGuiCol_Text, ImGuiUtilities::GetIconColor());
        ImGui::Text("%s ", folder_icon);
        ImGui::PopStyleColor();
        ImGui::SameLine();
        ImGui::TextUnformatted((const char*)dir_info->this_path.toStdString().c_str());

        ImVec2 vertical_line_start = ImGui::GetCursorScreenPos();

        if(ImGui::IsItemClicked()) {
            previous_directory_    = current_dir_;
            current_dir_           = const_cast<DirectoryInformation*>(dir_info);
            update_navigation_path_ = true;
        }

        if(is_open && contains_folder) {
            vertical_line_start.x += small_offset_x; // to nicely line up with the arrow symbol
            ImVec2 vertical_line_end = vertical_line_start;

            for(auto i : dir_info->children) {
                if(!i->is_file) {
                    auto current_pos = ImGui::GetCursorScreenPos();

                    ImGui::Indent(10.0f);

                    bool contains_folder_temp = false;
                    for(auto& file : i->children) {
                        if(!file->is_file) {
                            contains_folder_temp = true;
                            break;
                        }
                    }
                    float horizontal_tree_line_size = 16.0f * ImGui::GetWindowDpiScale(); // chosen arbitrarily

                    if(contains_folder_temp)
                        horizontal_tree_line_size *= 0.5f;
                    DrawFolder(i);

                    const ImRect child_rect = ImRect(current_pos, current_pos + ImVec2(0.0f, ImGui::GetFontSize()));

                    const float midpoint = (child_rect.Min.y + child_rect.Max.y) * 0.5f;
                    draw_list->AddLine(ImVec2(vertical_line_start.x, midpoint),
                                       ImVec2(vertical_line_start.x + horizontal_tree_line_size, midpoint),
                                       tree_line_color);
                    vertical_line_end.y = midpoint;

                    ImGui::Unindent(10.0f);
                }
            }

            draw_list->AddLine(vertical_line_start, vertical_line_end, tree_line_color);

            ImGui::TreePop();
        }

        if(is_open && !contains_folder)
            ImGui::TreePop();
    }

    if(is_dragging_ && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenBlockedByActiveItem)) {
        move_path_ = dir_info->global_path;
    }
}

void ResourcePanel::Refresh()
{
    project_path_ = ProjectSettings::GetInstance()->GetProjectPath();

    base_project_dir_ = ProcessDirectory("res://", nullptr);

    auto current_path = current_dir_->global_path;

    update_navigation_path_ = true;

    directories_.clear();
    previous_directory_ = nullptr;
    current_dir_ = nullptr;

    if(directories_.find(current_path) != directories_.end())
        current_dir_ = directories_[current_path].get();
    else
        ChangeDirectory(base_project_dir_);

}

}
