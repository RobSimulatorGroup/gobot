/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-3-10
*/


#include "gobot/editor/imgui/console_panel.hpp"
#include "gobot/editor/imgui/imgui_utilities.hpp"
#include "imgui_extension/icon_fonts/icons_material_design_icons.h"

#include "imgui.h"

namespace gobot {

ConsoleMessage::ConsoleMessage(const String& message, Level level, const String& source, int thread_id)
        : message_(message)
        , level_(level)
        , source_(source)
        , thread_id_(thread_id)
        , message_id_(std::hash<String>()(message))
{
}

void ConsoleMessage::OnImGUIRender()
{
    if(ConsolePanel::s_message_buffer_render_filter & level_)
    {
//        ImGuiUtilities::ScopedID((int)message_id_);
//        ImGui::PushStyleColor(ImGuiCol_Text, GetRenderColour(level_));
        auto levelIcon = GetLevelIcon(level_);
        ImGui::TextUnformatted(levelIcon);
//        ImGui::PopStyleColor();
        ImGui::SameLine();
        ImGui::TextUnformatted(message_.toStdString().c_str());
        if(ImGui::BeginPopupContextItem(message_.toStdString().c_str()))
        {
            if(ImGui::MenuItem("Copy"))
            {
                ImGui::SetClipboardText(message_.toStdString().c_str());
            }

            ImGui::EndPopup();
        }

        if(ImGui::IsItemHovered())
        {
            ImGui::SetTooltip("%s", source_.toStdString().c_str());
        }

        if(count_ > 1)
        {
            ImGui::SameLine(ImGui::GetContentRegionAvail().x - (count_ > 99 ? ImGui::GetFontSize() * 1.7f : ImGui::GetFontSize() * 1.5f));
            ImGui::Text("%d", count_);
        }
    }
}

const char* ConsoleMessage::GetLevelIcon(Level level)
{
    switch(level)
    {
        case ConsoleMessage::Level::Trace:
            return ICON_MDI_MESSAGE_TEXT;
        case ConsoleMessage::Level::Info:
            return ICON_MDI_INFORMATION;
        case ConsoleMessage::Level::Debug:
            return ICON_MDI_BUG;
        case ConsoleMessage::Level::Warn:
            return ICON_MDI_ALERT;
        case ConsoleMessage::Level::Error:
            return ICON_MDI_CLOSE_OCTAGON;
        case ConsoleMessage::Level::Critical:
            return ICON_MDI_ALERT_OCTAGRAM;
        default:
            return "Unknown name";
    }
}

const char* ConsoleMessage::GetLevelName(Level level)
{
    switch(level)
    {
        case ConsoleMessage::Level::Trace:
            return ICON_MDI_MESSAGE_TEXT " Trace";
        case ConsoleMessage::Level::Info:
            return ICON_MDI_INFORMATION " Info";
        case ConsoleMessage::Level::Debug:
            return ICON_MDI_BUG " Debug";
        case ConsoleMessage::Level::Warn:
            return ICON_MDI_ALERT " Warning";
        case ConsoleMessage::Level::Error:
            return ICON_MDI_CLOSE_OCTAGON " Error";
        case ConsoleMessage::Level::Critical:
            return ICON_MDI_ALERT_OCTAGRAM " Critical";
        default:
            return "Unknown name";
    }
}

Color ConsoleMessage::GetRenderColour(Level level)
{
    switch(level)
    {
        case ConsoleMessage::Level::Trace:
            return { 0.75f, 0.75f, 0.75f, 1.00f }; // Gray
        case ConsoleMessage::Level::Info:
            return { 0.40f, 0.70f, 1.00f, 1.00f }; // Blue
        case ConsoleMessage::Level::Debug:
            return { 0.00f, 0.50f, 0.50f, 1.00f }; // Cyan
        case ConsoleMessage::Level::Warn:
            return { 1.00f, 1.00f, 0.00f, 1.00f }; // Yellow
        case ConsoleMessage::Level::Error:
            return { 1.00f, 0.25f, 0.25f, 1.00f }; // Red
        case ConsoleMessage::Level::Critical:
            return { 0.6f, 0.2f, 0.8f, 1.00f }; // Purple
        default:
            return { 1.00f, 1.00f, 1.00f, 1.00f };
    }
}

uint32_t ConsolePanel::s_message_buffer_render_filter = 0;
uint16_t ConsolePanel::s_message_buffer_capacity = 200;
uint16_t ConsolePanel::s_message_buffer_size = 0;
uint16_t ConsolePanel::s_message_buffer_begin = 0;
std::vector<Ref<ConsoleMessage>> ConsolePanel::s_message_buffer = std::vector<Ref<ConsoleMessage>>(200);
bool ConsolePanel::s_allow_scrolling_to_bottom = true;
bool ConsolePanel::s_request_scroll_to_bottom = false;

ConsolePanel::ConsolePanel()
{
    name_                      = ICON_MDI_VIEW_LIST " Console###console";
    s_message_buffer_render_filter = ConsoleMessage::Level::Trace |
                                     ConsoleMessage::Level::Info |
                                     ConsoleMessage::Level::Debug |
                                     ConsoleMessage::Warn |
                                     ConsoleMessage::Error |
                                     ConsoleMessage::Critical;
}

void ConsolePanel::AddMessage(const Ref<ConsoleMessage>& message)
{
    if(message->level_ == 0)
        return;

    auto messageStart = s_message_buffer.begin() + s_message_buffer_begin;
    if(*messageStart) // If contains old message here
    {
        for(auto messIt = messageStart; messIt != s_message_buffer.end(); messIt++)
        {
            if(message->GetMessageID() == (*messIt)->GetMessageID())
            {
                (*messIt)->IncreaseCount();
                return;
            }
        }
    }

    if(s_message_buffer_begin != 0) // Skipped first messages in vector
    {
        for(auto messIt = s_message_buffer.begin(); messIt != messageStart; messIt++)
        {
            if(*messIt)
            {
                if(message->GetMessageID() == (*messIt)->GetMessageID())
                {
                    (*messIt)->IncreaseCount();
                    return;
                }
            }
        }
    }

    *(s_message_buffer.begin() + s_message_buffer_begin) = message;
    if(++s_message_buffer_begin == s_message_buffer_capacity)
        s_message_buffer_begin = 0;
    if(s_message_buffer_size < s_message_buffer_capacity)
        s_message_buffer_size++;

    if(s_allow_scrolling_to_bottom)
        s_request_scroll_to_bottom = true;
}

void ConsolePanel::Flush()
{
    for(auto message = s_message_buffer.begin(); message != s_message_buffer.end(); message++)
        (*message) = nullptr;
    s_message_buffer_begin = 0;
}

void ConsolePanel::OnImGui()
{
    auto flags = ImGuiWindowFlags_NoCollapse;
    ImGui::SetNextWindowSize(ImVec2(640, 480), ImGuiCond_FirstUseEver);
    ImGui::Begin(name_.toStdString().c_str(), &active_, flags);
    {
        ImGuiRenderHeader();
        ImGui::Separator();
        ImGuiRenderMessages();
    }
    ImGui::End();
}

void ConsolePanel::ImGuiRenderHeader()
{
    ImGuiStyle& style = ImGui::GetStyle();
    ImGui::AlignTextToFramePadding();
    // Button for advanced settings
    {
//        ImGuiUtilities::ScopedColour buttonColour(ImGuiCol_Button, ImVec4(0.0f, 0.0f, 0.0f, 0.0f));
        if(ImGui::Button(ICON_MDI_COGS))
            ImGui::OpenPopup("SettingsPopup");
    }
    if(ImGui::BeginPopup("SettingsPopup"))
    {
        // Checkbox for scrolling lock
        ImGui::Checkbox("Scroll to bottom", &s_allow_scrolling_to_bottom);

        // Button to clear the console
        if(ImGui::Button("Clear console"))
            Flush();

        ImGui::EndPopup();
    }

    ImGui::SameLine();
    ImGui::TextUnformatted(ICON_MDI_MAGNIFY);
    ImGui::SameLine();

    float spacing                   = ImGui::GetStyle().ItemSpacing.x;
    ImGui::GetStyle().ItemSpacing.x = 2;
    float levelButtonWidth          = (ImGui::CalcTextSize(ConsoleMessage::GetLevelIcon(ConsoleMessage::Level::Debug)) + ImGui::GetStyle().FramePadding * 2.0f).x;
    float levelButtonWidths         = (levelButtonWidth + ImGui::GetStyle().ItemSpacing.x) * 6;

    {
//        ImGuiUtilities::ScopedFont boldFont(ImGui::GetIO().Fonts->Fonts[1]);
//        ImGuiUtilities::ScopedStyle frameBorder(ImGuiStyleVar_FrameBorderSize, 0.0f);
//        ImGuiUtilities::ScopedColour frameColour(ImGuiCol_FrameBg, IM_COL32(0, 0, 0, 0));
        filter_.Draw("###ConsoleFilter", ImGui::GetContentRegionAvail().x - (levelButtonWidths));
//        ImGuiUtilities::DrawItemActivityOutline(2.0f, false);
    }

    ImGui::SameLine(); // ImGui::GetWindowWidth() - levelButtonWidths);

    for(int i = 0; i < 6; i++)
    {
//        ImGuiUtilities::ScopedColour buttonColour(ImGuiCol_Button, ImVec4(0.0f, 0.0f, 0.0f, 0.0f));
        ImGui::SameLine();
        auto level = ConsoleMessage::Level(std::pow(2, i));

        bool levelEnabled = s_message_buffer_render_filter & level;
        if(levelEnabled)
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.5f, 0.5, 0.5f, 0.5f));
        else
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.5f, 0.5, 0.5f, 0.5f));

        if(ImGui::Button(ConsoleMessage::GetLevelIcon(level)))
        {
            s_message_buffer_render_filter ^= level;
        }

        if(ImGui::IsItemHovered())
        {
            ImGui::SetTooltip("%s", ConsoleMessage::GetLevelName(level));
        }
        ImGui::PopStyleColor();
    }

    ImGui::GetStyle().ItemSpacing.x = spacing;

    if(!filter_.IsActive())
    {
        ImGui::SameLine();
//        ImGuiUtilities::ScopedFont boldFont(ImGui::GetIO().Fonts->Fonts[1]);
        ImGui::SetCursorPosX(ImGui::GetFontSize() * 4.0f);
//        ImGuiUtilities::ScopedStyle padding(ImGuiStyleVar_FramePadding, ImVec2(0.0f, ImGui::GetStyle().FramePadding.y));
        ImGui::TextUnformatted("Search...");
    }
}

void ConsolePanel::ImGuiRenderMessages()
{
    ImGui::BeginChild("ScrollRegion", ImVec2(0, 0), false, ImGuiWindowFlags_HorizontalScrollbar);
    {
        // ImGuiUtilities::AlternatingRowsBackground();

        auto messageStart = s_message_buffer.begin() + s_message_buffer_begin;
        if(*messageStart) // If contains old message here
        {
            for(auto message = messageStart; message != s_message_buffer.end(); message++)
            {
                if(filter_.IsActive())
                {
                    if(filter_.PassFilter((*message)->message_.toStdString().c_str()))
                    {
                        (*message)->OnImGUIRender();
                    }
                }
                else
                {
                    (*message)->OnImGUIRender();
                }
            }
        }

        if(s_message_buffer_begin != 0) // Skipped first messages in vector
        {
            for(auto message = s_message_buffer.begin(); message != messageStart; message++)
            {
                if(*message)
                {
                    if(filter_.IsActive())
                    {
                        if(filter_.PassFilter((*message)->message_.toStdString().c_str()))
                        {
                            (*message)->OnImGUIRender();
                        }
                    }
                    else
                    {
                        (*message)->OnImGUIRender();
                    }
                }
            }
        }

        if(s_request_scroll_to_bottom && ImGui::GetScrollMaxY() > 0)
        {
            ImGui::SetScrollHereY(1.0f);
            s_request_scroll_to_bottom = false;
        }
    }
    ImGui::EndChild();
}

}