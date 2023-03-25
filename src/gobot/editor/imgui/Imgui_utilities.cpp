/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-3-10
*/

#include "imgui_internal.h"
#include "gobot/editor/imgui/imgui_utilities.hpp"
#include "gobot/error_macros.hpp"
#include "gobot/rendering/rendering_server_globals.hpp"
#include "gobot/rendering/texture_storage.hpp"

namespace gobot {

Color ImGuiUtilities::s_selected_color = {0.28f, 0.56f, 0.9f, 1.0f};
Color ImGuiUtilities::s_icon_color = {0.2f, 0.2f, 0.2f, 1.0f};
char* ImGuiUtilities::s_multiline_buffer = nullptr;


Color ImGuiUtilities::GetSelectedColor() {
    return s_selected_color;
}

void ImGuiUtilities::Tooltip(const String& text) {
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(5, 5));
    Tooltip(text.toStdString().c_str());
    ImGui::PopStyleVar();
}

void ImGuiUtilities::Tooltip(const char* text) {
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(5, 5));

    if(ImGui::IsItemHovered()) {
        ImGui::BeginTooltip();
        ImGui::TextUnformatted(text);
        ImGui::EndTooltip();
    }

    ImGui::PopStyleVar();
}

void ImGuiUtilities::Image(const RenderRID& texture_id, const Vector2f& size, const Vector2f& uv0,
                           const Vector2f& uv1, const Color& tintCol, const Color& borderCol) {
    ImVec2 _uv0 = uv0;
    ImVec2 _uv1 = uv1;

    ERR_FAIL_COND_MSG(texture_id.IsNull(), "Input texture_id is invalid");

    if(RSG::texture_storage->IsRenderTarget(texture_id) && RSG::texture_storage->IsOriginBottomLeft()) {
        _uv0 = {0.0f, 1.0f};
        _uv1 = {1.0f, 0.0f};
    }

    ImGui::Image(texture_id, size, _uv0, _uv1, tintCol, borderCol);
}


void ImGuiUtilities::SetTheme(Theme theme)
{
    static const float max = 255.0f;

    auto& style     = ImGui::GetStyle();
    ImVec4* colors = style.Colors;
    s_selected_color  = Color(0.28f, 0.56f, 0.9f, 1.0f);

    if(theme == Black) {
        ImGui::StyleColorsDark();
        colors[ImGuiCol_Text]                  = ImVec4(1.00f, 1.00f, 1.00f, 1.00f);
        colors[ImGuiCol_TextDisabled]          = ImVec4(0.50f, 0.50f, 0.50f, 1.00f);
        colors[ImGuiCol_WindowBg]              = ImVec4(0.10f, 0.10f, 0.10f, 1.00f);
        colors[ImGuiCol_ChildBg]               = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
        colors[ImGuiCol_PopupBg]               = ImVec4(0.19f, 0.19f, 0.19f, 0.92f);
        colors[ImGuiCol_Border]                = ImVec4(0.19f, 0.19f, 0.19f, 0.29f);
        colors[ImGuiCol_BorderShadow]          = ImVec4(0.00f, 0.00f, 0.00f, 0.24f);
        colors[ImGuiCol_FrameBg]               = ImVec4(0.05f, 0.05f, 0.05f, 0.54f);
        colors[ImGuiCol_FrameBgHovered]        = ImVec4(0.19f, 0.19f, 0.19f, 0.54f);
        colors[ImGuiCol_FrameBgActive]         = ImVec4(0.20f, 0.22f, 0.23f, 1.00f);
        colors[ImGuiCol_TitleBg]               = ImVec4(0.00f, 0.00f, 0.00f, 1.00f);
        colors[ImGuiCol_TitleBgActive]         = ImVec4(0.06f, 0.06f, 0.06f, 1.00f);
        colors[ImGuiCol_TitleBgCollapsed]      = ImVec4(0.00f, 0.00f, 0.00f, 1.00f);
        colors[ImGuiCol_MenuBarBg]             = ImVec4(0.14f, 0.14f, 0.14f, 1.00f);
        colors[ImGuiCol_ScrollbarBg]           = ImVec4(0.05f, 0.05f, 0.05f, 0.54f);
        colors[ImGuiCol_ScrollbarGrab]         = ImVec4(0.34f, 0.34f, 0.34f, 0.54f);
        colors[ImGuiCol_ScrollbarGrabHovered]  = ImVec4(0.40f, 0.40f, 0.40f, 0.54f);
        colors[ImGuiCol_ScrollbarGrabActive]   = ImVec4(0.56f, 0.56f, 0.56f, 0.54f);
        colors[ImGuiCol_CheckMark]             = ImVec4(0.33f, 0.67f, 0.86f, 1.00f);
        colors[ImGuiCol_SliderGrab]            = ImVec4(0.34f, 0.34f, 0.34f, 0.54f);
        colors[ImGuiCol_SliderGrabActive]      = ImVec4(0.56f, 0.56f, 0.56f, 0.54f);
        colors[ImGuiCol_Button]                = ImVec4(0.05f, 0.05f, 0.05f, 0.54f);
        colors[ImGuiCol_ButtonHovered]         = ImVec4(0.19f, 0.19f, 0.19f, 0.54f);
        colors[ImGuiCol_ButtonActive]          = ImVec4(0.20f, 0.22f, 0.23f, 1.00f);
        colors[ImGuiCol_Header]                = ImVec4(0.00f, 0.00f, 0.00f, 0.52f);
        colors[ImGuiCol_HeaderHovered]         = ImVec4(0.00f, 0.00f, 0.00f, 0.36f);
        colors[ImGuiCol_HeaderActive]          = ImVec4(0.20f, 0.22f, 0.23f, 0.33f);
        colors[ImGuiCol_Separator]             = ImVec4(0.28f, 0.28f, 0.28f, 0.29f);
        colors[ImGuiCol_SeparatorHovered]      = ImVec4(0.44f, 0.44f, 0.44f, 0.29f);
        colors[ImGuiCol_SeparatorActive]       = ImVec4(0.40f, 0.44f, 0.47f, 1.00f);
        colors[ImGuiCol_ResizeGrip]            = ImVec4(0.28f, 0.28f, 0.28f, 0.29f);
        colors[ImGuiCol_ResizeGripHovered]     = ImVec4(0.44f, 0.44f, 0.44f, 0.29f);
        colors[ImGuiCol_ResizeGripActive]      = ImVec4(0.40f, 0.44f, 0.47f, 1.00f);
        colors[ImGuiCol_Tab]                   = ImVec4(0.00f, 0.00f, 0.00f, 0.52f);
        colors[ImGuiCol_TabHovered]            = ImVec4(0.14f, 0.14f, 0.14f, 1.00f);
        colors[ImGuiCol_TabActive]             = ImVec4(0.20f, 0.20f, 0.20f, 0.36f);
        colors[ImGuiCol_TabUnfocused]          = ImVec4(0.00f, 0.00f, 0.00f, 0.52f);
        colors[ImGuiCol_TabUnfocusedActive]    = ImVec4(0.14f, 0.14f, 0.14f, 1.00f);
        colors[ImGuiCol_DockingPreview]        = ImVec4(0.33f, 0.67f, 0.86f, 1.00f);
        colors[ImGuiCol_DockingEmptyBg]        = ImVec4(1.00f, 0.00f, 0.00f, 1.00f);
        colors[ImGuiCol_PlotLines]             = ImVec4(1.00f, 0.00f, 0.00f, 1.00f);
        colors[ImGuiCol_PlotLinesHovered]      = ImVec4(1.00f, 0.00f, 0.00f, 1.00f);
        colors[ImGuiCol_PlotHistogram]         = ImVec4(1.00f, 0.00f, 0.00f, 1.00f);
        colors[ImGuiCol_PlotHistogramHovered]  = ImVec4(1.00f, 0.00f, 0.00f, 1.00f);
        colors[ImGuiCol_TableHeaderBg]         = ImVec4(0.00f, 0.00f, 0.00f, 0.52f);
        colors[ImGuiCol_TableBorderStrong]     = ImVec4(0.00f, 0.00f, 0.00f, 0.52f);
        colors[ImGuiCol_TableBorderLight]      = ImVec4(0.28f, 0.28f, 0.28f, 0.29f);
        colors[ImGuiCol_TableRowBg]            = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
        colors[ImGuiCol_TableRowBgAlt]         = ImVec4(1.00f, 1.00f, 1.00f, 0.06f);
        colors[ImGuiCol_TextSelectedBg]        = ImVec4(0.20f, 0.22f, 0.23f, 1.00f);
        colors[ImGuiCol_DragDropTarget]        = ImVec4(0.33f, 0.67f, 0.86f, 1.00f);
        colors[ImGuiCol_NavHighlight]          = ImVec4(1.00f, 0.00f, 0.00f, 1.00f);
        colors[ImGuiCol_NavWindowingHighlight] = ImVec4(1.00f, 0.00f, 0.00f, 0.70f);
        colors[ImGuiCol_NavWindowingDimBg]     = ImVec4(0.80f, 0.80f, 0.80f, 0.35f);
        colors[ImGuiCol_ModalWindowDimBg]      = ImVec4(0.80f, 0.80f, 0.80f, 0.35f);
    } else if(theme == Dark) {
        ImGui::StyleColorsDark();
        ImVec4 title_bar    = ImVec4(40.0f / max, 42.0f / max, 54.0f / max, 1.0f);
        ImVec4 tab_active   = ImVec4(52.0f / max, 54.0f / max, 64.0f / max, 1.0f);
        ImVec4 tab_unactive = ImVec4(35.0f / max, 43.0f / max, 59.0f / max, 1.0f);

        s_selected_color              = ImVec4(155.0f / 255.0f, 130.0f / 255.0f, 207.0f / 255.0f, 1.00f);
        colors[ImGuiCol_Text]         = ImVec4(200.0f / 255.0f, 200.0f / 255.0f, 200.0f / 255.0f, 1.00f);
        colors[ImGuiCol_TextDisabled] = ImVec4(0.36f, 0.42f, 0.47f, 1.00f);

        s_icon_color              = colors[ImGuiCol_Text];
        colors[ImGuiCol_WindowBg] = tab_active;
        colors[ImGuiCol_ChildBg]  = tab_active;

        colors[ImGuiCol_PopupBg]        = ImVec4(42.0f / 255.0f, 38.0f / 255.0f, 47.0f / 255.0f, 1.00f);
        colors[ImGuiCol_Border]         = ImVec4(0.08f, 0.10f, 0.12f, 1.00f);
        colors[ImGuiCol_BorderShadow]   = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
        colors[ImGuiCol_FrameBg]        = ImVec4(65.0f / 255.0f, 79.0f / 255.0f, 92.0f / 255.0f, 1.00f);
        colors[ImGuiCol_FrameBgHovered] = ImVec4(0.12f, 0.20f, 0.28f, 1.00f);
        colors[ImGuiCol_FrameBgActive]  = ImVec4(0.09f, 0.12f, 0.14f, 1.00f);

        colors[ImGuiCol_TitleBg]          = title_bar;
        colors[ImGuiCol_TitleBgActive]    = title_bar;
        colors[ImGuiCol_TitleBgCollapsed] = title_bar;
        colors[ImGuiCol_MenuBarBg]        = title_bar;

        colors[ImGuiCol_ScrollbarBg]          = ImVec4(0.02f, 0.02f, 0.02f, 0.39f);
        colors[ImGuiCol_ScrollbarGrab]        = ImVec4(0.6f, 0.6f, 0.6f, 1.00f);
        colors[ImGuiCol_ScrollbarGrabActive]  = ImVec4(0.7f, 0.7f, 0.7f, 1.00f);
        colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.8f, 0.8f, 0.8f, 1.00f);

        colors[ImGuiCol_CheckMark]        = ImVec4(155.0f / 255.0f, 130.0f / 255.0f, 207.0f / 255.0f, 1.00f);
        colors[ImGuiCol_SliderGrab]       = ImVec4(155.0f / 255.0f, 130.0f / 255.0f, 207.0f / 255.0f, 1.00f);
        colors[ImGuiCol_SliderGrabActive] = ImVec4(185.0f / 255.0f, 160.0f / 255.0f, 237.0f / 255.0f, 1.00f);
        colors[ImGuiCol_Button]           = ImVec4(0.20f, 0.25f, 0.29f, 1.00f);
        colors[ImGuiCol_ButtonHovered]    = ImVec4(0.20f, 0.25f, 0.29f, 1.00f) + ImVec4(0.1f, 0.1f, 0.1f, 0.1f);
        colors[ImGuiCol_ButtonActive]     = ImVec4(0.20f, 0.25f, 0.29f, 1.00f) + ImVec4(0.1f, 0.1f, 0.1f, 0.1f);

        colors[ImGuiCol_Separator]        = ImVec4(0.20f, 0.25f, 0.29f, 1.00f);
        colors[ImGuiCol_SeparatorHovered] = ImVec4(0.10f, 0.40f, 0.75f, 0.78f);
        colors[ImGuiCol_SeparatorActive]  = ImVec4(0.10f, 0.40f, 0.75f, 1.00f);

        colors[ImGuiCol_ResizeGrip]        = ImVec4(0.26f, 0.59f, 0.98f, 0.25f);
        colors[ImGuiCol_ResizeGripHovered] = ImVec4(0.26f, 0.59f, 0.98f, 0.67f);
        colors[ImGuiCol_ResizeGripActive]  = ImVec4(0.26f, 0.59f, 0.98f, 0.95f);

        colors[ImGuiCol_PlotLines]             = ImVec4(0.61f, 0.61f, 0.61f, 1.00f);
        colors[ImGuiCol_PlotLinesHovered]      = ImVec4(1.00f, 0.43f, 0.35f, 1.00f);
        colors[ImGuiCol_PlotHistogram]         = ImVec4(0.90f, 0.70f, 0.00f, 1.00f);
        colors[ImGuiCol_PlotHistogramHovered]  = ImVec4(1.00f, 0.60f, 0.00f, 1.00f);
        colors[ImGuiCol_TextSelectedBg]        = ImVec4(0.26f, 0.59f, 0.98f, 0.35f);
        colors[ImGuiCol_DragDropTarget]        = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
        colors[ImGuiCol_NavHighlight]          = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
        colors[ImGuiCol_NavWindowingHighlight] = ImVec4(1.00f, 1.00f, 1.00f, 0.70f);
        colors[ImGuiCol_NavWindowingDimBg]     = ImVec4(0.80f, 0.80f, 0.80f, 0.20f);
        colors[ImGuiCol_ModalWindowDimBg]      = ImVec4(0.80f, 0.80f, 0.80f, 0.35f);

        colors[ImGuiCol_Header]        = tab_active + ImVec4(0.1f, 0.1f, 0.1f, 0.1f);
        colors[ImGuiCol_HeaderHovered] = tab_active + ImVec4(0.1f, 0.1f, 0.1f, 0.1f);
        colors[ImGuiCol_HeaderActive]  = tab_active + ImVec4(0.05f, 0.05f, 0.05f, 0.1f);

#ifdef IMGUI_HAS_DOCK

        colors[ImGuiCol_Tab]                = tab_unactive;
        colors[ImGuiCol_TabHovered]         = tab_active + ImVec4(0.1f, 0.1f, 0.1f, 0.1f);
        colors[ImGuiCol_TabActive]          = tab_active;
        colors[ImGuiCol_TabUnfocused]       = tab_unactive;
        colors[ImGuiCol_TabUnfocusedActive] = tab_active;
        colors[ImGuiCol_DockingEmptyBg]     = ImVec4(0.33f, 0.33f, 0.33f, 1.00f);
        colors[ImGuiCol_DockingPreview]     = ImVec4(0.33f, 0.33f, 0.33f, 1.00f);

#endif
    } else if(theme == Grey) {
        ImGui::StyleColorsDark();
        colors[ImGuiCol_Text]         = ImVec4(1.00f, 1.00f, 1.00f, 1.00f);
        colors[ImGuiCol_TextDisabled] = ImVec4(0.40f, 0.40f, 0.40f, 1.00f);
        s_icon_color                  = colors[ImGuiCol_Text];

        colors[ImGuiCol_ChildBg]               = ImVec4(0.25f, 0.25f, 0.25f, 1.00f);
        colors[ImGuiCol_WindowBg]              = ImVec4(0.25f, 0.25f, 0.25f, 1.00f);
        colors[ImGuiCol_PopupBg]               = ImVec4(0.25f, 0.25f, 0.25f, 1.00f);
        colors[ImGuiCol_Border]                = ImVec4(0.12f, 0.12f, 0.12f, 0.71f);
        colors[ImGuiCol_BorderShadow]          = ImVec4(1.00f, 1.00f, 1.00f, 0.06f);
        colors[ImGuiCol_FrameBg]               = ImVec4(0.42f, 0.42f, 0.42f, 0.54f);
        colors[ImGuiCol_FrameBgHovered]        = ImVec4(0.42f, 0.42f, 0.42f, 0.40f);
        colors[ImGuiCol_FrameBgActive]         = ImVec4(0.56f, 0.56f, 0.56f, 0.67f);
        colors[ImGuiCol_TitleBg]               = ImVec4(0.19f, 0.19f, 0.19f, 1.00f);
        colors[ImGuiCol_TitleBgActive]         = ImVec4(0.22f, 0.22f, 0.22f, 1.00f);
        colors[ImGuiCol_TitleBgCollapsed]      = ImVec4(0.17f, 0.17f, 0.17f, 0.90f);
        colors[ImGuiCol_MenuBarBg]             = ImVec4(0.335f, 0.335f, 0.335f, 1.000f);
        colors[ImGuiCol_ScrollbarBg]           = ImVec4(0.24f, 0.24f, 0.24f, 0.53f);
        colors[ImGuiCol_ScrollbarGrab]         = ImVec4(0.41f, 0.41f, 0.41f, 1.00f);
        colors[ImGuiCol_ScrollbarGrabHovered]  = ImVec4(0.52f, 0.52f, 0.52f, 1.00f);
        colors[ImGuiCol_ScrollbarGrabActive]   = ImVec4(0.76f, 0.76f, 0.76f, 1.00f);
        colors[ImGuiCol_CheckMark]             = ImVec4(0.65f, 0.65f, 0.65f, 1.00f);
        colors[ImGuiCol_SliderGrab]            = ImVec4(0.52f, 0.52f, 0.52f, 1.00f);
        colors[ImGuiCol_SliderGrabActive]      = ImVec4(0.64f, 0.64f, 0.64f, 1.00f);
        colors[ImGuiCol_Button]                = ImVec4(0.54f, 0.54f, 0.54f, 0.35f);
        colors[ImGuiCol_ButtonHovered]         = ImVec4(0.52f, 0.52f, 0.52f, 0.59f);
        colors[ImGuiCol_ButtonActive]          = ImVec4(0.76f, 0.76f, 0.76f, 1.00f);
        colors[ImGuiCol_Header]                = ImVec4(0.38f, 0.38f, 0.38f, 1.00f);
        colors[ImGuiCol_HeaderHovered]         = ImVec4(0.47f, 0.47f, 0.47f, 1.00f);
        colors[ImGuiCol_HeaderActive]          = ImVec4(0.76f, 0.76f, 0.76f, 0.77f);
        colors[ImGuiCol_Separator]             = ImVec4(0.000f, 0.000f, 0.000f, 0.137f);
        colors[ImGuiCol_SeparatorHovered]      = ImVec4(0.700f, 0.671f, 0.600f, 0.290f);
        colors[ImGuiCol_SeparatorActive]       = ImVec4(0.702f, 0.671f, 0.600f, 0.674f);
        colors[ImGuiCol_ResizeGrip]            = ImVec4(0.26f, 0.59f, 0.98f, 0.25f);
        colors[ImGuiCol_ResizeGripHovered]     = ImVec4(0.26f, 0.59f, 0.98f, 0.67f);
        colors[ImGuiCol_ResizeGripActive]      = ImVec4(0.26f, 0.59f, 0.98f, 0.95f);
        colors[ImGuiCol_PlotLines]             = ImVec4(0.61f, 0.61f, 0.61f, 1.00f);
        colors[ImGuiCol_PlotLinesHovered]      = ImVec4(1.00f, 0.43f, 0.35f, 1.00f);
        colors[ImGuiCol_PlotHistogram]         = ImVec4(0.90f, 0.70f, 0.00f, 1.00f);
        colors[ImGuiCol_PlotHistogramHovered]  = ImVec4(1.00f, 0.60f, 0.00f, 1.00f);
        colors[ImGuiCol_TextSelectedBg]        = ImVec4(0.73f, 0.73f, 0.73f, 0.35f);
        colors[ImGuiCol_ModalWindowDimBg]      = ImVec4(0.80f, 0.80f, 0.80f, 0.35f);
        colors[ImGuiCol_DragDropTarget]        = ImVec4(1.00f, 1.00f, 0.00f, 0.90f);
        colors[ImGuiCol_NavHighlight]          = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
        colors[ImGuiCol_NavWindowingHighlight] = ImVec4(1.00f, 1.00f, 1.00f, 0.70f);
        colors[ImGuiCol_NavWindowingDimBg]     = ImVec4(0.80f, 0.80f, 0.80f, 0.20f);

#ifdef IMGUI_HAS_DOCK
        colors[ImGuiCol_DockingEmptyBg]     = ImVec4(0.38f, 0.38f, 0.38f, 1.00f);
        colors[ImGuiCol_Tab]                = ImVec4(0.25f, 0.25f, 0.25f, 1.00f);
        colors[ImGuiCol_TabHovered]         = ImVec4(0.40f, 0.40f, 0.40f, 1.00f);
        colors[ImGuiCol_TabActive]          = ImVec4(0.33f, 0.33f, 0.33f, 1.00f);
        colors[ImGuiCol_TabUnfocused]       = ImVec4(0.25f, 0.25f, 0.25f, 1.00f);
        colors[ImGuiCol_TabUnfocusedActive] = ImVec4(0.33f, 0.33f, 0.33f, 1.00f);
        colors[ImGuiCol_DockingPreview]     = ImVec4(0.85f, 0.85f, 0.85f, 0.28f);
#endif
    } else if(theme == Light) {
        ImGui::StyleColorsLight();
        colors[ImGuiCol_Text]         = ImVec4(0.00f, 0.00f, 0.00f, 1.00f);
        colors[ImGuiCol_TextDisabled] = ImVec4(0.60f, 0.60f, 0.60f, 1.00f);
        s_icon_color                  = colors[ImGuiCol_Text];

        colors[ImGuiCol_WindowBg]             = ImVec4(0.94f, 0.94f, 0.94f, 0.94f);
        colors[ImGuiCol_PopupBg]              = ImVec4(1.00f, 1.00f, 1.00f, 0.94f);
        colors[ImGuiCol_Border]               = ImVec4(0.00f, 0.00f, 0.00f, 0.39f);
        colors[ImGuiCol_BorderShadow]         = ImVec4(1.00f, 1.00f, 1.00f, 0.10f);
        colors[ImGuiCol_FrameBg]              = ImVec4(1.00f, 1.00f, 1.00f, 0.94f);
        colors[ImGuiCol_FrameBgHovered]       = ImVec4(0.26f, 0.59f, 0.98f, 0.40f);
        colors[ImGuiCol_FrameBgActive]        = ImVec4(0.26f, 0.59f, 0.98f, 0.67f);
        colors[ImGuiCol_TitleBg]              = ImVec4(0.96f, 0.96f, 0.96f, 1.00f);
        colors[ImGuiCol_TitleBgCollapsed]     = ImVec4(1.00f, 1.00f, 1.00f, 0.51f);
        colors[ImGuiCol_TitleBgActive]        = ImVec4(0.82f, 0.82f, 0.82f, 1.00f);
        colors[ImGuiCol_MenuBarBg]            = ImVec4(0.86f, 0.86f, 0.86f, 1.00f);
        colors[ImGuiCol_ScrollbarBg]          = ImVec4(0.98f, 0.98f, 0.98f, 0.53f);
        colors[ImGuiCol_ScrollbarGrab]        = ImVec4(0.69f, 0.69f, 0.69f, 1.00f);
        colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.59f, 0.59f, 0.59f, 1.00f);
        colors[ImGuiCol_ScrollbarGrabActive]  = ImVec4(0.49f, 0.49f, 0.49f, 1.00f);
        colors[ImGuiCol_CheckMark]            = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
        colors[ImGuiCol_SliderGrab]           = ImVec4(0.24f, 0.52f, 0.88f, 1.00f);
        colors[ImGuiCol_SliderGrabActive]     = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
        colors[ImGuiCol_Button]               = ImVec4(0.26f, 0.59f, 0.98f, 0.40f);
        colors[ImGuiCol_ButtonHovered]        = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
        colors[ImGuiCol_ButtonActive]         = ImVec4(0.06f, 0.53f, 0.98f, 1.00f);
        colors[ImGuiCol_Header]               = ImVec4(0.26f, 0.59f, 0.98f, 0.31f);
        colors[ImGuiCol_HeaderHovered]        = ImVec4(0.26f, 0.59f, 0.98f, 0.80f);
        colors[ImGuiCol_HeaderActive]         = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
        colors[ImGuiCol_ResizeGrip]           = ImVec4(1.00f, 1.00f, 1.00f, 0.50f);
        colors[ImGuiCol_ResizeGripHovered]    = ImVec4(0.26f, 0.59f, 0.98f, 0.67f);
        colors[ImGuiCol_ResizeGripActive]     = ImVec4(0.26f, 0.59f, 0.98f, 0.95f);
        colors[ImGuiCol_PlotLines]            = ImVec4(0.39f, 0.39f, 0.39f, 1.00f);
        colors[ImGuiCol_PlotLinesHovered]     = ImVec4(1.00f, 0.43f, 0.35f, 1.00f);
        colors[ImGuiCol_PlotHistogram]        = ImVec4(0.90f, 0.70f, 0.00f, 1.00f);
        colors[ImGuiCol_PlotHistogramHovered] = ImVec4(1.00f, 0.60f, 0.00f, 1.00f);
        colors[ImGuiCol_TextSelectedBg]       = ImVec4(0.26f, 0.59f, 0.98f, 0.35f);
    } else if(theme == Cherry) {
        ImGui::StyleColorsDark();
#define HI(v) ImVec4(0.502f, 0.075f, 0.256f, v)
#define MED(v) ImVec4(0.455f, 0.198f, 0.301f, v)
#define LOW(v) ImVec4(0.232f, 0.201f, 0.271f, v)
#define BG(v) ImVec4(0.200f, 0.220f, 0.270f, v)
#define TEXTCol(v) ImVec4(0.860f, 0.930f, 0.890f, v)

        colors[ImGuiCol_Text]         = ImVec4(0.95f, 0.96f, 0.98f, 1.00f);
        colors[ImGuiCol_TextDisabled] = ImVec4(0.36f, 0.42f, 0.47f, 1.00f);
        s_icon_color                  = colors[ImGuiCol_Text];

        colors[ImGuiCol_WindowBg]             = ImVec4(0.13f, 0.14f, 0.17f, 1.00f);
        colors[ImGuiCol_PopupBg]              = BG(0.9f);
        colors[ImGuiCol_Border]               = ImVec4(0.31f, 0.31f, 1.00f, 0.00f);
        colors[ImGuiCol_BorderShadow]         = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
        colors[ImGuiCol_FrameBg]              = BG(1.00f);
        colors[ImGuiCol_FrameBgHovered]       = MED(0.78f);
        colors[ImGuiCol_FrameBgActive]        = MED(1.00f);
        colors[ImGuiCol_TitleBg]              = LOW(1.00f);
        colors[ImGuiCol_TitleBgActive]        = HI(1.00f);
        colors[ImGuiCol_TitleBgCollapsed]     = BG(0.75f);
        colors[ImGuiCol_MenuBarBg]            = BG(0.47f);
        colors[ImGuiCol_ScrollbarBg]          = BG(1.00f);
        colors[ImGuiCol_ScrollbarGrab]        = ImVec4(0.09f, 0.15f, 0.16f, 1.00f);
        colors[ImGuiCol_ScrollbarGrabHovered] = MED(0.78f);
        colors[ImGuiCol_ScrollbarGrabActive]  = MED(1.00f);
        colors[ImGuiCol_CheckMark]            = ImVec4(0.71f, 0.22f, 0.27f, 1.00f);
        colors[ImGuiCol_SliderGrab]           = ImVec4(0.47f, 0.77f, 0.83f, 0.14f);
        colors[ImGuiCol_SliderGrabActive]     = ImVec4(0.71f, 0.22f, 0.27f, 1.00f);
        colors[ImGuiCol_Button]               = ImVec4(0.47f, 0.77f, 0.83f, 0.14f);
        colors[ImGuiCol_ButtonHovered]        = MED(0.86f);
        colors[ImGuiCol_ButtonActive]         = MED(1.00f);
        colors[ImGuiCol_Header]               = MED(0.76f);
        colors[ImGuiCol_HeaderHovered]        = MED(0.86f);
        colors[ImGuiCol_HeaderActive]         = HI(1.00f);
        colors[ImGuiCol_ResizeGrip]           = ImVec4(0.47f, 0.77f, 0.83f, 0.04f);
        colors[ImGuiCol_ResizeGripHovered]    = MED(0.78f);
        colors[ImGuiCol_ResizeGripActive]     = MED(1.00f);
        colors[ImGuiCol_PlotLines]            = TEXTCol(0.63f);
        colors[ImGuiCol_PlotLinesHovered]     = MED(1.00f);
        colors[ImGuiCol_PlotHistogram]        = TEXTCol(0.63f);
        colors[ImGuiCol_PlotHistogramHovered] = MED(1.00f);
        colors[ImGuiCol_TextSelectedBg]       = MED(0.43f);
        colors[ImGuiCol_Border]               = ImVec4(0.539f, 0.479f, 0.255f, 0.162f);
        colors[ImGuiCol_TabHovered]           = colors[ImGuiCol_ButtonHovered];
    } else if(theme == Blue) {
        ImVec4 colour_for_text         = ImVec4(236.f / 255.f, 240.f / 255.f, 241.f / 255.f, 1.0f);
        ImVec4 colour_for_head         = ImVec4(41.f / 255.f, 128.f / 255.f, 185.f / 255.f, 1.0f);
        ImVec4 colour_for_area         = ImVec4(57.f / 255.f, 79.f / 255.f, 105.f / 255.f, 1.0f);
        ImVec4 colour_for_body         = ImVec4(44.f / 255.f, 62.f / 255.f, 80.f / 255.f, 1.0f);
        ImVec4 colour_for_pops         = ImVec4(33.f / 255.f, 46.f / 255.f, 60.f / 255.f, 1.0f);
        colors[ImGuiCol_Text]          = ImVec4(colour_for_text.x, colour_for_text.y, colour_for_text.z, 1.00f);
        colors[ImGuiCol_TextDisabled]  = ImVec4(colour_for_text.x, colour_for_text.y, colour_for_text.z, 0.58f);
        s_icon_color                   = colors[ImGuiCol_Text];

        colors[ImGuiCol_WindowBg]             = ImVec4(colour_for_body.x, colour_for_body.y, colour_for_body.z, 0.95f);
        colors[ImGuiCol_Border]               = ImVec4(colour_for_body.x, colour_for_body.y, colour_for_body.z, 0.00f);
        colors[ImGuiCol_BorderShadow]         = ImVec4(colour_for_body.x, colour_for_body.y, colour_for_body.z, 0.00f);
        colors[ImGuiCol_FrameBg]              = ImVec4(colour_for_area.x, colour_for_area.y, colour_for_area.z, 1.00f);
        colors[ImGuiCol_FrameBgHovered]       = ImVec4(colour_for_head.x, colour_for_head.y, colour_for_head.z, 0.78f);
        colors[ImGuiCol_FrameBgActive]        = ImVec4(colour_for_head.x, colour_for_head.y, colour_for_head.z, 1.00f);
        colors[ImGuiCol_TitleBg]              = ImVec4(colour_for_area.x, colour_for_area.y, colour_for_area.z, 1.00f);
        colors[ImGuiCol_TitleBgCollapsed]     = ImVec4(colour_for_area.x, colour_for_area.y, colour_for_area.z, 0.75f);
        colors[ImGuiCol_TitleBgActive]        = ImVec4(colour_for_head.x, colour_for_head.y, colour_for_head.z, 1.00f);
        colors[ImGuiCol_MenuBarBg]            = ImVec4(colour_for_area.x, colour_for_area.y, colour_for_area.z, 0.47f);
        colors[ImGuiCol_ScrollbarBg]          = ImVec4(colour_for_area.x, colour_for_area.y, colour_for_area.z, 1.00f);
        colors[ImGuiCol_ScrollbarGrab]        = ImVec4(colour_for_head.x, colour_for_head.y, colour_for_head.z, 0.21f);
        colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(colour_for_head.x, colour_for_head.y, colour_for_head.z, 0.78f);
        colors[ImGuiCol_ScrollbarGrabActive]  = ImVec4(colour_for_head.x, colour_for_head.y, colour_for_head.z, 1.00f);
        colors[ImGuiCol_CheckMark]            = ImVec4(colour_for_head.x, colour_for_head.y, colour_for_head.z, 0.80f);
        colors[ImGuiCol_SliderGrab]           = ImVec4(colour_for_head.x, colour_for_head.y, colour_for_head.z, 0.50f);
        colors[ImGuiCol_SliderGrabActive]     = ImVec4(colour_for_head.x, colour_for_head.y, colour_for_head.z, 1.00f);
        colors[ImGuiCol_Button]               = ImVec4(colour_for_head.x, colour_for_head.y, colour_for_head.z, 0.50f);
        colors[ImGuiCol_ButtonHovered]        = ImVec4(colour_for_head.x, colour_for_head.y, colour_for_head.z, 0.86f);
        colors[ImGuiCol_ButtonActive]         = ImVec4(colour_for_head.x, colour_for_head.y, colour_for_head.z, 1.00f);
        colors[ImGuiCol_Header]               = ImVec4(colour_for_head.x, colour_for_head.y, colour_for_head.z, 0.76f);
        colors[ImGuiCol_HeaderHovered]        = ImVec4(colour_for_head.x, colour_for_head.y, colour_for_head.z, 0.86f);
        colors[ImGuiCol_HeaderActive]         = ImVec4(colour_for_head.x, colour_for_head.y, colour_for_head.z, 1.00f);
        colors[ImGuiCol_ResizeGrip]           = ImVec4(colour_for_head.x, colour_for_head.y, colour_for_head.z, 0.15f);
        colors[ImGuiCol_ResizeGripHovered]    = ImVec4(colour_for_head.x, colour_for_head.y, colour_for_head.z, 0.78f);
        colors[ImGuiCol_ResizeGripActive]     = ImVec4(colour_for_head.x, colour_for_head.y, colour_for_head.z, 1.00f);
        colors[ImGuiCol_PlotLines]            = ImVec4(colour_for_text.x, colour_for_text.y, colour_for_text.z, 0.63f);
        colors[ImGuiCol_PlotLinesHovered]     = ImVec4(colour_for_head.x, colour_for_head.y, colour_for_head.z, 1.00f);
        colors[ImGuiCol_PlotHistogram]        = ImVec4(colour_for_text.x, colour_for_text.y, colour_for_text.z, 0.63f);
        colors[ImGuiCol_PlotHistogramHovered] = ImVec4(colour_for_head.x, colour_for_head.y, colour_for_head.z, 1.00f);
        colors[ImGuiCol_TextSelectedBg]       = ImVec4(colour_for_head.x, colour_for_head.y, colour_for_head.z, 0.43f);
        colors[ImGuiCol_PopupBg]              = ImVec4(colour_for_pops.x, colour_for_pops.y, colour_for_pops.z, 0.92f);
    } else if(theme == Classic) {
        ImGui::StyleColorsClassic();
        s_icon_color = colors[ImGuiCol_Text];
    } else if(theme == ClassicDark) {
        ImGui::StyleColorsDark();
        s_icon_color = colors[ImGuiCol_Text];
    } else if(theme == ClassicLight) {
        ImGui::StyleColorsLight();
        s_icon_color = colors[ImGuiCol_Text];
    } else if(theme == Cinder) {
        colors[ImGuiCol_Text]                 = ImVec4(0.95f, 0.96f, 0.98f, 1.00f);
        colors[ImGuiCol_TextDisabled]         = ImVec4(0.36f, 0.42f, 0.47f, 1.00f);
        s_icon_color                          = colors[ImGuiCol_Text];
        colors[ImGuiCol_WindowBg]             = ImVec4(0.13f, 0.14f, 0.17f, 1.00f);
        colors[ImGuiCol_Border]               = ImVec4(0.31f, 0.31f, 1.00f, 0.00f);
        colors[ImGuiCol_BorderShadow]         = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
        colors[ImGuiCol_FrameBg]              = ImVec4(0.20f, 0.22f, 0.27f, 1.00f);
        colors[ImGuiCol_FrameBgHovered]       = ImVec4(0.92f, 0.18f, 0.29f, 0.78f);
        colors[ImGuiCol_FrameBgActive]        = ImVec4(0.92f, 0.18f, 0.29f, 1.00f);
        colors[ImGuiCol_TitleBg]              = ImVec4(0.92f, 0.18f, 0.29f, 0.86f);
        colors[ImGuiCol_TitleBgCollapsed]     = ImVec4(0.20f, 0.22f, 0.27f, 0.75f);
        colors[ImGuiCol_TitleBgActive]        = ImVec4(0.92f, 0.18f, 0.29f, 1.00f);
        colors[ImGuiCol_MenuBarBg]            = ImVec4(0.92f, 0.18f, 0.29f, 0.86f);
        colors[ImGuiCol_ScrollbarBg]          = ImVec4(0.20f, 0.22f, 0.27f, 1.00f);
        colors[ImGuiCol_ScrollbarGrab]        = ImVec4(0.09f, 0.15f, 0.16f, 1.00f);
        colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.92f, 0.18f, 0.29f, 0.78f);
        colors[ImGuiCol_ScrollbarGrabActive]  = ImVec4(0.92f, 0.18f, 0.29f, 1.00f);
        colors[ImGuiCol_CheckMark]            = ImVec4(0.71f, 0.22f, 0.27f, 1.00f);
        colors[ImGuiCol_SliderGrab]           = ImVec4(0.47f, 0.77f, 0.83f, 0.14f);
        colors[ImGuiCol_SliderGrabActive]     = ImVec4(0.92f, 0.18f, 0.29f, 1.00f);
        colors[ImGuiCol_Button]               = ImVec4(0.47f, 0.77f, 0.83f, 0.14f);
        colors[ImGuiCol_ButtonHovered]        = ImVec4(0.92f, 0.18f, 0.29f, 0.86f);
        colors[ImGuiCol_ButtonActive]         = ImVec4(0.92f, 0.18f, 0.29f, 1.00f);
        colors[ImGuiCol_Header]               = ImVec4(0.92f, 0.18f, 0.29f, 0.76f);
        colors[ImGuiCol_HeaderHovered]        = ImVec4(0.92f, 0.18f, 0.29f, 0.86f);
        colors[ImGuiCol_HeaderActive]         = ImVec4(0.92f, 0.18f, 0.29f, 1.00f);
        colors[ImGuiCol_ResizeGrip]           = ImVec4(0.47f, 0.77f, 0.83f, 0.04f);
        colors[ImGuiCol_ResizeGripHovered]    = ImVec4(0.92f, 0.18f, 0.29f, 0.78f);
        colors[ImGuiCol_ResizeGripActive]     = ImVec4(0.92f, 0.18f, 0.29f, 1.00f);
        colors[ImGuiCol_PlotLines]            = ImVec4(0.86f, 0.93f, 0.89f, 0.63f);
        colors[ImGuiCol_PlotLinesHovered]     = ImVec4(0.92f, 0.18f, 0.29f, 1.00f);
        colors[ImGuiCol_PlotHistogram]        = ImVec4(0.86f, 0.93f, 0.89f, 0.63f);
        colors[ImGuiCol_PlotHistogramHovered] = ImVec4(0.92f, 0.18f, 0.29f, 1.00f);
        colors[ImGuiCol_TextSelectedBg]       = ImVec4(0.92f, 0.18f, 0.29f, 0.43f);
        colors[ImGuiCol_PopupBg]              = ImVec4(0.20f, 0.22f, 0.27f, 0.9f);
    } else if(theme == Dracula) {
        ImGui::StyleColorsDark();

        const ImVec4 title_bar    = ImVec4(33.0f / max, 34.0f / max, 44.0f / max, 1.0f);
        const ImVec4 tab_active   = ImVec4(40.0f / max, 42.0f / max, 54.0f / max, 1.0f);
        const ImVec4 tab_unactive = ImVec4(35.0f / max, 43.0f / max, 59.0f / max, 1.0f);

        s_icon_color                  = ImVec4(183.0f / 255.0f, 158.0f / 255.0f, 220.0f / 255.0f, 1.00f);
        s_selected_color              = ImVec4(145.0f / 255.0f, 111.0f / 255.0f, 186.0f / 255.0f, 1.00f);
        colors[ImGuiCol_Text]         = ImVec4(244.0f / 255.0f, 244.0f / 255.0f, 244.0f / 255.0f, 1.00f);
        colors[ImGuiCol_TextDisabled] = ImVec4(0.36f, 0.42f, 0.47f, 1.00f);

        colors[ImGuiCol_WindowBg] = tab_active;
        colors[ImGuiCol_ChildBg]  = tab_active;

        colors[ImGuiCol_PopupBg]        = ImVec4(42.0f / 255.0f, 38.0f / 255.0f, 47.0f / 255.0f, 1.00f);
        colors[ImGuiCol_Border]         = ImVec4(0.08f, 0.10f, 0.12f, 1.00f);
        colors[ImGuiCol_BorderShadow]   = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
        colors[ImGuiCol_FrameBg]        = ImVec4(65.0f / 255.0f, 79.0f / 255.0f, 92.0f / 255.0f, 1.00f);
        colors[ImGuiCol_FrameBgHovered] = ImVec4(0.12f, 0.20f, 0.28f, 1.00f);
        colors[ImGuiCol_FrameBgActive]  = ImVec4(0.09f, 0.12f, 0.14f, 1.00f);

        colors[ImGuiCol_TitleBg]          = title_bar;
        colors[ImGuiCol_TitleBgActive]    = title_bar;
        colors[ImGuiCol_TitleBgCollapsed] = title_bar;
        colors[ImGuiCol_MenuBarBg]        = title_bar;

        colors[ImGuiCol_ScrollbarBg]          = ImVec4(0.02f, 0.02f, 0.02f, 0.39f);
        colors[ImGuiCol_ScrollbarGrab]        = ImVec4(0.6f, 0.6f, 0.6f, 1.00f);
        colors[ImGuiCol_ScrollbarGrabActive]  = ImVec4(0.7f, 0.7f, 0.7f, 1.00f);
        colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.8f, 0.8f, 0.8f, 1.00f);

        colors[ImGuiCol_CheckMark]        = ImVec4(155.0f / 255.0f, 130.0f / 255.0f, 207.0f / 255.0f, 1.00f);
        colors[ImGuiCol_SliderGrab]       = ImVec4(155.0f / 255.0f, 130.0f / 255.0f, 207.0f / 255.0f, 1.00f);
        colors[ImGuiCol_SliderGrabActive] = ImVec4(185.0f / 255.0f, 160.0f / 255.0f, 237.0f / 255.0f, 1.00f);
        colors[ImGuiCol_Button]           = ImVec4(0.20f, 0.25f, 0.29f, 1.00f);
        colors[ImGuiCol_ButtonHovered]    = ImVec4(59.0f / 255.0f, 46.0f / 255.0f, 80.0f / 255.0f, 1.0f);
        colors[ImGuiCol_ButtonActive]     = colors[ImGuiCol_ButtonHovered] + ImVec4(0.1f, 0.1f, 0.1f, 0.1f);

        colors[ImGuiCol_Separator]        = ImVec4(0.20f, 0.25f, 0.29f, 1.00f);
        colors[ImGuiCol_SeparatorHovered] = ImVec4(0.10f, 0.40f, 0.75f, 0.78f);
        colors[ImGuiCol_SeparatorActive]  = ImVec4(0.10f, 0.40f, 0.75f, 1.00f);

        colors[ImGuiCol_ResizeGrip]        = ImVec4(0.26f, 0.59f, 0.98f, 0.25f);
        colors[ImGuiCol_ResizeGripHovered] = ImVec4(0.26f, 0.59f, 0.98f, 0.67f);
        colors[ImGuiCol_ResizeGripActive]  = ImVec4(0.26f, 0.59f, 0.98f, 0.95f);

        colors[ImGuiCol_PlotLines]             = ImVec4(0.61f, 0.61f, 0.61f, 1.00f);
        colors[ImGuiCol_PlotLinesHovered]      = ImVec4(1.00f, 0.43f, 0.35f, 1.00f);
        colors[ImGuiCol_PlotHistogram]         = ImVec4(0.90f, 0.70f, 0.00f, 1.00f);
        colors[ImGuiCol_PlotHistogramHovered]  = ImVec4(1.00f, 0.60f, 0.00f, 1.00f);
        colors[ImGuiCol_TextSelectedBg]        = ImVec4(0.26f, 0.59f, 0.98f, 0.35f);
        colors[ImGuiCol_DragDropTarget]        = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
        colors[ImGuiCol_NavHighlight]          = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
        colors[ImGuiCol_NavWindowingHighlight] = ImVec4(1.00f, 1.00f, 1.00f, 0.70f);
        colors[ImGuiCol_NavWindowingDimBg]     = ImVec4(0.80f, 0.80f, 0.80f, 0.20f);
        colors[ImGuiCol_ModalWindowDimBg]      = ImVec4(0.80f, 0.80f, 0.80f, 0.35f);
        colors[ImGuiCol_Header]        = tab_active + ImVec4(0.1f, 0.1f, 0.1f, 0.1f);
        colors[ImGuiCol_HeaderHovered] = tab_active + ImVec4(0.1f, 0.1f, 0.1f, 0.1f);
        colors[ImGuiCol_HeaderActive]  = tab_active + ImVec4(0.05f, 0.05f, 0.05f, 0.1f);

#ifdef IMGUI_HAS_DOCK

        colors[ImGuiCol_Tab]                = tab_unactive;
        colors[ImGuiCol_TabHovered]         = tab_active + ImVec4(0.1f, 0.1f, 0.1f, 0.1f);
        colors[ImGuiCol_TabActive]          = tab_active;
        colors[ImGuiCol_TabUnfocused]       = tab_unactive;
        colors[ImGuiCol_TabUnfocusedActive] = tab_active;
        colors[ImGuiCol_DockingEmptyBg]     = ImVec4(0.33f, 0.33f, 0.33f, 1.00f);
        colors[ImGuiCol_DockingPreview]     = ImVec4(0.33f, 0.33f, 0.33f, 1.00f);

#endif
    }

    colors[ImGuiCol_Separator]        = colors[ImGuiCol_TitleBg];
    colors[ImGuiCol_SeparatorActive]  = colors[ImGuiCol_Separator];
    colors[ImGuiCol_SeparatorHovered] = colors[ImGuiCol_Separator];

    colors[ImGuiCol_Tab]          = colors[ImGuiCol_MenuBarBg];
    colors[ImGuiCol_TabUnfocused] = colors[ImGuiCol_MenuBarBg];

    colors[ImGuiCol_TabUnfocusedActive] = colors[ImGuiCol_WindowBg];
    colors[ImGuiCol_TabActive]          = colors[ImGuiCol_WindowBg];
    colors[ImGuiCol_ChildBg]            = colors[ImGuiCol_TabActive];
    colors[ImGuiCol_ScrollbarBg]        = colors[ImGuiCol_TabActive];

    colors[ImGuiCol_TitleBgActive]    = colors[ImGuiCol_TitleBg];
    colors[ImGuiCol_TitleBgCollapsed] = colors[ImGuiCol_TitleBg];
    colors[ImGuiCol_MenuBarBg]        = colors[ImGuiCol_TitleBg];
    colors[ImGuiCol_PopupBg]          = colors[ImGuiCol_WindowBg] + ImVec4(0.05f, 0.05f, 0.05f, 0.0f);

    colors[ImGuiCol_Border]       = ImVec4(0.08f, 0.10f, 0.12f, 0.00f);
    colors[ImGuiCol_BorderShadow] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
}

void ImGuiUtilities::DrawItemActivityOutline(float rounding, bool draw_when_inactive, ImColor color_when_active) {
    auto* draw_list = ImGui::GetWindowDrawList();

    ImRect expanded_rect = ImRect(ImGui::GetItemRectMin(), ImGui::GetItemRectMax());
    expanded_rect.Min.x -= 1.0f;
    expanded_rect.Min.y -= 1.0f;
    expanded_rect.Max.x += 1.0f;
    expanded_rect.Max.y += 1.0f;

    const ImRect rect = expanded_rect;
    if(ImGui::IsItemHovered() && !ImGui::IsItemActive()) {
        draw_list->AddRect(rect.Min, rect.Max,
                           ImColor(60, 60, 60), rounding, 0, 1.5f);
    }
    if(ImGui::IsItemActive()) {
        draw_list->AddRect(rect.Min, rect.Max,
                           color_when_active, rounding, 0, 1.0f);
    } else if(!ImGui::IsItemHovered() && draw_when_inactive) {
        draw_list->AddRect(rect.Min, rect.Max,
                           ImColor(50, 50, 50), rounding, 0, 1.0f);
    }
}

}

