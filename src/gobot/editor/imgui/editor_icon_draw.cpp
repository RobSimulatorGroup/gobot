/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/editor/imgui/type_icons.hpp"

#include <algorithm>
#include <limits>

#include "imgui.h"

namespace gobot {

void DrawEditorIcon(const EditorIcon& icon, ImVec2 size) {
    const ImVec2 origin = ImGui::GetCursorScreenPos();
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    const float scale = std::min(size.x, size.y);
    const ImVec2 center(origin.x + size.x * 0.5f, origin.y + size.y * 0.5f);

    if (icon.kind == EditorIconKind::Axis3D) {
        const float radius = scale * 0.10f;
        const ImVec2 x_end(origin.x + size.x * 0.80f, origin.y + size.y * 0.28f);
        const ImVec2 y_end(origin.x + size.x * 0.50f, origin.y + size.y * 0.84f);
        const ImVec2 z_end(origin.x + size.x * 0.22f, origin.y + size.y * 0.33f);
        draw_list->AddLine(center, x_end, IM_COL32(235, 74, 74, 255), 2.0f);
        draw_list->AddLine(center, y_end, IM_COL32(70, 210, 88, 255), 2.0f);
        draw_list->AddLine(center, z_end, IM_COL32(64, 138, 255, 255), 2.0f);
        draw_list->AddCircleFilled(center, radius, IM_COL32(222, 228, 235, 255), 8);
        draw_list->AddCircleFilled(x_end, radius * 0.72f, IM_COL32(235, 74, 74, 255), 8);
        draw_list->AddCircleFilled(y_end, radius * 0.72f, IM_COL32(70, 210, 88, 255), 8);
        draw_list->AddCircleFilled(z_end, radius * 0.72f, IM_COL32(64, 138, 255, 255), 8);
        ImGui::Dummy(size);
        return;
    }

    if (icon.kind == EditorIconKind::Joint3D) {
        const float radius = scale * 0.16f;
        const float small_radius = scale * 0.055f;
        const float line_width = std::max(1.5f, scale * 0.11f);
        const ImVec2 left(origin.x + size.x * 0.20f, origin.y + size.y * 0.68f);
        const ImVec2 right(origin.x + size.x * 0.80f, origin.y + size.y * 0.32f);
        const ImU32 center_fill = IM_COL32(42, 48, 54, 255);
        draw_list->AddLine(left, center, icon.color, line_width);
        draw_list->AddLine(center, right, icon.color, line_width);
        draw_list->AddCircleFilled(left, small_radius, icon.color, 8);
        draw_list->AddCircleFilled(right, small_radius, icon.color, 8);
        draw_list->AddCircleFilled(center, radius, center_fill, 16);
        draw_list->AddCircle(center, radius, icon.color, 16, std::max(1.5f, scale * 0.10f));
        ImGui::Dummy(size);
        return;
    }

    ImFont* font = ImGui::GetFont();
    const float font_size = std::max(1.0f, scale);
    const ImVec2 text_size = font->CalcTextSizeA(font_size,
                                                 std::numeric_limits<float>::max(),
                                                 0.0f,
                                                 icon.glyph);
    const ImVec2 text_pos(origin.x + (size.x - text_size.x) * 0.5f,
                          origin.y + (size.y - text_size.y) * 0.5f);
    draw_list->AddText(font, font_size, text_pos, icon.color, icon.glyph);
    ImGui::Dummy(size);
}

} // namespace gobot
