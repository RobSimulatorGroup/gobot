/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/editor/imgui/type_icons.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "imgui.h"

namespace gobot {

namespace {

void DrawIconArrow(ImDrawList* draw_list,
                   const ImVec2& start,
                   const ImVec2& end,
                   ImU32 color,
                   float thickness,
                   float head_length,
                   float head_half_width) {
    const float dx = end.x - start.x;
    const float dy = end.y - start.y;
    const float length = std::sqrt(dx * dx + dy * dy);
    if (length <= 1.0e-4f) {
        return;
    }

    const float ux = dx / length;
    const float uy = dy / length;
    const float px = -uy;
    const float py = ux;
    const ImVec2 line_end{end.x - ux * head_length, end.y - uy * head_length};
    const ImVec2 left{end.x - ux * head_length + px * head_half_width,
                      end.y - uy * head_length + py * head_half_width};
    const ImVec2 right{end.x - ux * head_length - px * head_half_width,
                       end.y - uy * head_length - py * head_half_width};

    draw_list->AddLine(start, line_end, color, thickness);
    draw_list->AddTriangleFilled(end, left, right, color);
}

} // namespace

void DrawEditorIcon(const EditorIcon& icon, ImVec2 size) {
    const ImVec2 origin = ImGui::GetCursorScreenPos();
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    const float scale = std::min(size.x, size.y);
    const ImVec2 center(origin.x + size.x * 0.5f, origin.y + size.y * 0.5f);

    if (icon.kind == EditorIconKind::Axis3D) {
        const float line_width = std::max(2.0f, scale * 0.13f);
        const float head_length = std::max(5.0f, scale * 0.24f);
        const float head_half_width = std::max(3.0f, scale * 0.15f);
        const float radius = std::max(3.0f, scale * 0.16f);
        const ImVec2 axis_origin(origin.x + size.x * 0.28f, origin.y + size.y * 0.60f);
        const ImVec2 z_end(origin.x + size.x * 0.28f, origin.y + size.y * 0.04f);
        const ImVec2 y_end(origin.x + size.x * 0.88f, origin.y + size.y * 0.22f);
        const ImVec2 x_end(origin.x + size.x * 0.86f, origin.y + size.y * 0.88f);

        DrawIconArrow(draw_list, axis_origin, z_end, IM_COL32(0, 163, 225, 255), line_width, head_length, head_half_width);
        DrawIconArrow(draw_list, axis_origin, y_end, IM_COL32(139, 199, 73, 255), line_width, head_length, head_half_width);
        DrawIconArrow(draw_list, axis_origin, x_end, IM_COL32(255, 18, 18, 255), line_width, head_length, head_half_width);
        draw_list->AddCircleFilled(axis_origin, radius, IM_COL32(250, 252, 255, 255), 16);
        draw_list->AddCircle(axis_origin, radius, IM_COL32(12, 36, 45, 255), 16, std::max(1.4f, scale * 0.08f));
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
