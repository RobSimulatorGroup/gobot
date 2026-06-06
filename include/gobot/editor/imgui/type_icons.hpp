/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * This file is created by Qiqi Wu, 23-3-29
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <string_view>

#include "gobot_export.h"
#include "gobot/core/types.hpp"

struct ImVec2;

namespace gobot {

class Node;

enum class EditorIconKind {
    Font,
    Axis3D,
    Joint3D,
};

struct EditorIcon {
    const char* glyph = "";
    EditorIconKind kind = EditorIconKind::Font;
    std::uint32_t color = 0xFFE6E6E6;
};

GOBOT_EXPORT const char* GetTypeIcon(const Type& type);
GOBOT_EXPORT const char* GetTypeIcon(std::string_view type_name);
GOBOT_EXPORT const char* GetNodeIcon(const Node& node);
GOBOT_EXPORT const char* GetResourcePathIcon(std::string_view path, bool is_directory);
GOBOT_EXPORT EditorIcon GetTypeEditorIcon(const Type& type);
GOBOT_EXPORT EditorIcon GetTypeEditorIcon(std::string_view type_name);
GOBOT_EXPORT EditorIcon GetNodeEditorIcon(const Node& node);
GOBOT_EXPORT EditorIcon GetResourcePathEditorIcon(std::string_view path, bool is_directory);
GOBOT_EXPORT void DrawEditorIcon(const EditorIcon& icon, ImVec2 size);


}
