/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-3-10
*/

#pragma once

#include "gobot/editor/imgui/editor_panel.hpp"
#include "gobot/core/ref_counted.hpp"
#include "gobot/core/color.hpp"
#include "imgui.h"

namespace gobot {

class ConsoleMessage : public RefCounted {
    GOBCLASS(ConsoleMessage, RefCounted)
public:
    enum Level : uint32_t
    {
        Trace    = 1,
        Debug    = 2,
        Info     = 4,
        Warn     = 8,
        Error    = 16,
        Critical = 32,
    };

public:
    ConsoleMessage(const String& message = "", Level level = Level::Trace, String  source = "", int thread_id = 0);

    void OnImGUIRender();

    void IncreaseCount() { count_++; };

    [[nodiscard]] size_t GetMessageID() const { return message_id_; }

    static const char* GetLevelName(Level level);

    static const char* GetLevelIcon(Level level);

    static Color GetRenderColor(Level level);

public:
    const String message_;
    const Level level_;
    const String source_;
    const int thread_id_;
    int count_ = 1;
    size_t message_id_;
};


class ConsolePanel : public EditorPanel
{
public:

    ConsolePanel();

    ~ConsolePanel() = default;

    static void Flush();

    void OnImGui() override;

    static void AddMessage(const Ref<ConsoleMessage>& message);

private:
    void ImGuiRenderHeader();

    void ImGuiRenderMessages();

private:
    friend class ConsoleMessage;

    static uint16_t s_message_buffer_capacity;
    static uint16_t s_message_buffer_size;
    static uint16_t s_message_buffer_begin;
    static std::vector<Ref<ConsoleMessage>> s_message_buffer;
    static bool s_allow_scrolling_to_bottom;
    static bool s_request_scroll_to_bottom;
    static uint32_t s_message_buffer_render_filter;
    ImGuiTextFilter filter_;
};

}